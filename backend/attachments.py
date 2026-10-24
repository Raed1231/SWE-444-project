"""Shared size, signature and real-parser validation for all attachment endpoints."""
from dataclasses import dataclass
import io
import logging
from pathlib import Path
import warnings

from fastapi import APIRouter, UploadFile, File
from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader
from pypdf.errors import PyPdfError
from pypdf import filters
from pypdf.generic import ArrayObject, StreamObject
from starlette.concurrency import run_in_threadpool

from observability import measure
from service_errors import ServiceError

router = APIRouter()
MAX_SIZE = 5 * 1024 * 1024
MAX_PAGES = 20
MAX_IMAGE_PIXELS = 20_000_000
MAX_PDF_DECODED_BYTES = 16 * 1024 * 1024
filters.ZLIB_MAX_OUTPUT_LENGTH = 8 * 1024 * 1024
ALLOWED = {'.pdf': 'application/pdf', '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}
logging.getLogger('pypdf').setLevel(logging.ERROR)


@dataclass(frozen=True)
class Attachment:
    content: bytes
    mime_type: str
    size: int
    pages: int | None = None


def validate_content(content, filename, content_type=None):
    extension = Path(filename or '').suffix.lower()
    if extension not in ALLOWED or content_type not in (None, '', 'application/octet-stream', ALLOWED.get(extension)):
        raise ServiceError('UNSUPPORTED_FILE_TYPE', 400, 'اختر ملف JPG أو PNG أو PDF صالحًا.')
    if len(content) > MAX_SIZE:
        raise ServiceError('ATTACHMENT_TOO_LARGE', 413, 'حجم المرفق يتجاوز 5 ميجابايت.')
    if not content:
        raise ServiceError('EMPTY_FILE', 400, 'المرفق فارغ. اختر ملفًا آخر.')
    signature = b'%PDF-' if extension == '.pdf' else b'\x89PNG\r\n\x1a\n' if extension == '.png' else b'\xff\xd8\xff'
    if not content.startswith(signature):
        raise ServiceError('INVALID_FILE_CONTENT', 400, 'محتوى الملف لا يطابق امتداده.')
    if extension == '.pdf':
        try:
            reader = PdfReader(io.BytesIO(content), strict=True)
            if reader.is_encrypted:
                raise ServiceError('ENCRYPTED_PDF', 400, 'ملف PDF محمي. أرسل نسخة غير مشفرة.')
            pages = len(reader.pages)
            if not 1 <= pages <= MAX_PAGES:
                raise ServiceError('PDF_PAGE_LIMIT', 400, 'أرسل PDF يحتوي من صفحة واحدة إلى 20 صفحة.')
            # Force page resource/content resolution to reject header-only or
            # broken page trees locally, before any provider request.
            decoded_bytes = 0
            for page in reader.pages:
                if float(page.mediabox.width) <= 0 or float(page.mediabox.height) <= 0:
                    raise ValueError('Invalid page dimensions')
                contents = page.get('/Contents')
                if contents is not None:
                    contents = contents.get_object()
                    streams = contents if isinstance(contents, ArrayObject) else [contents]
                    for item in streams:
                        stream = item.get_object()
                        if not isinstance(stream, StreamObject):
                            raise ValueError('Invalid PDF content stream')
                        decoded_bytes += len(stream.get_data())
                        if decoded_bytes > MAX_PDF_DECODED_BYTES:
                            raise ValueError('PDF decompression budget exceeded')
        except ServiceError:
            raise
        except (PyPdfError, ValueError, TypeError, KeyError, AttributeError, IndexError, RecursionError, OverflowError):
            raise ServiceError('INVALID_PDF', 400, 'تعذّرت قراءة بنية PDF. أعد تصدير الملف ثم حاول.') from None
        return Attachment(content, ALLOWED[extension], len(content), pages)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                if image.format != ('PNG' if extension == '.png' else 'JPEG') or image.width * image.height > MAX_IMAGE_PIXELS or getattr(image, 'n_frames', 1) != 1:
                    raise ValueError('Unsupported image geometry or format')
                image.verify()
            with Image.open(io.BytesIO(content)) as image:
                image.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ServiceError('INVALID_IMAGE', 400, 'الصورة تالفة أو أبعادها كبيرة جدًا. أعد حفظها ثم حاول.') from None
    return Attachment(content, ALLOWED[extension], len(content))


async def read_attachment(file):
    try:
        with measure('attachment_validation'):
            content = await file.read(MAX_SIZE + 1)
            return await run_in_threadpool(validate_content, content, file.filename, file.content_type)
    finally:
        await file.close()


@router.post('/attachments')
async def upload_attachment(file: UploadFile = File(...)):
    filename = file.filename
    attachment = await read_attachment(file)
    return {'status': 'accepted', 'filename': filename, 'size': attachment.size, 'type': attachment.mime_type}
