"""Generate deterministic, synthetic attachment samples; contains no personal data.

Run with the project Python environment after installing Pillow, reportlab, pypdf.
The optional encrypted helper returns bytes for tests without a saved third PDF.
"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


FACTS = {
    "document_type": "Private-school fee change notice",
    "previous_tuition_sar": 10000,
    "new_tuition_sar": 12000,
    "change_percent": 20,
    "issue": "A parent disputes the increase and requests review of tuition fees.",
}


def image_font(size: int) -> ImageFont.ImageFont:
    # Bundled Pillow's default font is portable and does not use personal files.
    return ImageFont.load_default(size=size)


def generate_image(path: Path) -> None:
    image = Image.new("RGB", (1275, 1650), "#f4f7f8")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((90, 90, 1185, 1560), radius=24, fill="white")
    draw.rounded_rectangle((90, 90, 1185, 220), radius=24, fill="#143e43")
    draw.rectangle((90, 160, 1185, 220), fill="#143e43")
    draw.text((145, 127), "SYNTHETIC TEST DOCUMENT", font=image_font(40), fill="white")
    draw.text((145, 290), "Private-school fee", font=image_font(54), fill="#143e43")
    draw.text((145, 360), "change notice", font=image_font(54), fill="#143e43")
    draw.text((145, 480), "Generic private school", font=image_font(33), fill="#42555b")
    draw.text((145, 538), "Fictional data for software verification only", font=image_font(29), fill="#627279")
    draw.line((145, 635, 1130, 635), fill="#d8e3e6", width=3)
    for y, label, value in [
        (710, "Previous annual tuition", "SAR 10,000"),
        (820, "New annual tuition", "SAR 12,000"),
        (930, "Fee increase", "20%"),
    ]:
        draw.text((145, y), label, font=image_font(33), fill="#42555b")
        draw.text((825, y), value, font=image_font(33), fill="#143e43")
    draw.line((145, 1050, 1130, 1050), fill="#d8e3e6", width=3)
    draw.text((145, 1125), "Complaint context", font=image_font(34), fill="#143e43")
    for y, text in [
        (1190, "A parent disputes the tuition increase"),
        (1238, "and requests review of private-school fees."),
    ]:
        draw.text((145, y), text, font=image_font(31), fill="#42555b")
    draw.text((145, 1450), "No personal data. No official government document.", font=image_font(25), fill="#627279")
    image.save(path, format="PNG", optimize=True)


def generate_text_pdf(path: Path) -> None:
    pdf = canvas.Canvas(str(path), pagesize=letter, invariant=1, pageCompression=1)
    pdf.setTitle("Synthetic private-school fee notice")
    pdf.setAuthor("Wajhah synthetic test fixtures")
    pdf.setFillColor(colors.HexColor("#143e43"))
    pdf.rect(40, 690, 532, 62, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 17)
    pdf.drawString(58, 715, "SYNTHETIC TEST DOCUMENT")
    pdf.setFillColor(colors.HexColor("#143e43"))
    pdf.setFont("Helvetica-Bold", 24)
    pdf.drawString(58, 640, "Private-school fee change notice")
    pdf.setFillColor(colors.HexColor("#42555b"))
    pdf.setFont("Helvetica", 13)
    pdf.drawString(58, 601, "Generic private school")
    pdf.drawString(58, 575, "Fictional data for software verification only")
    pdf.setStrokeColor(colors.HexColor("#d8e3e6"))
    pdf.line(58, 547, 554, 547)
    for y, label, value in [
        (504, "Previous annual tuition", "SAR 10,000"),
        (460, "New annual tuition", "SAR 12,000"),
        (416, "Fee increase", "20%"),
    ]:
        pdf.drawString(58, y, label)
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(430, y, value)
        pdf.setFont("Helvetica", 13)
    pdf.line(58, 385, 554, 385)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(58, 345, "Complaint context")
    pdf.setFont("Helvetica", 13)
    pdf.drawString(58, 313, "A parent disputes the tuition increase")
    pdf.drawString(58, 288, "and requests review of private-school fees.")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(58, 74, "No personal data. No official government document.")
    pdf.showPage()
    pdf.save()


def generate_image_pdf(image_path: Path, pdf_path: Path) -> None:
    pdf = canvas.Canvas(str(pdf_path), pagesize=letter, invariant=1, pageCompression=1)
    pdf.setTitle("Synthetic image-only private-school fee notice")
    pdf.setAuthor("Wajhah synthetic test fixtures")
    pdf.drawImage(ImageReader(str(image_path)), 0, 0, width=612, height=792)
    pdf.showPage()
    pdf.save()


def encrypted_pdf_bytes(pdf_path: Path, password: str = "synthetic-test-password") -> bytes:
    """Return an encrypted sample in memory, for rejection tests only."""
    writer = PdfWriter()
    writer.append(PdfReader(str(pdf_path)))
    writer.encrypt(password)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def generate(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_image(output_dir / "image.png")
    Image.new("RGB", (600, 600), "white").save(output_dir / "blank.png", optimize=True)
    generate_text_pdf(output_dir / "text-based.pdf")
    generate_image_pdf(output_dir / "image.png", output_dir / "image-only.pdf")
    # Correct magic prefixes with invalid bodies exercise real decoding/parsing.
    (output_dir / "corrupt.pdf").write_bytes(b"%PDF-1.7\nSYNTHETIC INVALID PDF - no objects or trailer\n")
    (output_dir / "corrupt.png").write_bytes(b"\x89PNG\r\n\x1a\nSYNTHETIC INVALID PNG - no chunks\n")
    manifest = {"synthetic": True, "contains_personal_data": False, "facts": FACTS, "files": {}}
    for name in ["image.png", "image-only.pdf", "text-based.pdf", "blank.png", "corrupt.pdf", "corrupt.png"]:
        path = output_dir / name
        raw = path.read_bytes()
        entry = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        if name in {"image-only.pdf", "text-based.pdf"}:
            reader = PdfReader(path)
            entry.update({
                "pages": len(reader.pages),
                "text_characters": sum(len(page.extract_text() or "") for page in reader.pages),
                "embedded_images": sum(len(page.images) for page in reader.pages),
                "encrypted": reader.is_encrypted,
            })
        elif name in {"image.png", "blank.png"}:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                entry.update({"width": image.width, "height": image.height, "format": image.format})
        manifest["files"][name] = entry
    (output_dir / "fixture-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    options = parser.parse_args()
    print(json.dumps(generate(options.output_dir), indent=2))
