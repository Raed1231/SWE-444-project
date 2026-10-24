# Synthetic attachment fixtures

These files contain fictional, generic English text and no personal data. They
are not user-provided evidence or official government documents. The readable
samples all say that private-school annual tuition changed from SAR 10,000 to
SAR 12,000 (20%), and a parent requests review of that fee increase.

Regenerate with Python after installing Pillow, reportlab, and pypdf:

```sh
python generate_fixtures.py
```

The generator's default output directory is its own directory. Use
`--output-dir /path/to/fixtures` to choose a different destination. It uses
ReportLab invariant mode and verifies PDF/image structure when writing the
manifest. A repeated local generation produced identical bytes for all three
readable samples.

| File | Purpose | Inspection |
| --- | --- | --- |
| image.png | Readable synthetic evidence | 1275 x 1650 PNG, 82,683 bytes; Pillow verify succeeds |
| image-only.pdf | The same image embedded in a PDF | 1 page, 99,810 bytes, zero extracted text, one embedded image |
| text-based.pdf | Same facts as selectable PDF text | 1 page, 2,075 bytes, 360 extracted text characters, no images |
| blank.png | Valid but uninformative evidence | 600 x 600 all-white PNG, 1,884 bytes |
| corrupt.pdf | Invalid PDF body with valid magic prefix | pypdf raises PdfStreamError |
| corrupt.png | Invalid PNG body with valid magic prefix | Pillow verify raises OSError |

`fixture-manifest.json` records exact sizes, SHA-256 hashes, dimensions, and PDF
structure. Corrupt fixtures are intentionally rejected by real parsers even
though the file signatures are correct; signature-only validation is inadequate.

`encrypted_pdf_bytes(Path("text-based.pdf"))` returns an encrypted PDF in memory
using a fixed synthetic test password. It was verified as encrypted and readable
with that password. No third PDF is saved by default.

Both PDF pages were rendered and visually inspected at 1200-pixel height. Text,
fee numbers, complaint context, and the synthetic/no-personal-data labels are
readable, with no overlap or clipping. `image-only-preview.png` used Poppler.
Poppler could not find a writable font cache for the text PDF in this sandbox,
so `text-based-preview.png` used pypdfium2 for the successful visual inspection.
Preview images are local QA artifacts and need not be copied to backend tests.

No Gemini request was made while generating or checking these fixtures. A
successful local parsing check does not imply a successful AI analysis.
