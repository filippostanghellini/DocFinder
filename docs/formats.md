# Supported Formats

DocFinder supports a wide range of document formats. Each format is handled by a dedicated parser with automatic text extraction and page detection.

| Format | Extension | Parser | Page Model |
|--------|-----------|--------|------------|
| PDF | `.pdf` | PyMuPDF | Real pages (1-based) |
| Word | `.docx` | python-docx | Virtual pages (10 paragraphs) |
| Word (legacy) | `.doc` | olefile | Virtual pages (~3000 chars) |
| OpenDocument Text | `.odt` | odfpy | Virtual pages (10 paragraphs) |
| OpenDocument Presentation | `.odp` | odfpy | Virtual pages (10 paragraphs) |
| OpenDocument Drawing | `.odg` | odfpy | Virtual pages (10 paragraphs) |
| PowerPoint | `.pptx` | python-pptx | Slide numbers |
| PowerPoint (legacy) | `.ppt` | olefile | Virtual pages (~3000 chars) |
| HTML | `.html`, `.htm` | beautifulsoup4 | Virtual pages (~3000 chars) |
| EPUB | `.epub` | zipfile + beautifulsoup4 | Chapter numbers |
| Markdown | `.md` | Custom parser | Section numbers (headings) |
| Plain Text | `.txt` | Built-in | Virtual pages (~3000 chars) |

## Lazy Loading

All format-specific libraries are loaded lazily. If a library is not installed, DocFinder logs a warning and skips files of that type rather than crashing.

## Text Extraction Details

### PDF

PDFs are parsed with PyMuPDF. Tables are detected automatically and rendered as Markdown tables in the extracted text to preserve structure. Text blocks overlapping with tables are excluded to avoid duplication.

### Legacy Binary Formats (.doc, .ppt)

These formats use the OLE2 compound document structure (via `olefile`):

- **.doc** — Heuristic extraction tries UTF-16-LE decoding at multiple offsets within the `WordDocument` stream, falling back to printable ASCII extraction.
- **.ppt** — A recursive binary record parser walks the `PowerPoint Document` stream looking for `TextCharsAtom` and `TextBytesAtom` records.

### EPUB

EPUB parsing follows the standard OCF container structure:

1. Read `META-INF/container.xml` to locate the OPF file
2. Parse the OPF manifest and spine
3. Read each spine item in order
4. Extract clean text via BeautifulSoup

## Dependency Table

| Format | Required Library | Availability |
|--------|-----------------|--------------|
| PDF | PyMuPDF | Included by default |
| DOCX | python-docx | Included by default |
| DOC, PPT | olefile | Included by default |
| ODT, ODP, ODG | odfpy | Included by default |
| PPTX | python-pptx | Included by default |
| HTML, EPUB | beautifulsoup4 | Included by default |
