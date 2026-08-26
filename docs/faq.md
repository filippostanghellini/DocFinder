# FAQ

## General

### What is DocFinder?

DocFinder is a local-first semantic search tool for your documents. It indexes your files and lets you find them by meaning, not just keywords. It also includes an AI chat feature for asking questions about your documents.

### Is my data sent to the cloud?

No. Everything runs on your machine. Your documents never leave your computer.

### What document formats are supported?

The main ones: PDF, Word (DOCX/DOC), PowerPoint (PPTX/PPT), OpenDocument (ODT/ODP/ODG), HTML, EPUB, Markdown, and plain text — plus more. See the [Formats](formats.md) page for the complete list.

### Is it free?

Yes, DocFinder is open source under the AGPL-3.0 license.

## Installation & Setup

### Why does macOS show "unidentified developer"?

DocFinder is unsigned open-source software. Right-click the app and select **Open**, then click **Open** in the dialog. You only need to do this on first launch.

### Why does Windows SmartScreen block the installer?

Click **More info → Run anyway**. This is normal for open-source apps without a code signing certificate.

### Can I run DocFinder on a headless server?

Yes. Use the web interface (`make run-web`). It listens on `127.0.0.1` by default — pass `--host 0.0.0.0` to reach it from another machine.

## Indexing

### How long does indexing take?

It depends on the number and size of your documents. Indexing is CPU and memory intensive because each document is parsed and each chunk is embedded. DocFinder uses parallel processing on multi-core systems to speed this up.

### Can I index files across multiple directories?

Yes. You can select multiple folders from the application interface.

### Does DocFinder follow symlinks?

Symlinked files are indexed when they match a supported format. Symlinked directories are not recursed into.

### Why was my index cleared?

DocFinder records which embedding model built the index (in a small `meta` table). Vectors from different models are not comparable — so if the model changes, for example after upgrading DocFinder, the index is wiped automatically and everything is re-embedded on the next indexing run.

## Search

### How accurate is the search?

Semantic search finds documents by meaning, so it can find relevant results even when they don't contain your exact search terms. For example, searching "buy groceries" might find documents about "shopping list" or "weekly meal plan."

### Can I search by keywords only?

DocFinder uses semantic search by default. If you need exact keyword matching, the AI chat feature can incorporate your query more literally.

## AI Chat

### Do I need an internet connection for the AI chat?

No. The AI model runs locally on your machine.

### Why is the first chat response slow?

The first request loads the LLM model into memory, which can take 10–30 seconds depending on your hardware. Subsequent responses are faster.

### Which AI model does DocFinder use?

DocFinder uses Qwen3.5 models in GGUF format via `llama-cpp-python`. The model size is selected automatically based on your available RAM (9B / 4B / 2B tiers).

## Troubleshooting

### DocFinder won't start

1. Make sure you have Python 3.10+ installed
2. Try running from source: `make setup && make run`
3. Check the logs for error messages

### Indexing fails for some files

DocFinder logs errors for individual files without stopping the entire indexing process. Check the console output for messages about specific files.

### "odfpy not installed" warning

The format libraries ship as core dependencies, so this warning means the installation is incomplete or corrupted — reinstall DocFinder:

```bash
pip install --force-reinstall -e .
```

Similar warnings exist for the other format libraries (`python-docx`, `olefile`, `python-pptx`, `beautifulsoup4`). See [Formats](formats.md#dependency-table) for the full list.

### How do I reset the index?

Delete the database file (default location: `~/Documents/DocFinder/docfinder.db`, or `data/docfinder.db` if that already exists) and re-index.
