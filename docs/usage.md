# Usage

DocFinder provides a **desktop GUI** and a **web interface**.

## Desktop GUI

Launch the native desktop application:

```bash
make run
```

The GUI provides:

- Document browser and indexer
- Semantic search bar with relevance results
- AI chat panel for asking questions about your documents
- Global shortcut hotkey support

### Indexing Documents

Open the application and use the built-in file browser to select documents or folders. DocFinder will automatically discover all supported files and index them.

### Searching

Type your query in the search bar. Results are ranked by semantic relevance and displayed with page numbers and document titles.

### Global Hotkey

The global shortcut (configurable in Settings) lets you bring DocFinder to the front from anywhere. The default hotkey is `<alt>+d` on all platforms.

## Web Interface

Launch the web interface:

```bash
make run-web
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

The web UI mirrors the desktop GUI features. By default it listens on `127.0.0.1` (local only); start it with `docfinder web --host 0.0.0.0` to access it from other devices on your network.
