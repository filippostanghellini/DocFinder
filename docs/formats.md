<div class="df-page">

  <section class="df-page-hero df-wrap">
    <span class="df-eyebrow">Structure-Aware Multimodal Ingestion</span>
    <h1>Supported <span class="df-grad">Formats</span></h1>
    <p class="df-hero-sub">
      Every file type is parsed into its natural structural unit: real PDF pages, slides, worksheets, email messages, line-tracked source code, rendered page scans, and 30-second audio windows.
    </p>
  </section>

  <!-- ═══ FORMAT CATEGORIES BENTO ═══════════════════════════════════════════ -->
  <section class="df-wrap" style="margin-bottom: 3rem;">
    <div class="df-bento">

      <div class="df-card">
        <div>
          <div class="df-card-head">
            <span class="df-card-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M6.5 3.5h7.5l4.5 4.5v12.5h-12z"/><path d="M13.5 3.5v5h5M9 13h6M9 16.5h4"/></svg>
            </span>
            <span class="df-card-pill">Text Base · 270M</span>
          </div>
          <h3>Documents &amp; E-Books</h3>
          <p>
            PDFs preserve 1-based page numbers and convert embedded tables to Markdown. Word and OpenDocument group paragraphs; EPUB splits by spine chapter; Markdown splits by heading section.
          </p>
          <div class="df-ext-list">
            <span class="df-ext">.pdf</span><span class="df-ext">.docx</span><span class="df-ext">.doc</span>
            <span class="df-ext">.odt</span><span class="df-ext">.rtf</span><span class="df-ext">.epub</span>
            <span class="df-ext">.md</span><span class="df-ext">.txt</span><span class="df-ext">.rst</span><span class="df-ext">.tex</span>
          </div>
        </div>
      </div>

      <div class="df-card">
        <div>
          <div class="df-card-head">
            <span class="df-card-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3.5" y="4.5" width="17" height="15" rx="2"/><path d="M3.5 9.5h17M3.5 14.5h17M9.5 4.5v15"/></svg>
            </span>
            <span class="df-card-pill">Text Base · 270M</span>
          </div>
          <h3>Spreadsheets &amp; Presentations</h3>
          <p>
            Excel workbooks are parsed sheet by sheet using zero-dependency XML streaming; CSV/TSV files become 40-row Markdown tables; PowerPoint indexes slide by slide.
          </p>
          <div class="df-ext-list">
            <span class="df-ext">.xlsx</span><span class="df-ext">.ods</span><span class="df-ext">.csv</span>
            <span class="df-ext">.tsv</span><span class="df-ext">.pptx</span><span class="df-ext">.ppt</span>
            <span class="df-ext">.odp</span><span class="df-ext">.odg</span>
          </div>
        </div>
      </div>

      <div class="df-card">
        <div>
          <div class="df-card-head">
            <span class="df-card-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="m3 7 9 6 9-6"/></svg>
            </span>
            <span class="df-card-pill">Text Base · 270M</span>
          </div>
          <h3>Emails, Web &amp; Structured Data</h3>
          <p>
            Extracts <code>Subject</code>, <code>From</code>, <code>To</code>, <code>Date</code> headers and plain-text bodies from individual emails and multi-message mbox archives, plus clean HTML and config files.
          </p>
          <div class="df-ext-list">
            <span class="df-ext">.eml</span><span class="df-ext">.mbox</span><span class="df-ext">.html</span>
            <span class="df-ext">.htm</span><span class="df-ext">.json</span><span class="df-ext">.yaml</span>
            <span class="df-ext">.yml</span><span class="df-ext">.xml</span><span class="df-ext">.log</span>
          </div>
        </div>
      </div>

      <div class="df-card">
        <div>
          <div class="df-card-head">
            <span class="df-card-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M8.5 7 3.5 12l5 5M15.5 7l5 5-5 5"/></svg>
            </span>
            <span class="df-card-pill">Line-Tracked · Code Task</span>
          </div>
          <h3>Source Code Repositories</h3>
          <p>
            Chunked along line boundaries with exact <code>line_start</code> and <code>line_end</code> metadata. Prefix queries with <code>code:</code> for asymmetric code retrieval and direct opening at <code>file:line</code>.
          </p>
          <div class="df-ext-list">
            <span class="df-ext">.py</span><span class="df-ext">.ts</span><span class="df-ext">.tsx</span>
            <span class="df-ext">.js</span><span class="df-ext">.jsx</span><span class="df-ext">.rs</span>
            <span class="df-ext">.go</span><span class="df-ext">.java</span><span class="df-ext">.c</span>
            <span class="df-ext">.cpp</span><span class="df-ext">.h</span><span class="df-ext">.hpp</span>
            <span class="df-ext">.cs</span><span class="df-ext">.swift</span><span class="df-ext">.kt</span>
            <span class="df-ext">.rb</span><span class="df-ext">.php</span><span class="df-ext">.sh</span>
            <span class="df-ext">.sql</span><span class="df-ext">.toml</span><span class="df-ext">.ini</span>
          </div>
        </div>
      </div>

      <div class="df-card">
        <div>
          <div class="df-card-head">
            <span class="df-card-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3.5" y="4.5" width="17" height="15" rx="2.5"/><path d="M4 16.5l4.6-4.6 3.9 3.9 2.2-2.2 5 5"/><circle cx="15.6" cy="9.2" r="1.4" fill="currentColor" stroke="none"/></svg>
            </span>
            <span class="df-card-pill">+Vision Module · +170M</span>
          </div>
          <h3>Scanned PDFs &amp; Standalone Images</h3>
          <p>
            Embeds standalone images and automatically detects image-only PDF pages (invoices, receipts, scanned contracts), rendering them at 150 DPI via PyMuPDF without external OCR.
          </p>
          <div class="df-ext-list">
            <span class="df-ext">Scanned .pdf</span><span class="df-ext">.png</span><span class="df-ext">.jpg</span>
            <span class="df-ext">.jpeg</span><span class="df-ext">.webp</span><span class="df-ext">.tiff</span>
            <span class="df-ext">.tif</span><span class="df-ext">.bmp</span><span class="df-ext">.gif</span>
          </div>
        </div>
      </div>

      <div class="df-card">
        <div>
          <div class="df-card-head">
            <span class="df-card-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4.5 10.5v3M8.25 7.5v9M12 4.5v15M15.75 7.5v9M19.5 10.5v3"/></svg>
            </span>
            <span class="df-card-pill">+Audio Module · +300M</span>
          </div>
          <h3>Audio &amp; Video Recordings</h3>
          <p>
            Segments recordings into overlapping 30-second windows (25s stride) with <code>start_sec</code> and <code>M:SS</code> timestamp badges so results point to the exact moment.
          </p>
          <div class="df-ext-list">
            <span class="df-ext">.wav</span><span class="df-ext">.mp3</span><span class="df-ext">.m4a</span>
            <span class="df-ext">.flac</span><span class="df-ext">.ogg</span><span class="df-ext">.mp4</span>
            <span class="df-ext">.mov</span><span class="df-ext">.mkv</span><span class="df-ext">.webm</span>
          </div>
        </div>
      </div>

    </div>
  </section>

  <!-- ═══ SMART EXCLUSIONS & PRE-SCAN GUARD ═════════════════════════════════ -->
  <section class="df-section df-wrap">
    <div class="df-section-head">
      <span class="df-eyebrow">Clean Index by Default</span>
      <h2>Smart Exclusions &amp; Pre-Scan Modality Warnings</h2>
      <p>DocFinder keeps your index fast and clean without manual configuration.</p>
    </div>

    <div class="df-bento-2">
      <div class="df-card">
        <div>
          <div class="df-card-head">
            <span class="df-card-pill">Automatic Filtering</span>
          </div>
          <h3>Skips Vendored Code, Cloud Placeholders &amp; Secrets</h3>
          <p>
            Recursive scanning automatically ignores hidden paths, Office temporary files (<code>~$*</code>), un-downloaded cloud placeholders (<code>*.icloud</code>), private keys (<code>*.pem</code>, <code>*.key</code>, <code>id_rsa</code>, <code>id_ed25519</code>), lockfiles/minified bundles (<code>package-lock.json</code>, <code>*.min.js</code>), and dependency directories (<code>node_modules</code>, <code>venv</code>, <code>__pycache__</code>, <code>target</code>, <code>build</code>, <code>dist</code>).
          </p>
        </div>
      </div>

      <div class="df-card">
        <div>
          <div class="df-card-head">
            <span class="df-card-pill">Pre-Index Inspection</span>
          </div>
          <h3>Missing-Modality Detection</h3>
          <p>
            Before indexing starts, DocFinder probes candidate folders for standalone images, scanned PDFs, and audio/video files. If any require an inactive modality encoder (<code>+Vision</code> or <code>+Audio</code>), you get a one-click prompt to enable it on the spot.
          </p>
        </div>
      </div>
    </div>
  </section>

</div>
