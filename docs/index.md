<div class="df-page">

  <!-- ═══ HERO ═══════════════════════════════════════════════════════════════ -->
  <section class="df-hero df-wrap">
    <a class="df-pill-announce" href="changelog/">
      <span class="df-dot-live"></span>
      <span>DocFinder 2.4.0 · Google EmbeddingGemma 2 Multimodal + Hybrid Search + Local Qwen3.5</span>
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M9 6l6 6-6 6"/></svg>
    </a>

    <h1>
      Find anything inside your files.<br/>
      <span class="df-grad">Ask it questions, 100% on your machine.</span>
    </h1>

    <p class="df-hero-sub">
      DocFinder indexes your PDFs, Office documents, spreadsheets, emails, source code, scanned pages, images, and recordings into one local vector space powered by <strong>Google DeepMind EmbeddingGemma 2</strong>, and lets you chat with them offline using <strong>Qwen3.5</strong> or <strong>Ollama</strong>.
    </p>

    <div class="df-cta-group">
      <a class="df-btn df-btn-primary" href="https://github.com/filippostanghellini/DocFinder/releases/latest">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v11m-4.5-4.5L12 15l4.5-4.5M5 19.5h14"/></svg>
        Download for macOS
      </a>
      <a class="df-btn" href="https://github.com/filippostanghellini/DocFinder/releases/latest">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v11m-4.5-4.5L12 15l4.5-4.5M5 19.5h14"/></svg>
        Windows (64-bit)
      </a>
      <a class="df-btn" href="https://github.com/filippostanghellini/DocFinder/releases/latest">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v11m-4.5-4.5L12 15l4.5-4.5M5 19.5h14"/></svg>
        Linux AppImage
      </a>
      <a class="df-btn" href="install/">
        Installation Guide
      </a>
    </div>

    <p class="df-meta-line">
      v2.4.0 · Apple Silicon (Metal), NVIDIA (CUDA), AMD (ROCm) &amp; CPU · Free &amp; Open Source (AGPL-3.0) · Zero cloud telemetry
    </p>
  </section>

  <!-- ═══ MOTION VIDEO SHOWCASE (docs/assets/docfinder-motion.mp4) ═══════════ -->
  <section class="df-motion-wrap df-wrap">
    <div class="df-window-frame">
      <video class="df-motion-video" src="assets/docfinder-motion.mp4" autoplay loop muted playsinline controls poster="images/search.png"></video>
    </div>
  </section>


  <!-- ═══ INTERACTIVE 2D SHARED VECTOR SPACE ════════════════════════════════ -->
  <section class="df-section df-wrap df-reveal">
    <div class="df-section-head">
      <span class="df-eyebrow">Shared 768-D Coordinate Space</span>
      <h2>One model for text, scans, sound, and code</h2>
      <p>
        <strong>EmbeddingGemma 2</strong> places passages, spreadsheets, source code, photos, scanned pages, and 30-second audio windows into the same 768-dimensional space across 100+ languages. Select a query below to inspect its nearest vectors:
      </p>
    </div>

    <div class="df-space-wrap">
      <div class="df-space-canvas">
        <div id="df-space-glow" class="df-space-glow" style="left:26%;top:34%;"></div>
        <span id="df-space-pin" class="df-space-pin" style="left:26%;top:34%;">termination notice period</span>

        <span class="df-space-cluster" style="left:25%;top:20%;">Contracts &amp; Emails<small>PDF · DOCX · EML</small></span>
        <span class="df-space-cluster" style="left:69%;top:18%;">Scans, Photos &amp; Sheets<small>Vision +170M · XLSX</small></span>
        <span class="df-space-cluster" style="left:75%;top:82%;">Meetings &amp; Media<small>Audio +300M (30s windows)</small></span>
        <span class="df-space-cluster" style="left:29%;top:84%;">Repositories &amp; Configs<small>Python · TS · Rust · SQL</small></span>

        <svg class="df-space-svg" viewBox="0 0 1000 360" preserveAspectRatio="none" aria-hidden="true">
          <defs>
            <pattern id="df-grid" width="50" height="50" patternUnits="userSpaceOnUse">
              <path d="M 50 0 L 0 0 0 50" fill="none" stroke="rgba(255,255,255,0.035)" stroke-width="1"/>
            </pattern>
          </defs>
          <rect width="1000" height="360" fill="url(#df-grid)"/>

          <g fill="#818cf8" opacity="0.65">
            <circle cx="195" cy="88" r="4"/><circle cx="210" cy="98" r="4.5"/><circle cx="235" cy="82" r="3.5"/>
            <circle cx="280" cy="104" r="4"/><circle cx="305" cy="112" r="4.5"/><circle cx="245" cy="165" r="4.5"/>
            <circle cx="220" cy="148" r="3.5"/><circle cx="290" cy="142" r="4"/>
          </g>
          <g fill="#38bdf8" opacity="0.7">
            <circle cx="620" cy="82" r="4"/><circle cx="640" cy="92" r="4.5"/><circle cx="670" cy="76" r="3.5"/>
            <circle cx="710" cy="96" r="4"/><circle cx="725" cy="106" r="4.5"/><circle cx="695" cy="155" r="4.5"/>
            <circle cx="655" cy="140" r="3.5"/><circle cx="745" cy="134" r="4"/>
          </g>
          <g fill="#fbbf24" opacity="0.7">
            <circle cx="685" cy="215" r="4"/><circle cx="700" cy="225" r="4.5"/><circle cx="730" cy="208" r="3.5"/>
            <circle cx="770" cy="222" r="4"/><circle cx="785" cy="235" r="4.5"/><circle cx="755" cy="285" r="4.5"/>
            <circle cx="715" cy="275" r="3.5"/><circle cx="805" cy="268" r="4"/>
          </g>
          <g fill="#34d399" opacity="0.7">
            <circle cx="245" cy="225" r="4"/><circle cx="265" cy="235" r="4.5"/><circle cx="290" cy="218" r="3.5"/>
            <circle cx="330" cy="228" r="4"/><circle cx="345" cy="242" r="4.5"/><circle cx="300" cy="295" r="4.5"/>
            <circle cx="275" cy="280" r="3.5"/><circle cx="360" cy="275" r="4"/>
          </g>

          <g id="df-space-lines"></g>
        </svg>
      </div>

      <div class="df-space-controls">
        <div class="df-space-queries">
          <button type="button" class="df-space-qbtn active" data-df-space-q="0">termination notice period</button>
          <button type="button" class="df-space-qbtn" data-df-space-q="1">ricevuta ristorante Milano IVA (Cross-Lingual + Vision)</button>
          <button type="button" class="df-space-qbtn" data-df-space-q="2">database migration rollback risks (Audio 30s)</button>
          <button type="button" class="df-space-qbtn" data-df-space-q="3">code: retry exponential backoff</button>
        </div>
        <div id="df-space-hits" class="df-space-hits-row"></div>
      </div>
    </div>
  </section>

  <!-- ═══ MODULAR ARCHITECTURE BENTO GRID ═══════════════════════════════════ -->
  <section class="df-section df-wrap df-reveal">
    <div class="df-section-head">
      <span class="df-eyebrow">Modular Architecture</span>
      <h2>Load only the encoders you need. Never re-index.</h2>
      <p>
        Start lean with the 270M Text &amp; Code base (~250 MB RAM). Enable Vision or Audio at any time: because all three towers project into the exact same vector space, your existing index stays valid.
      </p>
    </div>

    <div class="df-bento">
      <div class="df-card">
        <div>
          <div class="df-card-head">
            <span class="df-card-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M6.5 3.5h7.5l4.5 4.5v12.5h-12z"/><path d="M13.5 3.5v5h5M9 13h6M9 16.5h4"/></svg>
            </span>
            <span class="df-card-pill">270M Params · ~250 MB RAM</span>
          </div>
          <h3>Text, Office &amp; Code (Base)</h3>
          <p>
            Always active. Indexes digital PDFs (converting tables to Markdown), Word, Excel (<code>.xlsx</code>, <code>.ods</code>, <code>.csv</code>), PowerPoint, EPUB, RTF, emails (<code>.eml</code>, <code>.mbox</code>), and 20+ programming languages with line numbers.
          </p>
        </div>
        <div class="df-ram-bar"><span style="width: 34%"></span></div>
      </div>

      <div class="df-card">
        <div>
          <div class="df-card-head">
            <span class="df-card-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3.5" y="4.5" width="17" height="15" rx="2.5"/><path d="M4 16.5l4.6-4.6 3.9 3.9 2.2-2.2 5 5"/><circle cx="15.6" cy="9.2" r="1.4" fill="currentColor" stroke="none"/></svg>
            </span>
            <span class="df-card-pill">+170M Params · +190 MB RAM</span>
          </div>
          <h3>Scanned PDFs &amp; Images (+Vision)</h3>
          <p>
            Optional toggle. Detects image-only/scanned PDF pages automatically, renders them via PyMuPDF, and embeds standalone photos, diagrams, and screenshots (<code>.png</code>, <code>.jpg</code>, <code>.webp</code>, <code>.tiff</code>) without external OCR.
          </p>
        </div>
        <div class="df-ram-bar"><span style="width: 58%"></span></div>
      </div>

      <div class="df-card">
        <div>
          <div class="df-card-head">
            <span class="df-card-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4.5 10.5v3M8.25 7.5v9M12 4.5v15M15.75 7.5v9M19.5 10.5v3"/></svg>
            </span>
            <span class="df-card-pill">+300M Params · +310 MB RAM</span>
          </div>
          <h3>Audio &amp; Video (+Audio)</h3>
          <p>
            Optional toggle. Slices recordings (<code>.wav</code>, <code>.mp3</code>, <code>.m4a</code>, <code>.flac</code>) and video soundtracks (<code>.mp4</code>, <code>.mov</code>, <code>.mkv</code>) into overlapping 30-second windows with timestamps.
          </p>
        </div>
        <div class="df-ram-bar"><span style="width: 92%"></span></div>
      </div>
    </div>
  </section>

  <!-- ═══ BUILT FOR DEEP WORK ═══════════════════════════════════════════════ -->
  <section class="df-section df-wrap df-reveal">
    <div class="df-section-head">
      <span class="df-eyebrow">Built for Deep Work</span>
      <h2>Every feature right where you need it</h2>
      <p>
        From grounded local RAG conversations to custom Ollama servers and strict per-document privacy enforcement.
      </p>
    </div>

    <div class="df-bento-2">
      <div class="df-card">
        <div>
          <div class="df-card-head">
            <span class="df-card-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
            </span>
            <span class="df-card-pill">Qwen3.5 · 9B / 4B / 2B</span>
          </div>
          <h3>Grounded Local AI Chat</h3>
          <p>Ask questions directly against any search result. DocFinder expands context along real PDF pages and Markdown tables so answers stay accurate and verifiable.</p>
        </div>
      </div>

      <div class="df-card">
        <div>
          <div class="df-card-head">
            <span class="df-card-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="4" width="18" height="6" rx="1.5"/><rect x="3" y="14" width="18" height="6" rx="1.5"/><circle cx="7" cy="7" r="1" fill="currentColor"/><circle cx="7" cy="17" r="1" fill="currentColor"/></svg>
            </span>
            <span class="df-card-pill">Local or Remote</span>
          </div>
          <h3>Bring Your Own Models (Ollama)</h3>
          <p>Connect to a local or remote Ollama server with one-click connection testing, independent embedding/chat model selection, and automatic modality tag mapping.</p>
        </div>
      </div>

      <div class="df-card">
        <div>
          <div class="df-card-head">
            <span class="df-card-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3.5 5 6.2v5.6c0 4.2 3 7.5 7 8.7 4-1.2 7-4.5 7-8.7V6.2z"/><path d="m9 12 2.2 2.2L15.2 10"/></svg>
            </span>
            <span class="df-card-pill">Strict Local Guard</span>
          </div>
          <h3>100% Privacy Shield Mode</h3>
          <p>Mark sensitive folders during indexing: DocFinder enforces local-only embeddings and blocks remote LLMs from ever accessing privacy-shielded documents.</p>
        </div>
      </div>

      <div class="df-card">
        <div>
          <div class="df-card-head">
            <span class="df-card-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3 7a2 2 0 0 1 2-2h4l2 2.5h8a2 2 0 0 1 2 2V17a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>
            </span>
            <span class="df-card-pill">Vector + Lexical + Rerank</span>
          </div>
          <h3>Folder Filters &amp; Instant Hybrid Ranking</h3>
          <p>Filter search results across indexed directories in real time without re-indexing, combining vector similarity, exact-word boosting, and cross-encoder reranking.</p>
        </div>
      </div>
    </div>
  </section>

  <!-- ═══ MADE FOR THE KEYBOARD ═════════════════════════════════════════════ -->
  <section class="df-section df-wrap df-reveal">
    <div class="df-section-head">
      <span class="df-eyebrow">Keyboard-First Navigation</span>
      <h2>Summon from anywhere</h2>
      <p>Keep your hands on the keyboard. Open the floating Spotlight bar over any application and jump straight to your file.</p>
    </div>

    <div class="df-kbd-grid">
      <div class="df-kbd-card">
        <div class="df-keycaps">
          <span class="df-keycap">⌥</span><span class="df-keycap">D</span>
          <span style="font-size:0.72rem;color:var(--df-text-3);margin-left:0.35rem;">or Alt + D</span>
        </div>
        <strong>Summon Spotlight Overlay</strong>
        <span style="font-size:0.8rem;color:var(--df-text-2);">Brings up the floating quick-search panel from any app (customizable in Settings).</span>
      </div>

      <div class="df-kbd-card">
        <div class="df-keycaps">
          <span class="df-keycap">↩</span>
          <span style="font-size:0.72rem;color:var(--df-text-3);margin-left:0.35rem;">Enter</span>
        </div>
        <strong>Open at Page or Line</strong>
        <span style="font-size:0.8rem;color:var(--df-text-2);">Opens documents, or launches VS Code, Cursor, or Zed at the exact matched line.</span>
      </div>

      <div class="df-kbd-card">
        <div class="df-keycaps">
          <span class="df-keycap">⌘</span><span class="df-keycap">↩</span>
          <span style="font-size:0.72rem;color:var(--df-text-3);margin-left:0.35rem;">or Ctrl + Enter</span>
        </div>
        <strong>Show in Finder / Explorer</strong>
        <span style="font-size:0.8rem;color:var(--df-text-2);">Reveals and highlights the file directly inside your system file manager.</span>
      </div>

      <div class="df-kbd-card">
        <div class="df-keycaps">
          <span class="df-keycap">⌘</span><span class="df-keycap">K</span>
          <span style="font-size:0.72rem;color:var(--df-text-3);margin-left:0.35rem;">or Ctrl + K</span>
        </div>
        <strong>Focus Search Input</strong>
        <span style="font-size:0.8rem;color:var(--df-text-2);">Jumps immediately to the search bar from any tab in the main window.</span>
      </div>
    </div>
  </section>

</div>
