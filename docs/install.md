<div class="df-page">

  <section class="df-page-hero df-wrap">
    <span class="df-eyebrow">Get Started in Seconds</span>
    <h1>Install <span class="df-grad">DocFinder</span></h1>
    <p class="df-hero-sub">
      Download a standalone desktop installer for macOS, Windows, or Linux, or run directly from source with automatic GPU acceleration.
    </p>
  </section>

  <!-- ═══ OS DOWNLOAD CARDS ═════════════════════════════════════════════════ -->
  <section class="df-wrap" style="margin-bottom: 3rem;">
    <div class="df-bento">

      <div class="df-card">
        <div>
          <div class="df-card-head">
            <span class="df-card-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M18.71 19.5c-.83 1.24-1.71 2.45-3.05 2.47-1.34.03-1.77-.79-3.29-.79-1.53 0-2 .77-3.27.82-1.31.05-2.3-1.32-3.14-2.53C4.25 17 2.94 12.45 4.7 9.39c.87-1.52 2.43-2.48 4.12-2.51 1.28-.02 2.5.87 3.29.87.78 0 2.26-1.07 3.81-.91.65.03 2.47.26 3.64 1.98-.09.06-2.17 1.28-2.15 3.81.03 3.02 2.65 4.03 2.68 4.04-.03.07-.42 1.44-1.38 2.83M15.96 6.32c.67-.81 1.12-1.94 1-3.07-.96.04-2.13.64-2.82 1.45-.61.7-.15 1.85-1.01 2.96 1.07.08 2.16-.53 2.83-1.34"/></svg>
            </span>
            <span class="df-card-pill">macOS 13+ · Metal GPU</span>
          </div>
          <h3>macOS (.dmg)</h3>
          <p>
            Native Cocoa window with floating Spotlight overlay panel (<code>NSPanel</code> + global shortcut) and PyTorch MPS hardware acceleration on Apple Silicon.
          </p>
        </div>
        <a class="df-btn df-btn-primary" style="justify-content:center;width:100%;" href="https://github.com/filippostanghellini/DocFinder/releases/latest">
          Download DocFinder-macOS.dmg
        </a>
      </div>

      <div class="df-card">
        <div>
          <div class="df-card-head">
            <span class="df-card-icon">
              <svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor"><path d="M3 5.55 10.94 4.4v7.15H3V5.55zm0 12.9L10.94 19.6v-7.15H3v6zm8.82 1.28L21 21v-8.55h-9.18v7.28zm0-15.46v7.28H21V3l-9.18 1.27z"/></svg>
            </span>
            <span class="df-card-pill">Windows 10 / 11 (64-bit)</span>
          </div>
          <h3>Windows (.exe)</h3>
          <p>
            Standalone desktop installer with NVIDIA CUDA auto-detection, system-wide hotkey listener, and native Explorer integration.
          </p>
        </div>
        <a class="df-btn df-btn-primary" style="justify-content:center;width:100%;" href="https://github.com/filippostanghellini/DocFinder/releases/latest">
          Download Windows Setup.exe
        </a>
      </div>

      <div class="df-card">
        <div>
          <div class="df-card-head">
            <span class="df-card-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="m7 9 3 3-3 3m5 0h5"/></svg>
            </span>
            <span class="df-card-pill">x86_64 · AppImage</span>
          </div>
          <h3>Linux (.AppImage)</h3>
          <p>
            Portable single-file binary supporting NVIDIA CUDA, AMD ROCm, and multi-process CPU indexing out of the box.
          </p>
        </div>
        <a class="df-btn df-btn-primary" style="justify-content:center;width:100%;" href="https://github.com/filippostanghellini/DocFinder/releases/latest">
          Download Linux AppImage
        </a>
      </div>

    </div>
  </section>

  <!-- ═══ RUN FROM SOURCE & CLI ═════════════════════════════════════════════ -->
  <section class="df-section df-wrap">
    <div class="df-section-head">
      <span class="df-eyebrow">Developer Workflow</span>
      <h2>Run from Source or CLI</h2>
      <p>Requires Python 3.10+ and <code>make</code>. Setup creates an isolated <code>.venv</code> and installs the desktop, web, and CLI tools.</p>
    </div>

    <div class="df-bento-2 df-bento-2-centered">
      <div class="df-card">
        <div>
          <div class="df-card-head">
            <span class="df-card-pill">1 · Clone &amp; Setup</span>
          </div>
          <h3>Quick Setup via Make</h3>
          <p>Installs core dependencies, FastAPI web server, and pywebview desktop GUI:</p>
          <pre class="df-pv-code" style="margin-bottom:0;">git clone https://github.com/filippostanghellini/DocFinder.git
cd DocFinder
make setup     # creates .venv and installs [dev,web,gui]
make run       # launch desktop GUI
make run-web   # launch web UI at http://127.0.0.1:8000</pre>
        </div>
      </div>

      <div class="df-card">
        <div>
          <div class="df-card-head">
            <span class="df-card-pill">2 · Optional Extras</span>
          </div>
          <h3>Enable Local AI Chat (RAG) &amp; CLI</h3>
          <p>Install the <code>[rag]</code> extra to enable local GGUF inference with Qwen3.5 via <code>llama-cpp-python</code>:</p>
          <pre class="df-pv-code" style="margin-bottom:0;">source .venv/bin/activate
pip install -e ".[rag]"

# Index with Vision and Audio towers enabled
docfinder index ~/Documents --modalities text,image,audio
docfinder search "termination clause"</pre>
        </div>
      </div>
    </div>
  </section>

  <!-- ═══ HARDWARE ACCELERATION MATRIX ══════════════════════════════════════ -->
  <section class="df-section df-wrap">
    <div class="df-section-head">
      <span class="df-eyebrow">Zero Configuration Hardware Tuning</span>
      <h2>Automatic GPU &amp; Precision Selection</h2>
      <p>DocFinder inspects your GPU and available system RAM at startup, selecting safe <code>bfloat16</code> or <code>float32</code> precision and adaptive batch sizes (4–64 chunks).</p>
    </div>

    <div class="df-bento">
      <div class="df-card">
        <div>
          <div class="df-card-head">
            <span class="df-card-pill">Apple Silicon (M1–M4)</span>
          </div>
          <h3>PyTorch MPS + Metal</h3>
          <p>Uses Apple's Metal Performance Shaders (<code>mps</code>) in <code>bfloat16</code> for 10–30× faster embedding generation and Metal offload for Qwen3.5.</p>
        </div>
      </div>

      <div class="df-card">
        <div>
          <div class="df-card-head">
            <span class="df-card-pill">NVIDIA &amp; AMD GPUs</span>
          </div>
          <h3>CUDA &amp; ROCm</h3>
          <p>Automatically detects NVIDIA CUDA or AMD ROCm devices, scaling embedding batch sizes up to 64 on systems with 16 GB+ RAM.</p>
        </div>
      </div>

      <div class="df-card">
        <div>
          <div class="df-card-head">
            <span class="df-card-pill">RAM-Aware RAG Tiers</span>
          </div>
          <h3>Qwen3.5 Auto-Selection</h3>
          <p>Selects <strong>Qwen3.5-9B</strong> (16 GB+ RAM), <strong>Qwen3.5-4B</strong> (8–16 GB RAM), or <strong>Qwen3.5-2B</strong> (any machine) automatically.</p>
        </div>
      </div>
    </div>
  </section>

</div>
