/* ═══════════════════════════════════════════════════════════════════════════
   DocFinder Interactive Website Controller (SVG Icons, Zero Emoji Slop)
   ═══════════════════════════════════════════════════════════════════════════ */

(function () {
  const ICONS = {
    doc: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M6.5 3.5h7.5l4.5 4.5v12.5h-12z"/><path d="M13.5 3.5v5h5M9 13h6M9 16.5h4"/></svg>',
    scan: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3.5" y="4.5" width="17" height="15" rx="2.5"/><path d="M4 16.5l4.6-4.6 3.9 3.9 2.2-2.2 5 5"/><circle cx="15.6" cy="9.2" r="1.4" fill="currentColor" stroke="none"/></svg>',
    audio: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4.5 10.5v3M8.25 7.5v9M12 4.5v15M15.75 7.5v9M19.5 10.5v3"/></svg>',
    code: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8.5 7 3.5 12l5 5M15.5 7l5 5-5 5"/></svg>',
    sheet: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><rect x="3.5" y="4.5" width="17" height="15" rx="2"/><path d="M3.5 9.5h17M3.5 14.5h17M9.5 4.5v15"/></svg>',
  };

  const SCENES = {
    pdf: {
      query: "the termination clause and notice period in the consulting contract",
      mode: "Hybrid · Vector + Lexical",
      caption: "Finds the exact clause across long PDFs and highlights the page, ready for one-click AI Chat.",
      footCount: "3 results · 94% top match",
      items: [
        {
          group: "Documents",
          icon: ICONS.doc,
          title: "Master_Consulting_Agreement_2026.pdf",
          tag: "Page 7 · 94%",
          sub: "~/Documents/Legal/Contracts",
          active: true,
        },
        {
          group: "Documents",
          icon: ICONS.doc,
          title: "Vendor_SLA_Addendum.docx",
          tag: "Page 3 · 81%",
          sub: "~/Documents/Legal/Contracts",
          active: false,
        },
        {
          group: "Messages",
          icon: ICONS.doc,
          title: "Re_Contract_Renewal_Terms.eml",
          tag: "Msg 1 · 76%",
          sub: "~/Mail/Legal",
          active: false,
        },
      ],
      previewHtml: `
        <div>
          <div class="df-pv-card">
            <div style="font-size:0.66rem;color:#9ca3af;margin-bottom:0.35rem;text-transform:uppercase;letter-spacing:0.05em;">Section 11.2 · Term &amp; Termination (Page 7 of 24)</div>
            “Either party may <mark>terminate</mark> this Agreement without cause upon thirty (30) calendar days’ prior written <mark>notice</mark> to the other party. In the event of a material breach, the non-breaching party may terminate after a fifteen (15) day cure period…”
          </div>
          <div style="font-size:0.84rem;font-weight:700;color:#f9fafb;">Master_Consulting_Agreement_2026.pdf</div>
          <div style="font-size:0.72rem;color:#9ca3af;margin-top:0.15rem;">~/Documents/Legal/Contracts · PDF · 24 pages · 1.4 MB · 100% Privacy Shield</div>
        </div>
        <div class="df-pv-meta-row">
          <span class="df-pv-why">Matched meaning + exact terms on Page 7</span>
          <div class="df-pv-actions">
            <span class="df-pv-btn primary">Open (p. 7)</span>
            <span class="df-pv-btn">Show in Folder</span>
            <span class="df-pv-btn accent">Chat with Qwen3.5</span>
          </div>
        </div>
      `,
    },
    scan: {
      query: "restaurant receipt from Milan with VAT breakdown",
      mode: "+Vision 170M · Zero OCR",
      caption: "Understands scanned PDF pages, photos, and screenshots visually, with no external OCR software needed.",
      footCount: "2 results · 89% visual match",
      items: [
        {
          group: "Scanned PDFs & Images",
          icon: ICONS.scan,
          title: "expenses_october_scans.pdf",
          tag: "Page 3 · 89%",
          sub: "~/Documents/Finance/Receipts",
          active: true,
        },
        {
          group: "Scanned PDFs & Images",
          icon: ICONS.scan,
          title: "IMG_4821_dinner_milan.jpg",
          tag: "Image · 84%",
          sub: "~/Pictures/Trips/Milan",
          active: false,
        },
      ],
      previewHtml: `
        <div>
          <div class="df-pv-card" style="background:rgba(18, 18, 30, 0.92);">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.45rem;">
              <span style="font-size:0.66rem;color:#38bdf8;font-weight:700;letter-spacing:0.04em;">EMBEDDINGGEMMA 2 VISION TOWER · 150 DPI PAGE RENDER</span>
              <span style="font-size:0.66rem;color:#9ca3af;">Scanned Page 3</span>
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;background:#090910;padding:0.6rem 0.75rem;border-radius:6px;color:#e2e8f0;border:1px solid rgba(255,255,255,0.06);">
              OSTERIA DEL DUOMO - MILANO (MI)<br/>
              ----------------------------------------<br/>
              2x Coperto &amp; Degustazione .... EUR 68.00<br/>
              <mark>IVA 10% (VAT Breakdown)</mark> ...... EUR  6.18<br/>
              TOTALE PAGATO ............... EUR 68.00
            </div>
          </div>
          <div style="font-size:0.84rem;font-weight:700;color:#f9fafb;">expenses_october_scans.pdf</div>
          <div style="font-size:0.72rem;color:#9ca3af;margin-top:0.15rem;">~/Documents/Finance/Receipts · Image-only PDF page · 3.8 MB</div>
        </div>
        <div class="df-pv-meta-row">
          <span class="df-pv-why">Matched visually on Scanned Page 3</span>
          <div class="df-pv-actions">
            <span class="df-pv-btn primary">Open (p. 3)</span>
            <span class="df-pv-btn">Show in Folder</span>
          </div>
        </div>
      `,
    },
    audio: {
      query: "where we discussed database migration risks and rollback",
      mode: "+Audio 300M · 30s Windows",
      caption: "Segments recordings and videos into overlapping 30-second windows and points straight to the timestamp.",
      footCount: "2 results · 88% audio match",
      items: [
        {
          group: "Audio & Video",
          icon: ICONS.audio,
          title: "architecture_sync_nov.wav",
          tag: "4:10 · 88%",
          sub: "~/Recordings/Engineering",
          active: true,
        },
        {
          group: "Audio & Video",
          icon: ICONS.audio,
          title: "release_readiness_review.mp4",
          tag: "12:25 · 79%",
          sub: "~/Movies/Meetings",
          active: false,
        },
      ],
      previewHtml: `
        <div>
          <div class="df-wave-box">
            <div style="width:32px;height:32px;border-radius:50%;background:#6366f1;display:flex;align-items:center;justify-content:center;color:#fff;flex-shrink:0;">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5.5v13l10.5-6.5z"/></svg>
            </div>
            <div class="df-wave-bars" aria-hidden="true">
              <i style="height:35%"></i><i style="height:55%"></i><i style="height:40%"></i><i style="height:70%"></i>
              <i class="hit" style="height:95%"></i><i class="hit" style="height:85%"></i><i class="hit" style="height:100%"></i><i class="hit" style="height:78%"></i><i class="hit" style="height:90%"></i>
              <i style="height:50%"></i><i style="height:62%"></i><i style="height:38%"></i><i style="height:45%"></i><i style="height:30%"></i>
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#38bdf8;flex-shrink:0;">04:10 – 04:40</div>
          </div>
          <div style="font-size:0.84rem;font-weight:700;color:#f9fafb;">architecture_sync_nov.wav</div>
          <div style="font-size:0.72rem;color:#9ca3af;margin-top:0.15rem;">~/Recordings/Engineering · Audio · 18:45 total · Window #10 (30s stride)</div>
        </div>
        <div class="df-pv-meta-row">
          <span class="df-pv-why">Matched spoken segment at 4:10</span>
          <div class="df-pv-actions">
            <span class="df-pv-btn primary">Open at 4:10</span>
            <span class="df-pv-btn">Show in Folder</span>
          </div>
        </div>
      `,
    },
    code: {
      query: "code: retry failed HTTP request with exponential backoff",
      mode: "Code Retrieval + Exact Boost",
      caption: "Finds functions by what they do or by exact identifier, and opens VS Code, Cursor, or Zed at the exact line.",
      footCount: "2 results · 96% match",
      items: [
        {
          group: "Source Code",
          icon: ICONS.code,
          title: "client_retry.py",
          tag: "Lines 42–68 · 96%",
          sub: "~/Projects/api-gateway/src/net",
          active: true,
        },
        {
          group: "Source Code",
          icon: ICONS.code,
          title: "backoff.ts",
          tag: "Lines 12–39 · 89%",
          sub: "~/Projects/web-sdk/src/utils",
          active: false,
        },
      ],
      previewHtml: `
        <div>
          <pre class="df-pv-code">42 │ def <mark>retry_with_backoff</mark>(fn, *, retries: int = 3, base_delay: float = 0.5):
43 │     """Retry a failed HTTP call with <mark>exponential backoff</mark> and jitter."""
44 │     for attempt in range(retries):
45 │         try:
46 │             return fn()
47 │         except TransientNetworkError:
48 │             if attempt == retries - 1:
49 │                 raise
50 │             time.sleep(base_delay * (2 ** attempt))</pre>
          <div style="font-size:0.84rem;font-weight:700;color:#f9fafb;">client_retry.py</div>
          <div style="font-size:0.72rem;color:#9ca3af;margin-top:0.15rem;">~/Projects/api-gateway/src/net · Python · Lines 42–68</div>
        </div>
        <div class="df-pv-meta-row">
          <span class="df-pv-why">Asymmetric task: code retrieval · Lines 42–68</span>
          <div class="df-pv-actions">
            <span class="df-pv-btn primary">Open at line 42</span>
            <span class="df-pv-btn">Show in Folder</span>
          </div>
        </div>
      `,
    },
    rag: {
      query: "INV-2026-9941",
      mode: "Local Qwen3.5 RAG + Exact ID",
      caption: "Exact invoice numbers get an automatic lexical boost, then local Qwen3.5 answers questions over the extracted Markdown tables.",
      footCount: "2 results · 99% hybrid score",
      items: [
        {
          group: "Documents & Spreadsheets",
          icon: ICONS.doc,
          title: "Q3_Cloud_Infrastructure_Invoice.pdf",
          tag: "Page 2 · 99%",
          sub: "~/Documents/Accounting/2026",
          active: true,
        },
        {
          group: "Spreadsheets",
          icon: ICONS.sheet,
          title: "vendor_ledger_2026.xlsx",
          tag: "Sheet 1 · 91%",
          sub: "~/Documents/Accounting/2026",
          active: false,
        },
      ],
      previewHtml: `
        <div>
          <div class="df-rag-thread">
            <div class="df-rag-q">Compare invoice <strong>INV-2026-9941</strong> with Q2 compute spend and list the due date.</div>
            <div class="df-rag-a">
              <strong>Qwen3.5 (Local · 100% Private):</strong><br/>
              Based on the table on <strong>Page 2</strong> of <code>Q3_Cloud_Infrastructure_Invoice.pdf</code>:<br/>
              • Reference: <mark>INV-2026-9941</mark> · Total: <strong>EUR 4,280.00</strong> (+11.4% vs Q2's EUR 3,842.00)<br/>
              • Payment Due Date: <strong>November 30, 2026</strong> (Net-30 terms).
            </div>
          </div>
          <div style="font-size:0.72rem;color:#9ca3af;">Context: Page 2 Markdown Table + Sheet 1 · Zero cloud calls</div>
        </div>
        <div class="df-pv-meta-row">
          <span class="df-pv-why">Answered locally via Qwen3.5 (or your Ollama server)</span>
          <div class="df-pv-actions">
            <span class="df-pv-btn primary">Open (p. 2)</span>
            <span class="df-pv-btn accent">Continue Chat</span>
          </div>
        </div>
      `,
    },
  };

  const SPACE_QUERIES = [
    {
      label: "termination notice period",
      x: "26%",
      y: "34%",
      coords: [260, 122],
      targets: [
        [210, 98],
        [305, 112],
        [245, 165],
      ],
      hits: [
        { role: "PDF · p.7", name: "Master_Consulting_Agreement.pdf", score: "0.94" },
        { role: "DOCX · p.3", name: "Vendor_SLA_Addendum.docx", score: "0.86" },
        { role: "EML · Msg 1", name: "Re_Contract_Renewal.eml", score: "0.81" },
      ],
    },
    {
      label: "ricevuta ristorante Milano IVA",
      x: "68%",
      y: "32%",
      coords: [680, 115],
      targets: [
        [640, 92],
        [725, 106],
        [695, 155],
      ],
      hits: [
        { role: "Scanned PDF · p.3", name: "expenses_october_scans.pdf", score: "0.89" },
        { role: "Image · JPG", name: "IMG_4821_dinner_milan.jpg", score: "0.84" },
        { role: "XLSX · Sheet 2", name: "rimborsi_trasferte.xlsx", score: "0.79" },
      ],
    },
    {
      label: "database migration rollback risks",
      x: "74%",
      y: "68%",
      coords: [740, 245],
      targets: [
        [700, 225],
        [785, 235],
        [755, 285],
      ],
      hits: [
        { role: "Audio · 4:10", name: "architecture_sync_nov.wav", score: "0.88" },
        { role: "Video · 12:25", name: "release_readiness.mp4", score: "0.82" },
        { role: "Markdown · Sec 3", name: "runbook_migration.md", score: "0.78" },
      ],
    },
    {
      label: "code: retry exponential backoff",
      x: "31%",
      y: "71%",
      coords: [310, 255],
      targets: [
        [265, 235],
        [345, 242],
        [300, 295],
      ],
      hits: [
        { role: "Python · L42–68", name: "src/net/client_retry.py", score: "0.96" },
        { role: "TypeScript · L12–39", name: "src/utils/backoff.ts", score: "0.89" },
        { role: "Rust · L18–54", name: "crates/http/retry.rs", score: "0.83" },
      ],
    },
  ];

  let typingTimer = null;
  let autoCycleTimer = null;

  function typeQueryText(el, text) {
    if (!el) return;
    clearInterval(typingTimer);
    el.textContent = "";
    let idx = 0;
    typingTimer = setInterval(() => {
      idx += 2;
      el.textContent = text.slice(0, idx);
      if (idx >= text.length) {
        el.textContent = text;
        clearInterval(typingTimer);
      }
    }, 18);
  }

  function renderScene(sceneKey, userInitiated) {
    const data = SCENES[sceneKey];
    if (!data) return;

    const queryEl = document.getElementById("df-spot-query");
    const modeEl = document.getElementById("df-spot-mode");
    const listEl = document.getElementById("df-spot-list");
    const previewEl = document.getElementById("df-spot-preview");
    const footCountEl = document.getElementById("df-spot-count");
    const captionEl = document.getElementById("df-scene-caption");

    document.querySelectorAll("[data-df-scene]").forEach((btn) => {
      btn.setAttribute("aria-selected", btn.getAttribute("data-df-scene") === sceneKey ? "true" : "false");
    });

    if (userInitiated && autoCycleTimer) {
      clearInterval(autoCycleTimer);
      autoCycleTimer = null;
    }

    typeQueryText(queryEl, data.query);
    if (modeEl) modeEl.textContent = data.mode;
    if (footCountEl) footCountEl.textContent = data.footCount;
    if (captionEl) captionEl.textContent = data.caption;

    if (listEl) {
      let lastGroup = "";
      listEl.innerHTML = data.items
        .map((item) => {
          const groupHeader =
            item.group !== lastGroup
              ? `<div class="df-spot-group-label">${item.group}</div>`
              : "";
          lastGroup = item.group;
          return `
            ${groupHeader}
            <div class="df-spot-item ${item.active ? "active" : ""}">
              <div class="df-spot-thumb">${item.icon}</div>
              <div class="df-spot-item-main">
                <div class="df-spot-item-title">
                  <span>${item.title}</span>
                  <span class="df-spot-tag">${item.tag}</span>
                </div>
                <div class="df-spot-item-sub">${item.sub}</div>
              </div>
            </div>
          `;
        })
        .join("");
    }

    if (previewEl) {
      previewEl.innerHTML = data.previewHtml;
    }
  }

  function renderSpaceQuery(idx) {
    const q = SPACE_QUERIES[idx];
    if (!q) return;

    const glow = document.getElementById("df-space-glow");
    const pin = document.getElementById("df-space-pin");
    const linesGroup = document.getElementById("df-space-lines");
    const hitsContainer = document.getElementById("df-space-hits");

    document.querySelectorAll("[data-df-space-q]").forEach((btn, i) => {
      btn.classList.toggle("active", i === idx);
    });

    if (glow) {
      glow.style.left = q.x;
      glow.style.top = q.y;
    }
    if (pin) {
      pin.style.left = q.x;
      pin.style.top = q.y;
      pin.textContent = q.label;
    }
    if (linesGroup) {
      const [qx, qy] = q.coords;
      linesGroup.innerHTML = q.targets
        .map(
          ([tx, ty]) =>
            `<line x1="${qx}" y1="${qy}" x2="${tx}" y2="${ty}" stroke="#818cf8" stroke-width="1.6" stroke-dasharray="4 4" opacity="0.85" />
             <circle cx="${tx}" cy="${ty}" r="5.5" fill="#38bdf8" stroke="#090910" stroke-width="2" />`
        )
        .join("");
    }
    if (hitsContainer) {
      hitsContainer.innerHTML = q.hits
        .map(
          (h, i) => `
          <div class="df-space-hit-card">
            <div>
              <div style="font-size:0.64rem;color:#9ca3af;text-transform:uppercase;">#${i + 1} · ${h.role}</div>
              <div style="font-weight:600;margin-top:0.1rem;color:#f3f4f6;">${h.name}</div>
            </div>
            <span class="df-space-hit-score">${h.score}</span>
          </div>
        `
        )
        .join("");
    }
  }

  function initSite() {
    // Scroll reveal observer across all pages
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
          }
        });
      },
      { threshold: 0.1 }
    );
    document.querySelectorAll(".df-reveal").forEach((el) => observer.observe(el));

    // Spotlight interactive simulator on Home page
    const spotQuery = document.getElementById("df-spot-query");
    if (spotQuery) {
      const sceneKeys = Object.keys(SCENES);
      let currentSceneIdx = 0;

      document.querySelectorAll("[data-df-scene]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const key = btn.getAttribute("data-df-scene");
          currentSceneIdx = sceneKeys.indexOf(key);
          renderScene(key, true);
        });
      });

      renderScene("pdf", false);

      clearInterval(autoCycleTimer);
      autoCycleTimer = setInterval(() => {
        currentSceneIdx = (currentSceneIdx + 1) % sceneKeys.length;
        renderScene(sceneKeys[currentSceneIdx], false);
      }, 5400);
    }

    // 2D Space interactive buttons on Home page
    if (document.getElementById("df-space-lines")) {
      document.querySelectorAll("[data-df-space-q]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const idx = parseInt(btn.getAttribute("data-df-space-q") || "0", 10);
          renderSpaceQuery(idx);
        });
      });
      renderSpaceQuery(0);
    }
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(initSite);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initSite);
  } else {
    initSite();
  }
})();
