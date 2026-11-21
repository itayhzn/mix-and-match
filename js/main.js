// Mix-and-Match project page: an automatic gallery that keeps swapping tiles, and an explorer for one library.
// A combination is drawn by stacking the chosen tile of every region (an RGBA image cut to the region's box) over
// the library's base image (tile 0 everywhere, which also holds the shared background).
"use strict";

const DATA = "data/";
const SWAP_MS = 2000;
// Region colors, as in the paper's figures.
const COLORS = ["#e6194b", "#3cb44b", "#f0a30a", "#0082c8", "#9a3fd1", "#f58231"];

const tileUrl = (run, r, t) => `${DATA}${run.id}/r${r}t${t}.webp`;
const randInt = (n) => Math.floor(Math.random() * n);
const pct = (v, total) => `${(100 * v) / total}%`;
const ucfirst = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const article = /^(a|an|the) /i;
const bare = (s) => s.replace(article, "");

/** The region slot under a point given as fractions of the image, or -1 for the shared background. */
function regionAt(run, fx, fy) {
  const rows = run.labels.length, cols = run.labels[0].length;
  const row = Math.min(rows - 1, Math.max(0, Math.floor(fy * rows)));
  const col = Math.min(cols - 1, Math.max(0, Math.floor(fx * cols)));
  const c = run.labels[row][col];
  return c === "." ? -1 : Number(c);
}

/** A stack of image layers showing one combination of a run, with cross-faded tile changes. */
class Composite {
  constructor(container, run, lazy = false) {
    this.run = run;
    this.el = container;
    this.combo = new Array(run.n).fill(0);
    this.layers = new Array(run.n).fill(null);
    const base = new Image();
    base.className = "base";
    base.alt = run.background;
    if (lazy) base.loading = "lazy";
    base.decoding = "async";
    base.src = `${DATA}${run.id}/base.webp`;
    container.appendChild(base);
  }

  /** Shows tile t in region r, fading it in once it has loaded. */
  set(r, t, fade = true) {
    if (this.combo[r] === t && this.layers[r]) return;
    this.combo[r] = t;
    const [W, H] = this.run.size;
    const [x0, y0, x1, y1] = this.run.regions[r].box;
    const img = new Image();
    img.decoding = "async";
    img.alt = "";
    Object.assign(img.style, { left: pct(x0, W), top: pct(y0, H), width: pct(x1 - x0, W), height: pct(y1 - y0, H) });
    if (fade) img.classList.add("fading");
    img.src = tileUrl(this.run, r, t);
    const shown = () => {
      if (this.combo[r] !== t) return; // superseded while loading
      const old = this.layers[r];
      this.layers[r] = img;
      this.el.appendChild(img);
      if (fade) {
        img.getBoundingClientRect(); // commit the transparent state before fading in
        requestAnimationFrame(() => img.classList.remove("fading"));
        if (old) setTimeout(() => old.remove(), 750);
      } else if (old) old.remove();
    };
    (img.decode ? img.decode() : Promise.resolve()).then(shown, shown);
  }

  setAll(combo, fade = true) {
    combo.forEach((t, r) => this.set(r, t, fade));
  }
}

/** Draws a run's regions on a canvas: a soft color tint and white outlines; `only` limits both to one region. */
// `keep` draws over what the canvas already shows.
function drawRegions(canvas, run, { tint = 0.4, outline = 1.5, only = null, keep = false } = {}) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  const ctx = canvas.getContext("2d");
  if (!keep) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const rows = run.labels.length, cols = run.labels[0].length;
  const slot = (row, col) => run.labels[row][col];

  if (tint > 0) {
    const small = document.createElement("canvas");
    small.width = cols;
    small.height = rows;
    const sctx = small.getContext("2d");
    const data = sctx.createImageData(cols, rows);
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const c = slot(row, col);
        if (c === "." || (only !== null && Number(c) !== only)) continue;
        const hex = COLORS[Number(c) % COLORS.length];
        const i = 4 * (row * cols + col);
        data.data[i] = parseInt(hex.slice(1, 3), 16);
        data.data[i + 1] = parseInt(hex.slice(3, 5), 16);
        data.data[i + 2] = parseInt(hex.slice(5, 7), 16);
        data.data[i + 3] = Math.round(255 * tint);
      }
    }
    sctx.putImageData(data, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(small, 0, 0, w, h);
  }

  if (outline > 0) {
    const sx = w / cols, sy = h / rows;
    const draws = (a, b) => a !== b && (only === null || Number(a) === only || Number(b) === only);
    ctx.beginPath();
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const c = slot(row, col);
        if (col + 1 < cols && draws(c, slot(row, col + 1))) {
          ctx.moveTo((col + 1) * sx, row * sy);
          ctx.lineTo((col + 1) * sx, (row + 1) * sy);
        }
        if (row + 1 < rows && draws(c, slot(row + 1, col))) {
          ctx.moveTo(col * sx, (row + 1) * sy);
          ctx.lineTo((col + 1) * sx, (row + 1) * sy);
        }
      }
    }
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "rgba(0,0,0,.35)";
    ctx.lineWidth = outline + 1.5;
    ctx.stroke();
    ctx.strokeStyle = "rgba(255,255,255,.95)";
    ctx.lineWidth = outline;
    ctx.stroke();
  }
}

/* ── Automatic gallery ──────────────────────────────────────── */

function buildAutoGallery(runs, onPick) {
  const grid = document.getElementById("auto-grid");
  const cards = runs.map((run) => {
    const card = document.createElement("div");
    card.className = "auto-card";
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-label", `${run.background}: open in the explorer`);
    const layers = document.createElement("div");
    layers.className = "layers";
    card.appendChild(layers);
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.innerHTML = `<span class="meta-nk">${run.n} regions × ${run.k} tiles = ${run.k ** run.n} images</span>`;
    meta.appendChild(document.createTextNode(run.background));
    card.appendChild(meta);
    grid.appendChild(card);
    const item = { run, card, composite: new Composite(layers, run, true), visible: false };
    const pick = () => onPick(run, item.composite.combo.slice());
    card.addEventListener("click", pick);
    card.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
    return item;
  });

  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      const item = cards.find((c) => c.card === entry.target);
      if (item) item.visible = entry.isIntersecting;
    }
  }, { rootMargin: "100px" });
  cards.forEach((c) => observer.observe(c.card));

  setInterval(() => {
    if (document.hidden) return;
    for (const { run, composite, visible } of cards) {
      if (!visible || run.k < 2) continue;
      let combo;
      do combo = composite.combo.map(() => randInt(run.k)); // a random tile in every region
      while (combo.every((t, r) => t === composite.combo[r])); // ... that differs from the current picture
      composite.setAll(combo);
    }
  }, SWAP_MS);
}

/* ── Explorer ───────────────────────────────────────────────── */

function buildExplorer(runs) {
  const strip = document.getElementById("strip");
  const layersEl = document.getElementById("stage-layers");
  const stage = document.getElementById("stage");
  const overlay = document.getElementById("stage-overlay");
  const tip = document.getElementById("stage-tip");
  const regionList = document.getElementById("region-list");
  const showRegions = document.getElementById("show-regions");
  const comboEl = document.getElementById("combo");
  let run = null, composite = null, hover = null, chipRows = [];

  // The strip: every library as its image with the segmentation drawn on it.
  const thumbs = runs.map((r, i) => {
    const button = document.createElement("button");
    button.className = "thumb";
    button.setAttribute("role", "option");
    button.title = r.background;
    button.innerHTML = `<div class="thumb-img"><img alt="" loading="lazy" src="${DATA}${r.id}/thumb.webp"><canvas></canvas></div>` +
      `<div class="thumb-cap"><b>${r.n}×${r.k}</b></div>`;
    button.querySelector(".thumb-cap").appendChild(document.createTextNode(bare(r.background)));
    button.addEventListener("click", () => select(i));
    strip.appendChild(button);
    return button;
  });
  const drawThumbs = () => thumbs.forEach((b, i) => drawRegions(b.querySelector("canvas"), runs[i], { tint: 0.38, outline: 1 }));
  const thumbObserver = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      const i = thumbs.indexOf(entry.target);
      drawRegions(entry.target.querySelector("canvas"), runs[i], { tint: 0.38, outline: 1 });
      thumbObserver.unobserve(entry.target);
    }
  }, { root: strip, rootMargin: "200px" });
  thumbs.forEach((b) => thumbObserver.observe(b));
  const prev = document.getElementById("strip-prev"), next = document.getElementById("strip-next");
  prev.addEventListener("click", () => strip.scrollBy({ left: -strip.clientWidth * 0.8 }));
  next.addEventListener("click", () => strip.scrollBy({ left: strip.clientWidth * 0.8 }));
  const arrows = () => {
    prev.disabled = strip.scrollLeft < 4;
    next.disabled = strip.scrollLeft + strip.clientWidth > strip.scrollWidth - 4;
  };
  strip.addEventListener("scroll", arrows, { passive: true });
  window.addEventListener("resize", arrows);
  setTimeout(arrows, 0);

  function redraw() {
    if (!run) return;
    if (showRegions.checked) {
      drawRegions(overlay, run, { tint: 0.3, outline: 2 });
      // The hovered region once more on top, so it stands out.
      if (hover !== null) drawRegions(overlay, run, { tint: 0.3, outline: 3.5, only: hover, keep: true });
    } else if (hover !== null) drawRegions(overlay, run, { tint: 0.28, outline: 2.2, only: hover });
    else drawRegions(overlay, run, { tint: 0, outline: 0 });
    chipRows.forEach((row, r) => row.region.classList.toggle("hot", r === hover));
  }

  function renderCombo() {
    comboEl.innerHTML = "";
    composite.combo.forEach((t, r) => {
      const chip = document.createElement("span");
      chip.className = "chip-n";
      chip.style.background = COLORS[r % COLORS.length];
      chip.textContent = t + 1;
      chip.title = `${run.regions[r].name}: ${run.regions[r].prompts[t]}`;
      comboEl.appendChild(chip);
    });
    const total = run.k ** run.n;
    comboEl.appendChild(document.createTextNode(`one of ${total.toLocaleString("en-US")} combinations`));
    chipRows.forEach((row, r) => row.chips.forEach((c, t) => c.classList.toggle("on", t === composite.combo[r])));
  }

  function setTile(r, t) {
    composite.set(r, t);
    renderCombo();
    if (hover === r) showTip(r);
  }

  function select(i, combo = null) {
    run = runs[i];
    thumbs.forEach((b, j) => { b.classList.toggle("active", j === i); b.setAttribute("aria-selected", j === i); });
    const strip = thumbs[i].parentElement;
    const target = thumbs[i].offsetLeft - (strip.clientWidth - thumbs[i].offsetWidth) / 2;
    strip.scrollTo({ left: target });
    layersEl.innerHTML = "";
    composite = new Composite(layersEl, run);
    if (combo) composite.setAll(combo, false);
    hover = null;
    tip.classList.remove("on");

    const gp = document.getElementById("global-prompt");
    gp.textContent = ucfirst(run.background);
    if (run.style) {
      const s = document.createElement("span");
      s.className = "style";
      s.textContent = ` ${run.style}`;
      gp.appendChild(s);
    }
    regionList.innerHTML = "";
    chipRows = run.regions.map((region, r) => {
      const color = COLORS[r % COLORS.length];
      const box = document.createElement("div");
      box.className = "region";
      box.style.setProperty("--rc", color);
      box.innerHTML = `<div class="region-head"><span class="region-swatch" style="background:${color}"></span><span class="region-name"></span></div><div class="chips"></div>`;
      box.querySelector(".region-name").textContent = bare(region.name);
      const chips = region.prompts.map((p, t) => {
        const chip = document.createElement("button");
        chip.className = "chip";
        chip.textContent = bare(p);
        chip.addEventListener("click", () => setTile(r, t));
        box.querySelector(".chips").appendChild(chip);
        return chip;
      });
      box.addEventListener("mouseenter", () => { hover = r; redraw(); });
      box.addEventListener("mouseleave", () => { hover = null; redraw(); });
      regionList.appendChild(box);
      return { region: box, chips };
    });
    renderCombo();
    redraw();
  }

  function showTip(r, x, y) {
    tip.innerHTML = `<i style="background:${COLORS[r % COLORS.length]}"></i>`;
    tip.appendChild(document.createTextNode(ucfirst(bare(run.regions[r].prompts[composite.combo[r]]))));
    const hint = document.createElement("small");
    hint.textContent = `tile ${composite.combo[r] + 1} of ${run.k} · click for the next`;
    tip.appendChild(hint);
    if (x !== undefined) {
      const w = stage.clientWidth;
      tip.style.left = `${Math.min(x + 14, w - 230)}px`;
      tip.style.top = `${y + 16}px`;
    }
    tip.classList.add("on");
  }

  function pointerRegion(e) {
    const rect = stage.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    return { r: regionAt(run, x / rect.width, y / rect.height), x, y };
  }
  stage.addEventListener("pointermove", (e) => {
    if (!run || e.pointerType === "touch") return;
    const { r, x, y } = pointerRegion(e);
    if (r !== hover) { hover = r < 0 ? null : r; redraw(); }
    if (hover !== null) showTip(hover, x, y); else tip.classList.remove("on");
    stage.style.cursor = hover !== null ? "pointer" : "default";
  });
  stage.addEventListener("pointerleave", () => { hover = null; tip.classList.remove("on"); redraw(); });
  stage.addEventListener("click", (e) => {
    if (!run) return;
    const { r, x, y } = pointerRegion(e);
    if (r < 0) return;
    setTile(r, (composite.combo[r] + 1) % run.k);
    if (e.pointerType === "touch" || e.pointerType === "") { hover = r; redraw(); showTip(r, x, y); }
  });
  showRegions.addEventListener("change", redraw);
  document.getElementById("btn-random").addEventListener("click", () => {
    if (!run) return;
    composite.combo.forEach((_, r) => setTile(r, randInt(run.k)));
  });
  document.getElementById("btn-reset").addEventListener("click", () => {
    if (!run) return;
    composite.combo.forEach((_, r) => setTile(r, 0));
  });
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { redraw(); drawThumbs(); }, 150);
  });

  return select;
}

/* ── Page chrome ────────────────────────────────────────────── */

function chrome() {
  const nav = document.getElementById("nav");
  const onScroll = () => nav.classList.toggle("solid", window.scrollY > 40);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) { e.target.classList.add("show"); io.unobserve(e.target); }
  }, { threshold: 0.08 });
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
}

/** Colors the BibTeX entry and wires up its copy button. The entry's text in the page is the one copied. */
function bibtex() {
  const pre = document.getElementById("bib-raw");
  const btn = document.getElementById("copy-btn");
  const raw = pre.textContent;
  const esc = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  pre.innerHTML = raw.split("\n").map((line) => {
    let m = line.match(/^@(\w+)\{([^,]*),$/);
    if (m) return `<span class="tok-p">@${m[1]}{</span><span class="tok-k">${esc(m[2])}</span><span class="tok-p">,</span>`;
    m = line.match(/^(\s*)(\w+)(\s*)= \{(.*)\}(,?)$/);
    if (m) return `${m[1]}<span class="tok-f">${m[2]}</span>${m[3]}<span class="tok-p">= {</span>` +
      `<span class="tok-v">${esc(m[4])}</span><span class="tok-p">}${m[5]}</span>`;
    return `<span class="tok-p">${esc(line)}</span>`;
  }).join("\n");
  let timer;
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(raw);
      btn.textContent = "Copied";
      btn.classList.add("done");
    } catch {
      getSelection().selectAllChildren(pre);
      btn.textContent = "Press Ctrl+C";
    }
    clearTimeout(timer);
    timer = setTimeout(() => { btn.textContent = "Copy"; btn.classList.remove("done"); }, 2000);
  });
}

async function main() {
  chrome();
  bibtex();
  const runs = window.MANIFEST;
  const select = buildExplorer(runs);
  select(0);
  buildAutoGallery(runs, (run, combo) => {
    select(runs.indexOf(run), combo);
    document.querySelector(".strip-wrap").scrollIntoView({ behavior: "smooth" });
  });
}

main().catch((error) => {
  console.error(error);
  document.getElementById("auto-grid").innerHTML = '<p class="loading">The examples could not be loaded.</p>';
});
