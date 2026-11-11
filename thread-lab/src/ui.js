// Thread Lab UI: block palette, drag-and-drop kernel, animated GPU stage.
(() => {
  const { LEVELS, runGPU, runCPU, blockText, threadLabel, makeThreads } = Engine;
  const PH = typeof Physics !== "undefined" ? Physics : null; // levels 5 and 6
  const isSim = (L) => L.kind === "physics" || L.kind === "host" || L.kind === "sweep";
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const range = (n) => Array.from({ length: n }, (_, i) => i);
  const SPEED = { slow: 1300, normal: 750, fast: 280 };
  const CATS = [
    ["index", "Indexing", "var(--c-index)"],
    ["control", "Control", "var(--c-control)"],
    ["memory", "Memory", "var(--c-memory)"],
    ["sync", "Sync and atomics", "var(--c-sync)"],
    ["physics", "Physics", "var(--c-physics)"],
    ["host", "CPU side", "var(--c-host)"],
  ];
  const PENDING_SHOWN = new Set(["partial", "total"]);
  const STORE = "threadlab-v2"; // v2: the "Build CUDA Balls" track comes first

  let uidSeq = 1;
  const newStmt = (type, fields) => ({ uid: uidSeq++, type, fields: { ...(fields || {}) } });

  const state = {
    level: 0,
    programs: LEVELS.map((lv) => lv.start.map((s) => newStmt(s.type, s.fields))),
    done: LEVELS.map(() => false),
    gpuSteps: LEVELS.map(() => null),
    run: null, // { kind: "gpu" | "cpu", data, frame, applied }
    playing: false,
    timer: null,
    selected: null,
    selectedBall: null, // a ball clicked on the canvas that no thread moves
    speed: "normal",
    ranGpu: new Set(), // levels run on the GPU this session
  };
  const lv = () => LEVELS[state.level];
  const prog = () => state.programs[state.level];
  const threadsOf = (level) => makeThreads(level.grid, level.block);

  // ------------------------------------------------------------ persistence
  function snapshotState() {
    return {
      level: state.level,
      done: state.done,
      gpuSteps: state.gpuSteps,
      programs: state.programs.map((p) => p.map((s) => ({ type: s.type, fields: s.fields }))),
    };
  }
  function save() {
    try { localStorage.setItem(STORE, JSON.stringify(snapshotState())); } catch (e) { /* storage unavailable */ }
  }
  function restore(data) {
    if (!data || !Array.isArray(data.programs)) return;
    data.programs.forEach((p, i) => {
      if (!LEVELS[i] || !Array.isArray(p)) return;
      let list = p.filter((s) => s && LEVELS[i].blocks[s.type]).map((s) => newStmt(s.type, s.fields));
      // host code must keep exactly one end-of-loop marker
      if (LEVELS[i].blocks.loopEnd && list.filter((s) => s.type === "loopEnd").length !== 1)
        list = LEVELS[i].start.map((s) => newStmt(s.type, s.fields));
      state.programs[i] = list;
    });
    if (Array.isArray(data.done)) state.done = LEVELS.map((_, i) => !!data.done[i]);
    if (Array.isArray(data.gpuSteps)) state.gpuSteps = LEVELS.map((_, i) => data.gpuSteps[i] ?? null);
    if (Number.isInteger(data.level) && LEVELS[data.level]) state.level = data.level;
  }

  // ------------------------------------------------------------ header + mission
  const TRACKS = [["physics", "Build CUDA Balls"], ["classic", "Classic examples"]];
  const trackOf = (i) => LEVELS[i].track;
  const trackLevels = (track) => LEVELS.map((L, i) => i).filter((i) => trackOf(i) === track);
  // A level unlocks once every earlier level in its track is solved. The level on screen is never locked,
  // so shared links (#level-3) still open directly.
  const blockerOf = (i) => trackLevels(trackOf(i)).find((j) => j < i && !state.done[j]);
  const isLocked = (i) => i !== state.level && !state.done[i] && blockerOf(i) !== undefined;
  function renderTabs() {
    const here = trackOf(state.level);
    const tracks = TRACKS.map(([id, name]) => {
      const all = trackLevels(id), solved = all.filter((i) => state.done[i]).length;
      return `<button type="button" class="track" data-track="${id}" aria-pressed="${id === here}">${esc(name)}` +
        `<span>${solved ? `${solved} of ${all.length} solved` : `${all.length} levels`}</span></button>`;
    }).join("");
    const tabs = trackLevels(here).map((i) => {
      const L = LEVELS[i];
      const locked = isLocked(i);
      const need = locked ? `Solve level ${LEVELS[blockerOf(i)].number} first` : "";
      return `<button type="button" class="level-tab${state.done[i] ? " done" : ""}${locked ? " locked" : ""}" role="tab" id="tab-${i}" aria-selected="${i === state.level}" data-level="${i}"` +
        (locked ? ` aria-disabled="true" title="${need}"` : "") + `>` +
        `<span class="num">${L.number}</span><span class="name">${esc(L.title)}</span>` +
        `<span class="state">${state.done[i] ? "Solved" : locked ? need : esc(L.concept)}</span></button>`;
    }).join("");
    $("levels").innerHTML = `<div class="tracks" role="group" aria-label="Tracks">${tracks}</div>` +
      `<div class="tabs" role="tablist" aria-label="Levels">${tabs}</div>`;
  }

  function renderMission() {
    const L = lv();
    $("mission").innerHTML =
      `<p class="eyebrow">${esc(TRACKS.find((t) => t[0] === L.track)[1])} · Level ${L.number} of ${trackLevels(L.track).length} · ${esc(L.concept)}</p>` +
      `<h1>${esc(L.title)}</h1><p>${esc(L.mission)}</p><p class="note">${esc(L.dataNote)}</p>` +
      `<details><summary>Hint</summary><p>${esc(L.hint)}</p></details>` +
      `<details><summary>${L.kind === "host" ? "How this level models the CPU and the GPU" : "How this lab models a GPU"}</summary><ul>` +
      (L.model ? L.model.map((m) => `<li>${esc(m)}</li>`).join("") :
        `<li>All live threads run the same block of your kernel together: one block is one step.</li>` +
        `<li>A thread sees its own writes at once. Other threads see them only after <code>__syncthreads()</code> or when the kernel ends.</li>` +
        `<li><code>atomicAdd</code> calls take turns, one per step.</li>` +
        `<li>Reading or writing outside an array crashes that thread. On a real GPU it is undefined behaviour, and tools like compute-sanitizer report it.</li>` +
        `<li>The CPU loop handles one element per step on one core. Step counts show the idea; real timings come from the benchmark notebook.</li>`) +
      `</ul></details>`;
  }

  // ------------------------------------------------------------ blocks
  function blockHTML(L, stmt, where) {
    const def = L.blocks[stmt.type];
    const fields = { ...(def.fields || {}), ...(stmt.fields || {}) };
    const inKernel = where.inKernel;
    const parts = def.parts.map((p) => {
      if (typeof p === "string") return `<span class="txt">${esc(p)}</span>`;
      if (p.m) return `<span class="slot mirror" data-mirror="${p.m}">${esc(fields[p.m])}</span>`;
      if (!inKernel) return `<span class="slot">${esc(fields[p.f])}</span>`; // pick values after adding
      const options = def.opts[p.f].map((o) => `<option${o === fields[p.f] ? " selected" : ""}>${esc(o)}</option>`).join("");
      return `<select class="slot" id="f-${stmt.uid}-${p.f}" data-field="${p.f}" aria-label="Choose a value">${options}</select>`;
    }).join("");
    const label = blockText(def, fields);
    if (inKernel && def.fixed)
      return `<li data-uid="${stmt.uid}" class="loop-li"><div class="block loop-mark" data-uid="${stmt.uid}" data-fixed="1">${parts}</div></li>`;
    if (inKernel)
      return `<li data-uid="${stmt.uid}"${where.inLoop ? ' class="in-loop"' : ""}><span class="step-no" aria-hidden="true">${where.number}</span>` +
        `<div class="block cat-${def.cat}" data-uid="${stmt.uid}">${parts}<span class="block-tools">` +
        `<button type="button" data-act="up" aria-label="Move ${esc(label)} up">↑</button>` +
        `<button type="button" data-act="down" aria-label="Move ${esc(label)} down">↓</button>` +
        `<button type="button" data-act="del" aria-label="Remove ${esc(label)}">✕</button></span></div></li>`;
    return `<div class="block cat-${def.cat}" data-type="${stmt.type}" role="button" tabindex="0" aria-label="Add block: ${esc(label)}">${parts}</div>`;
  }

  function renderPalette() {
    const L = lv();
    let html = `<h2>Blocks</h2><p class="help">Drag a block into your ${L.kind === "host" ? "code" : "kernel"}, or click it to add it at the end. Drag a block back here to remove it.</p>`;
    const cats = L.catOrder ? L.catOrder.map((c) => CATS.find((x) => x[0] === c)) : CATS;
    for (const [cat, name, color] of cats) {
      const types = L.palette.filter((t) => L.blocks[t].cat === cat);
      if (!types.length) continue;
      html += `<div class="palette-group" style="--dot:${color}">${esc(L.catNames?.[cat] ?? name)}</div><div class="palette-list">` +
        types.map((t) => blockHTML(L, { type: t, fields: {} }, { inKernel: false })).join("") + `</div>`;
    }
    $("palette").innerHTML = html;
  }

  function renderKernel() {
    const L = lv();
    const p = prog();
    const highlight = (line) => {
      const cut = line.indexOf("//");
      const code = cut < 0 ? line : line.slice(0, cut);
      return esc(code).replace(/\b(__global__|__device__|__shared__|__constant__|void|int|const|float2|float|bool|for)\b/g, '<span class="kw">$1</span>') +
        (cut < 0 ? "" : `<span class="cm">${esc(line.slice(cut))}</span>`);
    };
    const header = L.kernelHeader.map(highlight).join("\n");
    const loopEnd = p.findIndex((s) => L.blocks[s.type].fixed);
    let number = 0;
    const items = p.map((s, i) => blockHTML(L, s, { inKernel: true, index: i, inLoop: i < loopEnd, number: L.blocks[s.type].fixed ? 0 : ++number }));
    const host = L.kind === "host";
    $("kernel").innerHTML =
      `<h2>${esc(L.editorTitle || "Your kernel")}</h2><div class="stack-wrap"><div class="kernel-stack">` +
      `<div class="hat">${header}</div>` +
      `<ol id="kernel-list" aria-label="${host ? "Host code" : "Kernel statements"}, run top to bottom">${items.join("")}</ol>` +
      (p.length ? "" : `<div class="empty-drop">Drop blocks here</div>`) +
      `<div class="cap">}</div></div></div>` +
      `<div class="kernel-actions"><button class="btn" type="button" id="btn-clear">Remove all blocks</button>` +
      `<button class="btn" type="button" id="btn-restore">Reset code to starting version</button></div>`;
    markCurrentStatement();
  }

  function programChanged() {
    resetRun();
    renderKernel();
    save();
  }
  function addBlock(type) {
    prog().push(newStmt(type));
    programChanged();
  }

  // ------------------------------------------------------------ drag and drop
  let drag = null;
  document.addEventListener("pointerdown", (e) => {
    if (state.playing) return;
    const blk = e.target.closest(".block");
    if (!blk || blk.dataset.fixed || e.target.closest("select, button")) return;
    if (e.pointerType === "mouse" && e.button !== 0) return;
    const inKernel = !!blk.closest("#kernel-list");
    if (!inKernel && !blk.closest("#palette")) return;
    const r = blk.getBoundingClientRect();
    drag = {
      blk, inKernel, uid: inKernel ? Number(blk.dataset.uid) : null, type: blk.dataset.type,
      x0: e.clientX, y0: e.clientY, offX: e.clientX - r.left, offY: e.clientY - r.top, w: r.width,
      moved: false, ghost: null, marker: null, index: null, trash: false,
    };
    try { blk.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
  });
  document.addEventListener("pointermove", (e) => {
    if (!drag) return;
    if (!drag.moved) {
      if (Math.hypot(e.clientX - drag.x0, e.clientY - drag.y0) < 5) return;
      beginDrag();
    }
    e.preventDefault();
    drag.ghost.style.left = `${e.clientX - drag.offX}px`;
    drag.ghost.style.top = `${e.clientY - drag.offY}px`;
    updateDropTarget(e.clientX, e.clientY);
  });
  document.addEventListener("pointerup", () => {
    if (!drag) return;
    const d = drag;
    drag = null;
    if (!d.moved) {
      if (!d.inKernel) addBlock(d.type);
      return;
    }
    finishDrag(d);
  });
  document.addEventListener("pointercancel", () => {
    if (drag) cleanupDrag(drag);
    drag = null;
  });

  function beginDrag() {
    drag.moved = true;
    const g = drag.blk.cloneNode(true);
    g.querySelectorAll("[id]").forEach((n) => n.removeAttribute("id"));
    g.classList.add("drag-ghost");
    g.style.width = `${drag.w}px`;
    document.body.appendChild(g);
    drag.ghost = g;
    const m = document.createElement("li");
    m.className = "drop-marker";
    m.setAttribute("aria-hidden", "true");
    drag.marker = m;
    document.body.classList.add("dragging");
    if (drag.inKernel) drag.blk.closest("li").classList.add("drag-source");
  }

  function updateDropTarget(x, y) {
    const inside = (el) => {
      const r = el.getBoundingClientRect();
      return x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
    };
    const pal = $("palette");
    drag.trash = drag.inKernel && inside(pal);
    pal.classList.toggle("trash-target", drag.trash);
    const list = $("kernel-list");
    if (inside($("kernel"))) {
      const items = [...list.children].filter((li) => !li.classList.contains("drop-marker") && !li.classList.contains("drag-source"));
      let index = items.length;
      for (let i = 0; i < items.length; i++) {
        const r = items[i].getBoundingClientRect();
        if (y < r.top + r.height / 2) { index = i; break; }
      }
      drag.index = index;
      const ref = items[index];
      if (ref) list.insertBefore(drag.marker, ref);
      else list.appendChild(drag.marker);
    } else {
      drag.index = null;
      drag.marker.remove();
    }
  }

  function cleanupDrag(d) {
    d.ghost?.remove();
    d.marker?.remove();
    document.body.classList.remove("dragging");
    $("palette").classList.remove("trash-target");
    d.blk.closest("li")?.classList.remove("drag-source");
  }

  function finishDrag(d) {
    cleanupDrag(d);
    const p = prog();
    if (d.inKernel) {
      const from = p.findIndex((s) => s.uid === d.uid);
      if (from < 0) return;
      if (d.trash) { p.splice(from, 1); programChanged(); return; }
      if (d.index === null) return;
      const [s] = p.splice(from, 1);
      p.splice(d.index, 0, s);
      programChanged();
    } else if (d.index !== null) {
      p.splice(d.index, 0, newStmt(d.type));
      programChanged();
    }
  }

  // ------------------------------------------------------------ kernel events
  $("kernel").addEventListener("click", (e) => {
    if (state.playing) return;
    if (e.target.id === "btn-clear") {
      state.programs[state.level] = lv().start.filter((st) => lv().blocks[st.type].fixed).map((st) => newStmt(st.type));
      programChanged();
      return;
    }
    if (e.target.id === "btn-restore") {
      state.programs[state.level] = lv().start.map((s) => newStmt(s.type, s.fields));
      programChanged();
      return;
    }
    const btn = e.target.closest("button[data-act]");
    if (!btn) return;
    const uid = Number(btn.closest("li").dataset.uid);
    const p = prog();
    const i = p.findIndex((s) => s.uid === uid);
    if (i < 0) return;
    const act = btn.dataset.act;
    if (act === "del") p.splice(i, 1);
    if (act === "up" && i > 0) [p[i - 1], p[i]] = [p[i], p[i - 1]];
    if (act === "down" && i < p.length - 1) [p[i + 1], p[i]] = [p[i], p[i + 1]];
    programChanged();
    const again = document.querySelector(`#kernel-list li[data-uid="${uid}"] button[data-act="${act}"]`);
    again?.focus();
  });
  $("kernel").addEventListener("change", (e) => {
    const sel = e.target.closest("select.slot");
    if (!sel) return;
    const li = sel.closest("li");
    const stmt = prog().find((s) => s.uid === Number(li.dataset.uid));
    if (!stmt) return;
    stmt.fields[sel.dataset.field] = sel.value;
    li.querySelectorAll(`[data-mirror="${sel.dataset.field}"]`).forEach((m) => { m.textContent = sel.value; });
    resetRun();
    save();
  });
  $("palette").addEventListener("keydown", (e) => {
    const blk = e.target.closest(".block[data-type]");
    if (!blk || state.playing) return;
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); addBlock(blk.dataset.type); }
  });

  // ------------------------------------------------------------ running
  function currentRaw() {
    const r = state.run;
    return r ? r.data.frames[r.frame] : null;
  }
  const idleCache = {};
  function idleFrame() {
    return (idleCache[state.level] ||= runGPU(lv(), []).frames[0]);
  }
  // A frame with everything the renderer needs, for GPU, CPU and idle states.
  function renderFrame() {
    const r = state.run;
    const idle = idleFrame();
    if (!r) return { ...idle, idle: true };
    const f = r.data.frames[r.frame];
    if (r.kind === "cpu") return { ...idle, view: f.view, cpu: f.cpu, label: f.label, steps: f.steps, events: [], active: new Set(), cpuMode: true };
    return f;
  }
  const atEnd = () => state.run && state.run.frame >= state.run.data.frames.length - 1;

  let runSerial = 0; // changes whenever a run starts or is cleared, so the stage knows to rebuild
  function startRun(kind, paused) {
    stopTimer();
    runSerial++;
    const L = lv();
    if (kind === "gpu") state.ranGpu.add(state.level); // reveals Compare with CPU for this level
    if (kind === "cpu" && isSim(L)) { // physics: a note on what one core would do, no animation
      state.run = { kind: "cpunote", data: { frames: [null] }, frame: 0, applied: true };
      renderStage();
      return;
    }
    const data = kind === "cpu" ? runCPU(L) : isSim(L) ? PH.run(L, prog()) : runGPU(L, prog());
    state.run = { kind, data, frame: 0, applied: false };
    if (paused) { state.playing = false; renderStage(); return; }
    play();
  }
  function play() {
    if (!state.run) return;
    if (atEnd()) { finish(); return; }
    state.playing = true;
    document.body.classList.add("locked");
    renderStage();
    tick();
  }
  function tick() {
    const r = state.run;
    if (!r || !state.playing) return;
    if (atEnd()) { finish(); return; }
    state.timer = setTimeout(() => {
      const skip = isSim(lv()) && lv().kind !== "sweep" && state.speed === "fast" ? 2 : 1;
      r.frame = Math.min(r.frame + skip, r.data.frames.length - 1);
      if (atEnd()) finish();
      else { renderStage(); tick(); }
    }, delayFor(r));
  }
  function delayFor(r) {
    const base = SPEED[state.speed];
    if (r.data.kind === "sweep") return { slow: 1000 / 15, normal: 1000 / 30, fast: 1000 / 60 }[state.speed]; // half speed: watch the arcs
    if (r.data.kind === "physics" || r.data.kind === "host") return state.speed === "slow" ? 1000 / 30 : 1000 / 60;
    if (r.kind === "cpu") return Math.max(25, Math.min(base * 0.6, (base * 10) / r.data.frames.length));
    const next = r.data.frames[r.frame + 1];
    return next && next.atomicTurn ? base * 0.45 : base;
  }
  function stopTimer() {
    clearTimeout(state.timer);
    state.timer = null;
    state.playing = false;
    document.body.classList.remove("locked");
  }
  function pause() { stopTimer(); renderStage(); }
  function finish() {
    stopTimer();
    const r = state.run;
    if (r && r.kind === "gpu" && !r.applied) {
      r.applied = true;
      const v = r.data.verdict;
      if (v.ok) {
        state.done[state.level] = true;
        if (!isSim(lv())) state.gpuSteps[state.level] = r.data.steps; // read by the CPU-loop verdict
        renderTabs();
        save();
        celebrate();
      }
      if (v.who && v.who.length && state.selected === null) state.selected = v.who[0];
    }
    renderStage();
  }
  function stepOnce() {
    if (!state.run || state.run.kind !== "gpu" || atEnd()) { startRun("gpu", true); return; }
    stopTimer();
    state.run.frame++;
    if (atEnd()) finish();
    else renderStage();
  }
  function resetRun() {
    stopTimer();
    runSerial++;
    state.run = null;
    renderStage();
  }

  // ------------------------------------------------------------ stage
  function renderStage() {
    const L = lv();
    const [code, comment] = L.launch.split("//");
    $("launch").innerHTML = `${esc(code)}${comment ? `<span class="cm">//${esc(comment)}</span>` : ""}`;
    renderTicker();
    renderViz();
    renderVerdict();
    renderInspector();
    updateButtons();
    markCurrentStatement();
  }

  function markCurrentStatement() {
    const f = currentRaw();
    const k = !isSim(lv()) && state.run && state.run.kind === "gpu" && f && !f.done ? f.k : -1;
    document.querySelectorAll("#kernel-list > li[data-uid]").forEach((li, i) => li.classList.toggle("current", i === k));
  }

  function updateButtons() {
    const run = $("btn-run");
    const r = state.run;
    const L = lv();
    if (state.playing) run.textContent = "Pause";
    else if (r && r.kind === "gpu" && !atEnd()) run.textContent = "Resume";
    else run.textContent = L.kind === "host" ? "Run" : "Run on GPU";
    const cpu = $("btn-cpu");
    cpu.textContent = state.playing && r && r.kind === "cpu" ? "Running CPU loop…" : "Compare with CPU";
    // Comparison only makes sense once there is a GPU run to compare with; the host level is CPU code already.
    cpu.hidden = L.kind === "host" || !(state.ranGpu.has(state.level) || state.done[state.level]);
    $("btn-step").title = isSim(L) ? "Advance one frame (1/60 of a second)" : "Run one block of the kernel";
    $("btn-reset").hidden = !r;
  }

  function renderTicker() {
    const L = lv();
    if (isSim(L)) { $("ticker").innerHTML = simTicker(L); return; }
    const f = renderFrame();
    const r = state.run;
    const nThreads = L.grid.x * L.grid.y * L.block.x * L.block.y;
    let now;
    if (!r) now = `<b>Ready.</b> Run on GPU launches ${nThreads} threads; Step goes one block at a time.`;
    else if (r.kind === "cpu") now = `<b>CPU loop</b> · ${esc(f.label)}`;
    else if (f.k < 0) now = `<b>Launch</b> · ${nThreads} threads start at the top of the kernel`;
    else if (f.done) now = `<b>Done</b> · ${esc(f.label)}`;
    else if (f.barrier) now = `<b>Step ${f.k + 1}</b> · ${esc(f.label)} · ${f.active.size} threads wait here, then all writes become visible`;
    else {
      const working = new Set(f.events.filter((e) => e.kind !== "set").map((e) => e.t)).size;
      const extra = f.atomicTurn ? "" : ` · ${f.active.size} thread${f.active.size === 1 ? "" : "s"} running${working && working < f.active.size ? `, ${working} touching memory` : ""}`;
      now = `<b>Step ${f.k + 1}</b> · ${esc(f.label)}${extra}`;
    }
    const gpuSteps = r && r.kind === "gpu" ? f.steps : state.gpuSteps[state.level];
    const cpuSteps = r && r.kind === "cpu" ? `${f.steps} of ${L.cpuSteps}` : L.cpuSteps;
    $("ticker").innerHTML =
      `<span class="now">${now}</span><span class="counters">` +
      `<span class="counter gpu">GPU kernel: <b>${gpuSteps ?? "–"}</b> step${gpuSteps === 1 ? "" : "s"}</span>` +
      `<span class="counter cpu">CPU loop: <b>${cpuSteps}</b> steps</span>` +
      `<span class="counter">${nThreads} threads</span></span>`;
  }

  // Index of this frame's memory events, keyed "arr:idx".
  function eventIndex(f) {
    const ix = { read: new Set(), write: new Set(), stale: new Set(), fault: new Set() };
    for (const e of f.events || []) {
      if (e.kind === "read") ix[e.stale ? "stale" : "read"].add(`${e.arr}:${e.idx}`);
      if (e.kind === "write" || e.kind === "atomic") ix.write.add(`${e.arr}:${e.idx}`);
      if (e.kind === "fault" && e.info && e.info.kind === "oob") {
        const A = e.info;
        const idx = A.idx !== undefined ? A.idx : null;
        if (idx !== null) ix.fault.add(`${A.arr}:${idx}`);
      }
    }
    return ix;
  }
  // Cells the selected thread touched so far in this run.
  function mineSet() {
    const s = new Set();
    const r = state.run;
    if (!r || r.kind !== "gpu" || state.selected === null) return s;
    for (let i = 0; i <= r.frame; i++)
      for (const e of r.data.frames[i].events)
        if (e.t === state.selected && (e.kind === "read" || e.kind === "write" || e.kind === "atomic")) s.add(`${e.arr}:${e.idx}`);
    return s;
  }

  function chipClass(f, id) {
    const t = f.threads[id];
    const c = ["chip"];
    if (t.fault) c.push("fault");
    else if (t.returned) c.push("returned");
    else if (f.active.has(id)) c.push(f.barrier ? "waiting" : "active");
    if (f.done && state.run && state.run.kind === "gpu") {
      const conflict = state.run.data.mem.conflicts.find((x) => x.arr === "total");
      if (conflict && conflict.writers.includes(id)) c.push(conflict.kept === id ? "kept" : "lost");
    }
    if (state.selected === id) c.push("selected");
    return c.join(" ");
  }

  function cell(f, arr, idx, ix, mine, opts = {}) {
    const key = `${arr}:${idx}`;
    if (opts.ghost)
      return `<span class="cell ghost${ix.fault.has(key) ? " fault" : ""}" title="past the end of ${arr}">${ix.fault.has(key) ? "!" : ""}</span>`;
    const A = f.view[arr];
    const v = A.values[idx];
    const c = ["cell"];
    if (v === null || v === undefined) c.push("empty");
    if (ix.stale.has(key)) c.push("stale");
    else if (ix.read.has(key)) c.push("read");
    if (ix.write.has(key)) c.push("write");
    if (PENDING_SHOWN.has(arr) && A.pending.has(idx)) c.push("pending");
    if (mine.has(key)) c.push("mine");
    if (f.cpuMode && opts.cpuHit) c.push("cpu");
    return `<span class="${c.join(" ")}">${v === null || v === undefined ? (arr === "partial" ? "?" : "·") : v}</span>`;
  }

  const lane = (cols, label, items) =>
    `<div class="lane" style="--cols:${cols}"><span class="rowlbl">${label}</span>${items.join("")}</div>`;

  function chipRow(L, f, cols, content) {
    return lane(cols, "thread", threadsOf(L).map((t) =>
      `<button type="button" class="${chipClass(f, t.id)}" data-thread="${t.id}" aria-label="${esc(threadLabel(L, t))}">${content(t)}</button>`));
  }
  function blockRow(L, cols) {
    if (L.grid.x === 1) return lane(cols, "block", [`<span class="blockbox" style="grid-column: span ${L.block.x}">one block of ${L.block.x} threads</span>`]);
    return lane(cols, "block", range(L.grid.x).map((b) => `<span class="blockbox" style="grid-column: span ${L.block.x}">blockIdx.x = ${b}</span>`));
  }

  function legend(items) {
    return `<div class="legend">${items.map(([style, label]) => `<span><i style="${style}"></i>${label}</span>`).join("")}</div>`;
  }
  const LEG_CHIPS = [
    ["background:var(--accent)", "running this step"],
    ["background:repeating-linear-gradient(135deg,var(--stripe-a) 0 4px,var(--stripe-b) 4px 8px)", "returned early"],
    ["background:var(--bad)", "crashed"],
  ];
  const LEG_CELLS = [
    ["box-shadow:inset 0 0 0 2px var(--accent);background:var(--surface)", "read"],
    ["background:color-mix(in srgb, var(--c-memory) 22%, var(--surface));border:1px solid var(--c-memory)", "written"],
  ];

  function renderViz() {
    const L = lv();
    if (isSim(L)) { renderSimViz(L); return; }
    vizKey = null;
    const f = renderFrame();
    const ix = eventIndex(f);
    const mine = mineSet();
    let h = "";
    if (f.cpuMode) h += `<div class="cpu-core">One CPU core · <code>${esc(L.cpuCode)}</code></div>`;
    if (L.id === "vector-add") h += vizVector(L, f, ix, mine);
    else if (L.id === "image-blur") h += vizImage(L, f, ix, mine);
    else h += vizSum(L, f, ix, mine);
    $("viz").innerHTML = h;
  }

  function vizVector(L, f, ix, mine) {
    const cols = L.grid.x * L.block.x;
    const n = L.n;
    let h = `<p class="viz-title">Threads (each shows its threadIdx.x)</p>`;
    h += blockRow(L, cols);
    h += chipRow(L, f, cols, (t) => t.tx);
    h += lane(cols, "i", threadsOf(L).map((t) => {
      const v = f.threads[t.id].vars.i;
      return `<span class="ival${v !== undefined && v >= n ? " bad" : ""}">${v ?? ""}</span>`;
    }));
    h += `<p class="viz-title" style="margin-top:14px">Arrays in GPU memory (n = ${n}; dashed = past the end)</p>`;
    h += lane(cols, "index", range(cols).map((i) => `<span class="idx${i >= n ? " ghost" : ""}">${i}</span>`));
    for (const arr of ["a", "b", "c"])
      h += lane(cols, arr, range(cols).map((i) => cell(f, arr, i, ix, mine, { ghost: i >= n, cpuHit: f.cpu === i })));
    return h + legend([...LEG_CHIPS, ...LEG_CELLS]);
  }

  function gray(v) {
    return `hsl(214 18% ${Math.round(10 + v * 9.6)}%)`;
  }

  function vizImage(L, f, ix, mine) {
    const W = L.grid.x * L.block.x, H = L.grid.y * L.block.y;
    const inV = f.view.in.values, outV = f.view.out.values;
    const threadAt = {};
    for (const t of threadsOf(L)) threadAt[`${t.bx * L.block.x + t.tx},${t.by * L.block.y + t.ty}`] = t;
    let input = "";
    for (let y = 0; y < L.h; y++)
      for (let x = 0; x < L.w; x++) {
        const i = y * L.w + x;
        const cls = ["pix"];
        if (ix.read.has(`in:${i}`) || ix.stale.has(`in:${i}`)) cls.push("active");
        if (mine.has(`in:${i}`)) cls.push("selected");
        input += `<span class="${cls.join(" ")}" style="background:${gray(inV[i])}" title="in[${y}][${x}] = ${inV[i]}"></span>`;
      }
    let grid = "";
    for (let gy = 0; gy < H; gy++)
      for (let gx = 0; gx < W; gx++) {
        const t = threadAt[`${gx},${gy}`];
        const ts = f.threads[t.id];
        const inside = gx < L.w && gy < L.h;
        const i = gy * L.w + gx;
        const cls = ["pix"];
        if (gx % L.block.x === L.block.x - 1 && gx !== W - 1) cls.push("bx");
        if (gy % L.block.y === L.block.y - 1 && gy !== H - 1) cls.push("by");
        let style = "";
        if (!inside) cls.push(ts.returned ? "returned" : "ghost");
        else if (outV[i] === null) cls.push("unset");
        else style = `background:${gray(outV[i])}`;
        if (ts.fault) cls.push("fault");
        else if (f.active.has(t.id) && !f.done) cls.push("active");
        if (inside && ix.write.has(`out:${i}`)) cls.push("write");
        if (f.cpuMode && f.cpu === i) cls.push("cpu");
        if (state.selected === t.id || (inside && mine.has(`out:${i}`))) cls.push("selected");
        const label = `${threadLabel(L, t)}${inside ? `, pixel (${gx}, ${gy})` : ", no pixel here"}`;
        grid += `<button type="button" class="${cls.join(" ")}" style="${style}" data-thread="${t.id}" title="${esc(label)}" aria-label="${esc(label)}"></button>`;
      }
    return `<div class="images">` +
      `<div><p class="viz-title">Input image in[y][x]</p><div class="pixgrid" style="--w:${L.w};--px:20px">${input}</div>` +
      `<p class="imgcap">${L.w} × ${L.h} pixels.</p></div>` +
      `<div><p class="viz-title">Threads over the output out[y][x]</p><div class="pixgrid" style="--w:${W};--px:20px">${grid}</div>` +
      `<p class="imgcap">${W} × ${H} threads in ${L.grid.x} × ${L.grid.y} blocks (thick lines). Hatched threads have no pixel. Click one to inspect it.</p></div>` +
      `</div>` + legend([
        ["box-shadow:inset 0 0 0 2px var(--accent);background:var(--surface)", "running this step"],
        ["box-shadow:inset 0 0 0 2px var(--c-memory);background:var(--surface)", "written this step"],
        ["background:repeating-linear-gradient(135deg,var(--stripe-a) 0 3px,var(--stripe-b) 3px 6px)", "returned early"],
        ["background:var(--bad)", "crashed"],
      ]);
  }

  function vizSum(L, f, ix, mine) {
    const cols = L.grid.x * L.block.x;
    const tree = L.id === "tree";
    let h = `<p class="viz-title">Threads (each shows its threadIdx.x)</p>`;
    h += blockRow(L, cols);
    h += chipRow(L, f, cols, (t) => t.tx);
    if (!tree)
      h += lane(cols, "i", threadsOf(L).map((t) => `<span class="ival">${f.threads[t.id].vars.i ?? ""}</span>`));
    // what each thread wrote to total in this run, so lost updates are visible
    const r = state.run;
    if (r && r.kind === "gpu") {
      const wrote = {};
      for (let i = 0; i <= r.frame; i++)
        for (const e of r.data.frames[i].events)
          if (e.kind === "write" && e.arr === "total") wrote[e.t] = e.value;
      if (Object.keys(wrote).length)
        h += lane(cols, "wrote", threadsOf(L).map((t) => `<span class="ival">${wrote[t.id] ?? ""}</span>`));
    }
    h += `<p class="viz-title" style="margin-top:14px">${tree ? "Memory: x is global, partial is the block's shared memory" : "Global memory"}</p>`;
    h += lane(cols, "index", range(cols).map((i) => `<span class="idx">${i}</span>`));
    h += lane(cols, L.xLabel || "x", range(cols).map((i) => cell(f, "x", i, ix, mine, { cpuHit: f.cpu === i })));
    if (tree) h += lane(cols, "partial", range(cols).map((i) => cell(f, "partial", i, ix, mine)));
    const total = f.view.total.values[0];
    const expected = L.expectedTotal;
    const finished = (f.done && r && r.kind === "gpu") || (f.cpuMode && atEnd());
    const cls = finished ? (total === expected ? "good" : "bad") : "";
    let sub = `expected ${expected}`;
    if (f.atomicTurn) {
      const turns = r.data.frames.filter((x) => x.atomicTurn && x.k === f.k);
      sub = `turn ${turns.indexOf(f) + 1} of ${turns.length} · expected ${expected}`;
    } else if (f.queued) sub = `${f.queued} atomicAdd calls are queued · expected ${expected}`;
    else if (PENDING_SHOWN.has("total") && f.view.total.pending.has(0) && !f.done)
      sub = `the new value is not visible to other threads yet · expected ${expected}`;
    h += `<div class="registers"><div class="register ${cls}${ix.write.has("total:0") ? " flash" : ""}"><div class="lbl">total (global)</div><div class="big">${total ?? 0}</div><div class="sub">${sub}</div></div></div>`;
    const cellsLegend = [...LEG_CELLS];
    if (tree || r?.kind === "gpu") cellsLegend.push(["border:1.5px dashed var(--c-sync);background:var(--surface)", "written, not visible to other threads yet"]);
    if (tree) cellsLegend.push(["box-shadow:inset 0 0 0 2px var(--bad);background:var(--surface)", "read an old value"]);
    const chipsLegend = [LEG_CHIPS[0]];
    if (tree) chipsLegend.push(["background:color-mix(in srgb, var(--c-sync) 22%, var(--surface))", "waiting at __syncthreads()"]);
    else chipsLegend.push(["box-shadow:inset 0 0 0 2px var(--bad);background:var(--chip)", "update lost"], ["box-shadow:inset 0 0 0 2px var(--good);background:var(--chip)", "update kept"]);
    return h + legend([...chipsLegend, ...cellsLegend]);
  }

  function renderVerdict() {
    const el = $("verdict");
    const r = state.run;
    if (!r || !atEnd() || state.playing) { el.className = "verdict"; el.innerHTML = ""; return; }
    if (r.kind === "cpunote" && lv().kind === "sweep") {
      const P = PH.PIE, total = (P.throws * P.steps).toLocaleString("en-US");
      el.className = "verdict show";
      el.innerHTML = `<h3>One CPU core: ${total} steps</h3>` +
        `<p>A CPU loop flies the throws one after another: ${P.throws} throws × ${P.steps.toLocaleString("en-US")} steps = ${total}, against ${P.steps.toLocaleString("en-US")} on the GPU, where every throw has its own thread. ` +
        `Twenty throws is nothing for a real CPU. Sweeps pay off with thousands of inputs: Monte Carlo simulations, parameter searches, or every launch angle and speed at once.</p>`;
      return;
    }
    if (r.kind === "cpunote") {
      const ref = PH.stepCounts(prog());
      el.className = "verdict show";
      el.innerHTML = `<h3>One CPU core: ${ref.cpu} steps per substep</h3>` +
        `<p>A CPU loop runs the same stepBall code for ball 0, then ball 1, and so on: ${PH.PHYS.count} balls × ${ref.perBall} steps = ${ref.cpu}, against ${ref.gpu} for your kernel on the GPU, where every ball has its own thread. ` +
        `Step counts only show the shape of the work. A real CPU core still keeps up with a few hundred balls in real time; the GPU pays off with thousands. Run <code>cuda_balls --bench</code> to measure where that happens on your machine.</p>`;
      return;
    }
    if (r.kind === "cpu") {
      const L = lv();
      el.className = "verdict show";
      el.innerHTML = `<h3>The CPU loop needs ${r.data.steps} steps</h3><p>One core handles one ${L.unit || (L.id === "image-blur" ? "pixel" : "element")} per step, so the work grows with the data. ${state.gpuSteps[state.level] ? `Your GPU kernel took ${state.gpuSteps[state.level]} steps.` : "Now run your kernel on the GPU and compare."}</p>`;
      return;
    }
    const v = r.data.verdict;
    const L = lv(), track = trackLevels(L.track);
    const next = LEVELS[state.level + 1] && trackOf(state.level + 1) === L.track;
    const allSolved = track.every((i) => state.done[i]);
    const closing = L.track === "physics"
      ? `You built the heart of <code>cuda_balls</code>: the kernel, the loop that drives it and the energy check, then ran a parameter sweep. Splat. The program in the repository runs the same engine in 3-D with thousands of balls; <code>cuda_balls --bench</code> shows where the GPU overtakes one CPU core.</p>` +
        `<button class="btn" type="button" id="btn-classic">Try the classic examples</button>`
      : `These are the four examples in the tutorial; the benchmark notebook times them on a real GPU.</p>`;
    el.className = `verdict show ${v.ok ? "ok" : "bad"}`;
    const keepFlags = (h) => h.replace(/(^|\s)(--[a-z][a-z-]*)/g, '$1<span class="nowrap">$2</span>'); // never break "--bench"
    el.innerHTML = `<h3>${esc(v.title)}</h3><p>${keepFlags(esc(v.message))}</p>` +
      (L.id === "balls-energy" ? energyChart(r) : "") +
      (v.ok && next ? `<button class="btn primary" type="button" id="btn-next">Next level: ${esc(LEVELS[state.level + 1].title)}</button>` : "") +
      (v.ok && !next ? `<p style="margin-top:8px"><b>${allSolved ? `All ${track.length} levels solved.` : "Last level solved."}</b> ${closing}` : "");
  }

  function renderInspector() {
    const el = $("inspector");
    const L = lv();
    if (isSim(L)) { el.innerHTML = simInspector(L); return; }
    if (state.selected === null) {
      el.innerHTML = `Click any thread to see its own variables, and every read and write it made.`;
      return;
    }
    const t = threadsOf(L)[state.selected];
    if (!t) { state.selected = null; el.innerHTML = ""; return; }
    const f = renderFrame();
    const ts = f.threads[t.id];
    const ids = L.dims === 2
      ? `blockIdx = (${t.bx}, ${t.by}) · threadIdx = (${t.tx}, ${t.ty}) · blockDim = (${L.block.x}, ${L.block.y})`
      : `blockIdx.x = ${t.bx} · threadIdx.x = ${t.tx} · blockDim.x = ${L.block.x}`;
    const vars = Object.entries(ts.vars).map(([k, v]) => `${k} = ${v}`).join(" · ");
    const r = state.run;
    let log;
    if (r && r.kind === "gpu") {
      const upto = f.done ? Infinity : f.k + 1;
      const lines = r.data.threads[t.id].log.filter((line) => Number(line.slice(5, line.indexOf(":"))) <= upto);
      log = lines.length ? `<ol>${lines.map((l) => `<li>${esc(l)}</li>`).join("")}</ol>` : `<p class="vars">Nothing yet.</p>`;
    } else log = `<p class="vars">Run the kernel to see what this thread does.</p>`;
    const status = ts.fault ? `crashed: ${ts.fault}` : ts.returned ? "returned early" : "";
    el.innerHTML = `<div class="card"><h4>${esc(threadLabel(L, t))}${status ? ` · ${esc(status)}` : ""}</h4>` +
      `<p class="vars">${esc(ids)}${vars ? `<br>${esc(vars)}` : ""}</p>${log}</div>`;
  }

  // ------------------------------------------------------------ physics levels (5 and 6)
  const BLOCK_COLORS = ["#2f62d6", "#0e7c6a", "#c9771f"];
  // Faces for the balls: 22 of the 32 photos, picked at random each time the page loads (credits in the footer).
  const ballFaces = (() => {
    const order = FACES.map((f, k) => k);
    for (let k = order.length - 1; k > 0; k--) { const j = Math.floor(Math.random() * (k + 1)); [order[k], order[j]] = [order[j], order[k]]; }
    return order.slice(0, PH.PHYS.count).map((k) => {
      const img = new Image();
      img.onload = () => { if (lv().kind === "physics" || lv().kind === "host") paintCanvas(); };
      img.src = FACES[k].src;
      return { img, name: FACES[k].name };
    });
  })();
  const VIEW = { x0: -6.5, x1: 6.5, y0: -0.9, y1: 8.3 }; // world units; the box is x in [-6, 6], y in [0, 8]
  const OPS = {
    launch: ["L", "launch", "var(--c-physics)"],
    swap: ["S", "swap", "var(--c-host)"],
    copyBack: ["↓", "copy to CPU", "var(--c-memory)"],
    upload: ["↑", "copy to GPU", "#7a3a8f"],
    draw: ["D", "draw", "var(--c-hat)"],
  };
  let vizKey = null;
  const simRun = () => (state.run && state.run.kind === "gpu" ? state.run : null);
  const content = (c) => (!c ? "nothing yet" : c.g === 1 ? "uninitialized" : c.g ? "garbage" : `substep ${c.v.toLocaleString("en-US")}`);

  // What to draw right now: ball states, dashed ghost states, or nothing.
  function simPicture(L) {
    const r = simRun();
    if (L.kind === "physics") {
      if (!r) return { balls: PH.refState(0) };
      const ghost = r.data.verdict.ghost;
      return { balls: r.data.frames[r.frame], ghost: ghost ? ghost[Math.min(r.frame, ghost.length - 1)] : null };
    }
    if (!r) return { balls: PH.refState(0), note: "starting positions" };
    const shown = r.data.frames[r.frame].shown;
    if (!shown) return { balls: null };
    return { balls: shown.g ? PH.garbageState(shown.g) : PH.refState(shown.v), garbage: !!shown.g };
  }

  // Canvas colors come from the CSS theme tokens; refreshed whenever the theme changes.
  const TC = {};
  function readCanvasTheme() {
    const cs = getComputedStyle(document.documentElement);
    const v = (n) => cs.getPropertyValue(n).trim();
    Object.assign(TC, { bg: v("--canvas-bg"), box: v("--canvas-box"), grid: v("--canvas-grid"), wall: v("--canvas-wall"),
      muted: v("--muted"), ink: v("--ink"), room: v("--pie-room"), floor: v("--pie-floor"), tool: v("--canvas-tool") });
  }
  readCanvasTheme();
  function paintCanvas() {
    const cv = $("phys-canvas");
    if (!cv) return;
    const L = lv();
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth || 600, h = cv.clientHeight || 420;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
    }
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const s = w / (VIEW.x1 - VIEW.x0);
    const X = (x) => (x - VIEW.x0) * s, Y = (y) => (VIEW.y1 - y) * s;
    if (L.kind === "sweep") { paintPies(g, w, h, s, X, Y); return; }
    const { halfW, H } = PH.PHYS;
    g.fillStyle = TC.bg;
    g.fillRect(0, 0, w, h);
    g.fillStyle = TC.box;
    g.fillRect(X(-halfW), Y(H), 2 * halfW * s, H * s);
    g.strokeStyle = TC.grid;
    g.lineWidth = 1;
    for (let x = -halfW + 1; x < halfW; x++) { g.beginPath(); g.moveTo(X(x), Y(0)); g.lineTo(X(x), Y(H)); g.stroke(); }
    for (let y = 1; y < H; y++) { g.beginPath(); g.moveTo(X(-halfW), Y(y)); g.lineTo(X(halfW), Y(y)); g.stroke(); }
    g.strokeStyle = TC.wall;
    g.lineWidth = 3;
    g.strokeRect(X(-halfW), Y(H), 2 * halfW * s, H * s);

    const pic = simPicture(L);
    if (!pic.balls) {
      g.fillStyle = TC.muted;
      g.font = `600 15px "IBM Plex Sans", system-ui, sans-serif`;
      g.textAlign = "center";
      g.fillText("Nothing has been drawn yet", w / 2, h / 2);
      return;
    }
    const r = simRun();
    const done = r && atEnd() && !state.playing;
    const v = r ? r.data.verdict : null;
    const flagged = new Set(done && !v.ok && L.kind === "physics" ? v.balls || v.who || [] : []);
    const B = pic.balls;
    for (let i = 0; i < PH.PHYS.count; i++) {
      const x = X(B[4 * i]), y = Y(B[4 * i + 1]), rad = PH.RADIUS[i] * s;
      const color = BLOCK_COLORS[Math.floor(i / PH.PHYS.threads)], face = ballFaces[i];
      if (!pic.garbage && face.img.complete && face.img.naturalWidth) {
        // the face, clipped to the ball, ringed in the colour of the block whose thread moves it
        g.save();
        g.beginPath(); g.arc(x, y, rad, 0, Math.PI * 2); g.clip();
        g.drawImage(face.img, x - rad, y - rad, 2 * rad, 2 * rad);
        g.restore();
        g.beginPath(); g.arc(x, y, rad - 1.25, 0, Math.PI * 2);
        g.lineWidth = 2.5; g.strokeStyle = color; g.stroke();
      } else {
        g.beginPath();
        g.arc(x, y, rad, 0, Math.PI * 2);
        g.globalAlpha = pic.garbage ? 0.55 : 0.9;
        g.fillStyle = pic.garbage ? "#c73434" : color;
        g.fill();
        g.globalAlpha = 1;
        g.lineWidth = 1.5;
        g.strokeStyle = "rgba(22, 32, 42, .45)";
        g.stroke();
      }
      if (rad > 9 && !pic.garbage) {
        // the ball's number on a small badge at the bottom, so the face stays visible
        const fs = Math.round(Math.min(11, Math.max(8, rad * 0.5)));
        g.font = `600 ${fs}px "IBM Plex Mono", ui-monospace, monospace`;
        const tw = g.measureText(String(i)).width + 6, bh = fs + 3, by = y + rad - bh * 0.75;
        g.fillStyle = color;
        g.beginPath(); g.roundRect ? g.roundRect(x - tw / 2, by - bh / 2, tw, bh, bh / 2) : g.rect(x - tw / 2, by - bh / 2, tw, bh); g.fill();
        g.fillStyle = "white";
        g.textAlign = "center";
        g.textBaseline = "middle";
        g.fillText(String(i), x, by + 0.5);
      }
    }
    if (pic.ghost) {
      g.setLineDash([4, 3]);
      g.lineWidth = 1.5;
      g.strokeStyle = TC.ink;
      for (let i = 0; i < PH.PHYS.count; i++) {
        g.beginPath();
        g.arc(X(pic.ghost[4 * i]), Y(pic.ghost[4 * i + 1]), PH.RADIUS[i] * s, 0, Math.PI * 2);
        g.stroke();
      }
      g.setLineDash([]);
    }
    const ring = (i, color, width) => {
      g.beginPath();
      g.arc(X(B[4 * i]), Y(B[4 * i + 1]), PH.RADIUS[i] * s + width, 0, Math.PI * 2);
      g.lineWidth = width;
      g.strokeStyle = color;
      g.stroke();
    };
    for (const i of flagged) if (i < PH.PHYS.count) ring(i, "#c73434", 3);
    const first = r && L.kind === "physics" ? r.data.firstThreads : null;
    const sel = state.selected !== null ? (first ? first[state.selected].i : state.selected) : state.selectedBall;
    if (L.kind === "physics" && sel !== null && sel !== undefined && sel >= 0 && sel < PH.PHYS.count) {
      ring(sel, TC.ink, 3);
      const x = X(B[4 * sel]), y = Y(B[4 * sel + 1]);
      const vx = B[4 * sel + 2] * s * 0.12, vy = -B[4 * sel + 3] * s * 0.12; // arrow: 0.12 s of travel
      if (Math.hypot(vx, vy) > 4) {
        g.beginPath();
        g.moveTo(x, y);
        g.lineTo(x + vx, y + vy);
        g.lineWidth = 2.5;
        g.strokeStyle = TC.ink;
        g.stroke();
        const a = Math.atan2(vy, vx);
        g.beginPath();
        g.moveTo(x + vx, y + vy);
        g.lineTo(x + vx - 8 * Math.cos(a - 0.45), y + vy - 8 * Math.sin(a - 0.45));
        g.lineTo(x + vx - 8 * Math.cos(a + 0.45), y + vy - 8 * Math.sin(a + 0.45));
        g.closePath();
        g.fillStyle = TC.ink;
        g.fill();
      }
    }
  }

  function canvasBall(e) {
    const cv = $("phys-canvas");
    const pic = simPicture(lv());
    if (!cv || !pic.balls) return null;
    const rect = cv.getBoundingClientRect();
    const s = rect.width / (VIEW.x1 - VIEW.x0);
    const wx = (e.clientX - rect.left) / s + VIEW.x0, wy = VIEW.y1 - (e.clientY - rect.top) / s;
    let best = null, bestD = Infinity;
    for (let i = 0; i < PH.PHYS.count; i++) {
      const d = Math.hypot(wx - pic.balls[4 * i], wy - pic.balls[4 * i + 1]);
      if (d < PH.RADIUS[i] + 0.15 && d < bestD) { best = i; bestD = d; }
    }
    return best;
  }

  function renderSimViz(L) {
    const r = simRun();
    const key = [state.level, runSerial, state.selected, state.selectedBall, state.run ? state.run.kind : "idle", r && atEnd() && !state.playing ? "end" : ""].join("|");
    if (key !== vizKey) {
      vizKey = key;
      $("viz").innerHTML = L.kind === "physics" ? vizKernel(L) : L.kind === "sweep" ? vizPies(L) : vizHostLoop();
    }
    if (L.kind === "host") $("host-live").innerHTML = hostLive();
    paintCanvas();
  }

  function vizKernel(L) {
    const r = simRun();
    const cols = PH.PHYS.blocks * PH.PHYS.threads;
    const first = r ? r.data.firstThreads : null;
    const cpu = state.run && state.run.kind === "cpunote";
    let h = cpu ? `<div class="cpu-core">One CPU core · <code>${esc(L.cpuCode)}</code></div>` : "";
    h += `<p class="viz-title">The box, 12 × 8 units · each ball is coloured by the block whose thread moves it</p>`;
    h += `<canvas id="phys-canvas" class="phys-canvas" role="img" aria-label="The balls in the box. Click a ball to inspect its thread."></canvas>`;
    h += `<p class="viz-title" style="margin-top:12px">Threads · each chip shows threadIdx.x</p>`;
    h += blockRow(L, cols);
    h += lane(cols, "thread", threadsOf(L).map((t) => {
      const ft = first ? first[t.id] : null;
      const c = ["chip"];
      let style = "";
      if (ft && ft.fault) c.push("fault");
      else if (t.id >= PH.PHYS.count && ft) c.push("returned");
      else if (t.id < PH.PHYS.count) { c.push("ball"); style = ` style="--bc:${BLOCK_COLORS[t.bx]}"`; }
      if (state.selected === t.id) c.push("selected");
      return `<button type="button" class="${c.join(" ")}"${style} data-thread="${t.id}" aria-label="${esc(threadLabel(L, t))}">${t.tx}</button>`;
    }));
    h += lane(cols, "ball i", threadsOf(L).map((t) => {
      const ft = first ? first[t.id] : null;
      const i = ft ? ft.i : t.id < PH.PHYS.count ? t.id : undefined;
      return `<span class="ival${i !== undefined && i >= PH.PHYS.count ? " bad" : ""}">${i ?? ""}</span>`;
    }));
    const items = BLOCK_COLORS.map((c, b) => [`background:${c}`, `block ${b}`]);
    items.push(["background:repeating-linear-gradient(135deg,var(--stripe-a) 0 4px,var(--stripe-b) 4px 8px)", "no ball: returns early"]);
    if (r && r.data.faults.length) items.push(["background:var(--bad)", "crashed"]);
    if (r && r.data.verdict.ghost) items.push(["border:1.5px dashed var(--ink);border-radius:50%;background:var(--surface)", "same kernel, blocks run in reverse order"]);
    return h + legend(items);
  }

  function vizHostLoop() {
    const ops = [];
    const p = prog().map((s) => s.type);
    const end = p.indexOf("loopEnd");
    for (let s = 0; s < PH.PHYS.substeps; s++) ops.push(...p.slice(0, end));
    ops.push(...p.slice(end + 1));
    const used = [...new Set(ops)];
    let h = `<p class="viz-title">The screen · what draw(hostPos) shows</p>`;
    h += `<canvas id="phys-canvas" class="phys-canvas" role="img" aria-label="The picture on screen"></canvas>`;
    h += `<div id="host-live"></div>`;
    h += `<p class="viz-title" style="margin-top:14px">One call of frame(), in order · ${ops.length} operation${ops.length === 1 ? "" : "s"}</p>`;
    h += ops.length
      ? `<div class="timeline">${ops.map((o) => `<span class="op" style="--oc:${OPS[o][2]}" title="${esc(OPS[o][1])}">${OPS[o][0]}</span>`).join("")}</div>`
      : `<p class="imgcap">Empty.</p>`;
    if (used.length) h += legend(used.map((o) => [`background:${OPS[o][2]}`, `${OPS[o][0]} ${OPS[o][1]}`]));
    return h;
  }

  // Buffers, pointers and rates for the current frame of level 6.
  function hostLive() {
    const r = simRun();
    const f = r ? r.data.frames[r.frame] : { A: { v: 0 }, B: { g: 1 }, pos: "A", host: { v: 0 }, shown: null };
    const box = (name, c, ptr) =>
      `<div class="mem-box${c && c.g ? " bad" : ""}"><span class="mem-name">${name}</span><span class="mem-val">${esc(content(c))}</span>${ptr ? `<span class="ptr">← ${ptr}</span>` : ""}</div>`;
    let h = `<div class="mem-grid">` +
      `<div><p class="viz-title">GPU memory</p>${box("buffer A", f.A, f.pos === "A" ? "pos" : "nextPos")}${box("buffer B", f.B, f.pos === "B" ? "pos" : "nextPos")}</div>` +
      `<div><p class="viz-title">CPU memory</p>${box("hostPos", f.host)}</div>` +
      `<div><p class="viz-title">Screen</p>${box("picture", f.shown)}</div></div>`;
    if (r) {
      const c = r.data.counts, sec = (n) => (n * PH.PHYS.fps).toLocaleString("en-US");
      const speed = r.data.advance === null ? "–" : `${(r.data.advance / PH.PHYS.substeps).toFixed(2)}×`;
      h += `<div class="rates">` +
        [["kernel launches", sec(c.launch)], ["swaps", sec(c.swap)], ["copies to the CPU", sec(c.copyBack)], ["copies to the GPU", sec(c.upload)], ["draws", sec(c.draw)]]
          .map(([k, v]) => `<span class="rate"><b>${v}</b> ${k} a second</span>`).join("") +
        `<span class="rate${r.data.advance === PH.PHYS.substeps ? " good" : ""}"><b>${speed}</b> real-time speed</span></div>`;
    }
    return h;
  }

  function simTicker(L) {
    const r = simRun();
    if (L.kind === "sweep") return pieTicker(r);
    const P = PH.PHYS;
    let now, counters;
    if (L.kind === "physics") {
      const sc = PH.stepCounts(prog());
      if (state.run && state.run.kind === "cpunote") now = "<b>CPU loop</b> · one core, one ball after another";
      else if (!r) now = `<b>Ready.</b> Run on GPU launches ${P.blocks * P.threads} threads ${P.substeps} times per frame, for ${L.frames / P.fps} seconds.`;
      else if (r.data.frames.length === 1) now = "<b>Stopped</b> in the first substep";
      else {
        const early = r.data.frames.length - 1 < L.frames && atEnd() ? " · stopped early" : "";
        now = `<b>Frame ${r.frame} of ${L.frames}</b> · ${(r.frame / P.fps).toFixed(2)} s simulated · ${(r.frame * P.substeps).toLocaleString("en-US")} launches${early}`;
      }
      counters = `<span class="counter gpu">GPU kernel: <b>${sc.gpu}</b> steps per substep</span>` +
        `<span class="counter cpu">CPU loop: <b>${sc.cpu}</b> steps per substep</span>` +
        `<span class="counter">${P.blocks * P.threads} threads</span>`;
    } else {
      if (!r) now = `<b>Ready.</b> Run calls frame() ${P.fps} times a second, for 4 seconds.`;
      else {
        const f = r.data.frames[r.frame];
        now = `<b>Frame ${r.frame + 1} of ${P.frames}</b> · ${((r.frame + 1) / P.fps).toFixed(2)} s · the screen shows ${esc(content(f.shown))}`;
      }
      counters = `<span class="counter gpu">${P.substeps} substeps of 1/960 s per frame</span><span class="counter">${P.fps} frames a second</span>`;
    }
    return `<span class="now">${now}</span><span class="counters">${counters}</span>`;
  }

  function simInspector(L) {
    if (L.kind === "sweep") return pieInspector(L);
    if (L.kind === "host")
      return "Step advances one frame. Watch which substep each buffer holds, and where pos and nextPos point.";
    if (state.selected === null && state.selectedBall !== null && state.selectedBall !== undefined)
      return `<div class="card"><h4>Ball ${state.selectedBall} · no thread</h4><p class="vars">No thread computed i = ${state.selectedBall}, so this ball never moves. Thread t of block b should move ball b × 8 + t.</p></div>`;
    if (state.selected === null) return "Click a ball or a thread to see its position, velocity and what it read.";
    const t = threadsOf(L)[state.selected];
    if (!t) return "";
    const r = simRun();
    const first = r ? r.data.firstThreads : null;
    const ft = first ? first[t.id] : null;
    const head = `${threadLabel(L, t)}`;
    if (ft && ft.fault) return `<div class="card"><h4>${esc(head)} · crashed</h4><p class="vars">${esc(ft.fault)}</p></div>`;
    const i = ft && ft.i !== undefined ? ft.i : t.id;
    if (i >= PH.PHYS.count)
      return `<div class="card"><h4>${esc(head)} · no ball</h4><p class="vars">i = ${i}, and the last ball is 21. ${ft && ft.returned ? "It returned early." : "It must return before touching pos."}</p></div>`;
    const B = simPicture(L).balls;
    const fmt = (v) => (v >= 0 ? " " : "") + v.toFixed(2);
    const twins = first ? first.filter((x) => x.i === i && x.id !== t.id && !x.fault && !x.returned) : [];
    let h = `<div class="card"><h4>${esc(head)} · ball ${i} (${esc(ballFaces[i].name)})</h4>` +
      (twins.length ? `<p class="vars" style="color:var(--bad)">Ball ${i} is also moved by ${twins.map((x) => esc(threadLabel(L, x))).join(" and ")}.</p>` : "") +
      `<p class="vars">` +
      `p = (${fmt(B[4 * i])}, ${fmt(B[4 * i + 1])}) · v = (${fmt(B[4 * i + 2])}, ${fmt(B[4 * i + 3])})<br>` +
      `radius = ${PH.RADIUS[i].toFixed(2)} · mass = ${PH.MASS[i].toFixed(2)}</p>`;
    if (r && r.data.threadRaces[i])
      h += `<p class="vars" style="color:var(--bad)">Read another ball after its thread had already overwritten it ${r.data.threadRaces[i].toLocaleString("en-US")} times during the run.</p>`;
    else if (r) h += `<p class="vars">Read only positions from before this substep.</p>`;
    return h + `</div>`;
  }

  // ------------------------------------------------------------ celebration when a level is solved
  // The creature pops up over the page, confetti bursts out from behind it, and the sound plays
  // (unless the sound is switched off). Reduced motion: no bounce and no confetti.
  const SOUND_KEY = "threadlab-sound";
  let soundOn = true;
  try { soundOn = localStorage.getItem(SOUND_KEY) !== "off"; } catch (e) { /* storage unavailable */ }
  const showSoundButton = () => {
    const b = $("btn-sound");
    b.querySelector(".label").textContent = soundOn ? "Sound on" : "Sound off";
    b.setAttribute("aria-pressed", String(soundOn));
  };
  const yippee = new Audio(CELEBRATE.sound);
  yippee.preload = "auto";
  const CONFETTI = ["#2f62d6", "#f2a93b", "#0e7c6a", "#b3307d", "#6a46c4", "#eb6834", "#1c8048"];
  let party = null; // the overlay, made on first use
  const PIC_AREA = 76000; // px² each celebration picture covers, about the size of the first two
  let picTurn = 0; // the pictures take turns, one per solved level

  function celebrate() {
    if (soundOn) {
      try {
        yippee.currentTime = 0;
        const p = yippee.play();
        if (p && p.catch) p.catch(() => { /* the browser blocked sound; the picture still shows */ });
      } catch (e) { /* no audio here */ }
    }
    if (!party) {
      party = document.createElement("div");
      party.className = "celebrate";
      party.setAttribute("aria-hidden", "true");
      party.innerHTML = `<canvas class="confetti"></canvas>`;
      party.addEventListener("click", () => endParty());
      document.body.appendChild(party);
    }
    // a fresh <img> each time, so an animated picture starts from its first frame
    const pic = CELEBRATE.pics[picTurn++ % CELEBRATE.pics.length];
    party.querySelector(".yippee-img")?.remove();
    // a video (the fish) plays muted from its first frame; the pictures are plain <img>s
    const img = document.createElement(pic.video ? "video" : "img");
    Object.assign(img, { className: "yippee-img", src: pic.src, width: pic.w, height: pic.h });
    if (pic.video) Object.assign(img, { muted: true, loop: true, autoplay: true, playsInline: true });
    else img.alt = "";
    // every picture covers about the same area on screen, whatever its shape
    const aspect = pic.w / pic.h, shownH = Math.min(Math.sqrt(PIC_AREA / aspect), 340);
    img.style.aspectRatio = `${pic.w} / ${pic.h}`;
    img.style.width = "auto";
    img.style.height = `min(${Math.round(shownH)}px, 48vh)`;
    if (pic.card) { img.style.background = pic.card; img.style.borderRadius = "18px"; } // pictures with a solid backdrop get a rounded card
    if (pic.pixel) img.style.imageRendering = "pixelated"; // tiny pixel art stays crisp when enlarged
    img.style.animationDuration = `${pic.ms}ms`;
    party.appendChild(img);
    const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
    party.classList.remove("show", "calm");
    void party.offsetWidth; // restart the pop animation
    party.classList.add("show");
    if (calm) { party.classList.add("calm"); img.style.animationDuration = ""; }
    clearTimeout(party.timer);
    party.timer = setTimeout(endParty, calm ? 1800 : pic.ms);
    cancelAnimationFrame(party.frame);
    const cv = party.querySelector(".confetti");
    const g = cv.getContext("2d");
    const dpr = window.devicePixelRatio || 1, W = innerWidth, H = innerHeight;
    cv.width = W * dpr; cv.height = H * dpr;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    if (calm) return;
    const bits = Array.from({ length: 160 }, () => {
      const angle = -Math.PI / 2 + (Math.random() - 0.5) * Math.PI * 1.25, speed = 7 + Math.random() * 11;
      return {
        x: W / 2 + (Math.random() - 0.5) * 60, y: H / 2 + 40,
        vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed,
        w: 6 + Math.random() * 6, h: 9 + Math.random() * 8, round: Math.random() < 0.25,
        spin: (Math.random() - 0.5) * 0.4, turn: Math.random() * Math.PI,
        color: CONFETTI[Math.floor(Math.random() * CONFETTI.length)],
      };
    });
    const t0 = performance.now(), life = pic.ms / 1000; // the confetti fades out over the last 0.6 s
    const tick = (now) => {
      const t = (now - t0) / 1000;
      g.clearRect(0, 0, W, H);
      g.globalAlpha = t < life - 0.6 ? 1 : Math.max(0, (life - t) / 0.6);
      for (const b of bits) {
        b.vy += 0.32; b.vx *= 0.985; b.vy *= 0.985;
        b.x += b.vx + Math.sin(t * 6 + b.turn) * 0.8; b.y += b.vy;
        b.turn += b.spin;
        g.save();
        g.translate(b.x, b.y);
        g.rotate(b.turn);
        g.fillStyle = b.color;
        if (b.round) { g.beginPath(); g.arc(0, 0, b.w / 2, 0, Math.PI * 2); g.fill(); }
        else g.fillRect(-b.w / 2, (-b.h / 2) * Math.abs(Math.cos(b.turn)), b.w, b.h * Math.abs(Math.cos(b.turn)));
        g.restore();
      }
      if (t < life) party.frame = requestAnimationFrame(tick);
      else g.clearRect(0, 0, W, H);
    };
    party.frame = requestAnimationFrame(tick);
  }
  function endParty() {
    if (!party) return;
    party.classList.remove("show");
    cancelAnimationFrame(party.frame);
    clearTimeout(party.timer);
  }

  // ------------------------------------------------------------ level 6: pie the professor
  // A photo of the volunteer professor (photo.js), a pie launcher on a tripod, and one arc per thread.
  // Pies that reach the face stay there as a splat; the others squash where they land.
  const PIE_CREAM = "#fff7e6", PIE_CRUST = "#c98a45";
  const profImg = new Image();
  profImg.onload = () => { if (lv().kind === "sweep") paintCanvas(); };
  profImg.src = PROF_PHOTO.src;
  const pieFrame = () => {
    const r = simRun();
    return r ? r.frame : 0;
  };
  function piePositions() {
    const r = simRun();
    if (r) return r.data.frames[r.frame];
    const P = PH.PIE, a = new Float32Array(2 * P.throws);
    for (let k = 0; k < P.throws; k++) { a[2 * k] = P.hand.x; a[2 * k + 1] = P.hand.y; }
    return a;
  }
  function paintPies(g, w, h, s, X, Y) {
    const P = PH.PIE, { halfW, H } = PH.PHYS, r = simRun();
    const step = pieFrame() * P.every;
    const face = r ? r.data.throws.filter((t) => t.hit && t.hitStep <= step) : [];
    const done = r && atEnd() && !state.playing;
    // the room: a wall, a floor and a banner
    g.fillStyle = TC.bg; g.fillRect(0, 0, w, h);
    g.fillStyle = TC.room; g.fillRect(X(-halfW), Y(H), 2 * halfW * s, H * s);
    g.fillStyle = TC.floor; g.fillRect(X(-halfW), Y(0.55), 2 * halfW * s, 0.55 * s);
    g.fillStyle = TC.wall;
    g.font = `800 ${Math.round(s * 0.42)}px "Big Shoulders Display", "Arial Narrow", sans-serif`;
    g.textAlign = "center"; g.textBaseline = "middle";
    g.fillText("CHARITY PIE DAY", X(0), Y(7.3));
    g.font = `500 ${Math.round(s * 0.2)}px "IBM Plex Sans", system-ui, sans-serif`;
    g.fillStyle = TC.muted;
    g.fillText("one thread per throw · find the gentlest that lands", X(0), Y(6.85));
    g.strokeStyle = TC.wall; g.lineWidth = 3;
    g.strokeRect(X(-halfW), Y(H), 2 * halfW * s, H * s);

    // the launcher: a tripod with a spoon arm at the hand position
    g.strokeStyle = TC.tool; g.lineWidth = Math.max(2, s * 0.07); g.lineCap = "round";
    const hx = P.hand.x, hy = P.hand.y;
    for (const dx of [-0.45, 0.45]) { g.beginPath(); g.moveTo(X(hx - 0.2), Y(hy - 0.35)); g.lineTo(X(hx - 0.2 + dx), Y(0.02)); g.stroke(); }
    g.beginPath(); g.moveTo(X(hx - 0.2), Y(hy - 0.35)); g.lineTo(X(hx + 0.05), Y(hy - 0.05)); g.stroke();
    g.fillStyle = TC.tool; g.beginPath(); g.arc(X(hx - 0.2), Y(hy - 0.35), s * 0.09, 0, Math.PI * 2); g.fill();

    // the professor: the photo, scaled so his face is at the head position and his feet on the floor
    const hd = P.head, k = (hd.y - 0.02) / (PROF_PHOTO.h - PROF_PHOTO.faceY), hr = PROF_PHOTO.headR * k;
    if (profImg.complete && profImg.naturalWidth)
      g.drawImage(profImg, X(hd.x - PROF_PHOTO.faceX * k), Y(hd.y + PROF_PHOTO.faceY * k), PROF_PHOTO.w * k * s, PROF_PHOTO.h * k * s);

    // the arcs so far, one per thread, coloured by block
    const sel = state.selected, best = done && r.data.verdict.ok ? r.data.truth : null;
    if (r) {
      for (let k = 0; k < P.throws; k++) {
        const strong = k === sel || k === best;
        g.strokeStyle = BLOCK_COLORS[Math.floor(k / PH.PHYS.threads)];
        g.globalAlpha = strong ? 0.95 : sel !== null || best !== null ? 0.18 : 0.45;
        g.lineWidth = strong ? 3 : 1.5;
        g.beginPath();
        for (let f = 0; f <= r.frame; f++) {
          const F = r.data.frames[f], x = X(F[2 * k]), y = Y(F[2 * k + 1]);
          f ? g.lineTo(x, y) : g.moveTo(x, y);
        }
        g.stroke();
      }
      g.globalAlpha = 1;
    }

    // the pies: in the air, squashed where they landed, or on the face
    const pos = piePositions();
    const rad = P.pieR * s;
    for (let k = 0; k < P.throws; k++) {
      const t = r ? r.data.throws[k] : null;
      const x = X(pos[2 * k]), y = Y(pos[2 * k + 1]);
      const landed = t && t.stop && t.stop.step <= step;
      if (landed && t.stop.kind === "face") continue; // drawn as the splat below
      g.fillStyle = PIE_CRUST;
      g.beginPath();
      if (landed) g.ellipse(x, y, rad * 1.5, rad * 0.6, 0, 0, Math.PI * 2);
      else g.arc(x, y, rad, 0, Math.PI * 2);
      g.fill();
      g.fillStyle = PIE_CREAM;
      g.beginPath();
      if (landed) g.ellipse(x, y - rad * 0.2, rad * 1.1, rad * 0.35, 0, 0, Math.PI * 2);
      else g.arc(x, y - rad * 0.15, rad * 0.65, 0, Math.PI * 2);
      g.fill();
      if (k === sel) { g.strokeStyle = TC.ink; g.lineWidth = 2; g.beginPath(); g.arc(x, y, rad + 4, 0, Math.PI * 2); g.stroke(); }
    }
    if (face.length) {
      // cream over the face, a few drips, and the tin on top
      g.fillStyle = PIE_CREAM; g.strokeStyle = "#e8dcc4"; g.lineWidth = 1.5;
      // sizes in units of the head radius in the photo, so the cream covers his face and not much more
      const blobs = [[0, -0.05, 0.95], [-0.6, 0.3, 0.5], [0.62, -0.2, 0.5], [-0.3, -0.85, 0.38], [0.45, 0.7, 0.34], [0.8, 0.5, 0.26]];
      for (const [dx, dy, br] of blobs) { g.beginPath(); g.arc(X(hd.x + dx * hr), Y(hd.y + dy * hr), br * hr * s, 0, Math.PI * 2); g.fill(); g.stroke(); }
      for (const [dx, len] of [[-0.5, 0.8], [0.1, 1.15], [0.6, 0.7]]) {
        g.fillRect(X(hd.x + dx * hr) - s * hr * 0.1, Y(hd.y - 0.8 * hr), s * hr * 0.2, len * hr * s);
        g.beginPath(); g.arc(X(hd.x + dx * hr), Y(hd.y - (0.8 + len) * hr), s * hr * 0.13, 0, Math.PI * 2); g.fill();
      }
      g.fillStyle = "#b9c0c8";
      g.beginPath(); g.ellipse(X(hd.x - 0.1 * hr), Y(hd.y + 0.15 * hr), s * hr * 0.85, s * hr * 0.34, -0.3, 0, Math.PI * 2); g.fill();
      g.fillStyle = PIE_CRUST;
      g.beginPath(); g.ellipse(X(hd.x - 0.1 * hr), Y(hd.y + 0.15 * hr), s * hr * 0.62, s * hr * 0.2, -0.3, 0, Math.PI * 2); g.fill();
      g.fillStyle = TC.ink;
      g.font = `800 ${Math.round(s * 0.36)}px "Big Shoulders Display", "Arial Narrow", sans-serif`;
      g.fillText(face.length > 1 ? `SPLAT ×${face.length}` : "SPLAT", X(hd.x - 1.3), Y(hd.y + 0.85));
    }
    if (best !== null) {
      g.fillStyle = TC.ink;
      g.font = `600 ${Math.round(s * 0.2)}px "IBM Plex Sans", system-ui, sans-serif`;
      g.textAlign = "right";
      g.fillText(`gentlest: throw ${best} · ${r.data.throws[best].speed.toFixed(2)} m/s`, X(P.jacket.x0 - 0.3), Y(0.9));
      g.textAlign = "center";
    }
  }
  function canvasPie(e) {
    const cv = $("phys-canvas");
    if (!cv) return null;
    const rect = cv.getBoundingClientRect();
    const s = rect.width / (VIEW.x1 - VIEW.x0);
    const wx = (e.clientX - rect.left) / s + VIEW.x0, wy = VIEW.y1 - (e.clientY - rect.top) / s;
    const pos = piePositions();
    let best = null, bestD = 0.45;
    for (let k = 0; k < PH.PIE.throws; k++) {
      const d = Math.hypot(wx - pos[2 * k], wy - pos[2 * k + 1]);
      if (d < bestD) { best = k; bestD = d; }
    }
    return best;
  }
  const pieOutcome = (t, step = Infinity) =>
    !t.stop || t.stop.step > step ? "in the air"
      : t.stop.kind === "face" ? "hit the face" : `landed on the ${t.stop.kind}`;
  function vizPies(L) {
    const r = simRun(), P = PH.PIE;
    const cols = PH.PHYS.blocks * PH.PHYS.threads;
    const done = r && atEnd() && !state.playing;
    const cpu = state.run && state.run.kind === "cpunote";
    let h = cpu ? `<div class="cpu-core">One CPU core · <code>for (int i = 0; i &lt; 20; ++i) fly(i);</code></div>` : "";
    h += `<p class="viz-title">The lecture hall, 12 × 8 units · each arc is one thread's throw, coloured by its block</p>`;
    h += `<canvas id="phys-canvas" class="phys-canvas" role="img" aria-label="Twenty pie throws towards a volunteer professor. Click a pie to inspect its thread."></canvas>`;
    h += `<p class="viz-title" style="margin-top:12px">Threads · each chip shows threadIdx.x</p>`;
    h += blockRow(L, cols);
    h += lane(cols, "thread", threadsOf(L).map((t) => {
      const c = ["chip"];
      let style = "";
      if (t.id >= P.throws) { if (r) c.push("returned"); }
      else { c.push("ball"); style = ` style="--bc:${BLOCK_COLORS[t.bx]}"`; }
      if (state.selected === t.id) c.push("selected");
      return `<button type="button" class="${c.join(" ")}"${style} data-thread="${t.id}" aria-label="${esc(threadLabel(L, t))}">${t.tx}</button>`;
    }));
    h += lane(cols, "throw i", threadsOf(L).map((t) => `<span class="ival${t.id >= P.throws ? " bad" : ""}">${t.id}</span>`));
    if (done)
      h += lane(cols, "result", threadsOf(L).map((t) => {
        const th = r.data.throws[t.id];
        return `<span class="ival${th && th.hit ? " hit" : ""}">${th ? (th.hit ? "hit" : "–") : ""}</span>`;
      }));
    const items = BLOCK_COLORS.map((c, b) => [`background:${c}`, `block ${b}`]);
    items.push(["background:repeating-linear-gradient(135deg,var(--stripe-a) 0 4px,var(--stripe-b) 4px 8px)", "no throw: returns early"]);
    return h + legend(items);
  }
  function pieTicker(r) {
    const P = PH.PIE;
    let now;
    if (state.run && state.run.kind === "cpunote") now = "<b>CPU loop</b> · one core, one throw after another";
    else if (!r) now = `<b>Ready.</b> One launch: ${P.throws} threads each fly a whole throw, ${P.steps.toLocaleString("en-US")} steps of 1/960 s.`;
    else {
      const step = r.frame * P.every;
      const onFace = r.data.throws.filter((t) => t.hit && t.hitStep <= step).length;
      const end = atEnd() && r.data.afterOps.some((o) => o.startsWith("record"))
        ? ` · gentlest = <b>${r.data.gentlest}</b>` : "";
      now = `<b>t = ${(step / 960).toFixed(2)} s</b> · step ${step.toLocaleString("en-US")} of ${P.steps.toLocaleString("en-US")} · ${onFace} pie${onFace === 1 ? "" : "s"} on the face${end}`;
    }
    return `<span class="now">${now}</span><span class="counters"><span class="counter gpu">GPU: 1 launch, ${P.steps.toLocaleString("en-US")} steps per thread</span>` +
      `<span class="counter cpu">CPU loop: ${(P.throws * P.steps).toLocaleString("en-US")} steps</span><span class="counter">${P.throws} throws</span></span>`;
  }
  function pieInspector(L) {
    if (state.selected === null) return "Click a pie or a thread to see its throw.";
    const t = threadsOf(L)[state.selected];
    if (!t) return "";
    const head = esc(threadLabel(L, t));
    if (t.id >= PH.PIE.throws)
      return `<div class="card"><h4>${head} · no throw</h4><p class="vars">i = ${t.id}, and n = ${PH.PIE.throws}, so this thread returns before throwing.</p></div>`;
    const r = simRun();
    const speed = (9.25 + 0.25 * t.id).toFixed(2);
    let h = `<div class="card"><h4>${head} · throw ${t.id}</h4><p class="vars">speed[${t.id}] = ${speed} m/s at 35°</p>`;
    if (r) {
      const th = r.data.throws[t.id], step = r.frame * PH.PIE.every;
      const when = th.stop && th.stop.step <= step ? ` at t = ${((th.stop.step + 1) / 960).toFixed(2)} s` : "";
      h += `<p class="vars">${esc(pieOutcome(th, step))}${when}${th.hit ? " · hit = true" : ""}</p>`;
      if (atEnd() && r.data.racers.includes(t.id))
        h += `<p class="vars" style="color:var(--bad)">Wrote gentlest = ${t.id} after reading ${PH.PIE.throws}, at the same moment as the other hits.</p>`;
    }
    return h + `</div>`;
  }

  // ------------------------------------------------------------ energy chart (level 5)
  // The student's kernel run on every other frame of the 4-second simulation, against the true total.
  // Colors: slots 1 and 2 of the dataviz reference palette, checked with its validator. Blue always
  // means "your kernel", orange always means "correct total".
  const SERIES = { got: ["#2a78d6", "your kernel"], truth: ["#eb6834", "correct total"] };
  let chartData = null;
  function niceStep(raw) {
    const p = 10 ** Math.floor(Math.log10(raw));
    return [1, 2, 2.5, 5, 10].map((m) => m * p).find((v) => v >= raw);
  }
  function energyChart(r) {
    if (r.kind !== "gpu" || r.data.diag.faults.length) return "";
    r.data.series ??= PH.energySeries(lv(), prog(), 2);
    const S = r.data.series;
    if (S.some((p) => p.got === null)) return "";
    const wrong = S.some((p) => Math.abs(p.got - p.truth) > 1e-6 * p.truth);
    const keys = wrong ? ["got", "truth"] : ["got"];
    const W = 600, H = 232, l = 44, right = 116, t = 12, b = 28;
    const pw = W - l - right, ph = H - t - b;
    const top = Math.max(...S.map((p) => Math.max(p.truth, p.got)));
    const step = niceStep(top / 4), ymax = Math.ceil(top / step) * step;
    const tmax = S.at(-1).t;
    const X = (x) => l + (x / tmax) * pw, Y = (v) => t + ph - (v / ymax) * ph;
    chartData = { S, keys, X, W };
    let svg = "";
    for (let v = 0; v <= ymax + 1e-9; v += step)
      svg += `<line x1="${l}" x2="${l + pw}" y1="${Y(v)}" y2="${Y(v)}" class="grid"/>` +
        `<text x="${l - 8}" y="${Y(v) + 4}" class="tick" text-anchor="end">${v.toLocaleString("en-US")}</text>`;
    for (let x = 0; x <= tmax + 1e-9; x++)
      svg += `<text x="${X(x)}" y="${t + ph + 19}" class="tick" text-anchor="middle">${x} s</text>`;
    for (const k of keys)
      svg += `<path class="ln" stroke="${SERIES[k][0]}" d="${S.map((p, i) => `${i ? "L" : "M"}${X(p.t).toFixed(1)},${Y(p[k]).toFixed(1)}`).join("")}"/>`;
    for (const k of keys) {
      const p = S.at(-1);
      svg += `<circle class="dot" cx="${X(p.t)}" cy="${Y(p[k])}" r="4" fill="${SERIES[k][0]}"/>` +
        `<text class="endlbl" x="${X(p.t) + 10}" y="${Y(p[k]) + 4}"><tspan class="v">${Math.round(p[k]).toLocaleString("en-US")}</tspan>${keys.length > 1 ? ` ${SERIES[k][1]}` : ""}</text>`;
    }
    svg += `<line class="xh" x1="0" x2="0" y1="${t}" y2="${t + ph}" visibility="hidden"/>`;
    const first = S[0], last = S.at(-1);
    const sub = wrong
      ? "Your kernel loses most of the total on every frame."
      : `It falls from ${Math.round(first.truth).toLocaleString("en-US")} to ${Math.round(last.truth).toLocaleString("en-US")}: collisions and air drag turn motion into heat.`;
    const legend = keys.length > 1
      ? `<div class="keys">${keys.map((k) => `<span><i style="background:${SERIES[k][0]}"></i>${SERIES[k][1]}</span>`).join("")}</div>` : "";
    const rows = S.filter((p, i) => i % 15 === 0)
      .map((p) => `<tr><td>${p.t.toFixed(1)} s</td>${keys.map((k) => `<td>${Math.round(p[k]).toLocaleString("en-US")}</td>`).join("")}</tr>`).join("");
    return `<div class="echart"><h4>Total energy, every frame of the 4-second run</h4><p class="sub">${esc(sub)}</p>${legend}` +
      `<div class="echart-plot" tabindex="0" aria-label="Chart of total energy over time. Use the left and right arrow keys to read values.">` +
      `<svg viewBox="0 0 ${W} ${H}" role="img" aria-hidden="true">${svg}</svg><div class="tip" hidden></div></div>` +
      `<details><summary>Show the numbers</summary><table><thead><tr><th>time</th>${keys.map((k) => `<th>${SERIES[k][1]}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></details></div>`;
  }
  function chartShow(plot, idx) {
    if (!chartData) return;
    const { S, keys, X, W } = chartData;
    idx = Math.max(0, Math.min(S.length - 1, idx));
    plot.dataset.idx = idx;
    const p = S[idx], x = X(p.t);
    const xh = plot.querySelector(".xh");
    xh.setAttribute("x1", x);
    xh.setAttribute("x2", x);
    xh.setAttribute("visibility", "visible");
    const tip = plot.querySelector(".tip");
    tip.hidden = false;
    tip.innerHTML = `<div class="when">${p.t.toFixed(2)} s</div>` +
      keys.map((k) => `<div><i style="background:${SERIES[k][0]}"></i><strong>${Math.round(p[k]).toLocaleString("en-US")}</strong> ${SERIES[k][1]}</div>`).join("");
    const frac = x / W;
    tip.style.left = frac < 0.6 ? `calc(${(frac * 100).toFixed(2)}% + 10px)` : "";
    tip.style.right = frac < 0.6 ? "" : `calc(${((1 - frac) * 100).toFixed(2)}% + 10px)`;
  }
  function chartHide(plot) {
    plot.querySelector(".xh")?.setAttribute("visibility", "hidden");
    const tip = plot.querySelector(".tip");
    if (tip) tip.hidden = true;
  }
  document.addEventListener("pointermove", (e) => {
    const plot = e.target.closest?.(".echart-plot");
    if (!plot || !chartData) return;
    const rect = plot.getBoundingClientRect();
    const sx = ((e.clientX - rect.left) / rect.width) * chartData.W;
    let best = 0;
    chartData.S.forEach((p, i) => { if (Math.abs(chartData.X(p.t) - sx) < Math.abs(chartData.X(chartData.S[best].t) - sx)) best = i; });
    chartShow(plot, best);
  });
  document.addEventListener("pointerout", (e) => {
    const plot = e.target.closest?.(".echart-plot");
    if (plot && !plot.contains(e.relatedTarget) && document.activeElement !== plot) chartHide(plot);
  });
  document.addEventListener("focusin", (e) => { if (e.target.classList?.contains("echart-plot")) chartShow(e.target, Number(e.target.dataset.idx ?? 0)); });
  document.addEventListener("focusout", (e) => { if (e.target.classList?.contains("echart-plot")) chartHide(e.target); });
  document.addEventListener("keydown", (e) => {
    const plot = e.target.classList?.contains("echart-plot") ? e.target : null;
    if (!plot || (e.key !== "ArrowLeft" && e.key !== "ArrowRight")) return;
    e.preventDefault();
    chartShow(plot, Number(plot.dataset.idx ?? 0) + (e.key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 10 : 1));
  });

  window.addEventListener("resize", () => { if (isSim(lv())) paintCanvas(); });

  // ------------------------------------------------------------ wiring
  $("btn-run").addEventListener("click", () => {
    if (state.playing) { pause(); return; }
    if (state.run && state.run.kind === "gpu" && !atEnd()) { play(); return; }
    startRun("gpu", false);
  });
  $("btn-step").addEventListener("click", stepOnce);
  $("btn-cpu").addEventListener("click", () => {
    if (state.playing && state.run?.kind === "cpu") return;
    startRun("cpu", false);
  });
  $("btn-reset").addEventListener("click", () => {
    resetRun();
    $("btn-run").focus(); // the reset button hides itself once there is nothing to reset
  });
  $("btn-sound").addEventListener("click", () => {
    soundOn = !soundOn;
    try { localStorage.setItem(SOUND_KEY, soundOn ? "on" : "off"); } catch (e) { /* storage unavailable */ }
    showSoundButton();
  });
  showSoundButton();
  // Theme: follows the system until the learner picks one with the header toggle.
  const THEME_KEY = "threadlab-theme";
  const showThemeButton = () => {
    const dark = document.documentElement.dataset.theme === "dark";
    $("btn-theme").setAttribute("aria-pressed", String(dark));
  };
  const applyTheme = (theme) => {
    document.documentElement.dataset.theme = theme;
    showThemeButton();
    readCanvasTheme();
    paintCanvas();
  };
  $("btn-theme").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* storage unavailable */ }
    applyTheme(next);
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", (e) => {
    let saved = null;
    try { saved = localStorage.getItem(THEME_KEY); } catch (e2) { /* storage unavailable */ }
    if (saved !== "dark" && saved !== "light") applyTheme(e.matches ? "dark" : "light");
  });
  showThemeButton();
  $("speed").addEventListener("change", (e) => { state.speed = e.target.value; });
  $("levels").addEventListener("click", (e) => {
    const tab = e.target.closest("[data-level]");
    if (tab) { if (tab.getAttribute("aria-disabled") !== "true") setLevel(Number(tab.dataset.level)); return; }
    const track = e.target.closest("[data-track]");
    if (track && track.dataset.track !== trackOf(state.level)) {
      const all = trackLevels(track.dataset.track);
      setLevel(all.find((i) => !state.done[i]) ?? all[0]);
    }
  });
  $("stage").addEventListener("click", (e) => {
    if (e.target.id === "btn-next") { setLevel(state.level + 1); return; }
    if (e.target.id === "btn-classic") { setLevel(trackLevels("classic")[0]); return; }
    if (e.target.id === "phys-canvas" && lv().kind === "sweep") {
      const k = canvasPie(e);
      state.selected = k === null || k === state.selected ? null : k;
      renderStage();
      return;
    }
    if (e.target.id === "phys-canvas") {
      if (lv().kind !== "physics") return;
      const ball = canvasBall(e);
      const first = simRun()?.data.firstThreads;
      const owner = ball === null ? -1 : first ? first.findIndex((t) => t.i === ball && !t.fault) : ball;
      state.selectedBall = owner < 0 ? ball : null;
      state.selected = owner < 0 || owner === state.selected ? null : owner;
      renderStage();
      return;
    }
    const chip = e.target.closest("[data-thread]");
    if (!chip) return;
    const id = Number(chip.dataset.thread);
    state.selected = state.selected === id ? null : id;
    state.selectedBall = null;
    renderStage();
  });

  const hashFor = (i) => `#${trackOf(i) === "classic" ? "classic" : "level"}-${LEVELS[i].number}`;
  function levelFromHash() {
    const m = /^#(level|classic)-(\d+)$/.exec(location.hash);
    if (!m) return -1;
    const track = m[1] === "classic" ? "classic" : "physics";
    return LEVELS.findIndex((L) => L.track === track && L.number === Number(m[2]));
  }
  function setLevel(i) {
    if (!LEVELS[i]) return;
    stopTimer();
    state.level = i;
    state.run = null;
    state.selected = null;
    state.selectedBall = null;
    renderAll();
    save();
    try { history.replaceState(null, "", hashFor(i)); } catch (e) { /* ignore */ }
    window.scrollTo({ top: 0, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  }

  function renderAll() {
    renderTabs();
    renderMission();
    renderPalette();
    renderKernel();
    renderStage();
  }

  function renderCredits() {
    const items = FACES.map((f) =>
      `<li><b>${esc(f.name)}</b>: “${esc(f.title)}” by ${esc(f.creator)}, <a href="${esc(f.source)}" target="_blank" rel="noopener">Wikimedia Commons</a>, ` +
      `<a href="${esc(f.licenseUrl)}" target="_blank" rel="noopener">${esc(f.license)}</a>, cropped.</li>`).join("");
    $("credits").innerHTML = `<details><summary>Photo credits: the faces on the balls</summary>` +
      `<p>${FACES.length} photos from Wikimedia Commons, each cropped to the face; ${PH.PHYS.count} are picked at random each time the page loads.</p>` +
      `<ul>${items}</ul></details>`;
  }

  // ------------------------------------------------------------ guided walkthrough
  // Spotlights one part of the workflow at a time (goal → blocks → kernel → run → inspect) and greys out the rest.
  const TOUR = [
    { target: "#mission", step: "Your goal", title: "Read the mission",
      text: () => `This is what “${lv().title}” asks you to do. Read it first. Open Hint below the mission if you get stuck.`, next: "Got it" },
    { target: "#palette", step: "Step 1", title: "Choose blocks",
      text: () => `Each block is one line of ${lv().kind === "host" ? "CPU code" : "CUDA"}. Click a block to add it to the end, or drag it to exactly where it belongs.` },
    { target: "#kernel", step: "Step 2", title: () => lv().kind === "host" ? "Build your code" : "Build your kernel",
      text: () => "Statements run top to bottom. Reorder them with the arrows, pick values from the dropdowns, and drag a block back to Blocks to remove it." },
    { target: "#stage .stage-head", step: "Step 3", title: "Run it",
      text: () => isSim(lv())
        ? `Run on GPU plays the whole simulation. Step advances one frame at a time so you can follow what happens.${lv().kind === "host" ? "" : " After your first run, Compare with CPU shows what one core would do."}`
        : "Run on GPU plays the whole kernel. Step runs one statement at a time so you can follow every thread. After your first run, Compare with CPU shows how long one core would take." },
    { target: "#viz", step: "Step 4", title: "Inspect the result",
      text: () => "Watch the threads and memory change. Click a thread to see its own variables, then read the verdict below to find out what to fix.", next: "Start building" },
  ];
  const tourCard = document.createElement("div");
  tourCard.className = "tour-card";
  tourCard.setAttribute("role", "dialog");
  tourCard.setAttribute("aria-modal", "true");
  tourCard.setAttribute("aria-labelledby", "tour-title");
  tourCard.setAttribute("aria-describedby", "tour-text");
  tourCard.hidden = true;
  document.body.appendChild(tourCard);
  let tourIndex = -1;
  // The walkthrough plays only once per browser; finishing or skipping it marks it as seen.
  const TOUR_KEY = "threadlab-tour-seen";
  const tourSeen = () => { try { return localStorage.getItem(TOUR_KEY) === "1"; } catch (e) { return false; } };

  function clearSpot() {
    document.querySelectorAll(".tour-dim").forEach((n) => n.classList.remove("tour-dim"));
    document.querySelectorAll(".tour-focus").forEach((n) => n.classList.remove("tour-focus"));
  }
  // Grey out every sibling of the target and of each of its ancestors, so only the target stays in colour.
  function spotlight(target) {
    clearSpot();
    target.classList.add("tour-focus");
    for (let el = target; el.parentElement && el !== document.body; el = el.parentElement)
      for (const sib of el.parentElement.children)
        if (sib !== el && sib !== tourCard && !["SCRIPT", "STYLE"].includes(sib.tagName)) sib.classList.add("tour-dim");
  }
  function placeCard() {
    if (tourIndex < 0) return;
    const target = document.querySelector(TOUR[tourIndex].target);
    if (!target) return;
    const r = target.getBoundingClientRect(), m = 16, gap = 18;
    const w = tourCard.offsetWidth, h = tourCard.offsetHeight, W = innerWidth, H = innerHeight;
    const clampY = (y) => Math.max(m, Math.min(y, H - h - m));
    const clampX = (x) => Math.max(m, Math.min(x, W - w - m));
    let x, y;
    if (W - r.right >= w + gap + m) { x = r.right + gap; y = clampY(r.top); }
    else if (r.left >= w + gap + m) { x = r.left - w - gap; y = clampY(r.top); }
    else if (H - r.bottom >= h + gap + m) { x = clampX(r.left); y = r.bottom + gap; }
    else if (r.top >= h + gap + m) { x = clampX(r.left); y = r.top - h - gap; }
    else { x = clampX((W - w) / 2); y = H - h - m; }
    tourCard.style.left = `${Math.round(x)}px`;
    tourCard.style.top = `${Math.round(y)}px`;
  }
  function showTourStep(i) {
    tourIndex = i;
    const s = TOUR[i];
    const target = document.querySelector(s.target);
    if (!target) { endTour(); return; }
    const val = (v) => (typeof v === "function" ? v() : v);
    const last = i === TOUR.length - 1;
    tourCard.innerHTML =
      `<div class="tour-step"><p class="eyebrow">${esc(s.step)}</p>` +
      `<span class="tour-dots" aria-label="Part ${i + 1} of ${TOUR.length}">${TOUR.map((_, k) => `<i class="${k <= i ? "on" : ""}"></i>`).join("")}</span></div>` +
      `<h2 id="tour-title">${esc(val(s.title))}</h2><p id="tour-text">${esc(val(s.text))}</p>` +
      `<div class="tour-actions">` +
      (last ? "" : `<button type="button" class="tour-skip" data-tour="skip">Skip tour</button>`) +
      (i > 0 ? `<button type="button" class="btn" data-tour="back"${last ? ' style="margin-right:auto"' : ""}>Back</button>` : "") +
      `<button type="button" class="btn primary" data-tour="next">${esc(s.next || "Next")}</button></div>`;
    spotlight(target);
    target.scrollIntoView({ block: innerWidth < 900 || target.offsetHeight > innerHeight * .7 ? "start" : "center", behavior: "auto" });
    tourCard.hidden = false;
    placeCard();
    tourCard.querySelector('[data-tour="next"]').focus({ preventScroll: true });
  }
  // Keyboard handling shared by the start screen and the tour: Escape closes, Tab stays inside the dialog.
  function trapFocus(dialog, close) {
    dialog.addEventListener("keydown", (e) => {
      if (e.key === "Escape") { e.preventDefault(); close(); return; }
      if (e.key !== "Tab") return;
      const items = [...dialog.querySelectorAll("button")];
      const first = items[0], last = items[items.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    });
  }
  function startTour() {
    if (tourSeen()) return;
    stopTimer();
    backdrop().forEach((n) => { n.inert = true; });
    // let the splash fade and the new level render before measuring positions
    setTimeout(() => showTourStep(0), matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 280);
  }
  function endTour() {
    try { localStorage.setItem(TOUR_KEY, "1"); } catch (e) { /* storage unavailable */ }
    tourIndex = -1;
    clearSpot();
    tourCard.hidden = true;
    tourCard.innerHTML = "";
    backdrop().forEach((n) => { n.inert = false; });
    window.scrollTo({ top: 0, behavior: "auto" });
    const h1 = document.querySelector("#mission h1");
    if (h1) { h1.tabIndex = -1; h1.focus({ preventScroll: true }); }
  }
  tourCard.addEventListener("click", (e) => {
    const act = e.target.closest("[data-tour]")?.dataset.tour;
    if (act === "skip") endTour();
    else if (act === "back") showTourStep(tourIndex - 1);
    else if (act === "next") tourIndex < TOUR.length - 1 ? showTourStep(tourIndex + 1) : endTour();
  });
  trapFocus(tourCard, endTour);
  addEventListener("resize", placeCard);
  addEventListener("scroll", placeCard, { passive: true });

  // ------------------------------------------------------------ splash screen
  const backdrop = () => document.querySelectorAll("body > header, body > main, body > footer");
  let splashReturnFocus = null;
  function openSplash() {
    const el = $("splash");
    const L = lv();
    const started = state.done.some(Boolean) || state.programs.some((p, i) => p.length !== LEVELS[i].start.length) || state.level !== 0;
    const tracks = TRACKS.map(([id, name]) => {
      const all = trackLevels(id), solved = all.filter((i) => state.done[i]).length;
      const meta = solved === all.length ? `All ${all.length} levels solved` : solved ? `${solved} of ${all.length} solved` : `${all.length} levels`;
      return `<button type="button" class="splash-track" data-splash-track="${id}"><strong>${esc(name)}</strong><span>${meta}</span></button>`;
    }).join("");
    const blocks = range(4).map((b) =>
      `<div class="splash-block"><div class="splash-block-label">block ${b}</div><div class="splash-chips" style="--b:${b}">` +
      range(8).map((t) => `<span class="splash-chip">${b * 8 + t}</span>`).join("") + `</div></div>`).join("");
    el.innerHTML =
      `<div class="splash-inner">` +
      `<div class="splash-copy">` +
      `<p class="eyebrow">A CUDA workshop game</p>` +
      `<h1 class="splash-title" id="splash-title">Thread <span>Lab</span></h1>` +
      `<p class="splash-lede">Learn how a GPU splits work across thousands of threads by building CUDA kernels from blocks and watching each thread run.</p>` +
      `<button type="button" class="btn primary splash-start" data-splash-start>${started ? "Continue" : "Start"} <b>· Level ${L.number}: ${esc(L.title)}</b></button>` +
      `<p class="splash-or">Or choose a track</p><div class="splash-tracks">${tracks}</div>` +
      `</div>` +
      `<div class="splash-gpu" aria-hidden="true">` +
      `<div class="splash-gpu-head"><h2>The GPU</h2><span>4 blocks × 8 threads</span></div>` +
      `<div class="splash-code"><span class="kw">int</span> i = blockIdx.x * blockDim.x + threadIdx.x;\n<span class="cm">// every thread gets its own i</span></div>` +
      blocks +
      `<p class="splash-caption">Each block's threads run the same step together, each on its own element.</p>` +
      `</div></div>`;
    splashReturnFocus = document.activeElement;
    el.classList.remove("leaving");
    el.hidden = false;
    backdrop().forEach((n) => { n.inert = true; });
    document.documentElement.style.overflow = "hidden";
    el.querySelector("[data-splash-start]").focus();
  }
  function closeSplash(level) {
    const el = $("splash");
    if (el.hidden || el.classList.contains("leaving")) return;
    backdrop().forEach((n) => { n.inert = false; });
    document.documentElement.style.overflow = "";
    const changed = level !== undefined && level !== state.level;
    if (changed) setLevel(level);
    const done = () => { el.hidden = true; el.classList.remove("leaving"); el.innerHTML = ""; };
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) done();
    else { el.classList.add("leaving"); setTimeout(done, 250); }
    const target = !changed && splashReturnFocus && splashReturnFocus.id === "btn-home" ? splashReturnFocus : document.querySelector("#mission h1");
    if (target) {
      if (target.tagName === "H1") target.tabIndex = -1;
      target.focus({ preventScroll: true });
    }
  }
  $("splash").addEventListener("click", (e) => {
    if (e.target.closest("[data-splash-start]")) { closeSplash(); startTour(); return; }
    const track = e.target.closest("[data-splash-track]");
    if (!track) return;
    const all = trackLevels(track.dataset.splashTrack);
    closeSplash(all.find((i) => !state.done[i]) ?? all[0]);
    startTour();
  });
  trapFocus($("splash"), closeSplash);
  $("btn-home").addEventListener("click", () => { pause(); openSplash(); });

  function start() {
    renderCredits();
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(STORE) || "null"); } catch (e) { /* no saved progress */ }
    restore(saved);
    const fromHash = levelFromHash();
    if (fromHash >= 0) state.level = fromHash;
    renderAll();
    if (fromHash < 0) openSplash(); // deep links to a level skip the start screen
  }
  window.addEventListener("hashchange", () => {
    const i = levelFromHash();
    if (i >= 0 && i !== state.level) setLevel(i);
  });
  start();
})();

