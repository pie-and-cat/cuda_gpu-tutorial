// Thread Lab engine: a tiny, deterministic model of a CUDA GPU.
// No DOM here, so it can be tested in Node.
//
// Execution model (kept simple on purpose, and explained in the game):
// - All live threads run statement k together, in lockstep ("step k").
// - A thread's own writes are visible to it immediately. Writes by OTHER threads
//   become visible only after a barrier (__syncthreads) or when the kernel ends.
//   A read that misses another thread's newer value is recorded as "stale".
// - atomicAdd calls take turns: each one is its own step, and is visible at once.
// - Reading or writing outside an array faults that thread (it stops).
const Engine = (() => {
  class ThreadFault extends Error {
    constructor(message, info) {
      super(message);
      this.info = info;
    }
  }

  // ---------------------------------------------------------------- threads
  function makeThreads(grid, block) {
    const out = [];
    for (let by = 0; by < grid.y; by++)
      for (let bx = 0; bx < grid.x; bx++)
        for (let ty = 0; ty < block.y; ty++)
          for (let tx = 0; tx < block.x; tx++) out.push({ id: out.length, bx, by, tx, ty });
    return out;
  }

  function threadLabel(level, t) {
    if (level.dims === 2) return `block (${t.bx}, ${t.by}), thread (${t.tx}, ${t.ty})`;
    if (level.grid.x === 1) return `thread ${t.tx}`;
    return `block ${t.bx}, thread ${t.tx}`;
  }

  // ---------------------------------------------------------------- memory
  class Memory {
    constructor(spec) {
      this.arrays = {};
      for (const [name, s] of Object.entries(spec)) {
        this.arrays[name] = {
          name,
          kind: s.kind || "global",
          w: s.w || null,
          h: s.h || null,
          len: s.data.length,
          committed: s.data.slice(),
          pending: new Map(), // idx -> [{ value, writer }]
          writers: new Map(), // idx -> Set(thread ids), whole run
        };
      }
      this.conflicts = []; // { arr, idx, writers: [ids], kept }
    }
    read(arr, idx, tid) {
      const A = this.arrays[arr];
      const pend = A.pending.get(idx);
      if (pend) {
        const own = [...pend].reverse().find((p) => p.writer === tid);
        if (own) return { value: own.value, stale: false };
        const other = pend.at(-1);
        return { value: A.committed[idx], stale: true, writer: other.writer, newer: other.value };
      }
      return { value: A.committed[idx], stale: false };
    }
    write(arr, idx, value, tid) {
      const A = this.arrays[arr];
      if (!A.pending.has(idx)) A.pending.set(idx, []);
      A.pending.get(idx).push({ value, writer: tid });
      if (!A.writers.has(idx)) A.writers.set(idx, new Set());
      A.writers.get(idx).add(tid);
    }
    atomicAdd(arr, idx, value, tid) {
      const A = this.arrays[arr];
      A.committed[idx] = (A.committed[idx] ?? 0) + value;
      if (!A.writers.has(idx)) A.writers.set(idx, new Set());
      A.writers.get(idx).add(tid);
      return A.committed[idx];
    }
    commit() {
      for (const A of Object.values(this.arrays)) {
        for (const [idx, list] of A.pending) {
          const kept = list.at(-1);
          const distinct = [...new Set(list.map((p) => p.writer))];
          if (distinct.length > 1) this.conflicts.push({ arr: A.name, idx, writers: distinct, kept: kept.writer });
          A.committed[idx] = kept.value;
        }
        A.pending.clear();
      }
    }
    view() {
      const v = {};
      for (const A of Object.values(this.arrays)) {
        const values = A.committed.slice();
        const pending = new Set();
        for (const [idx, list] of A.pending) {
          values[idx] = list.at(-1).value;
          pending.add(idx);
        }
        v[A.name] = { values, pending };
      }
      return v;
    }
  }

  // ---------------------------------------------------------------- blocks
  const F = (f) => ({ f }); // a dropdown field
  const M = (f) => ({ m: f }); // a read-only mirror of a field

  function blockText(def, fields) {
    return def.parts
      .map((p) => (typeof p === "string" ? p : fields[p.f ?? p.m] ?? def.fields[p.f ?? p.m]))
      .join("");
  }

  const IDX_1D = ["blockIdx.x", "blockDim.x", "threadIdx.x", "gridDim.x"];
  const IDX_2D = ["blockIdx.x", "blockIdx.y", "blockDim.x", "blockDim.y", "threadIdx.x", "threadIdx.y"];

  function indexBlock(name, defaults, opts) {
    return {
      cat: "index",
      parts: [`int ${name} = `, F("a"), " * ", F("b"), " + ", F("c"), ";"],
      fields: { a: defaults[0], b: defaults[1], c: defaults[2] },
      opts: { a: opts, b: opts, c: opts },
      exec(ctx, f) {
        ctx.set(name, ctx.builtin(f.a) * ctx.builtin(f.b) + ctx.builtin(f.c));
      },
    };
  }

  // ---------------------------------------------------------------- levels
  const PI = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4, 3, 3, 8, 3];
  const E = [2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5, 2, 3, 5, 3, 6, 0, 2, 8, 7, 4, 7, 1];
  const X16 = [5, 3, 8, 1, 9, 2, 7, 4, 6, 8, 3, 5, 9, 1, 4, 7];

  // 14 x 10 image: the letters G P U, 9 = bright, 1 = background
  const GPU_ROWS = [
    "..............",
    ".###.###..#..#",
    "#....#..#.#..#",
    "#....#..#.#..#",
    "#.##.###..#..#",
    "#..#.#....#..#",
    "#..#.#....#..#",
    ".###.#.....##.",
    "..............",
    "..............",
  ];
  const IMG_W = 14, IMG_H = 10;
  const IMG = GPU_ROWS.join("").split("").map((ch) => (ch === "#" ? 9 : 1));
  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
  function blurAt(get, x, y) {
    let sum = 0;
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) sum += get(clamp(y + dy, 0, IMG_H - 1), clamp(x + dx, 0, IMG_W - 1));
    return Math.floor((sum + 4) / 9);
  }
  const BLURRED = [];
  for (let y = 0; y < IMG_H; y++)
    for (let x = 0; x < IMG_W; x++) BLURRED.push(blurAt((r, c) => IMG[r * IMG_W + c], x, y));

  const sum16 = X16.reduce((s, v) => s + v, 0);

  // Blocks that several levels use as they are (physics.js reuses them too).
  const IDX_FIXED = {
    cat: "index",
    parts: ["int i = blockIdx.x * blockDim.x + threadIdx.x;"],
    exec(ctx) {
      ctx.set("i", ctx.builtin("blockIdx.x") * ctx.builtin("blockDim.x") + ctx.builtin("threadIdx.x"));
    },
  };
  const GUARD = {
    cat: "control",
    parts: ["if (i >= n) return;"],
    exec(ctx) {
      if (ctx.get("i") >= ctx.n) ctx.ret();
    },
  };

  const LEVELS = [
    {
      id: "vector-add",
      number: 1,
      title: "One thread per element",
      concept: "Thread indexing",
      dims: 1,
      grid: { x: 4, y: 1 },
      block: { x: 8, y: 1 },
      n: 28,
      kernelHeader: ["__global__ void vectorAdd(const int* a, const int* b, int* c, int n) {"],
      launch: "vectorAdd<<<4, 8>>>(a, b, c, 28);   // 4 blocks × 8 threads = 32 threads",
      cpuCode: "for (int i = 0; i < 28; ++i) c[i] = a[i] + b[i];",
      mission:
        "Add two 28-element arrays with 32 threads. Each thread must work out its own element i, stay inside the arrays, and write c[i] = a[i] + b[i].",
      hint: "Thread 3 of block 2 should get i = 19, because 2 × 8 + 3 = 19. The last 4 threads have no element to work on.",
      dataNote: "a holds the first digits of π, b the first digits of e.",
      palette: ["idx_i", "idx_tid", "guard", "add", "mul"],
      start: [{ type: "add" }],
      blocks: {
        idx_i: indexBlock("i", ["threadIdx.x", "blockDim.x", "blockIdx.x"], IDX_1D),
        idx_tid: {
          cat: "index",
          parts: ["int i = threadIdx.x;"],
          exec(ctx) {
            ctx.set("i", ctx.builtin("threadIdx.x"));
          },
        },
        guard: GUARD,
        add: {
          cat: "memory",
          parts: ["c[i] = a[i] + b[i];"],
          exec(ctx) {
            const i = ctx.get("i");
            ctx.write("c", i, ctx.read("a", i) + ctx.read("b", i));
          },
        },
        mul: {
          cat: "memory",
          parts: ["c[i] = a[i] * b[i];"],
          exec(ctx) {
            const i = ctx.get("i");
            ctx.write("c", i, ctx.read("a", i) * ctx.read("b", i));
          },
        },
      },
      memory: () => ({ a: { data: PI }, b: { data: E }, c: { data: new Array(28).fill(null) } }),
      expected: PI.map((v, i) => v + E[i]),
      cpuSteps: 28,
    },
    {
      id: "image-blur",
      number: 2,
      title: "One thread per pixel",
      concept: "2-D grids",
      dims: 2,
      grid: { x: 4, y: 3 },
      block: { x: 4, y: 4 },
      w: IMG_W,
      h: IMG_H,
      kernelHeader: [
        "// blur_pixel: the rounded average of",
        "// the 3 × 3 pixels around (x, y)",
        "__device__ int blur_pixel(",
        "    const int* in, int w, int h,",
        "    int x, int y);",
        "__global__ void blur(const int* in,",
        "    int* out, int w, int h) {",
      ],
      launch: "blur<<<dim3(4, 3), dim3(4, 4)>>>(in, out, 14, 10);   // 12 blocks × 16 threads = 192 threads",
      cpuCode: "for (int y = 0; y < 10; ++y) for (int x = 0; x < 14; ++x) out[y][x] = blur_pixel(in, w, h, x, y);",
      mission:
        "Blur a 14 × 10 image with one thread per pixel. The grid is 4 × 3 blocks of 4 × 4 threads, which is 16 × 12 threads, so some threads fall off the right and bottom edges.",
      hint: "x comes from the .x variables and y from the .y variables. Threads with x ≥ 14 or y ≥ 10 have no pixel.",
      dataNote: "blur_pixel returns the rounded average of the 3 × 3 input pixels around (x, y), the same function as in the tutorial's image_blur.cu.",
      palette: ["idx_x", "idx_y", "guard2", "blur", "copy", "blurT"],
      start: [{ type: "idx_x" }, { type: "idx_y" }],
      blocks: {
        idx_x: indexBlock("x", ["blockIdx.x", "blockDim.x", "threadIdx.x"], IDX_2D),
        idx_y: indexBlock("y", ["blockIdx.x", "blockDim.x", "threadIdx.x"], IDX_2D),
        guard2: {
          cat: "control",
          parts: ["if (x >= w || y >= h) return;"],
          exec(ctx) {
            if (ctx.get("x") >= IMG_W || ctx.get("y") >= IMG_H) ctx.ret();
          },
        },
        blur: {
          cat: "memory",
          parts: ["out[y][x] = blur_pixel(in, w, h, x, y);"],
          exec(ctx) {
            const x = ctx.get("x"), y = ctx.get("y");
            const v = blurAt((r, c) => ctx.read2("in", r, c), x, y);
            ctx.write2("out", y, x, v);
          },
        },
        copy: {
          cat: "memory",
          parts: ["out[y][x] = in[y][x];"],
          exec(ctx) {
            const x = ctx.get("x"), y = ctx.get("y");
            ctx.write2("out", y, x, ctx.read2("in", y, x));
          },
        },
        blurT: {
          cat: "memory",
          parts: ["out[x][y] = blur_pixel(in, w, h, y, x);"],
          exec(ctx) {
            const x = ctx.get("x"), y = ctx.get("y");
            ctx.read2("in", x, y); // the swapped coordinates fault here for x >= 10
            const v = blurAt((r, c) => ctx.read2("in", r, c), y, x);
            ctx.write2("out", x, y, v);
          },
        },
      },
      memory: () => ({
        in: { data: IMG, w: IMG_W, h: IMG_H },
        out: { data: new Array(IMG_W * IMG_H).fill(null), w: IMG_W, h: IMG_H },
      }),
      expected: BLURRED,
      cpuSteps: IMG_W * IMG_H,
    },
    {
      id: "race",
      number: 3,
      atomicLesson: true,
      title: "Race to the total",
      concept: "Race conditions and atomics",
      dims: 1,
      grid: { x: 2, y: 1 },
      block: { x: 8, y: 1 },
      n: 16,
      kernelHeader: ["__device__ int total = 0;", "__global__ void sum(const int* x, int n) {"],
      launch: "sum<<<2, 8>>>(x, 16);   // 2 blocks × 8 threads = 16 threads",
      cpuCode: "for (int i = 0; i < 16; ++i) total += x[i];",
      mission:
        "Sixteen threads each add one number to a shared total. The kernel on the left compiles and runs, but its answer is wrong. Run it, find out why, and fix it.",
      hint: "total += x[i] is really three steps: read total, add, write total back. What if 16 threads read total at the same moment?",
      dataNote: `The right answer is ${sum16}.`,
      palette: ["idx_fixed", "guard", "race", "atomic"],
      start: [{ type: "idx_fixed" }, { type: "race" }],
      blocks: {
        idx_fixed: IDX_FIXED,
        guard: GUARD,
        race: {
          cat: "memory",
          parts: ["total += x[i];"],
          exec(ctx) {
            const i = ctx.get("i");
            const v = ctx.read("total", 0) + ctx.read("x", i);
            ctx.write("total", 0, v);
          },
        },
        atomic: {
          cat: "sync",
          parts: ["atomicAdd(&total, x[i]);"],
          exec(ctx) {
            const i = ctx.get("i");
            ctx.atomicAdd("total", 0, ctx.read("x", i));
          },
        },
      },
      memory: () => ({ x: { data: X16 }, total: { data: [0] } }),
      expectedTotal: sum16,
      cpuSteps: 16,
    },
    {
      id: "tree",
      number: 4,
      title: "Fast and correct: the tree",
      concept: "Shared memory and __syncthreads()",
      dims: 1,
      grid: { x: 1, y: 1 },
      block: { x: 16, y: 1 },
      n: 16,
      kernelHeader: ["__device__ int total = 0;", "__global__ void treeSum(const int* x) {", "    __shared__ int partial[16];"],
      launch: "treeSum<<<1, 16>>>(x);   // 1 block × 16 threads",
      cpuCode: "for (int i = 0; i < 16; ++i) total += x[i];",
      mission:
        "Make the sum correct without making threads take turns. Pairs of threads add their values in shared memory, halving the number of active threads each step: 16 values, then 8, 4, 2, 1. Use __syncthreads() so that no thread reads a value before another thread has finished writing it.",
      hint: "Strides 8, 4, 2, 1, each one followed by __syncthreads(). Thread 0 then copies partial[0] into total.",
      dataNote: `The right answer is ${sum16}. In this lab, a thread sees another thread's write only after a __syncthreads() (its own writes it sees at once).`,
      palette: ["tid", "load", "sync", "stride", "final", "raceTotal"],
      start: [{ type: "tid" }, { type: "load" }],
      blocks: {
        tid: {
          cat: "index",
          parts: ["int t = threadIdx.x;"],
          exec(ctx) {
            ctx.set("t", ctx.builtin("threadIdx.x"));
          },
        },
        load: {
          cat: "memory",
          parts: ["partial[t] = x[t];"],
          exec(ctx) {
            const t = ctx.get("t");
            ctx.write("partial", t, ctx.read("x", t));
          },
        },
        sync: { cat: "sync", parts: ["__syncthreads();"], barrier: true, exec() {} },
        stride: {
          cat: "memory",
          parts: ["if (t < ", F("s"), ") partial[t] += partial[t + ", M("s"), "];"],
          fields: { s: "8" },
          opts: { s: ["8", "4", "2", "1"] },
          exec(ctx, f) {
            const t = ctx.get("t"), s = Number(f.s);
            if (t < s) ctx.write("partial", t, ctx.read("partial", t) + ctx.read("partial", t + s));
          },
        },
        final: {
          cat: "memory",
          parts: ["if (t == 0) total = partial[0];"],
          exec(ctx) {
            if (ctx.get("t") === 0) ctx.write("total", 0, ctx.read("partial", 0));
          },
        },
        raceTotal: {
          cat: "memory",
          parts: ["total += partial[t];"],
          exec(ctx) {
            const t = ctx.get("t");
            ctx.write("total", 0, ctx.read("total", 0) + ctx.read("partial", t));
          },
        },
      },
      memory: () => ({ x: { data: X16 }, partial: { data: new Array(16).fill(null), kind: "shared" }, total: { data: [0] } }),
      expectedTotal: sum16,
      cpuSteps: 16,
    },
  ];

  // ---------------------------------------------------------------- GPU run
  function runGPU(level, program) {
    const threads = makeThreads(level.grid, level.block);
    const mem = new Memory(level.memory());
    const st = threads.map((t) => ({ ...t, vars: {}, alive: true, returned: false, fault: null, log: [] }));
    const diag = { faults: [], stale: [], atomics: 0 };
    const frames = [];
    let steps = 0;

    const snapshot = (k, label, active, events, extra = {}) => ({
      k,
      label,
      steps,
      active: new Set(active),
      events,
      view: mem.view(),
      threads: st.map((s) => ({ alive: s.alive, returned: s.returned, fault: s.fault, vars: { ...s.vars } })),
      ...extra,
    });

    frames.push(snapshot(-1, "Launch: every thread starts at the top of the kernel", [], []));

    for (let k = 0; k < program.length; k++) {
      const stmt = program[k];
      const def = level.blocks[stmt.type];
      const fields = { ...(def.fields || {}), ...(stmt.fields || {}) };
      const label = blockText(def, fields);
      const live = st.filter((s) => s.alive);
      if (!live.length) break;

      if (def.barrier) {
        mem.commit();
        steps++;
        frames.push(snapshot(k, label, live.map((s) => s.id), [], { barrier: true }));
        continue;
      }

      const events = [];
      const queue = [];
      for (const s of live) {
        const ctx = makeCtx(level, s, mem, events, queue, diag, k);
        try {
          def.exec(ctx, fields);
        } catch (e) {
          if (!(e instanceof ThreadFault)) throw e;
          s.alive = false;
          s.fault = e.message;
          diag.faults.push({ thread: s.id, k, message: e.message, info: e.info });
          events.push({ t: s.id, kind: "fault", message: e.message, info: e.info });
          s.log.push(`step ${k + 1}: ${e.message}`);
        }
      }

      if (queue.length) {
        // atomics take turns, one per step
        frames.push(snapshot(k, label, live.map((s) => s.id), events, { queued: queue.length }));
        for (const op of queue) {
          steps++;
          diag.atomics++;
          const after = mem.atomicAdd(op.arr, op.idx, op.value, op.t);
          st[op.t].log.push(`step ${k + 1}: atomicAdd(&${op.arr}, ${op.value}) → ${op.arr} = ${after}`);
          frames.push(snapshot(k, `${label}  (thread ${st[op.t].tx} of block ${st[op.t].bx} takes its turn)`, [op.t],
            [{ t: op.t, kind: "atomic", arr: op.arr, idx: op.idx, value: op.value, after }], { atomicTurn: true }));
        }
      } else {
        steps++;
        frames.push(snapshot(k, label, live.map((s) => s.id), events));
      }
    }
    mem.commit(); // the end of the kernel makes every write visible
    frames.push(snapshot(program.length, "Kernel finished: all writes are now visible", [], [], { done: true }));

    const result = { level, program, threads: st, mem, diag, frames, steps };
    result.verdict = judge(level, program, result);
    return result;
  }

  function makeCtx(level, s, mem, events, queue, diag, k) {
    const fault = (msg, info) => {
      throw new ThreadFault(msg, info);
    };
    return {
      n: level.n,
      builtin(name) {
        switch (name) {
          case "blockIdx.x": return s.bx;
          case "blockIdx.y": return s.by;
          case "threadIdx.x": return s.tx;
          case "threadIdx.y": return s.ty;
          case "blockDim.x": return level.block.x;
          case "blockDim.y": return level.block.y;
          case "gridDim.x": return level.grid.x;
          case "gridDim.y": return level.grid.y;
          default: throw new Error("unknown builtin " + name);
        }
      },
      get(name) {
        if (!(name in s.vars)) fault(`used ${name} before setting it`, { kind: "undef", name });
        return s.vars[name];
      },
      set(name, v) {
        s.vars[name] = v;
        s.log.push(`step ${k + 1}: ${name} = ${v}`);
        events.push({ t: s.id, kind: "set", name, value: v });
      },
      read(arr, idx) {
        const A = mem.arrays[arr];
        if (idx < 0 || idx >= A.len)
          fault(`read ${arr}[${idx}], past the end of ${arr} (it has ${A.len} elements)`, { kind: "oob", arr, idx, op: "read" });
        const r = mem.read(arr, idx, s.id);
        if (r.stale) diag.stale.push({ t: s.id, k, arr, idx, writer: r.writer, saw: r.value, newer: r.newer });
        else if (r.value === null && A.kind === "shared") diag.stale.push({ t: s.id, k, arr, idx, writer: null, saw: null });
        s.log.push(`step ${k + 1}: read ${arr}[${idx}] = ${r.value === null ? "?" : r.value}${r.stale ? " (old value!)" : ""}`);
        events.push({ t: s.id, kind: "read", arr, idx, value: r.value, stale: r.stale });
        return r.value ?? 0;
      },
      read2(arr, row, col) {
        const A = mem.arrays[arr];
        if (row < 0 || row >= A.h || col < 0 || col >= A.w)
          fault(`read ${arr}[${row}][${col}], outside the ${A.w} × ${A.h} image`, { kind: "oob", arr, row, col, op: "read" });
        return this.read(arr, row * A.w + col);
      },
      write(arr, idx, v) {
        const A = mem.arrays[arr];
        if (idx < 0 || idx >= A.len)
          fault(`wrote ${arr}[${idx}], past the end of ${arr} (it has ${A.len} elements)`, { kind: "oob", arr, idx, op: "write" });
        mem.write(arr, idx, v, s.id);
        s.log.push(`step ${k + 1}: wrote ${arr}[${idx}] = ${v}`);
        events.push({ t: s.id, kind: "write", arr, idx, value: v });
      },
      write2(arr, row, col, v) {
        const A = mem.arrays[arr];
        if (row < 0 || row >= A.h || col < 0 || col >= A.w)
          fault(`wrote ${arr}[${row}][${col}], outside the ${A.w} × ${A.h} image`, { kind: "oob", arr, row, col, op: "write" });
        this.write(arr, row * A.w + col, v);
      },
      atomicAdd(arr, idx, v) {
        queue.push({ t: s.id, arr, idx, value: v });
        events.push({ t: s.id, kind: "queue", arr, idx, value: v });
      },
      ret() {
        s.alive = false;
        s.returned = true;
        s.log.push(`step ${k + 1}: returned early`);
        events.push({ t: s.id, kind: "return" });
      },
    };
  }

  // ---------------------------------------------------------------- verdicts
  const has = (program, type) => program.some((s) => s.type === type);
  const indexOf = (program, type) => program.findIndex((s) => s.type === type);
  const fmtList = (xs, max = 4) => (xs.length <= max ? xs.join(", ") : `${xs.slice(0, max).join(", ")} and ${xs.length - max} more`);
  function ranges(nums) {
    const s = [...nums].sort((a, b) => a - b);
    const out = [];
    for (let i = 0; i < s.length; i++) {
      let j = i;
      while (j + 1 < s.length && s[j + 1] === s[j] + 1) j++;
      out.push(i === j ? `${s[i]}` : `${s[i]}–${s[j]}`);
      i = j;
    }
    return out;
  }

  function judge(level, program, res) {
    if (!program.length)
      return fail("The kernel is empty", "Drag blocks from the palette into the kernel, then press Run on GPU.");
    const { diag, mem } = res;
    const L = (t) => threadLabel(level, res.threads[t]);

    // 1. threads that crashed
    if (diag.faults.length) {
      const f = diag.faults[0];
      if (f.info.kind === "undef") {
        const setter = { i: "int i = …", x: "int x = …", y: "int y = …", t: "int t = …" }[f.info.name];
        return fail(`Every thread used ${f.info.name} before setting it`,
          `Put the ${setter} block above the first block that uses ${f.info.name}.`);
      }
      const oob = diag.faults.filter((d) => d.info.kind === "oob");
      const who = oob.map((d) => L(d.thread));
      const guardType = level.dims === 2 ? "guard2" : "guard";
      const guardAt = indexOf(program, guardType);
      let hint;
      if (level.dims === 2 && has(program, "blurT"))
        hint = "The swapped block uses x as the row. Rows go with y: out[y][x].";
      else if (guardAt === -1)
        hint = level.dims === 2
          ? "Add the block if (x >= w || y >= h) return; right after computing x and y."
          : "Add if (i >= n) return; right after computing i, before touching the arrays.";
      else if (guardAt > oob[0].k)
        hint = "The bounds check comes too late: move it above the block that reads and writes the arrays.";
      else hint = "Check the index formula: some threads compute an index that is too large.";
      return fail(`${oob.length} thread${oob.length > 1 ? "s" : ""} went outside the data`,
        `For example ${who[0]} ${oob[0].message}. ${hint}`, { who: oob.map((d) => d.thread) });
    }

    if (level.id === "vector-add" || level.id === "image-blur") {
      const outName = level.id === "vector-add" ? "c" : "out";
      const A = mem.arrays[outName];
      const missing = [], dup = [];
      for (let i = 0; i < A.len; i++) {
        const w = A.writers.get(i);
        if (!w) missing.push(i);
        else if (w.size > 1) dup.push(i);
      }
      const cell = (i) => (level.dims === 2 ? `out[${Math.floor(i / level.w)}][${i % level.w}]` : `c[${i}]`);
      if (missing.length === A.len)
        return fail(`Nothing was written to ${outName}`,
          level.id === "vector-add" ? "Add the block that computes c[i]." : "Add out[y][x] = blur_pixel(in, w, h, x, y);");
      if (dup.length || missing.length) {
        let msg = "";
        if (dup.length) msg += `${cell(dup[0])} was written by ${A.writers.get(dup[0]).size} different threads. `;
        if (missing.length)
          msg += level.dims === 2
            ? `${missing.length} of the ${A.len} pixels were never written. `
            : `c[${fmtList(ranges(missing))}] ${missing.length === 1 ? "was" : "were"} never written. `;
        let hint = "Every thread needs a different index. ";
        if (level.dims === 1) {
          const ib = program.find((s) => s.type === "idx_i");
          if (has(program, "idx_tid")) hint += "threadIdx.x alone repeats in every block: blocks also need blockIdx.x.";
          else if (ib) hint += "Which variable tells blocks apart, and how many elements does each block cover?";
        } else {
          const yb = program.find((s) => s.type === "idx_y");
          const yf = { ...level.blocks.idx_y.fields, ...(yb?.fields || {}) };
          if (yb && Object.values(yf).some((v) => v.endsWith(".x"))) hint += "y should be built from the .y variables only.";
          else hint += "x uses the .x variables, y the .y variables, in the same pattern as level 1.";
        }
        return fail("Some outputs were written twice and others never", msg + hint);
      }
      const wrong = [];
      for (let i = 0; i < A.len; i++) if (A.committed[i] !== level.expected[i]) wrong.push(i);
      if (wrong.length) {
        const i = wrong[0];
        const why = level.id === "vector-add" ? `a[${i}] + b[${i}] = ${level.expected[i]}` : `the 3 × 3 average, ${level.expected[i]}`;
        return fail(`${wrong.length} output values are wrong`,
          `${cell(i)} is ${A.committed[i]} but should be ${why}. ${level.id === "vector-add" ? "Use the + block." : "Use the blur_pixel block."}`);
      }
      const returned = res.threads.filter((t) => t.returned).length;
      return pass(
        level.id === "vector-add" ? "All 28 elements are correct" : "Every pixel is blurred correctly",
        `${res.threads.length} threads ran the kernel at the same time, finishing in ${res.steps} steps. The CPU loop needs ${level.cpuSteps} steps, one per ${level.id === "vector-add" ? "element" : "pixel"}. ${returned} thread${returned === 1 ? "" : "s"} had nothing to do and returned early.`
      );
    }

    // summation levels
    const total = mem.arrays.total.committed[0];
    const lost = mem.conflicts.filter((c) => c.arr === "total");
    if (total === level.expectedTotal) {
      if (level.atomicLesson)
        return pass(`total = ${total}: correct`, level.passText ? level.passText(res, diag) :
          `But the ${diag.atomics} atomicAdd calls had to take turns, so the kernel took ${res.steps} steps, more than the CPU loop's ${level.cpuSteps}. Level 4 shows how to be correct and fast.`);
      return pass(`total = ${total}: correct and fast`,
        `${res.steps} steps, against ${level.cpuSteps} for the CPU loop. Each step halves the active threads, so 1,024 numbers would take about 23 steps with the same pattern, instead of 1,024.`);
    }
    if (level.atomicLesson && has(program, "race") && has(program, "atomic"))
      return fail(`total = ${total}, expected ${level.expectedTotal}`,
        "The kernel still contains total += x[i], whose racing writes land at the end and overwrite the atomic result. Remove that block and keep only atomicAdd.");
    if (lost.length) {
      const c = lost[0];
      const survivor = threadLabel(level, res.threads[c.kept]);
      return fail(`total = ${total}, expected ${level.expectedTotal}: a race condition`,
        `${c.writers.length} threads read total at the same moment, each added its own number, and each wrote back. Only the write by ${survivor} survived; ${c.writers.length - 1} updates were lost. ${level.atomicLesson ? "Use atomicAdd, which makes read, add and write one indivisible step." : "Let one thread write the finished sum instead: if (t == 0) total = partial[0];"}`);
    }
    if (diag.stale.length) {
      const s0 = diag.stale[0];
      const loadAt = indexOf(program, "load");
      if (s0.writer === null && (loadAt === -1 || loadAt > s0.k))
        return fail(`total = ${total}, expected ${level.expectedTotal}`,
          "The threads read partial[] before anything was stored in it, so they added garbage. Start by copying the input into shared memory: partial[t] = x[t];",
          { who: [s0.t] });
      const reader = threadLabel(level, res.threads[s0.t]);
      const stepName = blockText(level.blocks[program[s0.k].type], { ...(level.blocks[program[s0.k].type].fields || {}), ...(program[s0.k].fields || {}) });
      const what = s0.writer === null
        ? `which no thread had written yet, so it got garbage`
        : `before ${threadLabel(level, res.threads[s0.writer])}'s new value (${s0.newer}) was visible, so it got the old value`;
      return fail(`total = ${total}, expected ${level.expectedTotal}: a thread read too early`,
        `At step ${s0.k + 1} (${stepName}) ${reader} read ${s0.arr}[${s0.idx}] ${what}. Put a __syncthreads() block right before that step, so every thread waits until the writes are visible.`,
        { who: [s0.t] });
    }
    if (total === 0)
      return fail("total is still 0",
        level.atomicLesson ? "No thread added anything to total." : "Nothing copied the finished sum into total. Add if (t == 0) total = partial[0]; at the end.");
    return fail(`total = ${total}, expected ${level.expectedTotal}`,
      "Even with perfect timing, some values are added twice or not at all. Use the strides 8, 4, 2 and 1 once each, largest first.");
  }

  const pass = (title, message) => ({ ok: true, title, message });
  const fail = (title, message, extra = {}) => ({ ok: false, title, message, ...extra });

  // ---------------------------------------------------------------- CPU loop
  function runCPU(level) {
    const mem = new Memory(level.memory());
    const frames = [];
    const view = () => mem.view();
    if (level.id === "vector-add") {
      for (let i = 0; i < level.n; i++) {
        const A = mem.arrays;
        A.c.committed[i] = A.a.committed[i] + A.b.committed[i];
        frames.push({ cpu: i, label: `i = ${i}: c[${i}] = ${A.a.committed[i]} + ${A.b.committed[i]}`, steps: i + 1, view: view() });
      }
    } else if (level.id === "image-blur") {
      const A = mem.arrays;
      for (let y = 0; y < IMG_H; y++)
        for (let x = 0; x < IMG_W; x++) {
          A.out.committed[y * IMG_W + x] = blurAt((r, c) => A.in.committed[r * IMG_W + c], x, y);
          frames.push({ cpu: y * IMG_W + x, label: `y = ${y}, x = ${x}`, steps: y * IMG_W + x + 1, view: view() });
        }
    } else {
      const A = mem.arrays;
      for (let i = 0; i < level.n; i++) {
        A.total.committed[0] += A.x.committed[i];
        frames.push({ cpu: i, label: `i = ${i}: total += ${A.x.committed[i]} → ${A.total.committed[0]}`, steps: i + 1, view: view() });
      }
    }
    return { frames, steps: frames.length };
  }

  return { LEVELS, runGPU, runCPU, blockText, makeThreads, threadLabel, IDX_FIXED, GUARD };
})();

if (typeof module !== "undefined") module.exports = Engine;

