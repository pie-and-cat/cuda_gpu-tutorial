// Thread Lab, track "Build CUDA Balls": five levels that build the CUDA Balls physics engine in 2-D.
//
//   1 Move the balls     one thread per ball: the index formula and the bounds check
//   2 Gravity and walls  each thread's own variables, statement order, a kernel launched every substep
//   3 Collisions         every thread reads every ball: the in-place race and double buffering
//   4 The time loop      the CPU side: launch, swap, one copy back, draw
//   5 Total energy       one shared total: the race on +=, fixed with atomicAdd
//
// Levels 1 to 3 run the student's kernel for every thread and every substep. Blocks of threads run one
// after another (block 0, then 1, then 2, as a GPU might schedule them); inside a block the threads run
// each statement in lockstep. A thread that reads another ball's slot after that ball's thread already
// overwrote it during the same substep is a race: its result depends on block order.
// Level 4 simulates the frame loop with two GPU buffers, the CPU's copy and the screen.
// Level 5 is an ordinary Thread Lab level run by the engine, fed with the balls' energies.
const Physics = (() => {
  const E = typeof Engine !== "undefined" ? Engine : require("./engine.js");

  // Same constants as cuda_balls/src/physics.cu, minus friction and the third dimension.
  const PHYS = Object.freeze({
    halfW: 6, H: 8, g: 9.81, k: 4000, zeta: 0.11, dt: 1 / 960, drag: 0.05, vmax: 40,
    substeps: 16, frames: 240, fps: 60, count: 22, blocks: 3, threads: 8,
    overlapLimit: 0.25, // deepest overlap allowed; correct physics stays well below it
  });
  const N = PHYS.count;

  function mulberry32(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const makeBuf = () => ({ px: new Float64Array(N), py: new Float64Array(N), vx: new Float64Array(N), vy: new Float64Array(N) });
  const copyBuf = (b) => ({ px: b.px.slice(), py: b.py.slice(), vx: b.vx.slice(), vy: b.vy.slice() });
  const garbageBuf = () => {
    const b = makeBuf();
    for (const a of Object.values(b)) a.fill(NaN);
    return b;
  };

  // The balls: rows of 8 high in the box, moving sideways a little (levels 2 to 5).
  const RADIUS = new Float64Array(N), MASS = new Float64Array(N);
  const START = makeBuf();
  // Level 1 has no gravity: the same balls, lower down and spaced out, drifting slowly in all directions.
  const DRIFT = makeBuf();
  {
    const rnd = mulberry32(2026);
    for (let i = 0; i < N; i++) {
      const row = Math.floor(i / 8), col = i % 8;
      const inRow = Math.min(8, N - row * 8);
      RADIUS[i] = 0.35 + 0.2 * rnd();
      MASS[i] = Math.PI * RADIUS[i] * RADIUS[i];
      START.px[i] = (col - (inRow - 1) / 2) * 1.3 + (rnd() - 0.5) * 0.1;
      START.py[i] = 4.2 + row * 1.3;
      START.vx[i] = -2 + 4 * rnd();
      START.vy[i] = 0;
      const angle = 2 * Math.PI * rnd(), speed = 0.5 + 0.25 * rnd();
      DRIFT.px[i] = (col - (inRow - 1) / 2) * 1.15;
      DRIFT.py[i] = 2.8 + row * 1.2;
      DRIFT.vx[i] = speed * Math.cos(angle);
      DRIFT.vy[i] = speed * Math.sin(angle);
    }
  }

  // ---------------------------------------------------------------- forces
  // Force from one contact: a spring pushes the bodies apart and a damper takes out energy.
  // `n` points from the surface towards the body; contacts push, never pull.
  function addContact(f, nx, ny, overlap, rvx, rvy, m) {
    const normalSpeed = rvx * nx + rvy * ny;
    const push = PHYS.k * overlap - 2 * PHYS.zeta * Math.sqrt(PHYS.k * m) * normalSpeed;
    if (push > 0) {
      f.x += nx * push;
      f.y += ny * push;
    }
  }
  function addPair(f, i, p, v, j, qx, qy, wx, wy) {
    const dx = p.x - qx, dy = p.y - qy;
    const reach = RADIUS[i] + RADIUS[j];
    const d2 = dx * dx + dy * dy;
    if (d2 >= reach * reach || d2 < 1e-10) return;
    const d = Math.sqrt(d2);
    const reduced = (MASS[i] * MASS[j]) / (MASS[i] + MASS[j]);
    addContact(f, dx / d, dy / d, reach - d, v.x - wx, v.y - wy, reduced);
  }
  function addWalls(f, i, p, v) {
    const r = RADIUS[i], m = MASS[i];
    const gaps = [p.y, PHYS.H - p.y, p.x + PHYS.halfW, PHYS.halfW - p.x];
    const normals = [[0, 1], [0, -1], [1, 0], [-1, 0]];
    for (let w = 0; w < 4; w++) if (r > gaps[w]) addContact(f, normals[w][0], normals[w][1], r - gaps[w], v.x, v.y, m);
  }
  const gravityAndDrag = (i, v) => ({ x: -v.x * PHYS.drag * MASS[i], y: -PHYS.g * MASS[i] - v.y * PHYS.drag * MASS[i] });
  function applyForce(i, v, f) {
    const s = PHYS.dt / MASS[i];
    let x = v.x + f.x * s, y = v.y + f.y * s;
    const speed = Math.hypot(x, y);
    if (speed > PHYS.vmax) {
      x *= PHYS.vmax / speed;
      y *= PHYS.vmax / speed;
    }
    return { x, y };
  }
  const move = (p, v) => ({ x: p.x + v.x * PHYS.dt, y: p.y + v.y * PHYS.dt });

  // The reference substep: the correct kernel, one ball after another.
  function refStep(cur, next) {
    for (let i = 0; i < N; i++) {
      let p = { x: cur.px[i], y: cur.py[i] }, v = { x: cur.vx[i], y: cur.vy[i] };
      const f = gravityAndDrag(i, v);
      for (let j = 0; j < N; j++) if (j !== i) addPair(f, i, p, v, j, cur.px[j], cur.py[j], cur.vx[j], cur.vy[j]);
      addWalls(f, i, p, v);
      v = applyForce(i, v, f);
      p = move(p, v);
      next.px[i] = p.x;
      next.py[i] = p.y;
      next.vx[i] = v.x;
      next.vy[i] = v.y;
    }
  }

  // Snapshot layout: [x0, y0, vx0, vy0, x1, ...]
  function snap(b) {
    const s = new Float32Array(4 * N);
    for (let i = 0; i < N; i++) {
      s[4 * i] = b.px[i];
      s[4 * i + 1] = b.py[i];
      s[4 * i + 2] = b.vx[i];
      s[4 * i + 3] = b.vy[i];
    }
    return s;
  }

  // Correct states by substep number, computed on demand (levels 4 and 5 use these).
  const refCache = { states: [snap(START)], cur: copyBuf(START), next: makeBuf() };
  function refState(v) {
    while (refCache.states.length <= v) {
      refStep(refCache.cur, refCache.next);
      [refCache.cur, refCache.next] = [refCache.next, refCache.cur];
      refCache.states.push(snap(refCache.cur));
    }
    return refCache.states[v];
  }

  // Uninitialized memory, drawn as balls scattered anywhere. Deterministic per garbage id.
  function garbageState(g) {
    const rnd = mulberry32(9001 + g * 7919);
    const s = new Float32Array(4 * N);
    for (let i = 0; i < N; i++) {
      s[4 * i] = (rnd() - 0.5) * 2 * (PHYS.halfW + 1.5);
      s[4 * i + 1] = rnd() * (PHYS.H + 3) - 1.5;
    }
    return s;
  }

  // Energy of every ball in a snapshot: ½ m v² + m g y.
  function energies(S) {
    const e = [];
    for (let i = 0; i < N; i++) e.push(0.5 * MASS[i] * (S[4 * i + 2] ** 2 + S[4 * i + 3] ** 2) + MASS[i] * PHYS.g * S[4 * i + 1]);
    return e;
  }

  // ---------------------------------------------------------------- kernel blocks (levels 1 to 3)
  const FORCE_SETTER = "float2 f = gravity(i) + airDrag(i, v);";
  const LOAD = "float2 p = pos[i], v = vel[i];";
  const IDX_OPTS = ["blockIdx.x", "blockDim.x", "threadIdx.x", "gridDim.x"];
  const kernelBlocks = {
    idx_tid: {
      cat: "index",
      parts: ["int i = threadIdx.x;"],
      exec(c) {
        c.set("i", c.builtin("threadIdx.x"));
      },
    },
    idx_i: {
      cat: "index",
      parts: ["int i = ", { f: "a" }, " * ", { f: "b" }, " + ", { f: "c" }, ";"],
      fields: { a: "threadIdx.x", b: "blockDim.x", c: "blockIdx.x" },
      opts: { a: IDX_OPTS, b: IDX_OPTS, c: IDX_OPTS },
      exec(c, f) {
        c.set("i", c.builtin(f.a) * c.builtin(f.b) + c.builtin(f.c));
      },
    },
    idx: {
      cat: "index",
      parts: ["int i = blockIdx.x * blockDim.x + threadIdx.x;"],
      exec(c) {
        c.set("i", c.builtin("blockIdx.x") * c.builtin("blockDim.x") + c.builtin("threadIdx.x"));
      },
    },
    guard: {
      cat: "control",
      parts: ["if (i >= n) return;"],
      exec(c) {
        if (c.get("i") >= N) c.ret();
      },
    },
    drift: {
      cat: "memory",
      parts: ["pos[i] += vel[i] * dt;"],
      exec(c) {
        const i = c.get("i");
        const b = c.readBall(i);
        const v = { x: b[2], y: b[3] };
        c.writeBall("cur", i, move({ x: b[0], y: b[1] }, v), v);
      },
    },
    load: {
      cat: "memory",
      parts: [LOAD],
      exec(c) {
        const b = c.readBall(c.get("i"));
        c.set("p", { x: b[0], y: b[1] });
        c.set("v", { x: b[2], y: b[3] });
      },
    },
    forces: {
      cat: "physics",
      parts: [FORCE_SETTER],
      exec(c) {
        const i = c.get("i");
        c.set("f", gravityAndDrag(i, c.get("v")));
      },
    },
    collide: {
      cat: "physics",
      parts: ["for (int j = 0; j < n; ++j)\n  if (j != i)\n    f += push(i, j);"],
      cost: N,
      exec(c) {
        const i = c.get("i"), p = c.get("p"), v = c.get("v"), f = c.get("f");
        for (let j = 0; j < N; j++) {
          if (j === i) continue;
          const b = c.readBall(j);
          addPair(f, i, p, v, j, b[0], b[1], b[2], b[3]);
        }
      },
    },
    walls: {
      cat: "physics",
      parts: ["f += pushFromWalls(i, p, v);"],
      exec(c) {
        const i = c.get("i"), p = c.get("p"), v = c.get("v"), f = c.get("f");
        addWalls(f, i, p, v);
      },
    },
    velocity: {
      cat: "physics",
      parts: ["v += f / mass[i] * dt;"],
      exec(c) {
        const i = c.get("i"), v = c.get("v"), f = c.get("f");
        c.set("v", applyForce(i, v, f));
      },
    },
    position: {
      cat: "physics",
      parts: ["p += v * dt;"],
      exec(c) {
        const p = c.get("p"), v = c.get("v");
        c.set("p", move(p, v));
      },
    },
    store: {
      cat: "memory",
      parts: ["nextPos[i] = p;  nextVel[i] = v;"],
      exec(c) {
        const i = c.get("i"), p = c.get("p"), v = c.get("v");
        c.writeBall("next", i, p, v);
      },
    },
    storeInPlace: {
      cat: "memory",
      parts: ["pos[i] = p;  vel[i] = v;"],
      exec(c) {
        const i = c.get("i"), p = c.get("p"), v = c.get("v");
        c.writeBall("cur", i, p, v);
      },
    },
  };
  const INDEX_TYPES = ["idx_tid", "idx_i", "idx"];

  // ---------------------------------------------------------------- kernel interpreter
  class Fault extends Error {
    constructor(message, info) {
      super(message);
      this.info = info;
    }
  }

  function makeCtx(t, sim) {
    const scratch = [0, 0, 0, 0];
    const fault = (message, info) => {
      throw new Fault(message, info);
    };
    return {
      builtin(name) {
        switch (name) {
          case "blockIdx.x": return t.bx;
          case "threadIdx.x": return t.tx;
          case "blockDim.x": return PHYS.threads;
          case "gridDim.x": return PHYS.blocks;
          default: throw new Error("unknown builtin " + name);
        }
      },
      get(name) {
        const v = t.vars[name];
        if (v === undefined) fault(`used ${name} before setting it`, { kind: "undef", name });
        return v;
      },
      set(name, v) {
        t.vars[name] = v;
      },
      readBall(j) {
        if (j < 0 || j >= N) fault(`read pos[${j}], past the end of pos (it has ${N} elements)`, { kind: "oob", idx: j, op: "read" });
        if (j !== t.vars.i && sim.stamp[j] === sim.substep) sim.race(t, j);
        const B = sim.cur;
        scratch[0] = B.px[j];
        scratch[1] = B.py[j];
        scratch[2] = B.vx[j];
        scratch[3] = B.vy[j];
        return scratch;
      },
      writeBall(which, i, p, v) {
        const name = which === "next" ? "nextPos" : "pos";
        if (i < 0 || i >= N) fault(`wrote ${name}[${i}], past the end of ${name} (it has ${N} elements)`, { kind: "oob", idx: i, op: "write" });
        const B = which === "next" ? sim.next : sim.cur;
        B.px[i] = p.x;
        B.py[i] = p.y;
        B.vx[i] = v.x;
        B.vy[i] = v.y;
        if (sim.substep === 0) sim.movers[i]++;
        if (which === "next") sim.nextWrites++;
        else {
          sim.stamp[i] = sim.substep;
          sim.inPlaceWrites++;
          if (sim.substep === 0) sim.inPlaceWriter[i] = t.id;
        }
      },
      ret() {
        t.alive = false;
        t.returned = true;
      },
    };
  }

  // Runs `program` for up to `maxFrames` frames from the level's starting state, scheduling the blocks in `order`.
  function simulate(level, program, order = [0, 1, 2], maxFrames = level.frames) {
    const steps = program.map((s) => {
      const def = kernelBlocks[s.type];
      return { def, fields: { ...(def.fields || {}), ...(s.fields || {}) } };
    });
    const threads = [];
    for (let bx = 0; bx < PHYS.blocks; bx++)
      for (let tx = 0; tx < PHYS.threads; tx++)
        threads.push({ id: threads.length, bx, tx, vars: {}, alive: true, returned: false, fault: null, races: 0 });
    const start = level.startState;
    const sim = {
      cur: copyBuf(start),
      next: garbageBuf(), // nextPos starts uninitialized
      stamp: new Int32Array(N).fill(-1),
      inPlaceWriter: new Int32Array(N).fill(-1),
      movers: new Int32Array(N), // how many threads wrote each ball in the first substep
      substep: 0,
      nextWrites: 0,
      inPlaceWrites: 0,
      races: 0,
      firstRace: null,
      race(t, j) {
        this.races++;
        t.races++;
        if (!this.firstRace) this.firstRace = { thread: t.id, ball: t.vars.i, j };
      },
    };
    const ctxs = threads.map((t) => makeCtx(t, sim));
    const groups = order.map((b) => threads.slice(b * PHYS.threads, (b + 1) * PHYS.threads));

    const frames = [snap(sim.cur)];
    const faults = [];
    let firstThreads = null, nothingStored = false;
    let maxOverlap = 0, overlapPair = null;
    let escaped = null, moved = 0;

    run: for (let fr = 1; fr <= maxFrames; fr++) {
      for (let s = 0; s < PHYS.substeps; s++) {
        sim.nextWrites = 0;
        sim.inPlaceWrites = 0;
        for (const group of groups) {
          for (const t of group) {
            t.vars = {};
            t.alive = true;
            t.returned = false;
          }
          for (let k = 0; k < steps.length; k++)
            for (const t of group) {
              if (!t.alive) continue;
              try {
                steps[k].def.exec(ctxs[t.id], steps[k].fields);
              } catch (e) {
                if (!(e instanceof Fault)) throw e;
                t.alive = false;
                t.fault = e.message;
                faults.push({ thread: t.id, message: e.message, info: e.info, k });
              }
            }
        }
        if (sim.substep === 0) {
          firstThreads = threads.map((t) => ({ id: t.id, bx: t.bx, tx: t.tx, i: t.vars.i, returned: t.returned, fault: t.fault }));
          if (faults.length) break run;
          if (!sim.nextWrites && !sim.inPlaceWrites) {
            nothingStored = true;
            break run;
          }
        }
        if (sim.nextWrites) [sim.cur, sim.next] = [sim.next, sim.cur];
        sim.substep++;
      }
      const S = snap(sim.cur);
      frames.push(S);

      for (let i = 0; i < N; i++) {
        const x = S[4 * i], y = S[4 * i + 1], r = RADIUS[i];
        if (!escaped && (x < -PHYS.halfW - r || x > PHYS.halfW + r || y < -r || y > PHYS.H + r)) escaped = { ball: i, frame: fr };
        moved = Math.max(moved, Math.hypot(x - start.px[i], y - start.py[i]));
        for (let j = i + 1; j < N; j++) {
          const depth = r + RADIUS[j] - Math.hypot(x - S[4 * j], y - S[4 * j + 1]);
          if (depth > maxOverlap) {
            maxOverlap = depth;
            overlapPair = [i, j];
          }
        }
      }
      if (escaped && level.checks.escape && fr >= escaped.frame + 40) break;
    }

    return {
      frames, faults, firstThreads, nothingStored, escaped, moved, maxOverlap, overlapPair,
      movers: Array.from(sim.movers), races: sim.races, firstRace: sim.firstRace, inPlaceWriter: sim.inPlaceWriter,
      threadRaces: threads.map((t) => t.races),
    };
  }

  // Steps per substep: every statement is one step, except the collision loop, which visits every ball.
  function stepCounts(program) {
    const cost = (s) => kernelBlocks[s.type]?.cost ?? 1;
    const gpu = program.reduce((a, s) => a + cost(s), 0);
    const perBall = program.filter((s) => !INDEX_TYPES.includes(s.type) && s.type !== "guard").reduce((a, s) => a + cost(s), 0);
    return { gpu, cpu: N * perBall, perBall };
  }

  // ---------------------------------------------------------------- kernel verdicts
  const has = (program, type) => program.some((s) => s.type === type);
  const at = (program, type) => program.findIndex((s) => s.type === type);
  const pass = (title, message, extra = {}) => ({ ok: true, title, message, ...extra });
  const fail = (title, message, extra = {}) => ({ ok: false, title, message, ...extra });
  const text = (type) => kernelBlocks[type].parts.map((p) => (typeof p === "string" ? p : "…")).join("").replace(/\s+/g, " ");
  // [0,1,2,3,8,9] -> "0–3, 8 and 9"
  function ballList(list) {
    const parts = [];
    for (let k = 0; k < list.length; ) {
      let e = k;
      while (e + 1 < list.length && list[e + 1] === list[e] + 1) e++;
      parts.push(e - k >= 2 ? `${list[k]}–${list[e]}` : e === k ? `${list[k]}` : `${list[k]}, ${list[e]}`);
      k = e + 1;
    }
    return parts.length > 1 ? parts.slice(0, -1).join(", ") + " and " + parts.at(-1) : parts[0];
  }

  function judgeKernel(level, program, res, rerun) {
    if (!program.length) return fail("The kernel is empty", "Drag blocks from the palette into the kernel, then press Run on GPU.");
    if (program.filter((s) => INDEX_TYPES.includes(s.type)).length > 1)
      return fail("Two blocks set i", "Keep one of the int i = … blocks.");
    const seen = new Set();
    for (const s of program) {
      if (seen.has(s.type))
        return fail(`${text(s.type)} is in the kernel twice`, "Each step happens once per substep. Remove the extra copy.");
      seen.add(s.type);
    }
    const L = (id) => E.threadLabel(level, res.firstThreads[id]);

    if (res.faults.length) {
      const f = res.faults[0];
      if (f.info.kind === "undef") {
        const setter = { i: "int i = …", p: LOAD, v: LOAD, f: FORCE_SETTER }[f.info.name];
        return fail(`Every thread used ${f.info.name} before setting it`,
          `Put ${setter} above the first block that uses ${f.info.name}.`, { who: res.faults.map((x) => x.thread) });
      }
      const oob = res.faults.filter((x) => x.info.kind === "oob");
      const guardAt = at(program, "guard");
      const hint = guardAt === -1
        ? `There are ${N} balls but ${PHYS.blocks * PHYS.threads} threads. Add if (i >= n) return; right after computing i.`
        : "The bounds check comes too late: move it above the block that reads pos.";
      return fail(`${oob.length} thread${oob.length > 1 ? "s" : ""} went outside the data`,
        `For example ${L(oob[0].thread)} ${oob[0].message}. ${hint}`, { who: oob.map((x) => x.thread) });
    }

    if (res.nothingStored)
      return fail("Nothing was stored", level.nothingStored);

    if (level.checks.coverage) {
      const missing = [], extra = [];
      res.movers.forEach((m, i) => (m === 0 ? missing : m > 1 ? extra : []).push(i));
      if (missing.length || extra.length) {
        const times = extra.length ? res.movers[extra[0]] : 0;
        const title = missing.length ? `Ball${missing.length > 1 ? "s" : ""} ${ballList(missing)} never moved` : `Ball${extra.length > 1 ? "s" : ""} ${ballList(extra)} moved too fast`;
        let msg = extra.length ? `Balls ${ballList(extra)} were each moved by ${times} threads, so they went ${times} times too fast. ` : "";
        msg += has(program, "idx_tid")
          ? "i = threadIdx.x starts again at 0 in every block. Use the int i = … block with blockIdx.x * blockDim.x + threadIdx.x, so thread t of block b moves ball b × 8 + t."
          : "Check the formula: thread t of block b should move ball b × 8 + t.";
        return fail(title, msg, { balls: missing.length ? missing : extra });
      }
    }

    const iv = at(program, "velocity"), ip = at(program, "position");
    if (level.checks.escape && res.escaped) {
      const b = res.escaped.ball;
      let hint;
      if (iv === -1) hint = "Nothing applies the forces to the velocity: add v += f / mass[i] * dt; after the force blocks.";
      else if (!has(program, "walls")) hint = "Nothing stops a ball at the floor and walls: add f += pushFromWalls(i, p, v); before v is updated.";
      else hint = "pushFromWalls runs after v += …, so its force is never used. Move it above the velocity update.";
      return fail(`Ball ${b} left the box`, hint, { who: [b] });
    }

    if (level.checks.frozen && res.moved < 0.05) {
      return fail("The balls never move", ip === -1
        ? "The velocity changes but nothing moves the ball: add p += v * dt; after the velocity update."
        : "The store runs before p changes, so it saves the old position. Put the store last.");
    }

    if (level.checks.overlap && res.maxOverlap > PHYS.overlapLimit) {
      const [a, b] = res.overlapPair;
      const hint = !has(program, "collide")
        ? "Nothing pushes balls apart: add the for (j …) f += push(i, j); loop before v is updated."
        : "push(i, j) runs after v += …, so its force is never used. Move the loop above the velocity update.";
      return fail(`Balls ${a} and ${b} passed through each other`,
        `They overlapped by ${res.maxOverlap.toFixed(2)} units. ${hint}`, { who: [a, b] });
    }

    if (level.checks.race && res.races) {
      const r0 = res.firstRace;
      const writer = res.inPlaceWriter[r0.j];
      const other = rerun();
      const last = Math.min(res.frames.length, other.frames.length) - 1;
      let div = 0;
      for (let i = 0; i < N; i++)
        div = Math.max(div, Math.hypot(res.frames[last][4 * i] - other.frames[last][4 * i], res.frames[last][4 * i + 1] - other.frames[last][4 * i + 1]));
      const writerText = writer >= 0 ? `${L(writer)} had already overwritten it` : "its thread had already overwritten it";
      return fail("A race: the result depends on which block runs first",
        `${L(r0.thread)} (ball ${r0.ball}) read pos[${r0.j}] after ${writerText} with ball ${r0.j}'s next position, so the two balls were one substep apart. ` +
        `Run again with the blocks in the opposite order (dashed rings), the balls end up as much as ${div.toFixed(2)} units apart. ` +
        "Write to nextPos and nextVel instead; the host swaps the buffers after each launch.",
        { who: [r0.thread], ghost: other.frames, divergence: div });
    }

    return pass(level.passTitle, level.passText(stepCounts(program)));
  }

  function runKernelLevel(level, program) {
    const res = simulate(level, program, [0, 1, 2]);
    const verdict = judgeKernel(level, program, res, () => simulate(level, program, [2, 1, 0], res.frames.length - 1));
    return { kind: "physics", level, program, ...res, verdict };
  }

  // ---------------------------------------------------------------- level 4: the time loop
  const hostBlocks = {
    launch: { cat: "physics", parts: ["stepKernel<<<3, 8>>>(pos, vel, nextPos, nextVel, n);"] },
    swap: { cat: "host", parts: ["swap(pos, nextPos);  swap(vel, nextVel);"] },
    copyBack: { cat: "memory", parts: ["cudaMemcpy(hostPos, pos, bytes, cudaMemcpyDeviceToHost);"] },
    upload: { cat: "memory", parts: ["cudaMemcpy(pos, hostPos, bytes, cudaMemcpyHostToDevice);"] },
    draw: { cat: "host", parts: ["draw(hostPos);"] },
    loopEnd: { cat: "loop", fixed: true, parts: ["}   // below this: once per frame"] },
  };

  // Buffer contents are { v: substep } or { g: id } for garbage. They never change once made.
  function runHost(level, program) {
    const types = program.map((s) => s.type);
    const end = types.indexOf("loopEnd");
    const loopOps = types.slice(0, end), frameOps = types.slice(end + 1);
    const buf = { A: { v: 0 }, B: { g: 1 } };
    let pos = "A", next = "B";
    let host = { v: 0 }, shown = null;
    let garbage = 1, rewound = false, firstGarbage = null;
    const timeline = [];
    for (let s = 0; s < PHYS.substeps; s++) timeline.push(...loopOps);
    timeline.push(...frameOps);

    const apply = (op) => {
      if (op === "launch") {
        const src = buf[pos];
        buf[next] = src.g ? { g: ++garbage } : { v: src.v + 1 };
      } else if (op === "swap") [pos, next] = [next, pos];
      else if (op === "copyBack") host = buf[pos];
      else if (op === "upload") {
        if (host.v !== undefined && buf[pos].v !== undefined && host.v < buf[pos].v) rewound = true;
        buf[pos] = host;
      } else if (op === "draw") shown = host;
    };

    const frames = [];
    for (let f = 0; f < PHYS.frames; f++) {
      for (const op of timeline) apply(op);
      const latest = buf[pos];
      if (firstGarbage === null && (latest.g || (shown && shown.g))) firstGarbage = f;
      frames.push({ shown, A: buf.A, B: buf.B, pos, host, latest });
    }

    const per = (t) => timeline.filter((o) => o === t).length;
    const counts = { launch: per("launch"), swap: per("swap"), copyBack: per("copyBack"), upload: per("upload"), draw: per("draw") };
    const a = frames[20].latest, b = frames[21].latest;
    const advance = a.v !== undefined && b.v !== undefined ? b.v - a.v : null;
    const sh = frames[21].shown;
    const lag = sh && sh.v !== undefined && b.v !== undefined ? b.v - sh.v : null;
    const res = { kind: "host", level, program, frames, counts, advance, lag, rewound, firstGarbage, loopOps, frameOps };
    res.verdict = judgeHost(res);
    return res;
  }

  function judgeHost(res) {
    const { counts: c, loopOps, frameOps, advance, lag } = res;
    const sec = (n) => (n * PHYS.fps).toLocaleString("en-US");
    if (!loopOps.length && !frameOps.length)
      return fail("The frame function is empty", "Drag blocks into it. Blocks above the } run 16 times per frame; blocks below it run once.");
    if (!c.launch) return fail("Nothing runs the simulation", "Add the stepKernel<<<3, 8>>> launch inside the loop.");
    if (res.firstGarbage !== null)
      return fail("The balls turned into garbage",
        "A swap makes pos point at the nextPos buffer before any kernel has written it, so the GPU reads uninitialized memory. Inside the loop, launch first, then swap.");
    if (!c.draw) return fail("Nothing is drawn", "Add draw(hostPos); below the }, so each frame ends with a picture.");
    if (c.draw >= PHYS.substeps)
      return fail(`Drawing ${sec(c.draw)} times a second`,
        "draw is inside the loop, so it runs after every substep, but the screen shows only one picture per frame. Move it below the }.");
    if (c.upload)
      return res.rewound
        ? fail("The simulation keeps rewinding",
          "cudaMemcpy(pos, hostPos, …, cudaMemcpyHostToDevice) overwrites the GPU's newer positions with the CPU's older copy. The balls live on the GPU; the CPU needs a copy only to draw. Remove the upload.")
        : fail(`${sec(c.upload)} copies a second to the GPU, for nothing`,
          "The upload sends positions the GPU already has. Every cudaMemcpy crosses the PCIe bus and waits for the GPU to finish. Remove it.");
    if (!c.copyBack)
      return fail("The picture never changes",
        "The GPU is simulating, but draw shows hostPos, the CPU's copy from before the first frame. After the loop, copy pos back: cudaMemcpy(hostPos, pos, bytes, cudaMemcpyDeviceToHost);");
    if (c.copyBack >= PHYS.substeps)
      return fail(`Copying ${sec(c.copyBack)} times a second`,
        "The copy is inside the loop, so it runs after every substep, and each cudaMemcpy makes the CPU wait for the GPU. Only the last copy reaches the screen. Move it below the }.");
    if (!advance)
      return fail("The balls never move", c.swap
        ? "Each launch must be followed by exactly one swap, so the next launch starts from its result. Put the launch, then the swap, inside the loop."
        : "Every launch reads pos and writes nextPos, but nothing swaps them, so each launch starts from the same positions. Add swap(pos, nextPos); swap(vel, nextVel); right after the launch.");
    if (advance < PHYS.substeps) {
      const launchInLoop = loopOps.includes("launch");
      return fail(`Slow motion: ${advance === 1 ? "16" : (PHYS.substeps / advance).toFixed(1)} times slower than real time`, launchInLoop
        ? "The swap runs once per frame, so all 16 launches read the same positions and compute the same substep. Move the swap inside the loop, right after the launch."
        : "The launch is below the }, so it runs once per frame: 60 substeps a second instead of 960. Move the launch and the swap inside the loop.");
    }
    if (advance > PHYS.substeps)
      return fail(`Fast forward: ${(advance / PHYS.substeps).toFixed(0)} times faster than real time`,
        "The loop runs more than one launch and swap per substep. Keep one launch and one swap inside the loop.");
    if (c.launch > advance)
      return fail(`${sec(c.launch - advance)} launches a second are wasted`,
        "An extra launch recomputes a substep that is already done, because pos has not been swapped yet. Keep one launch per substep.");
    if (c.copyBack > 1 || c.draw > 1)
      return fail("Extra work every frame", "Below the loop you need one copy and one draw per frame. Remove the duplicates.");
    if (lag)
      return fail("Always one frame behind",
        "draw runs before the copy, so each frame shows the positions copied in the frame before: 16 substeps old. Copy first, then draw.");
    return pass("Real time, with one copy per frame",
      "Every frame: 16 launches and swaps on the GPU, then one copy of the 22 positions to the CPU and one draw. That is 960 launches but only 60 copies a second, because the balls stay on the GPU the whole time. " +
      "Swapping two pointers costs nothing; no data moves. This is the loop in CUDA Balls: physicsStep, then physicsRead.");
  }

  // ---------------------------------------------------------------- level 5: total energy
  // The balls' energies one second in, rounded so they fit in the memory view. The chart uses exact values.
  const ENERGY_FRAME = 60;
  const ENERGY_X = energies(refState(ENERGY_FRAME * PHYS.substeps)).map(Math.round);
  const ENERGY_TOTAL = ENERGY_X.reduce((a, b) => a + b, 0);

  // ---------------------------------------------------------------- the levels
  const MODEL_KERNEL = [
    "Each thread moves one ball forward by one substep, 1/960 of a second. The kernel is launched 16 times per frame, 60 frames a second.",
    "Blocks run one after another here: block 0, then 1, then 2. A real GPU may pick any order, or run them at the same time.",
    "Inside a block, all threads run each statement together.",
    "Step counts show the idea: a loop over all the balls costs one step per ball. Real timings come from cuda_balls --bench.",
  ];
  const MODEL_HOST = [
    "This code runs on the CPU. A kernel launch only queues work for the GPU, so the CPU can carry on.",
    "pos and nextPos are pointers to two GPU buffers. swap exchanges the pointers; no data moves.",
    "cudaMemcpy waits until the GPU has finished, then moves the data over the PCIe bus. It is the slow call here.",
    "The display calls frame() 60 times a second, so 16 substeps per frame keep the simulation in real time.",
  ];
  const kernelCommon = {
    kind: "physics",
    track: "physics",
    dims: 1,
    grid: { x: PHYS.blocks, y: 1 },
    block: { x: PHYS.threads, y: 1 },
    n: N,
    model: MODEL_KERNEL,
    blocks: kernelBlocks,
    cpuCode: "for (int i = 0; i < 22; ++i) stepBall(i, …);",
  };
  const STEP_HEADER = [
    "__global__ void stepKernel(",
    "    float2* pos, float2* vel,",
    "    float2* nextPos, float2* nextVel,",
    "    int n) {",
  ];

  const moveLevel = {
    ...kernelCommon,
    id: "balls-move",
    number: 1,
    title: "Move the balls",
    concept: "One thread per ball",
    frames: 120,
    startState: DRIFT,
    checks: { coverage: true },
    kernelHeader: ["__global__ void stepKernel(float2* pos,", "    const float2* vel, int n) {"],
    launch: "stepKernel<<<3, 8>>>(pos, vel, 22);   // 960 launches a second",
    mission:
      "This is the start of CUDA Balls, a ball simulation. 24 threads, in 3 blocks of 8, should each move one of the 22 balls a tiny step, 960 times a second. The starting kernel runs, but most balls never move. Fix it so every ball has exactly one thread.",
    hint: "Thread 3 of block 2 should move ball 19, because 2 × 8 + 3 = 19. The last 2 threads have no ball.",
    dataNote: "pos[i] and vel[i] are ball i's position and velocity. No gravity or walls yet: the balls just drift, and pass through each other.",
    nothingStored: "No thread moved its ball. Add pos[i] += vel[i] * dt;",
    palette: ["idx_tid", "idx_i", "guard", "drift"],
    start: ["idx_tid", "drift"].map((type) => ({ type })),
    passTitle: "Every ball has exactly one thread",
    passText: (sc) =>
      `Thread t of block b moved ball b × 8 + t, and the 2 spare threads returned early. All 22 balls moved at once: ${sc.gpu} steps per substep on the GPU, against ${sc.cpu} for one CPU core going ball by ball. ` +
      "Every CUDA program starts like this: work out which piece of the data is yours.",
  };

  const gravityLevel = {
    ...kernelCommon,
    id: "balls-gravity",
    number: 2,
    title: "Gravity and walls",
    concept: "Each thread's own variables",
    frames: 180,
    startState: START,
    checks: { escape: true, frozen: true },
    kernelHeader: STEP_HEADER,
    launch: "stepKernel<<<3, 8>>>(pos, vel, nextPos, nextVel, 22);   // 960 launches a second",
    mission:
      "Now make the balls fall and bounce. Each thread copies its ball into its own variables p and v, works out the force f on it, updates v and then p, and stores them back. The starting kernel has a block in the wrong place, and no walls.",
    hint: "One thread runs its statements top to bottom: load, force, walls, v += …, p += …, then store. A store that runs before p changes saves the old position.",
    dataNote: "p, v and f belong to each thread; no other thread sees them. The balls still pass through each other: collisions come next.",
    nothingStored: "Every thread works out its ball's new p and v, then throws them away. Add the block that stores them.",
    palette: ["idx", "guard", "load", "forces", "walls", "velocity", "position", "storeInPlace"],
    start: ["idx", "guard", "load", "forces", "velocity", "storeInPlace", "position"].map((type) => ({ type })),
    passTitle: "The balls fall and bounce",
    passText: (sc) =>
      `Each thread kept its ball in its own variables, and the kernel ran 960 times a second, each launch starting from the last one's result. ${sc.gpu} steps per substep on the GPU, ${sc.cpu} on one CPU core. ` +
      "The balls still pass through each other, because each thread looks only at its own ball. Next: collisions.",
  };

  const collideLevel = {
    ...kernelCommon,
    id: "balls-collide",
    number: 3,
    title: "Collisions",
    concept: "Double buffering",
    frames: PHYS.frames,
    startState: START,
    checks: { escape: true, frozen: true, overlap: true, race: true },
    kernelHeader: ["__constant__ float radius[22];", "__constant__ float mass[22];", ...STEP_HEADER],
    launch: "stepKernel<<<3, 8>>>(pos, vel, nextPos, nextVel, 22);   // 960 launches a second",
    mission:
      "Make the balls push each other apart. Each thread now has to look at every other ball, which brings a new problem: threads read positions that other threads are overwriting. Fix it so the result is the same whichever block the GPU runs first.",
    hint: "Add the for (j …) push loop above v += …. Then the race: no thread may overwrite pos while other threads still read it, so store into nextPos and nextVel instead.",
    dataNote: "If your kernel writes back into pos and vel, the host skips its usual swap. With nextPos and nextVel, the host swaps the two buffers after every launch.",
    nothingStored: "Every thread works out its ball's new p and v, then throws them away. Add the block that stores them.",
    palette: ["idx", "guard", "load", "forces", "collide", "walls", "velocity", "position", "store", "storeInPlace"],
    start: ["idx", "guard", "load", "forces", "walls", "velocity", "position", "storeInPlace"].map((type) => ({ type })),
    passTitle: "The balls bounce, stack and stay in the box",
    passText: (sc) =>
      "Each of the 22 balls had its own thread, and every thread read only the old positions, so the result is the same in any block order. " +
      `One substep took ${sc.gpu} steps on the GPU; one CPU core would need ${N} × ${sc.perBall} = ${sc.cpu}. ` +
      "At 22 balls a real CPU still keeps up; run cuda_balls --bench to see where your GPU pulls ahead.",
  };

  const hostLevel = {
    id: "balls-loop",
    kind: "host",
    track: "physics",
    number: 4,
    title: "The time loop",
    concept: "Host code and copies",
    editorTitle: "Your host code",
    kernelHeader: [
      "// CPU code, once per screen frame",
      "void frame() {",
      "  for (int s = 0; s < 16; ++s) {",
    ],
    launch: "frame();   // the display calls this 60 times a second",
    mission:
      "The kernel from level 3 is done. Now write the CPU code that drives it: each screen frame must run 16 substeps on the GPU, then bring the positions back and draw them. The starting code runs, but the picture never changes. Make the balls move in real time, without copying more than you need.",
    hint:
      "Inside the loop: launch, then swap. Below the }: copy pos to the CPU, then draw. The GPU keeps its own copy of the balls, so nothing needs to go back to it.",
    dataNote:
      "At the start, pos holds the balls and nextPos is uninitialized. Blocks above the } run 16 times per frame; blocks below it run once.",
    model: MODEL_HOST,
    catNames: { physics: "GPU work", host: "CPU side" },
    catOrder: ["physics", "host", "memory"],
    palette: ["launch", "swap", "copyBack", "upload", "draw"],
    start: ["launch", "loopEnd", "swap", "draw"].map((type) => ({ type })),
    blocks: hostBlocks,
  };

  const energyLevel = {
    id: "balls-energy",
    track: "physics",
    number: 5,
    title: "Total energy",
    concept: "Race conditions and atomics",
    dims: 1,
    grid: { x: PHYS.blocks, y: 1 },
    block: { x: PHYS.threads, y: 1 },
    n: N,
    atomicLesson: true,
    xLabel: "energy",
    unit: "ball",
    kernelHeader: ["__device__ float total = 0;", "__global__ void totalEnergy(", "    const float2* pos, const float2* vel,", "    int n) {"],
    launch: "totalEnergy<<<3, 8>>>(pos, vel, 22);   // once per frame",
    cpuCode: "for (int i = 0; i < 22; ++i) total += energy(i);",
    mission:
      "Is the simulation losing energy, as real bouncing balls do? Each thread works out its ball's energy and adds it to one shared total. The kernel runs, but the total is far too small. Run it, find out why, and fix it.",
    hint: "total += … is really three steps: read total, add, write total back. What happens when 22 threads read total at the same moment?",
    dataNote: `energy(i) = ½ m v² + m g y for ball i, one second into the simulation, rounded. The right total is ${ENERGY_TOTAL}.`,
    palette: ["idx_fixed", "guard", "race", "atomic"],
    start: ["idx_fixed", "guard", "race"].map((type) => ({ type })),
    blocks: {
      idx_fixed: E.IDX_FIXED,
      guard: E.GUARD,
      race: {
        cat: "memory",
        parts: ["total += energy(i);"],
        exec(ctx) {
          const i = ctx.get("i");
          ctx.write("total", 0, ctx.read("total", 0) + ctx.read("x", i));
        },
      },
      atomic: {
        cat: "sync",
        parts: ["atomicAdd(&total, energy(i));"],
        exec(ctx) {
          const i = ctx.get("i");
          ctx.atomicAdd("total", 0, ctx.read("x", i));
        },
      },
    },
    memory: () => ({ x: { data: ENERGY_X }, total: { data: [0] } }),
    expectedTotal: ENERGY_TOTAL,
    cpuSteps: N,
    passText: (res, diag) =>
      `The ${diag.atomics} atomicAdd calls had to take turns, so this took ${res.steps} steps, about as many as the CPU loop. That is fine for 22 balls; for millions, a tree of partial sums in shared memory is faster (the classic track's last level). ` +
      "The chart below runs your kernel on every frame: the total falls as collisions and air drag turn motion into heat. CUDA Balls checks the same thing in --selftest.",
  };

  // ---------------------------------------------------------------- level 6: pie the professor
  // A parameter sweep: each thread flies one whole throw inside the kernel, then the threads that hit
  // the professor's face agree on the gentlest one. The pies never touch each other, so no swaps and no
  // copies are needed until the answer. The floor, the jacket and the walls only stop the drawing of a
  // pie; the kernel's result depends on face hits alone.
  const PIE = Object.freeze({
    throws: 20, steps: 1440, every: 16, angle: (35 * Math.PI) / 180, pieR: 0.16,
    hand: { x: -4.8, y: 1.6 },
    head: { x: 4.55, y: 3.55, r: 0.38 },
    // where missed pies stick, matching the photo as the page places it (face at the head, feet on the floor)
    jacket: { x0: 3.8, x1: 5.15, y0: 1.7, y1: 3.2 },
    legs: { x0: 4.35, x1: 5.0, top: 1.7 },
  });
  const pieSpeed = (i) => 9.25 + 0.25 * i;

  const sweepBlocks = {
    gravity: { cat: "physics", parts: ["v.y -= g * dt;"] },
    fly: { cat: "physics", parts: ["p += v * dt;"] },
    hitCheck: { cat: "control", parts: ["if (hitsFace(p)) {\n  hit = true;  break;\n}"] },
    recordRace: { cat: "memory", parts: ["if (hit && i < gentlest)\n  gentlest = i;"] },
    recordAtomic: { cat: "sync", parts: ["if (hit)\n  atomicMin(&gentlest, i);"] },
    loopEnd: { cat: "loop", fixed: true, parts: ["}   // end of the flight"] },
  };

  function runSweep(level, program) {
    const types = program.map((s) => s.type);
    const end = types.indexOf("loopEnd");
    const loopOps = types.slice(0, end), afterOps = types.slice(end + 1);
    const T = PIE.throws, nThreads = PHYS.blocks * PHYS.threads;
    const throws = [];
    for (let i = 0; i < T; i++) {
      const sp = pieSpeed(i);
      throws.push({
        i, speed: sp, p: { ...PIE.hand }, v: { x: sp * Math.cos(PIE.angle), y: sp * Math.sin(PIE.angle) },
        inLoop: true, hit: false, hitStep: null, shown: { ...PIE.hand }, stop: null,
      });
    }
    const snapShown = () => {
      const a = new Float32Array(2 * T);
      throws.forEach((t, k) => { a[2 * k] = t.shown.x; a[2 * k + 1] = t.shown.y; });
      return a;
    };
    const exec = (op, t, step) => {
      if (op === "gravity") t.v.y -= PHYS.g * PHYS.dt;
      else if (op === "fly") { t.p.x += t.v.x * PHYS.dt; t.p.y += t.v.y * PHYS.dt; }
      else if (op === "hitCheck") {
        if (t.inLoop && Math.hypot(t.p.x - PIE.head.x, t.p.y - PIE.head.y) < PIE.head.r + PIE.pieR / 2) {
          t.hit = true;
          t.hitStep = step;
          t.inLoop = false; // break
        }
      }
    };
    // where the drawing of a pie stops: the face (a hit), or the floor, the jacket, a wall or the ceiling
    const settle = (t, step) => {
      if (t.stop) return;
      const { x, y } = t.p, r = PIE.pieR, J = PIE.jacket, G = PIE.legs;
      let kind = null;
      if (t.hit) kind = "face";
      else if (x > J.x0 - r && x < J.x1 + r && y > J.y0 && y < J.y1) kind = "jacket";
      else if (x > G.x0 - r && x < G.x1 + r && y < G.top) kind = "legs";
      else if (y < r) kind = "floor";
      else if (x > PHYS.halfW - r) kind = "wall";
      else if (y > PHYS.H - r) kind = "ceiling";
      t.shown = { x, y };
      if (kind) t.stop = { kind, step };
    };
    const frames = [snapShown()];
    for (let step = 0; step < PIE.steps; step++) {
      for (const op of loopOps) for (const t of throws) if (t.inLoop) exec(op, t, step);
      for (const t of throws) settle(t, step);
      if ((step + 1) % PIE.every === 0) frames.push(snapShown());
    }
    // after the loop: statements below the } run once; every thread reaches them at the same moment
    let gentlest = T;
    const racers = [];
    for (const op of afterOps) {
      if (op === "recordRace") {
        const seen = gentlest, writers = throws.filter((t) => t.hit && t.i < seen);
        for (const t of writers) gentlest = t.i; // all read the same old value; the last write lands
        racers.push(...writers.map((t) => t.i));
      } else if (op === "recordAtomic") {
        for (const t of throws) if (t.hit) gentlest = Math.min(gentlest, t.i); // one at a time
      } else for (const t of throws) exec(op, t, PIE.steps);
    }
    const hitters = throws.filter((t) => t.hit).map((t) => t.i);
    const res = {
      kind: "sweep", level, program, frames, loopOps, afterOps, gentlest, hitters, racers,
      truth: hitters.length ? hitters[0] : null,
      throws: throws.map((t) => ({ i: t.i, speed: t.speed, hit: t.hit, hitStep: t.hitStep, stop: t.stop })),
      firstThreads: Array.from({ length: nThreads }, (_, id) => ({ id, bx: Math.floor(id / PHYS.threads), tx: id % PHYS.threads, i: id, returned: id >= T, fault: null })),
    };
    res.verdict = judgeSweep(res);
    return res;
  }

  function judgeSweep(res) {
    const { loopOps: L, afterOps: A, hitters } = res;
    const where = (op) => (L.includes(op) ? "loop" : A.includes(op) ? "after" : null);
    const T = PIE.throws;
    if (!L.length && !A.length) return fail("The kernel is empty", "Drag blocks into it. Blocks above the } run once per step of the flight; blocks below it run once, after the flight.");
    for (const op of new Set([...L, ...A]))
      if ([...L, ...A].filter((o) => o === op).length > 1)
        return fail(`${sweepBlocks[op].parts.join("").replace(/\s+/g, " ")} is in the kernel twice`, "Each step of a throw happens once. Remove the extra copy.");
    if (where("fly") !== "loop")
      return fail("The pies never leave the launcher", where("fly")
        ? "p += v * dt is below the }, so it runs once, after the flight. Move it inside the loop."
        : "Nothing moves the pies. Add p += v * dt; inside the loop.");
    if (where("gravity") !== "loop")
      return fail("The pies fly in straight lines", where("gravity")
        ? "v.y -= g * dt is below the }, so gravity acts once, after the flight is over. Move it inside the loop."
        : "Nothing pulls the pies down, so every throw sails over the professor. Add v.y -= g * dt; inside the loop.");
    if (where("hitCheck") === "after")
      return fail("The hit check is outside the loop",
        "Below the }, break has no loop to leave, and the check would only look at where each pie ended up. Move it inside the loop.");
    if (!where("hitCheck"))
      return fail("Pies fly straight through the professor",
        "Nothing checks for a hit, so no thread ever sets hit. Add the hitsFace(p) check inside the loop.");
    if (!where("recordRace") && !where("recordAtomic"))
      return fail("Nobody wrote down the answer",
        `${hitters.length} throws hit the face, but the kernel never records one. Below the }, add the block that keeps the gentlest hit.`);
    if (where("recordRace") === "loop" || where("recordAtomic") === "loop")
      return fail("gentlest is still 20: nothing was recorded",
        "The record is inside the loop, but a thread that hits leaves the loop at break, so it never reaches the record. Move it below the }.");
    if (!hitters.length) return fail("No pie hit the face", "Check that gravity and the move both run inside the loop.");
    const truth = res.truth;
    const range = hitters.length > 2 ? `${hitters[0]}–${hitters.at(-1)}` : hitters.join(" and ");
    if (where("recordRace") && where("recordAtomic"))
      return fail("Two records", "Keep only the atomicMin block: the plain compare-and-write is the race you are replacing.");
    if (where("recordRace") && res.gentlest !== truth)
      return fail(`gentlest = ${res.gentlest}, but throw ${truth} is gentler: a race`,
        `Throws ${range} all hit. Each of those threads compared its i with gentlest = ${T} at the same moment, so all ${res.racers.length} wrote, and the last write, throw ${res.gentlest}, survived. ` +
        "atomicMin compares and writes in one indivisible step: use if (hit) atomicMin(&gentlest, i);");
    if (where("recordRace"))
      return fail("Right answer, by luck", "Only one thread wrote this time. Use atomicMin so the answer stays right however many throws hit.");
    const before = res.throws[truth - 1];
    const slower = before ? ` Throw ${truth - 1}, at ${before.speed.toFixed(2)} m/s, lands on the ${before.stop ? before.stop.kind : "floor"} instead.` : "";
    return pass(`Splat! Throw ${truth} is the gentlest that lands`,
      `It leaves the launcher at ${res.throws[truth].speed.toFixed(2)} m/s.${slower} ` +
      `Every thread flew a whole 1.5-second throw, ${PIE.steps.toLocaleString("en-US")} steps, inside a single launch, with no copies until the answer. ` +
      `Trying many independent inputs at once is called a parameter sweep, one of the easiest ways to put a GPU to work. One CPU core would fly the throws one after another: ${T} × ${PIE.steps.toLocaleString("en-US")} = ${(T * PIE.steps).toLocaleString("en-US")} steps.`);
  }

  const pieLevel = {
    id: "balls-pie",
    kind: "sweep",
    track: "physics",
    number: 6,
    title: "Pie the professor",
    concept: "A parameter sweep",
    dims: 1,
    grid: { x: PHYS.blocks, y: 1 },
    block: { x: PHYS.threads, y: 1 },
    n: PIE.throws,
    kernelHeader: [
      "__device__ int gentlest = 20;   // no hit yet",
      "__global__ void throwPies(",
      "    const float* speed, int n) {",
      "  int i = blockIdx.x * blockDim.x",
      "        + threadIdx.x;",
      "  if (i >= n) return;",
      "  float2 p = hand;",
      "  float2 v = throwAt(speed[i]);",
      "  bool hit = false;",
      "  for (int s = 0; s < 1440; ++s) {",
    ],
    launch: "throwPies<<<3, 8>>>(speed, 20);   // 20 throws, one launch",
    mission:
      "Finale: the department's charity fundraiser, where a volunteer professor takes a pie to the face. Twenty pies are ready, thrown from soft to fast, and each thread flies one whole throw. Find the gentlest throw that still lands on the face. The starting kernel has no gravity, and its record of the answer is not safe.",
    hint:
      "Inside the loop: gravity, then the move, then the hit check. Below the }: record the gentlest hit. Several threads hit at once, so the compare and the write must happen in one step.",
    dataNote:
      "Throw i leaves the launcher at 9.25 + 0.25 × i m/s, always at 35°. The flight is simulated for 1.5 seconds in steps of 1/960 s. Blocks above the } run every step; blocks below it run once.",
    model: [
      "Each thread flies one throw from start to finish, so one kernel launch covers the whole 1.5 seconds.",
      "The throws never touch each other, so threads share nothing until they record the answer.",
      "Every thread reaches the line below the } at the same moment, as in level 5.",
      "A pie that misses the face stops on the jacket, the legs, the floor or a wall in the drawing; only face hits count.",
    ],
    catOrder: ["physics", "control", "memory", "sync"],
    palette: ["gravity", "fly", "hitCheck", "recordRace", "recordAtomic"],
    start: ["fly", "hitCheck", "loopEnd", "recordRace"].map((type) => ({ type })),
    blocks: sweepBlocks,
  };

  // The "Build CUDA Balls" track comes first; the original four levels become the classic track.
  for (const l of E.LEVELS) l.track = l.track || "classic";
  E.LEVELS.unshift(moveLevel, gravityLevel, collideLevel, hostLevel, energyLevel, pieLevel);

  // Total energy per frame as the student's kernel computes it, next to the true total.
  function energySeries(level, program, every = 4) {
    const out = [];
    for (let f = 0; f <= PHYS.frames; f += every) {
      const e = energies(refState(f * PHYS.substeps));
      const variant = { ...level, memory: () => ({ x: { data: e }, total: { data: [0] } }) };
      const res = E.runGPU(variant, program);
      const got = res.diag.faults.length ? null : res.mem.arrays.total.committed[0];
      out.push({ t: f / PHYS.fps, truth: e.reduce((a, b) => a + b, 0), got });
    }
    return out;
  }

  function run(level, program) {
    if (level.kind === "host") return runHost(level, program);
    if (level.kind === "sweep") return runSweep(level, program);
    return runKernelLevel(level, program);
  }

  return { PHYS, PIE, RADIUS, MASS, run, simulate, refState, garbageState, stepCounts, energySeries, ENERGY_TOTAL };
})();

if (typeof module !== "undefined") module.exports = Physics;

