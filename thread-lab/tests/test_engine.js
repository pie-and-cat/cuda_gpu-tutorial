// Node tests: correct programs pass, typical mistakes get the right diagnosis.
const E = require("../src/engine.js");
const [L1, L2, L3, L4] = E.LEVELS;
let failures = 0;
function check(name, level, program, expectOk, mustInclude) {
  const r = E.runGPU(level, program);
  const v = r.verdict;
  const text = `${v.title} | ${v.message}`;
  const okMatch = v.ok === expectOk && (!mustInclude || text.includes(mustInclude));
  console.log(`${okMatch ? "ok  " : "FAIL"} L${level.number} ${name}: [${v.ok ? "pass" : "fail"}] ${text}  (steps ${r.steps}, frames ${r.frames.length})`);
  if (!okMatch) failures++;
  return r;
}
const idx = (a, b, c, type = "idx_i") => ({ type, fields: { a, b, c } });

// ---- level 1
check("starting program", L1, L1.start, false, "before setting it");
check("default index formula, no guard", L1, [{ type: "idx_i" }, { type: "add" }], false, "went outside");
check("default formula with guard", L1, [{ type: "idx_i" }, { type: "guard" }, { type: "add" }], false, "never written");
check("threadIdx only", L1, [{ type: "idx_tid" }, { type: "guard" }, { type: "add" }], false, "blockIdx.x");
check("right formula, no guard", L1, [idx("blockIdx.x", "blockDim.x", "threadIdx.x"), { type: "add" }], false, "Add if (i >= n)");
check("guard after add", L1, [idx("blockIdx.x", "blockDim.x", "threadIdx.x"), { type: "add" }, { type: "guard" }], false, "too late");
check("multiply instead of add", L1, [idx("blockIdx.x", "blockDim.x", "threadIdx.x"), { type: "guard" }, { type: "mul" }], false, "should be");
const ok1 = check("correct", L1, [idx("blockIdx.x", "blockDim.x", "threadIdx.x"), { type: "guard" }, { type: "add" }], true, "3 steps");
check("correct, swapped operands", L1, [idx("blockDim.x", "blockIdx.x", "threadIdx.x"), { type: "guard" }, { type: "add" }], true);
check("empty", L1, [], false, "empty");
console.log("   c =", ok1.mem.arrays.c.committed.join(" "));

// ---- level 2
const X = idx("blockIdx.x", "blockDim.x", "threadIdx.x", "idx_x");
const Y = idx("blockIdx.y", "blockDim.y", "threadIdx.y", "idx_y");
check("starting program", L2, L2.start, false, "Nothing was written");
check("y from .x vars", L2, [X, { type: "idx_y" }, { type: "guard2" }, { type: "blur" }], false, ".y variables");
check("no guard", L2, [X, Y, { type: "blur" }], false, "outside the 14 × 10 image");
check("copy, not blur", L2, [X, Y, { type: "guard2" }, { type: "copy" }], false, "should be");
check("transposed", L2, [X, Y, { type: "guard2" }, { type: "blurT" }], false, "out[y][x]");
const ok2 = check("correct", L2, [X, Y, { type: "guard2" }, { type: "blur" }], true, "4 steps");
const out = ok2.mem.arrays.out.committed;
for (let y = 0; y < 10; y++) console.log("   " + out.slice(y * 14, y * 14 + 14).join(""));

// ---- level 3
check("starting program (race)", L3, L3.start, false, "race condition");
const ok3 = check("atomic", L3, [{ type: "idx_fixed" }, { type: "atomic" }], true, "take turns");
check("atomic + guard", L3, [{ type: "idx_fixed" }, { type: "guard" }, { type: "atomic" }], true);
check("race + atomic both", L3, [{ type: "idx_fixed" }, { type: "race" }, { type: "atomic" }], false, "Remove that block");
check("atomic without index", L3, [{ type: "atomic" }], false, "before setting it");
console.log("   atomic frames:", ok3.frames.length, "steps:", ok3.steps);

// ---- level 4
const S = (s) => ({ type: "stride", fields: { s: String(s) } });
const sync = { type: "sync" };
const good = [{ type: "tid" }, { type: "load" }, sync, S(8), sync, S(4), sync, S(2), sync, S(1), { type: "final" }];
const ok4 = check("correct", L4, good, true, "11 steps");
check("correct with extra final sync", L4, [...good.slice(0, 10), sync, { type: "final" }], true);
check("starting program", L4, L4.start, false, "total is still 0");
check("no syncs at all", L4, [{ type: "tid" }, { type: "load" }, S(8), S(4), S(2), S(1), { type: "final" }], false, "read too early");
check("missing sync after load", L4, [{ type: "tid" }, { type: "load" }, S(8), sync, S(4), sync, S(2), sync, S(1), { type: "final" }], false, "thread 8");
check("missing sync before stride 1", L4, [{ type: "tid" }, { type: "load" }, sync, S(8), sync, S(4), sync, S(2), S(1), { type: "final" }], false, "step 9");
check("stride 8 twice", L4, [{ type: "tid" }, { type: "load" }, sync, S(8), sync, S(8), sync, S(4), sync, S(2), sync, S(1), { type: "final" }], false, "added twice");
check("ascending strides", L4, [{ type: "tid" }, { type: "load" }, sync, S(1), sync, S(2), sync, S(4), sync, S(8), { type: "final" }], false);
check("race on total", L4, [{ type: "tid" }, { type: "load" }, sync, { type: "raceTotal" }], false, "race condition");
check("no load", L4, [{ type: "tid" }, sync, S(8), sync, S(4), sync, S(2), sync, S(1), { type: "final" }], false, "partial[t] = x[t]");
console.log("   partial after correct run:", ok4.mem.arrays.partial.committed.join(" "));

// ---- CPU loops
for (const L of E.LEVELS) {
  const c = E.runCPU(L);
  console.log(`   CPU L${L.number}: ${c.steps} steps, last frame: ${c.frames.at(-1).label}`);
}
console.log(failures ? `\n${failures} FAILED` : "\nall engine checks passed");
process.exit(failures ? 1 : 0);
