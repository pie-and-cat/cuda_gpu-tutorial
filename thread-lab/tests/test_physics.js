// Node tests for the "Build CUDA Balls" track: correct programs pass, typical mistakes get the right diagnosis.
const E = require("../src/engine.js");
global.Engine = E;
const P = require("../src/physics.js");
const lvl = (id) => E.LEVELS.find((l) => l.id === id);
const [L1, L2, L3, L4, L5, L6] = ["balls-move", "balls-gravity", "balls-collide", "balls-loop", "balls-energy", "balls-pie"].map(lvl);
let failures = 0;
const prog = (s) => s.split(/\s+/).filter(Boolean).map((type) => ({ type }));
const RIGHT_IDX = { type: "idx_i", fields: { a: "blockIdx.x", b: "blockDim.x", c: "threadIdx.x" } };
const runAny = (level, program) => (level.kind ? P.run(level, program) : E.runGPU(level, program));

function check(name, level, program, expectOk, mustInclude) {
  const r = runAny(level, program);
  const v = r.verdict;
  const text = `${v.title} | ${v.message}`;
  const good = v.ok === expectOk && (!mustInclude || text.includes(mustInclude));
  console.log(`${good ? "ok  " : "FAIL"} L${level.number} ${name}: [${v.ok ? "pass" : "fail"}] ${v.title}`);
  if (!good) {
    console.log("     " + v.message);
    failures++;
  }
  return r;
}
const expect = (cond, okText, failText) => {
  console.log(cond ? `ok   ${okText}` : `FAIL ${failText}`);
  if (!cond) failures++;
};

// ---- track layout
const order = E.LEVELS.map((l) => `${l.track}:${l.number}`).join(" ");
expect(order === "physics:1 physics:2 physics:3 physics:4 physics:5 physics:6 classic:1 classic:2 classic:3 classic:4", `track order ${order}`, `track order ${order}`);

// ---- level 1: move the balls
check("starting kernel", L1, L1.start, false, "3 times too fast");
check("formula with default dropdowns", L1, prog("idx_i drift"), false, "Add if (i >= n)");
check("default dropdowns + guard", L1, prog("idx_i guard drift"), false, "Check the formula");
check("right formula, no guard", L1, [RIGHT_IDX, { type: "drift" }], false, "Add if (i >= n)");
check("guard too late", L1, [RIGHT_IDX, { type: "drift" }, { type: "guard" }], false, "too late");
check("correct", L1, [RIGHT_IDX, { type: "guard" }, { type: "drift" }], true, "b × 8 + t");
check("no index", L1, prog("guard drift"), false, "used i before");
check("two index blocks", L1, [RIGHT_IDX, { type: "idx_tid" }, { type: "guard" }, { type: "drift" }], false, "Two blocks");
check("nothing moves", L1, [RIGHT_IDX, { type: "guard" }], false, "pos[i] += vel[i]");

// ---- level 2: gravity and walls
check("starting kernel", L2, L2.start, false, "store runs before p");
check("store moved last", L2, prog("idx guard load forces velocity position storeInPlace"), false, "pushFromWalls");
check("correct", L2, prog("idx guard load forces walls velocity position storeInPlace"), true, "pass through each other");
check("walls after velocity", L2, prog("idx guard load forces velocity walls position storeInPlace"), false, "never used");
check("no velocity update", L2, prog("idx guard load forces walls position storeInPlace"), false, "v += f / mass[i] * dt");
check("no position update", L2, prog("idx guard load forces walls velocity storeInPlace"), false, "p += v * dt");
check("forces before load", L2, prog("idx guard forces load walls velocity position storeInPlace"), false, "used v before");

// ---- level 3: collisions
const CORRECT = "idx guard load forces collide walls velocity position store";
check("starting kernel", L3, L3.start, false, "Nothing pushes");
const race = check("collisions with in-place store", L3, prog("idx guard load forces collide walls velocity position storeInPlace"), false, "race");
const ok3 = check("correct", L3, prog(CORRECT), true, "616");
check("walls before collisions", L3, prog("idx guard load forces walls collide velocity position store"), true);
check("explicit Euler order", L3, prog("idx guard load forces collide walls position velocity store"), true);
check("both stores", L3, prog("idx guard load forces collide walls velocity position store storeInPlace"), false, "race");
check("collisions after velocity", L3, prog("idx guard load forces walls velocity collide position store"), false, "never used");
check("no guard", L3, prog("idx load forces collide walls velocity position store"), false, "if (i >= n) return;");

const sim = P.simulate(L3, prog(CORRECT), [0, 1, 2], 20);
const ref = P.refState(20 * P.PHYS.substeps);
let maxDiff = 0;
for (let k = 0; k < ref.length; k++) maxDiff = Math.max(maxDiff, Math.abs(ref[k] - sim.frames[20][k]));
expect(maxDiff === 0, "interpreter matches the reference physics exactly", `interpreter differs from reference by ${maxDiff}`);
const rev = P.simulate(L3, prog(CORRECT), [2, 1, 0]);
let orderDiff = 0;
for (let k = 0; k < rev.frames.at(-1).length; k++) orderDiff = Math.max(orderDiff, Math.abs(rev.frames.at(-1)[k] - ok3.frames.at(-1)[k]));
expect(orderDiff === 0, "correct kernel gives identical results in either block order", `correct kernel depends on block order: ${orderDiff}`);
expect(race.verdict.divergence > 0.5, `racy kernel diverges by ${race.verdict.divergence?.toFixed(2)} units with the blocks reversed`, "racy kernel diverges too little to see");

// ---- level 4: the time loop
check("correct", L4, prog("launch swap loopEnd copyBack draw"), true, "960 a second");
check("starting code", L4, L4.start, false, "never changes");
check("start + copy", L4, prog("launch loopEnd swap copyBack draw"), false, "swap runs once per frame");
check("swap before launch", L4, prog("swap launch loopEnd copyBack draw"), false, "garbage");
check("no swap", L4, prog("launch loopEnd copyBack draw"), false, "nothing swaps them");
check("copy inside the loop", L4, prog("launch swap copyBack loopEnd draw"), false, "Copying 960");
check("draw inside the loop", L4, prog("launch swap copyBack draw loopEnd"), false, "Drawing 960");
check("draw before copy", L4, prog("launch swap loopEnd draw copyBack"), false, "one frame behind");
check("upload in the loop", L4, prog("upload launch swap loopEnd copyBack draw"), false, "rewinding");
check("two substeps per iteration", L4, prog("launch swap launch swap loopEnd copyBack draw"), false, "Fast forward");

// ---- level 5: total energy
check("starting kernel", L5, L5.start, false, "race condition");
const ok5 = check("correct", L5, prog("idx_fixed guard atomic"), true, "chart");
expect(ok5.mem.arrays.total.committed[0] === P.ENERGY_TOTAL, `atomic total is ${P.ENERGY_TOTAL}`, `atomic total is ${ok5.mem.arrays.total.committed[0]}`);
check("no guard", L5, prog("idx_fixed atomic"), false, "outside the data");
check("both adds", L5, prog("idx_fixed guard race atomic"), false, "still contains");
const good = P.energySeries(L5, prog("idx_fixed guard atomic"));
const bad = P.energySeries(L5, prog("idx_fixed guard race"));
expect(good.every((p) => Math.abs(p.got - p.truth) < 1e-6 * p.truth), `atomic kernel matches the true total on all ${good.length} sampled frames`, "atomic kernel differs from the true total");
expect(bad.every((p) => p.got < 0.2 * p.truth), "racy kernel loses most of the total on every frame", "racy kernel is sometimes close to right");
console.log(`     energy: ${good[0].truth.toFixed(0)} at 0 s, ${good[15].truth.toFixed(0)} at 1 s, ${good.at(-1).truth.toFixed(0)} at 4 s (rounded total at 1 s: ${P.ENERGY_TOTAL})`);
expect(good.at(-1).truth < 0.6 * good[0].truth, "energy falls over the run", "energy does not fall");

// ---- level 6: pie the professor
check("starting kernel", L6, L6.start, false, "straight lines");
const pieRace = check("gravity added", L6, prog("gravity fly hitCheck loopEnd recordRace"), false, "a race");
check("correct", L6, prog("gravity fly hitCheck loopEnd recordAtomic"), true, "Throw 8");
check("move before gravity", L6, prog("fly gravity hitCheck loopEnd recordAtomic"), true);
check("gravity below the loop", L6, prog("fly hitCheck loopEnd gravity recordAtomic"), false, "acts once");
check("move below the loop", L6, prog("gravity hitCheck loopEnd fly recordAtomic"), false, "runs once");
check("hit check below the loop", L6, prog("gravity fly loopEnd hitCheck recordAtomic"), false, "outside the loop");
check("no hit check", L6, prog("gravity fly loopEnd recordAtomic"), false, "Nothing checks");
check("record inside the loop", L6, prog("gravity fly hitCheck recordAtomic loopEnd"), false, "never reaches");
check("no record", L6, prog("gravity fly hitCheck loopEnd"), false, "never records");
check("both records", L6, prog("gravity fly hitCheck loopEnd recordRace recordAtomic"), false, "Two records");
expect(pieRace.gentlest === 12 && pieRace.truth === 8, "the racy record keeps throw 12; the gentlest hit is throw 8", `racy record kept ${pieRace.gentlest}, truth ${pieRace.truth}`);
expect(pieRace.hitters.join() === "8,9,10,11,12", "throws 8–12 hit the face, all in block 1", `hitters ${pieRace.hitters}`);

console.log(failures ? `\n${failures} FAILED` : "\nall physics checks passed");
process.exit(failures ? 1 : 0);
