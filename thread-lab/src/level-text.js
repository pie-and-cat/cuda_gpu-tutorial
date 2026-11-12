// All the words students read in the levels: title, mission, hint, the note about the data and the
// "you solved it" message. Edit the text here; the level code in engine.js and physics.js does the rest.
//
// Each level is an object keyed by its id. A few need numbers from the level code, so they are written
// as ({ name }) => ({ ... }): the names in the braces are filled in by the code and can be used in the text
// as ${name}. passText is a function of the run's score and returns the sentence shown after a pass.
//
// Further down are the messages shown after a run, when it goes wrong or right (kernelMessages, hostMessages,
// sweepMessages, classicMessages). The judge code in engine.js and physics.js decides what went wrong and names
// it with a short key (a "why"); the sentence for that key lives here, so you can reword it freely.
const LEVEL_TEXT = (() => {
  const levels = {
    // CUDA Balls · level 1
    "balls-move": {
      title: "Balls",
      concept: "",
      mission:
        "Your first task- give each ball a thread to move it. A free hint for the first one: blockId.x * blockDim.x + threadIdx.x. (Candy for the first person to explain why it works.) ",
      hint: "There are more threads than balls. How do you end the extra threads early?",
      metaphorHint:
        "We hired 24 workers (3 teams of 8), but there are only 22 balls. How do you tell the 2 extra workers with no ball to clock out early so they don't break anything?",
      dataNote: "pos[i] and vel[i] are ball i's position and velocity. No gravity or walls yet!.",
      nothingStored: "No thread moved its ball. Add pos[i] += vel[i] * dt;",
      passTitle: "Every ball has a thread!",
      passText: (sc) =>
        `Thread t of block b moved ball b × 8 + t, and the 2 spare threads returned early. The GPU was faster than the CPU, with ${sc.gpu} steps per substep on the GPU, against the CPU's ${sc.cpu}. Congrats! `
    },

    // CUDA Balls · level 2
    "balls-gravity": {
      title: "Gravity & walls",
      concept: "",
      mission:
        "Now make the balls fall and bounce! Each thread copies its ball into its own variables p and v, works out the force f on it, updates v and then p, and stores them back.",
      hint: "A store that runs before p changes saves the old position.",
      metaphorHint:
        "If a worker writes back to the warehouse (pos[i] = p) before updating p on their clipboard, they'll just save the old position and the ball won't move!",
      dataNote: "p, v and f belong to each thread; no other thread sees them. The balls still pass through each other.",
      nothingStored: "Every thread works out its ball's new p and v, then throws them away. Add the block that stores them.",
      passTitle: "The balls fall and bounce!",
      passText: (sc) =>
        `The balls still pass through each other, because each thread looks only at its own ball. ${sc.gpu} steps per substep on the GPU, ${sc.cpu} on one CPU core. Congrats! `
    },

    // CUDA Balls · level 3
    "balls-collide": ({ N }) => ({
      title: "Collisions",
      concept: "",
      mission:
        "Make the balls collide! I believe in you! :)",
      hint: "Which of the blocks consider the positions of other balls?",
      metaphorHint:
        "Look for the purple loop block (for (int j = 0...)). That's your worker scanning every other ball (j) to see if it's touching theirs (i).",
      dataNote: "If your kernel writes back into pos and vel, the host skips its usual swap. With nextPos and nextVel, the host swaps the two buffers after every launch.",
      nothingStored: "Every thread works out its ball's new p and v, then throws them away. Add the block that stores them.",
      passTitle: "The balls bounce, collide and stay in the box!",
      passText: (sc) =>
        "Each of the 22 balls had its own thread, and every thread read only the old positions, so the result is the same in any block order. " +
        `One substep took ${sc.gpu} steps on the GPU; one CPU core would need ${N} × ${sc.perBall} = ${sc.cpu}. ` +
        "At 22 balls a real CPU still keeps up; run cuda_balls --bench to see where your GPU pulls ahead.",
    }),

    // CUDA Balls · level 4
    "balls-loop": {
      title: "The Big Loop. 0.0",
      concept: "",
      editorTitle: "Your host code",
      mission:
        "It's time to write the CPU code! Make the balls move in real time. The code from the previous level is running.",
      hint:
        "Do you want to copy the positions back to the CPU every screen frame, or every frame?",
      metaphorHint:
        "Are you going to ship the results from the GPU warehouse to your CPU desk (cudaMemcpy) for every tiny physics tick (16 times per frame), or just once at the end when you're ready to draw (hostPos)?",
      dataNote:
        "At the start, pos holds the balls and nextPos is uninitialized. Blocks above the } run 16 times per frame; blocks below it run once.",
      passTitle: "The balls move in real time!",
      passText: (res) =>
        "The balls move in real time, and the GPU keeps its own copy of their positions. " +
        `Each screen frame ran ${res.counts.launch} launches on the GPU, ${res.counts.launch * 60} a second. Congrats!`,
    },

    // CUDA Balls · level 5
    "balls-energy": ({ ENERGY_TOTAL }) => ({
      title: "Total energy",
      concept: "Race conditions and atomics",
      mission:
        "Are the balls losing energy, like in real life? Each thread works out its ball's energy and adds it to a shared total, but the total is smaller than it should be. Fix it plz",
      hint: "Are all of the threads adding to total at the same time?",
      metaphorHint:
        "Traffic Jam at the Scoreboard: With total += energy(i), all 8 workers on a team read total and add their own energy on their clipboard, and write back at the exact same time. Look at the pink SYNC AND ATOMICS block (atomicAdd): it puts a velvet rope around total so workers form a single-file line and add their numbers one by one.",
      dataNote: `energy(i) = ½ m v² + m g y for ball i, one second into the simulation, rounded. The right total is ${ENERGY_TOTAL}.`,
      passText: (res, diag) =>
        `The ${diag.atomics} atomicAdd calls had to take turns, so this took ${res.steps} steps, about as many as the CPU loop. That is fine for 22 balls; for millions, a tree of partial sums in shared memory is faster. Congrats!`
    }),

    // CUDA Balls · level 6
    "balls-pie": {
      title: "Pie the professor!",
      concept: "A parameter sweep",
      mission:
        "Make a pie thrower simulation with 16 throws at once, and find the gentlest that hits the professor. " + 
        "Each thread flies one throw, and if it hits, compares its throw number i with gentlest. If i is smaller, it writes i into gentlest.",
      hint:
        "What are the forces that act on a pie? What is the minimum amount of force that hits the professor?",
      dataNote:
        "Throw i leaves the launcher at 9.25 + 0.25 × i m/s, always at 35°. The flight is simulated for 1.5 seconds in steps of 1/960 s. Blocks above the } run every step; blocks below it run once.",
    },

    // Classic · level 1
    "vector-add": {
      title: "One thread per element",
      concept: "Thread indexing",
      mission:
        "Add two 28-element arrays with 32 threads. Each thread must work out its own element i, stay inside the arrays, and write c[i] = a[i] + b[i].",
      hint: "Thread 3 of block 2 should get i = 19, because 2 × 8 + 3 = 19. The last 4 threads have no element to work on.",
      dataNote: "a holds the first digits of π, b the first digits of e.",
    },

    // Classic · level 2
    "image-blur": {
      title: "One thread per pixel",
      concept: "2-D grids",
      mission:
        "Blur a 14 × 10 image with one thread per pixel. The grid is 4 × 3 blocks of 4 × 4 threads, which is 16 × 12 threads, so some threads fall off the right and bottom edges.",
      hint: "x comes from the .x variables and y from the .y variables. Threads with x ≥ 14 or y ≥ 10 have no pixel.",
      dataNote: "blur_pixel returns the rounded average of the 3 × 3 input pixels around (x, y), the same function as in the tutorial's image_blur.cu.",
    },

    // Classic · level 3
    "race": ({ sum16 }) => ({
      title: "Race to the total",
      concept: "Race conditions and atomics",
      mission:
        "Sixteen threads each add one number to a shared total. The kernel on the left compiles and runs, but its answer is wrong. Run it, find out why, and fix it.",
      hint: "total += x[i] is really three steps: read total, add, write total back. What if 16 threads read total at the same moment?",
      dataNote: `The right answer is ${sum16}.`,
    }),

    // Classic · level 4
    "tree": ({ sum16 }) => ({
      title: "Fast and correct: the tree",
      concept: "Shared memory and __syncthreads()",
      mission:
        "Make the sum correct without making threads take turns. Pairs of threads add their values in shared memory, halving the number of active threads each step: 16 values, then 8, 4, 2, 1. Use __syncthreads() so that no thread reads a value before another thread has finished writing it.",
      hint: "Strides 8, 4, 2, 1, each one followed by __syncthreads(). Thread 0 then copies partial[0] into total.",
      dataNote: `The right answer is ${sum16}. In this lab, a thread sees another thread's write only after a __syncthreads() (its own writes it sees at once).`,
    }),
  };

  // ---------------------------------------------------------------- messages after a run
  // What the lab tells the student when a run goes wrong, or right. Each message is [title, details].
  // The names in the braces are values the code works out (a thread's label, a count) and hands over;
  // the code in engine.js and physics.js picks which message to show, and these functions only write it.

  // CUDA Balls levels 1–3: kernel levels
  const kernelMessages = ({ text, N, PHYS, ballList }) => ({
    kernelEmpty: () => [
      "The kernel is empty",
      "Drag blocks from the palette into the kernel, then press Run on GPU.",
    ],
    twoIndexBlocks: () => [
      "Two blocks set i",
      "Keep one of the int i = … blocks.",
    ],
    blockTwice: ({ s }) => [
      `${text(s.type)} is in the kernel twice`,
      "Each step happens once per substep. Remove the extra copy.",
    ],
    usedBeforeSet: ({ f, setter }) => [
      `Every thread used ${f.info.name} before setting it`,
      `Put ${setter} above the first block that uses ${f.info.name}.`,
    ],
    outsideTheData: ({ oob, L, why }) => {
      const hint = {
        noGuard: `There are ${N} balls but ${PHYS.blocks * PHYS.threads} threads. Add if (i >= n) return; right after computing i.`,
        guardTooLate: "The bounds check comes too late: move it above the block that reads pos.",
      }[why];
      return [
        `${oob.length} thread${oob.length > 1 ? "s" : ""} went outside the data`,
        `For example ${L(oob[0].thread)} ${oob[0].message}. ${hint}`,
      ];
    },
    nothingStored: ({ level }) => [
      "Nothing was stored",
      level.nothingStored,
    ],
    coverage: ({ missing, extra, times, usesThreadIdxOnly }) => [
      missing.length
        ? `Ball${missing.length > 1 ? "s" : ""} ${ballList(missing)} never moved`
        : `Ball${extra.length > 1 ? "s" : ""} ${ballList(extra)} moved too fast`,
      (extra.length ? `Balls ${ballList(extra)} were each moved by ${times} threads, so they went ${times} times too fast. ` : "") +
      (usesThreadIdxOnly
        ? "i = threadIdx.x starts again at 0 in every block. Use the int i = … block with blockIdx.x * blockDim.x + threadIdx.x, so thread t of block b moves ball b × 8 + t."
        : "Check the formula: thread t of block b should move ball b × 8 + t."),
    ],
    ballLeftBox: ({ b, why }) => [
      `Ball ${b} left the box`,
      {
        noVelocityUpdate: "Nothing applies the forces to the velocity: add v += f / mass[i] * dt; after the force blocks.",
        noWalls: "Nothing stops a ball at the floor and walls: add f += pushFromWalls(i, p, v); before v is updated.",
        wallsTooLate: "pushFromWalls runs after v += …, so its force is never used. Move it above the velocity update.",
      }[why],
    ],
    ballsNeverMove: ({ ip }) => [
      "The balls never move",
      ip === -1
      ? "The velocity changes but nothing moves the ball: add p += v * dt; after the velocity update."
      : "The store runs before p changes, so it saves the old position. Put the store last.",
    ],
    ballsOverlap: ({ a, b, res, why }) => [
      `Balls ${a} and ${b} passed through each other`,
      `They overlapped by ${res.maxOverlap.toFixed(2)} units. ${{
        noCollide: "Nothing pushes balls apart: add the for (j …) f += push(i, j); loop before v is updated.",
        collideTooLate: "push(i, j) runs after v += …, so its force is never used. Move the loop above the velocity update.",
      }[why]}`,
    ],
    raceInPlace: ({ L, r0, writer, div }) => {
      const writerText = writer >= 0 ? `${L(writer)} had already overwritten it` : "its thread had already overwritten it";
      return [
        "A race: the result depends on which block runs first",
        `${L(r0.thread)} (ball ${r0.ball}) read pos[${r0.j}] after ${writerText} with ball ${r0.j}'s next position, so the two balls were one substep apart. ` +
        `Run again with the blocks in the opposite order (dashed rings), the balls end up as much as ${div.toFixed(2)} units apart. ` +
        "Write to nextPos and nextVel instead; the host swaps the buffers after each launch.",
      ];
    },
  });

  // CUDA Balls level 4: the time loop
  const hostMessages = ({ PHYS }) => ({
    frameEmpty: () => [
      "The frame function is empty",
      "Drag blocks into it. Blocks above the } run 16 times per frame; blocks below it run once.",
    ],
    nothingRunsSimulation: () => [
      "Nothing runs the simulation",
      "Add the stepKernel<<<3, 8>>> launch inside the loop.",
    ],
    garbage: () => [
      "The balls turned into garbage",
      "A swap makes pos point at the nextPos buffer before any kernel has written it, so the GPU reads uninitialized memory. Inside the loop, launch first, then swap.",
    ],
    nothingDrawn: () => [
      "Nothing is drawn",
      "Add draw(hostPos); below the }, so each frame ends with a picture.",
    ],
    drawTooOften: ({ sec, c }) => [
      `Drawing ${sec(c.draw)} times a second`,
      "draw is inside the loop, so it runs after every substep, but the screen shows only one picture per frame. Move it below the }.",
    ],
    keepsRewinding: () => [
      "The simulation keeps rewinding",
      "cudaMemcpy(pos, hostPos, …, cudaMemcpyHostToDevice) overwrites the GPU's newer positions with the CPU's older copy. The balls live on the GPU; the CPU needs a copy only to draw. Remove the upload.",
    ],
    uselessUpload: ({ sec, c }) => [
      `${sec(c.upload)} copies a second to the GPU, for nothing`,
      "The upload sends positions the GPU already has. Every cudaMemcpy crosses the PCIe bus and waits for the GPU to finish. Remove it.",
    ],
    pictureNeverChanges: () => [
      "The picture never changes",
      "The GPU is simulating, but draw shows hostPos, the CPU's copy from before the first frame. After the loop, copy pos back: cudaMemcpy(hostPos, pos, bytes, cudaMemcpyDeviceToHost);",
    ],
    copyTooOften: ({ sec, c }) => [
      `Copying ${sec(c.copyBack)} times a second`,
      "The copy is inside the loop, so it runs after every substep, and each cudaMemcpy makes the CPU wait for the GPU. Only the last copy reaches the screen. Move it below the }.",
    ],
    ballsNeverMove: ({ c }) => [
      "The balls never move",
      c.swap
      ? "Each launch must be followed by exactly one swap, so the next launch starts from its result. Put the launch, then the swap, inside the loop."
      : "Every launch reads pos and writes nextPos, but nothing swaps them, so each launch starts from the same positions. Add swap(pos, nextPos); swap(vel, nextVel); right after the launch.",
    ],
    slowMotion: ({ advance, launchInLoop }) => [
      `Slow motion: ${advance === 1 ? "16" : (PHYS.substeps / advance).toFixed(1)} times slower than real time`,
      launchInLoop
      ? "The swap runs once per frame, so all 16 launches read the same positions and compute the same substep. Move the swap inside the loop, right after the launch."
      : "The launch is below the }, so it runs once per frame: 60 substeps a second instead of 960. Move the launch and the swap inside the loop.",
    ],
    fastForward: ({ advance }) => [
      `Fast forward: ${(advance / PHYS.substeps).toFixed(0)} times faster than real time`,
      "The loop runs more than one launch and swap per substep. Keep one launch and one swap inside the loop.",
    ],
    wastedLaunches: ({ sec, c, advance }) => [
      `${sec(c.launch - advance)} launches a second are wasted`,
      "An extra launch recomputes a substep that is already done, because pos has not been swapped yet. Keep one launch per substep.",
    ],
    extraWork: () => [
      "Extra work every frame",
      "Below the loop you need one copy and one draw per frame. Remove the duplicates.",
    ],
    oneFrameBehind: () => [
      "Always one frame behind",
      "draw runs before the copy, so each frame shows the positions copied in the frame before: 16 substeps old. Copy first, then draw.",
    ],
  });

  // CUDA Balls level 6: pie the professor
  const sweepMessages = ({ sweepBlocks, PIE }) => ({
    kernelEmpty: () => [
      "The kernel is empty",
      "Drag blocks into it. Blocks above the } run once per step of the flight; blocks below it run once, after the flight.",
    ],
    blockTwice: ({ op }) => [
      `${sweepBlocks[op].parts.join("").replace(/\s+/g, " ")} is in the kernel twice`,
      "Each step of a throw happens once. Remove the extra copy.",
    ],
    neverLeaveLauncher: ({ where }) => [
      "The pies never leave the launcher",
      where("fly")
      ? "p += v * dt is below the }, so it runs once, after the flight. Move it inside the loop."
      : "Nothing moves the pies. Add p += v * dt; inside the loop.",
    ],
    straightLines: ({ where }) => [
      "The pies fly in straight lines",
      where("gravity")
      ? "v.y -= g * dt is below the }, so gravity acts once, after the flight is over. Move it inside the loop."
      : "Nothing pulls the pies down, so every throw sails over the professor. Add v.y -= g * dt; inside the loop.",
    ],
    hitCheckOutside: () => [
      "The hit check is outside the loop",
      "Below the }, break has no loop to leave, and the check would only look at where each pie ended up. Move it inside the loop.",
    ],
    noHitCheck: () => [
      "Pies fly straight through the professor",
      "Nothing checks for a hit, so no thread ever sets hit. Add the hitsFace(p) check inside the loop.",
    ],
    noRecord: ({ hitters }) => [
      "Nobody wrote down the answer",
      `${hitters.length} throws hit the face, but the kernel never records one. Below the }, add the block that keeps the gentlest hit.`,
    ],
    recordInsideLoop: () => [
      "gentlest is still 20: nothing was recorded",
      "The record is inside the loop, but a thread that hits leaves the loop at break, so it never reaches the record. Move it below the }.",
    ],
    noHit: () => [
      "No pie hit the face",
      "Check that gravity and the move both run inside the loop.",
    ],
    twoRecords: () => [
      "Two records",
      "Keep only the atomicMin block: the plain compare-and-write is the race you are replacing.",
    ],
    raceLost: ({ res, truth, range, T }) => [
      `gentlest = ${res.gentlest}, but throw ${truth} is gentler: a race`,
      `Throws ${range} all hit. Each of those threads compared its i with gentlest = ${T} at the same moment, so all ${res.racers.length} wrote, and the last write, throw ${res.gentlest}, survived. ` +
      "atomicMin compares and writes in one indivisible step: use if (hit) atomicMin(&gentlest, i);",
    ],
    luckyRace: () => [
      "Right answer, by luck",
      "Only one thread wrote this time. Use atomicMin so the answer stays right however many throws hit.",
    ],
    passSplat: ({ truth, res, before, T }) => {
      const slower = before ? ` Throw ${truth - 1}, at ${before.speed.toFixed(2)} m/s, lands on the ${before.stop ? before.stop.kind : "floor"} instead.` : "";
      return [
        `Splat! Throw ${truth} is the gentlest that lands`,
        `It leaves the launcher at ${res.throws[truth].speed.toFixed(2)} m/s.${slower} ` +
        `Every thread flew a whole 1.5-second throw, ${PIE.steps.toLocaleString("en-US")} steps, inside a single launch, with no copies until the answer. ` +
        `Trying many independent inputs at once is called a parameter sweep, one of the easiest ways to put a GPU to work. One CPU core would fly the throws one after another: ${T} × ${PIE.steps.toLocaleString("en-US")} = ${(T * PIE.steps).toLocaleString("en-US")} steps.`,
      ];
    },
  });

  // Classic examples
  const classicMessages = ({ fmtList, ranges }) => ({
    kernelEmpty: () => [
      "The kernel is empty",
      "Drag blocks from the palette into the kernel, then press Run on GPU.",
    ],
    usedBeforeSet: ({ f, setter }) => [
      `Every thread used ${f.info.name} before setting it`,
      `Put the ${setter} block above the first block that uses ${f.info.name}.`,
    ],
    outsideTheData: ({ oob, who, why }) => {
      const hint = {
        swappedXY: "The swapped block uses x as the row. Rows go with y: out[y][x].",
        noGuard2D: "Add the block if (x >= w || y >= h) return; right after computing x and y.",
        noGuard1D: "Add if (i >= n) return; right after computing i, before touching the arrays.",
        guardTooLate: "The bounds check comes too late: move it above the block that reads and writes the arrays.",
        badIndex: "Check the index formula: some threads compute an index that is too large.",
      }[why];
      return [
        `${oob.length} thread${oob.length > 1 ? "s" : ""} went outside the data`,
        `For example ${who[0]} ${oob[0].message}. ${hint}`,
      ];
    },
    nothingWritten: ({ outName, level }) => [
      `Nothing was written to ${outName}`,
      level.id === "vector-add" ? "Add the block that computes c[i]." : "Add out[y][x] = blur_pixel(in, w, h, x, y);",
    ],
    writtenTwiceOrNever: ({ dup, missing, A, cell, level, why }) => {
      let what = "";
      if (dup.length) what += `${cell(dup[0])} was written by ${A.writers.get(dup[0]).size} different threads. `;
      if (missing.length)
        what += level.dims === 2
          ? `${missing.length} of the ${A.len} pixels were never written. `
          : `c[${fmtList(ranges(missing))}] ${missing.length === 1 ? "was" : "were"} never written. `;
      const hint = {
        tidRepeats: "threadIdx.x alone repeats in every block: blocks also need blockIdx.x.",
        whichVariable: "Which variable tells blocks apart, and how many elements does each block cover?",
        yUsesX: "y should be built from the .y variables only.",
        xyPattern: "x uses the .x variables, y the .y variables, in the same pattern as level 1.",
      }[why] || "";
      return ["Some outputs were written twice and others never", `${what}Every thread needs a different index. ${hint}`];
    },
    wrongValues: ({ wrong, cell, i, A, level }) => [
      `${wrong.length} output values are wrong`,
      `${cell(i)} is ${A.committed[i]} but should be ${level.id === "vector-add" ? `a[${i}] + b[${i}] = ${level.expected[i]}` : `the 3 × 3 average, ${level.expected[i]}`}. ${level.id === "vector-add" ? "Use the + block." : "Use the blur_pixel block."}`,
    ],
    passElementwise: ({ level, res, returned }) => [
      level.id === "vector-add" ? "All 28 elements are correct" : "Every pixel is blurred correctly",
      `${res.threads.length} threads ran the kernel at the same time, finishing in ${res.steps} steps. The CPU loop needs ${level.cpuSteps} steps, one per ${level.id === "vector-add" ? "element" : "pixel"}. ${returned} thread${returned === 1 ? "" : "s"} had nothing to do and returned early.`,
    ],
    passTotalAtomic: ({ total, level, res, diag }) => [
      `total = ${total}: correct`,
      level.passText ? level.passText(res, diag) :
      `But the ${diag.atomics} atomicAdd calls had to take turns, so the kernel took ${res.steps} steps, more than the CPU loop's ${level.cpuSteps}. Level 4 shows how to be correct and fast.`,
    ],
    passTotalTree: ({ total, res, level }) => [
      `total = ${total}: correct and fast`,
      `${res.steps} steps, against ${level.cpuSteps} for the CPU loop. Each step halves the active threads, so 1,024 numbers would take about 23 steps with the same pattern, instead of 1,024.`,
    ],
    atomicStillHasRace: ({ total, level }) => [
      `total = ${total}, expected ${level.expectedTotal}`,
      "The kernel still contains total += x[i], whose racing writes land at the end and overwrite the atomic result. Remove that block and keep only atomicAdd.",
    ],
    raceCondition: ({ total, level, c, survivor }) => [
      `total = ${total}, expected ${level.expectedTotal}: a race condition`,
      `${c.writers.length} threads read total at the same moment, each added its own number, and each wrote back. Only the write by ${survivor} survived; ${c.writers.length - 1} updates were lost. ${level.atomicLesson ? "Use atomicAdd, which makes read, add and write one indivisible step." : "Let one thread write the finished sum instead: if (t == 0) total = partial[0];"}`,
    ],
    nothingInSharedMemory: ({ total, level }) => [
      `total = ${total}, expected ${level.expectedTotal}`,
      "The threads read partial[] before anything was stored in it, so they added garbage. Start by copying the input into shared memory: partial[t] = x[t];",
    ],
    readTooEarly: ({ total, level, s0, stepName, reader, writer }) => {
      const what = writer === null
        ? `which no thread had written yet, so it got garbage`
        : `before ${writer}'s new value (${s0.newer}) was visible, so it got the old value`;
      return [
        `total = ${total}, expected ${level.expectedTotal}: a thread read too early`,
        `At step ${s0.k + 1} (${stepName}) ${reader} read ${s0.arr}[${s0.idx}] ${what}. Put a __syncthreads() block right before that step, so every thread waits until the writes are visible.`,
      ];
    },
    totalStillZero: ({ level }) => [
      "total is still 0",
      level.atomicLesson ? "No thread added anything to total." : "Nothing copied the finished sum into total. Add if (t == 0) total = partial[0]; at the end.",
    ],
    totalWrong: ({ total, level }) => [
      `total = ${total}, expected ${level.expectedTotal}`,
      "Even with perfect timing, some values are added twice or not at all. Use the strides 8, 4, 2 and 1 once each, largest first.",
    ],
  });

  // The text of one level. Levels that need numbers from the code get them in `consts`.
  function get(id, consts) {
    const entry = levels[id];
    return typeof entry === "function" ? entry(consts) : entry;
  }

  return { get, kernelMessages, hostMessages, sweepMessages, classicMessages };
})();

if (typeof module !== "undefined") module.exports = LEVEL_TEXT;
