# Thread Lab

A block-based game that teaches CUDA by building the CUDA Balls physics engine.

## Running it

Open `index.html` in a browser (double-click works) or serve the folder, for example with GitHub Pages. There is no build step.

`index.html` loads everything from `src/`, so keep the folder together: sending someone `index.html` on its own will not work.

```
node tests/test_engine.js && node tests/test_physics.js   # run the tests
```

## What lives where

| File | What it holds |
| --- | --- |
| `index.html` | The page markup, and the list of scripts and styles it loads. |
| `src/style.css` | All styling, including the light and dark themes (CSS variables at the top). |
| `src/theme-init.js` | Tiny script that applies the saved theme before the page paints. |
| `src/engine.js` | The GPU model and the classic levels. No DOM, so it also runs in Node. |
| `src/physics.js` | The CUDA Balls levels 1–6 and the physics simulation. No DOM. |
| `src/ui.js` | The interface: palette, drag and drop, canvas drawing, tour, celebration. |
| `src/data/*.js` | The lists of ball faces (with photo credits), celebration pictures and the professor photo. |
| `src/assets/` | The image, video and sound files those lists point to. |
| `tests/` | Node tests for the engine and the physics. |

The scripts load in order (`engine.js`, `physics.js`, the data files, then `ui.js`), and each one uses the ones before it.

## Common changes

- **Add a celebration picture:** put the file in `src/assets/celebrate/` and add an entry to `src/data/celebrate.js` (`src`, `w`, `h`, `ms`; optional `video`, `card` for a backdrop color, `pixel` for crisp pixel art). Every picture is scaled to cover about the same area.
- **Change the celebration sound:** replace `src/assets/celebrate/yippee.mp3`.
- **Add a ball face:** put a square-ish webp in `src/assets/faces/` and add an entry with its credit to `src/data/faces.js`. The credits footer is built from that file.
- **Change colors:** edit the variables in `src/style.css`. The canvas drawing in `src/ui.js` has its own colors, so check there too.

Keep videos and GIFs short (a couple of seconds): they download every time the page loads.
