# cesipy.github.io

## Mountain artwork

The 34 transparent pencil-sketch mountain illustrations are stored in
[`assets/mountains/`](assets/mountains/). They share the same quiet graphite style,
with different ridge and valley shapes. The filenames describe each variation.

- `home.png` appears below the introduction on the homepage.
- `wide-saddle.png` appears above the post years on the archive page.
- `gentle-arc.png` separates the introduction and links on the About page.
- `horizon-whisper.png` sits below the return link on the 404 page.
- `layered-distance.png` appears after the bachelor thesis post.
- `long-ridge.png` appears after the torch autograd post.
- `distant-chain.png` appears after the Cetris post and is the default in `template.html`.
- `broken-crest.png` appears at the end of the legacy research notes page.

Eight distinct sketches decorate the static pages, leaving 26 available for
future pages or swaps. The interactive studio generates its own artwork and
does not use these images as inputs.

To use another sketch, change the image `src` on its page or in `template.html`.
The main site pages use the sizing and light/dark rules in `styles.css`; the
legacy research page has a small scoped style of its own. These assets were
generated with the built-in image tool using prompts for minimal graphite
mountain line art on transparent backgrounds. The shared prompt direction was:
"sparse graphite pencil contours of distant Alpine ridges, wide shallow panorama,
genuine transparency, no sky, trees, text, or solid fills"; each filename reflects
the variation in the mountain silhouette.

## Mountain studio

[`mountains.html`](mountains.html) is a procedural pencil landscape generator,
linked from the main navigation. It creates new mountain geometry and pencil
strokes entirely in the browser. No mountain PNGs, image collages, external AI
services, backend, API key, or build step are involved.

Every click on **new landscape** obtains a fresh random seed using
`crypto.getRandomValues`. Seeded randomness then generates the terrain and
strokes, so a saved seed or shared link can reproduce the drawing. Nearer terrain
masks the ranges behind it, including in transparent exports.

- The **character** slider is the main control, from quiet to expressive. It maps
  `c` from 0 to 100 to sharpness `15 + .7c`, bumpiness `10 + .85c`, entropy
  `20 + .7c`, creativity `.85c`, depth `25 + .6c`, pencil strength `50 + .4c`,
  and shading `40 + .6c`, rounding each value. A new empty landscape starts at
  character 50 (including shading 70). Terrain style, layer count, and seed stay
  independent of this slider.
- Fine adjustments are under the closed-by-default **advanced settings** section.
  It contains foreground, middle, and distant terrain styles; one to three layers;
  pencil strength; depth; sharpness; bumpiness; entropy; creativity; shading; and
  the seed. Shading runs from outline-only (0) through the original version 3
  appearance (50) to bold, high-contrast shading (100). Above 50, broader shadow
  tones, denser hatching, and darker strokes build rapidly; the default of 70
  is deliberately more shaded. Manual edits mark the character as custom;
  moving the main slider applies its mapping again.
- Enter a seed to reproduce a composition, or return to the previous one. Copy a
  link that contains the seed, character, and every setting, including each
  layer's terrain style. Older shared links without layer styles use `same` for
  middle and distant terrains. Missing creativity defaults to 0, missing
  sharpness, bumpiness, and entropy to 50, and missing shading to 50. If an older
  link has no character, the slider position is inferred from its settings while
  the saved numeric values remain intact. Links using the earlier `wildness`
  parameter are still read as creativity.
- Links preserve their renderer version: version 2 links use the archived drawing
  engine, while new landscapes use version 3. Older landscape hashes without a
  `v` field continue to use version 2. Character and shading are disabled on
  version 2 until **use updated drawing** is selected; upgrading preserves its
  settings and uses shading 50. **Previous** returns to the saved version.
  Unsupported explicit versions fall back to version 3 with a notice. Creating a
  new landscape keeps the current settings while changing its seed.
- Download a 2400 × 800 PNG with transparency or a white background.
  The preview always shows white paper, including in dark mode.

The current procedural drawing engine lives in [`mountain-terrain.js`](mountain-terrain.js).
[`mountain-terrain-v2.js`](mountain-terrain-v2.js) preserves the previous renderer
for existing links. [`mountains.js`](mountains.js) handles controls, random seeds,
history, sharing, and downloads; [`mountains.css`](mountains.css) contains the
scoped styles.
Drawings are created in memory and saved to the visitor's device when downloaded;
they are not added to `assets/mountains/` automatically. Shared landscapes
reproduce the same result within their renderer version. Keep the archived renderer
unchanged when adding a future version so saved links remain reproducible.

For a local preview, run `python3 -m http.server 8765` from the repository root
and open `http://localhost:8765/mountains.html`. Use a local server instead of
opening the HTML directly so canvas downloads work consistently.

## Vibed projects

[`vibed.html`](vibed.html) is the small experiments hub linked from the site
navigation. It currently features the Mountain Studio and reuses the existing
transparent pencil artwork from `assets/mountains/`. To add a project, append
another `<article class="vibed-project">` inside the `vibed-project-list`
section, with a concise description and a working link to the project. Reuse an
existing asset where it helps; keep the hub focused on projects that exist.
