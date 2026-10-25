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

The small background ridges in [`assets/mountains/background/`](assets/mountains/background/)
are different: these 12 transparent PNGs were rendered with the Mountain Studio
version 4 algorithm in [`mountain-terrain.js`](mountain-terrain.js), not generated
with the image tool. Their exact seeds and settings are stored in
[`manifest.json`](assets/mountains/background/manifest.json). Each began as a
2400 × 800 transparent Studio canvas and preserves the complete mountain group,
reduced to a 720 × 240 transparent sketch for the page.
A second outline-only pass with thicker strokes and an alpha correction keep
the contours readable at small sizes; the manifest records these treatments.
[`mountain-background.js`](mountain-background.js) places them as small tilted
sketches with the same tilt across the page background. Seeded spacing avoids
repeated rows. Nearby sketches gently move away from the mouse and spring back
when it leaves; each drawing stays intact. Sketches behind text are softened
for legibility.
[`mountain-background.css`](mountain-background.css)
keeps them faint, below the existing cursor glow, and adapts them to dark mode.
The motion is disabled for reduced-motion and coarse-pointer settings.
[`scripts/generate-background-mountains.cjs`](scripts/generate-background-mountains.cjs)
recreates the assets from the Studio renderer using an isolated Chrome debugging
tab; the script describes how to run it.

## Mountain studio

[`mountains.html`](mountains.html) is a procedural pencil landscape generator,
linked from the main navigation. It creates new mountain geometry and pencil
strokes entirely in the browser. No mountain PNGs, image collages, external AI
services, backend, API key, or build step are involved in the pencil generator.

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
  the seed. In version 4, shading runs from outline-only (0) through light face
  tone (50) to denser graphite and stronger contrast (100). The shadow side is
  consistent across layers, and the default of 70 is deliberately shaded.
  Version 3 links retain their original shading model. Manual edits mark the
  character as custom; moving the main slider applies its mapping again.
- Enter a seed to reproduce a composition, or return to the previous one. Copy a
  link that contains the seed, character, and every setting, including each
  layer's terrain style. Older shared links without layer styles use `same` for
  middle and distant terrains. Missing creativity defaults to 0, missing
  sharpness, bumpiness, and entropy to 50, and missing shading to 50. If an older
  link has no character, the slider position is inferred from its settings while
  the saved numeric values remain intact. Links using the earlier `wildness`
  parameter are still read as creativity.
- Links preserve their renderer version: version 2 links use the archived drawing
  engine, version 3 links keep their original appearance, and new landscapes use
  version 4. Older landscape hashes without a `v` field continue to use version
  2. Character and shading are disabled on version 2 until **use updated drawing**
  is selected; upgrading preserves its settings and uses shading 50. Version 3
  controls remain available, and upgrading preserves its settings. **Previous**
  returns to the saved version. Unsupported explicit versions fall back to version
  4 with a notice. Creating a new landscape keeps the current settings while
  changing its seed.
- Download a 2400 × 800 PNG with transparency or a white background.
  The preview always shows white paper, including in dark mode.

The current procedural drawing engine lives in [`mountain-terrain.js`](mountain-terrain.js).
[`mountain-terrain-v2.js`](mountain-terrain-v2.js) preserves the previous renderer
for existing links. [`mountains.js`](mountains.js) handles controls, random seeds,
history, sharing, and downloads; [`mountains.css`](mountains.css) contains the
scoped styles.
Drawings are created in memory and saved to the visitor's device when downloaded;
they are not added to `assets/mountains/` automatically. Shared landscapes
reproduce the same result within their renderer version. Keep existing renderer
versions unchanged when adding a future version so saved links remain reproducible.

For a local preview, run `python3 -m http.server 8765` from the repository root
and open `http://localhost:8765/mountains.html`. Use a local server instead of
opening the HTML directly so canvas downloads work consistently.

## Local sketch-to-photo

The optional photo panel in [`mountains.html`](mountains.html) turns the current
mountain composition into a realistic landscape using a model running on your Mac.
The visible pencil drawing, its downloads, and shared links stay as they are.
The default engine is quantized FLUX.2 Klein 4B running through Apple MLX. It
uses the visible pencil drawing as a reference to make a photographic landscape;
ridge positions can change. The alternative Stable Diffusion 1.5 line-art
ControlNet engine follows the contours more closely. Both run on your Mac, with
no API key or per-image fee. For ControlNet, the page draws a temporary clean
contour sketch from the same seed. Only start the server for the engine you select.

### FLUX.2 Klein 4B · MLX 4-bit

Use Python 3.12. This option downloads the pre-quantized
[`Runpod/FLUX.2-klein-4B-mflux-4bit`](https://huggingface.co/Runpod/FLUX.2-klein-4B-mflux-4bit)
checkpoint (about 4.6 GB) on first generation, without fetching the full
precision checkpoint first. Allow additional disk space for Python packages and
temporary download files.

```sh
python3.12 -m venv .venv-klein
source .venv-klein/bin/activate
python -m pip install -r requirements-klein.txt
python local_klein_server.py
```

Keep that terminal running. The Klein server listens at `127.0.0.1:8767` and
currently generates 768 × 384 pixel photos.

### SD 1.5 · line-art ControlNet

This alternative downloads roughly 5–6 GB of model weights on first generation.
Use Python 3.12 and install [`requirements-photo.txt`](requirements-photo.txt):

```sh
python3.12 -m venv .venv-photo
source .venv-photo/bin/activate
python -m pip install -r requirements-photo.txt
python local_photo_server.py
```

The ControlNet server listens at `127.0.0.1:8766`. Its output is 768 × 384 pixels.

In another terminal, run `python3 -m http.server 8765` from the repository root
and open `http://localhost:8765/mountains.html`. Expand **turn this sketch into a
real landscape**, enter a scene description, and select **make real landscape**. Keep
the selected local Python server running while generating. The photo can be downloaded
separately from the pencil sketch.

Both servers listen only on your computer; sketches and generated photos are
processed locally. Model weights are downloaded from Hugging Face on first use.
If you open the public GitHub Pages site, your browser
may ask permission for that site to access the local server. The localhost
preview is the simplest setup.

## Vibed projects

[`vibed.html`](vibed.html) is the small experiments hub linked from the site
navigation. It currently features the Mountain Studio and reuses the existing
transparent pencil artwork from `assets/mountains/`. To add a project, append
another `<article class="vibed-project">` inside the `vibed-project-list`
section, with a concise description and a working link to the project. Reuse an
existing asset where it helps; keep the hub focused on projects that exist.

## Cursor halo

`cursor-glow.css` and `cursor-glow.js` add a faint `#06c078` halo behind page
content, without replacing the native cursor or intercepting clicks. Both are
included by the page template. The effect runs only for a fine, hovering pointer
when reduced motion is off; it ignores touch/pen movement, fades on exit or
keyboard navigation, and stops its animation loop when the pointer settles.
