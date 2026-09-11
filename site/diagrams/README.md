# Penrose diagrams

`tensor-parallel.svg` is generated from the matching `.domain`, `.substance`,
and `.style` files using Penrose 3.3.0. The PNG is a preview of that SVG.

The style uses serif mathematical labels, muted blue and amber for matching
partitions, thin neutral arrows, and explicit matrix dimensions. Layout is
fixed in the Style program for reproducibility.

To regenerate, from this directory:

```sh
pnpm install --frozen-lockfile
pnpm render
```

The renderer uses installed Google Chrome on macOS. Set `CHROME_PATH` to use
another Chromium executable. Generated images are checked in, so the Lean/Verso
build does not need Node or Chrome.

Here A₁ and A₂ denote the n × k column blocks of A; B₁ and B₂ denote
the k × p row blocks of B. Both partial products have shape n × p and are
added, not concatenated.
