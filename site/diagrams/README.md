# Penrose diagrams

The tensor-parallel, row-equivariance, and data-parallel SVGs are generated
from their matching `.domain`, `.substance`, and `.style` files using Penrose
3.3.0. All three use `common.style` for typography, colors, boxes, and arrows.
The PNGs are previews of the SVGs.

The style uses serif mathematical labels, muted blue and amber for matching
partitions, thin neutral arrows, and explicit matrix dimensions. Layout is
fixed in the Style program for reproducibility.

To regenerate, from this directory:

```sh
pnpm install --frozen-lockfile
pnpm render
```

To render just one diagram, run `pnpm render row-equivariance` (or
`tensor-parallel` / `data-parallel`).

The renderer uses installed Google Chrome on macOS. Set `CHROME_PATH` to use
another Chromium executable. Generated images are checked in, so the Lean/Verso
build does not need Node or Chrome.

Here A₁ and A₂ denote the n × k column blocks of A; B₁ and B₂ denote
the k × p row blocks of B. Both partial products have shape n × p and are
added, not concatenated.

Row equivariance illustrates a two-row swap; the Lean theorem is more general
and permits arbitrary row selection. Data parallelism uses the same network
parameters on both devices and retains original batch indices in `point_loss`.
