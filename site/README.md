# Stylesheet sources

- `tufte.css`: [DiffRast's stylesheet](https://github.com/srush/DiffRast/blob/main/docs/tufte.css),
  retrieved September 10, 2026. The only changes are a provenance comment
  and absolute font URLs pointing to the original site's ET Book fonts.
- The stylesheet derives from [Tufte CSS](https://github.com/edwardtufte/tufte-css)
  and uses [ET Book](https://github.com/edwardtufte/et-book).
- `verso-tufte.css`: project-specific overrides for Verso's literate renderer.
  These preserve the article typography while leaving Lean's interactive code intact.

Run `make docs` from the project root after editing either stylesheet.

For automatic rebuilds and browser refresh, run `python3 watch.py --open` from
the project root. The preview uses port 8001 so it can run alongside the old
server on port 8000. Open the new preview URL once; subsequent successful builds
refresh that page automatically, preserving its URL fragment. Lean errors are
printed in the terminal and do not trigger a refresh. Stop with Ctrl-C.
Use `--port 8002` or `--lake /path/to/lake` to override the defaults.
