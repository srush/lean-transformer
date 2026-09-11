LAKE ?= lake

.PHONY: check docs serve
check:
	$(LAKE) build

docs:
	$(LAKE) build :literateHtml
	# Re-render even for CSS-only edits (Verso's facet does not track extra_css).
	$(LAKE) exe verso-literate-html .lake/build/literate-html .lake/build/literate-module-map literate.toml

serve: docs
	python3 -m http.server 8000 --bind 127.0.0.1 --directory .lake/build/literate-html
