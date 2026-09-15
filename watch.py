#!/usr/bin/env python3
"""Build and serve Verso with automatic rebuilds and browser refresh."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import subprocess
import threading
import time
import webbrowser

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / ".lake/build/literate-html"
WATCHED = ["Transformer.lean", "literate.toml", "lakefile.toml",
           "lean-toolchain", "Makefile"]


def fingerprint():
    paths = [ROOT / name for name in WATCHED]
    paths += list((ROOT / "site").glob("*.css"))
    paths += list((ROOT / "site").glob("*.js"))
    paths += list((ROOT / "site/diagrams").glob("*.svg"))
    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size)
                 for p in sorted(paths) if p.is_file())


class Preview(SimpleHTTPRequestHandler):
    revision = 0

    def log_message(self, format, *args):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path == "/__verso_revision":
            data = str(Preview.revision).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        path = Path(self.translate_path(self.path))
        if path.is_dir():
            path /= "index.html"
        if path.suffix == ".html" and path.is_file():
            script = """<script>
const versoRevision = %d;
setInterval(async () => {
  try {
    const r = await fetch('/__verso_revision', {cache: 'no-store'});
    if (r.ok && Number(await r.text()) !== versoRevision) location.reload();
  } catch (_) {}
}, 1000);
</script>""" % Preview.revision
            data = path.read_bytes() + script.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        super().do_GET()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--lake", default=shutil.which("lake") or
                        str(Path.home() / ".elan/bin/lake"))
    parser.add_argument("--open", action="store_true", help="Open the preview in your browser")
    args = parser.parse_args()
    if not shutil.which(args.lake):
        parser.error("Lake not found; pass --lake /path/to/lake")
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                    partial(Preview, directory=str(OUTPUT)))
    except OSError as error:
        parser.error(f"Cannot start preview: {error}. Try --port 8002")

    def build():
        print("Building Verso…", flush=True)
        result = subprocess.run(["make", "docs", f"LAKE={args.lake}"], cwd=ROOT)
        if result.returncode == 0:
            Preview.revision += 1
            print("Build succeeded; refreshing browsers.", flush=True)
        else:
            print("Build failed; fix the errors and save again. No browser refresh.", flush=True)

    try:
        seen = fingerprint()
        build()
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{args.port}/Transformer/"
        print(f"Watching Transformer.lean and site assets. Preview: {url}\nCtrl-C to stop.", flush=True)
        if args.open:
            webbrowser.open(url)
        while True:
            time.sleep(0.3)
            current = fingerprint()
            if current == seen:
                continue
            # Wait until saves settle, then build serially. Changes made during
            # a build remain detectable on the next iteration.
            time.sleep(0.3)
            if fingerprint() != current:
                continue
            seen = current
            build()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
