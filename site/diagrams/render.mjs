import { readFile, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const root = new URL("./", import.meta.url);
const bundle = await readFile(new URL("node_modules/@penrose/core/dist/bundle/index.js", root));
const server = createServer((req, res) => {
  res.setHeader("Content-Type", req.url === "/penrose.js" ? "text/javascript" : "text/html");
  res.end(req.url === "/penrose.js" ? bundle : '<!doctype html><body style="margin:0;background:white"></body>');
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
let browser;
try {
  browser = await chromium.launch({ executablePath: process.env.CHROME_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless: true });
  const page = await browser.newPage({ viewport: { width: 920, height: 500 }, deviceScaleFactor: 2 });
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const trio = {
    domain: await readFile(new URL("tensor-parallel.domain", root), "utf8"),
    substance: await readFile(new URL("tensor-parallel.substance", root), "utf8"),
    style: await readFile(new URL("tensor-parallel.style", root), "utf8"),
    variation: "tensor-parallel-v1"
  };
  const svg = await page.evaluate(async trio => {
    const penrose = await import("/penrose.js");
    await penrose.__tla;
    const { compile, optimize, toSVG, showError } = penrose;
    const compiled = await compile(trio);
    if (compiled.isErr()) throw Error(showError(compiled.error));
    const optimized = optimize(compiled.value);
    if (optimized.isErr()) throw Error(showError(optimized.error));
    const svg = await toSVG(optimized.value, async () => undefined);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Tensor parallel matrix multiplication: split A by columns and B by rows; compute A1 B1 and A2 B2 independently, then add the two n by p results.");
    document.body.append(svg);
    return svg.outerHTML;
  }, trio);
  await writeFile(new URL("tensor-parallel.svg", root), svg + "\n");
  await page.screenshot({ path: fileURLToPath(new URL("tensor-parallel.png", root)) });
  console.log("Rendered tensor-parallel.svg with Penrose 3.3.0");
} finally {
  if (browser) await browser.close();
  server.close();
}
