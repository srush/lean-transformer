import { readFile, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const root = new URL("./", import.meta.url);
const descriptions = {
  "tensor-parallel": "Split A by columns and B by rows; compute the two matmuls independently, then add their results.",
  "row-equivariance": "Swapping the rows of A before multiplying by B gives the same result as swapping the rows of AB. Blue and amber rows retain their identities.",
  "data-parallel": "Split a batch into X1 and X2. Apply the same neural network to each half, sum point losses with their original batch indices, then add the two scalar losses. This equals the full-batch loss."
};
const names = process.argv.slice(2);
if (!names.length) names.push(...Object.keys(descriptions));
for (const name of names) if (!(name in descriptions)) throw Error(`Unknown diagram: ${name}`);
const bundle = await readFile(new URL("node_modules/@penrose/core/dist/bundle/index.js", root));
const server = createServer((req, res) => {
  res.setHeader("Content-Type", req.url === "/penrose.js" ? "text/javascript" : "text/html");
  res.end(req.url === "/penrose.js" ? bundle : '<!doctype html><body style="margin:0;background:white"></body>');
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
let browser;
try {
  browser = await chromium.launch({ executablePath: process.env.CHROME_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless: true });
  for (const name of names) {
  const page = await browser.newPage({ viewport: { width: 920, height: 500 }, deviceScaleFactor: 2 });
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const trio = {
    domain: await readFile(new URL(`${name}.domain`, root), "utf8"),
    substance: await readFile(new URL(`${name}.substance`, root), "utf8"),
    style: await readFile(new URL("common.style", root), "utf8") +
      await readFile(new URL(`${name}.style`, root), "utf8"),
    variation: `${name}-v1`
  };
  const svg = await page.evaluate(async ({trio, description}) => {
    const penrose = await import("/penrose.js");
    await penrose.__tla;
    const { compile, optimize, toSVG, showError } = penrose;
    const compiled = await compile(trio);
    if (compiled.isErr()) throw Error(showError(compiled.error));
    const optimized = optimize(compiled.value);
    if (optimized.isErr()) throw Error(showError(optimized.error));
    const svg = await toSVG(optimized.value, async () => undefined);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", description);
    document.body.append(svg);
    return svg.outerHTML;
  }, {trio, description: descriptions[name]});
  await writeFile(new URL(`${name}.svg`, root), svg + "\n");
  await page.screenshot({ path: fileURLToPath(new URL(`${name}.png`, root)) });
  await page.close();
  console.log(`Rendered ${name}.svg with Penrose 3.3.0`);
  }
} finally {
  if (browser) await browser.close();
  server.close();
}
