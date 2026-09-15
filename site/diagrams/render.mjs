import { readFile, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const root = new URL("./", import.meta.url);
const descriptions = {
  "chunkwise-ssm": "Equal-size chunks pass a matrix carry from left to right. Each chunk combines its decayed incoming carry with weighted local updates. A partial chunk gives every position's state and query readout, identical to the recurrent SSM.",
  "flash-attention": "For one query, process equal-size key/value tiles while carrying a scalar weight sum Z and a vector weighted sum U. Normalize U by Z only after all tiles, exactly matching the softmax-like attention used here.",
  "ssm": "The same decayed SSM can be evaluated recurrently, as masked linear attention, or in equal-size chunks. The mask is lower triangular with powers of alpha; each chunk combines a decayed incoming state with its weighted local updates.",
  "selection-equivariance": "The selection p = [2, 0, 2] reorders, repeats, and selects three positions from four. Applying the same selection-equivariant function before or after selection gives identical outputs.",
  "region-invariance": "Two sequences agree inside the radius-one window around position s, but differ outside it. A region-invariant function gives the same output at s; outputs at other positions need not agree.",
  "tensor-parallel": "Split A by columns and B by rows; compute the two matmuls independently, then add their results.",
  "row-equivariance": "Swapping the rows of A before multiplying by B gives the same result as swapping the rows of AB. Blue and amber rows retain their identities.",
  "data-parallel": "Split a batch into X1 and X2. Apply the same neural network to each half, sum point losses with their original batch indices, then add the two scalar losses. This equals the full-batch loss.",
  "equivariance-invariance": "Equivariance: transforming the input by T transforms the output by S. Invariance: transforming the input leaves the output unchanged. Both diagrams commute.",
  "batch-invariance": "Select example b before or after applying the same selection-equivariant network. Processing the example alone at batch index zero gives the same output as its original row in the full batch."
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
    await document.fonts.ready;
    const bounds = svg.getBBox();
    const width = bounds.width + 40;
    const height = bounds.height + 40;
    svg.setAttribute("viewBox", `${bounds.x - 20} ${bounds.y - 20} ${width} ${height}`);
    svg.setAttribute("width", "920");
    svg.setAttribute("height", String(920 * height / width));
    return svg.outerHTML;
  }, {trio, description: descriptions[name]});
  await writeFile(new URL(`${name}.svg`, root), svg + "\n");
  await page.locator('svg').first().screenshot({ path: fileURLToPath(new URL(`${name}.png`, root)) });
  await page.close();
  console.log(`Rendered ${name}.svg with Penrose 3.3.0`);
  }
} finally {
  if (browser) await browser.close();
  server.close();
}
