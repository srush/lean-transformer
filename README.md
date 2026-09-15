# Lean Verified Transformers

[Read the article](https://srush.github.io/lean-transformer/), or explore the
definitions and proofs in [Transformer.lean](Transformer.lean).

## Open in VS Code

1. Install [VS Code](https://code.visualstudio.com/) and the official
   [Lean 4 extension](https://marketplace.visualstudio.com/items?itemName=leanprover.lean4).
   Follow its setup guide to install Lean's toolchain manager, Elan, and Git.
   See the [Lean installation guide](https://lean-lang.org/install/) if needed.
2. Clone this repository (GitHub access is required while it is private):

   ```sh
   git clone https://github.com/srush/lean-transformer.git
   cd lean-transformer
   ```

3. Open this entire folder in VS Code with **File → Open Folder** (or `code .`).
   In VS Code's terminal, run:

   ```sh
   lake build
   ```

   The project pins Lean in `lean-toolchain` and Verso in `lake-manifest.json`.
   The first build downloads dependencies and can take a few minutes.
4. Open `Transformer.lean`. Click inside a proof to see its goals and hypotheses
   in Lean's **InfoView**. Edits are checked automatically.

If `lake` is not found, finish the extension's setup guide and restart the terminal.

## Local article preview

With Python 3 installed, run `python3 watch.py --open` from the project folder.
It rebuilds the Verso article and refreshes the browser after successful edits.
