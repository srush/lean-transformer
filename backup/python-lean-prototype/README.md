# Minimal verified tensor language

This repository contains a first Python-to-Lean execution slice for a small,
mutation-free tensor language inspired by Tensor Puzzles.

The Python frontend accepts exactly one function containing immutable local
bindings and a final `return`. It supports:

- `arange(n)`
- `where(condition, if_true, if_false)`
- `unsqueeze(x)` and `x.unsqueeze()` (append a size-1 dimension)
- `+`, `-`, `*`, exact rational `/`
- rank-2 matrix multiplication with `@`
- `==`, `!=`, `<`, `<=`, `>`, and `>=`
- NumPy-style pointwise broadcasting

Dimension parameters use `int`; tensor parameters use symbolic annotations:

```python
def arithmetic(n: int, a: Tensor["n", "n"], b: Tensor["n", "n"]):
    mixed = (a + b) * (a - b) / 2
    return mixed @ b
```

The annotation is parsed as syntax and needs no Python `Tensor` package.
`tensor_parser.py` uses only the Python standard library.

## Generate and check an example

```bash
python3 tensor_parser.py examples/triu.py \
  -o TensorPuzzles/Generated/Triu.lean
lake env lean TensorPuzzles/Generated/Triu.lean
lake build
```

The generated file contains:

1. The tensor program as Lean `Expr` data.
2. The symbolic output shape claimed by the frontend.
3. A proof, checked by Lean computation, that Lean's independent inference
   agrees with the claimed dtype and shape.

Run the expression for `n = 4` with:

```bash
lake env lean TensorPuzzles/Examples/TriuRun.lean
```

Tensors are pure functions from indices to values; materialization happens
only for display.

## AST Infoview widget

Import `TensorPuzzles.AstWidget`, then use either form:

```lean
#tensor_ast triuProgram

example : True := by
  tensor_ast triuProgram
  trivial
```

With the Lean language server and Infoview open, put the cursor on
`#tensor_ast` or `tensor_ast`. A collapsible tree displays the expression's
constructors, operators, inputs, dimensions, and the three roles under
`where`. The tactic form leaves the proof goal unchanged, so it can be placed
inside a real proof while inspecting the program.

See `TensorPuzzles/Examples/AstWidgetDemo.lean` for a ready-to-open example.

## Verified `triu` specification

`TensorPuzzles/Specs/Triu.lean` defines `TriuSpec` independently of the tensor
program. It requires an output shape of `[n, n]` and, for every valid pair of
indices, the value `1` when `i <= j` and `0` otherwise.

In Lean, the definition reads:

```lean
def TriuSpec (n : Nat) (output : Tensor Rat) : Prop :=
  output.shape = [n, n] ∧
    ∀ i j, i < n → j < n →
      output.get [i, j] = if i ≤ j then 1 else 0
```

Read it from the outside inward:

- `TriuSpec ... : Prop` means the specification is a proposition that can be
  proved or disproved.
- `output.shape = [n, n] ∧ ...` joins two obligations: the shape is correct
  **and** all entries are correct.
- `∀ i j` chooses arbitrary row and column indices.
- `i < n → j < n →` restricts the claim to indices inside the tensor.
- `if i ≤ j then 1 else 0` includes the diagonal in the upper triangle.

The bounds matter because this prototype represents a tensor as a total Lean
function `List Nat → Rat`. Such a function technically returns something even
for malformed or out-of-bounds index lists. The mathematical contract ignores
those unobservable values. For `n = 0`, the shape is `[0, 0]` and the entry
clause holds vacuously because no valid indices exist.

One theorem connects execution of the generated AST to the specification,
assuming `n > 1`:

```lean
theorem triu_correct (n : Nat) (hn : 1 < n) :
    ∃ output,
      triuProgram.eval [("n", n)] [] = .ok (.num output) ∧
      TriuSpec n output
```

The proof unfolds the evaluator, uses its computed tensor as the output, and
checks the shape and entries directly. Broadcasting reads coordinate zero for
any source dimension of size one; `hn` rules out that case for `n`, so no case
split is needed. The theorem now covers dimensions greater than one; the spec
itself still makes sense for every natural-number dimension.

Put the cursor on the `tensor_ast` line inside the proof to view the verified
program beside the proof state.

## Current trust boundary

Lean owns type/shape inference, broadcasting, arithmetic, `where`, `arange`,
`unsqueeze`, matrix multiplication, and execution. The Python parser is still
trusted to translate the displayed source into the intended AST. Lean proves
the generated AST is well typed; it does not yet prove the Python source and
AST correspond token-for-token.

Numbers are exact Lean `Rat` values. Division by zero is rejected by the
executor. `@` currently supports only matrix-by-matrix multiplication.

## Python tests

```bash
python3 -m unittest discover -s tests -v
```
