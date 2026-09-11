# Minimal rational vectors

Read [Tensors.lean](Tensors.lean) from top to bottom. This single Lean file
contains section headings, definitions, proofs, and printed computations.
The proofs use only Lean's standard library; the HTML
renderer uses a pinned Verso release compatible with Lean 4.33.

## Reading order

1. **Vectors:** fixed-size rational vectors and pointwise arithmetic.
2. **Matrices:** dot products, transpose, and pointwise products.
3. **Tensor parallelism:** split a reduction dimension and add two matmuls.
4. **Neural networks and data parallelism:**
   - **4a:** linear layers, ReLU, and a stack of arbitrary depth.
   - **4b:** run the whole network independently on two batch halves and prove
     that their summed losses equal the full-batch loss.
5. **Sequences and attention:** per-position layers, sequence pooling, and Q/K/V.
6. **Stacks:** one traversal for bias-free linear-plus-attention blocks.
7. **Position and batch invariance:** reorder positions; run batch shards separately.
8. **Sliding windows:** prove the receptive field grows by one radius per block.
9. **Tiled attention:** accumulate weighted values and normalization across tiles and prove equivalence.
10. **SSMs:** one recurrence, forward/bidirectional scans, and order sensitivity.
11. **Chunkwise SSM:** carry a state across chunks and prove equivalence to the recurrence.

The `#eval` commands print concrete results; the theorems cover arbitrary inputs.
Vectors, matrices, and sequences have `Repr` instances that print nested lists.
Definitions live in `TensorPuzzles` to avoid Lean's built-in `Vector` name;
use `open TensorPuzzles` from another file.

## Conventions

- Shapes are enforced by types; arithmetic is exact rational arithmetic.
- Matrix `+` and `*` are pointwise; `a.matmul b` is matrix multiplication.
- `neural_network` takes a list of `NeuralLayer` parameters and applies ReLU
  after every linear layer. An empty list is the identity.
- `data_parallel_loss_correct` covers any such network and scorer on an even
  batch, retaining original batch indices when scoring either shard.
- Equal-halves splitting uses a dimension written as `k + k` and returns two
  halves of size `k`, without division or a separate evenness proof.
  For concrete sizes, supply `(k := 1)` to split a dimension of size two.
  Flash attention uses a sequence length of `tiles * tile_size`, so every tile
  is full. `flash_attention tiles tile_size qkv` needs no positivity
  proof: zero tiles or zero tile size give an empty sequence. `tile_fold`
  recurses structurally on the tile count, with no termination argument.
  Chunkwise SSM uses the same `tiles * tile_size` shape and shares the `tile`
  slicing helper with `tile_fold`. Neither needs partial final chunks or a
  positivity proof.
- Only `row_split` is defined. To split columns, transpose, split rows, and
  transpose the resulting matrices back.
- The final loss sums hidden vectors over sequence positions, scores once per
  example, then sums over the batch. The scorer receives a batch index and the
  pooled hidden vector, with no sequence-position argument.
- `transformer_loss_batch_split` proves that a batch of size `n + 1` has the
  same loss as independently running its first `n` examples and its last one.
  This covers every nonempty batch, including a singleton, without evenness.
  Scorers retain original batch indices on the smaller batches.
- Position permutations leave the pooled loss unchanged with the same scorer.
  SWA locality describes individual outputs before pooling, not the pooled loss.
- `softmax_like` normalizes `1 + relu(zᵢ)` by their sum. On nonempty
  vectors, the weights are positive and sum to one. This is not
  exponential softmax; negative scores tie and translation invariance is not guaranteed.
  Empty vectors have no weights. The retained theorem proves permutation equivariance.
- Full attention uses that rule; sliding-window attention uses raw dot products.
- `linear_attention qkv` uses raw Q/K dot products without normalization.
  Its optional final mask defaults to all ones. Use
  `fun qkv => linear_attention qkv` as a mixer with the default mask.
- All layers are bias-free, including `NeuralLayer` and `sequence_layer`.
- Both bidirectional SSM scans include the current input, counting it twice.
- Parallel and tiled forms are mathematical models, not performance guarantees.

## Build and read

```sh
lake build
```

Open `Tensors.lean` in a Lean editor to inspect proof states.

To render the same file as an article:

```sh
make docs
make serve
```

Open http://127.0.0.1:8000 after starting the server. HTML is generated in
`.lake/build/literate-html/`; no separate copy of the book is maintained.
`make docs` also picks up CSS-only changes.

`literate.toml` selects the module and stylesheets. `site/tufte.css` is the
stylesheet from [DiffRast](https://srush.github.io/DiffRast/), with font URLs
made absolute; `site/verso-tufte.css` adapts it to Verso's article structure.
The ET Book fonts load from DiffRast, with local serif fallbacks when offline.
Lean highlighting, declaration links, copy buttons, and proof hovers remain
provided by Verso. This configures a local build, not automatic deployment.

## Simplified API and backups

`project_qkv input wq wk wv` constructs one `QKV` input shared by all mixers.
The transformer performs this projection; a `Mixer` is simply a function from
this projected input to an output sequence.
`ssm_layer α qkv` and `bidirectional_ssm_layer α qkv`
scan a hidden-by-hidden matrix state:
`Sₜ = α Sₜ₋₁ + kₜᵀ vₜ`, with output `qₜ Sₜ`.
Use `α = 1` for the unweighted case. Compose layers with
`sequence_loss` or `transformer_loss` directly instead of using a separate
loss wrapper for every layer. The shared `transformer` takes a mixer function:
`transformer attention_layer blocks input` for full attention,
`transformer (swa radius) blocks input` for sliding-window attention, or
`transformer (ssm_layer α) blocks input` for a forward SSM.
Use `bidirectional_ssm_layer α` for bidirectional scans (the current update is counted twice).
`transformer_loss` takes the same mixer as its first argument.
At `α = 1`, an entire stack of
bidirectional SSM blocks has permutation-invariant pooled loss.

`chunkwise_ssm_layer tiles n α qkv` implements the constant-decay
specialization of the chunkwise linear-attention equations in
[Gated Delta Networks, §2.1, equations (1)–(2)](https://arxiv.org/pdf/2412.06464).
It is not the paper's later gated-delta recurrence. In our transposed state convention,
each chunk advances its incoming state by `α^length` and adds the decay-weighted
outer products of its keys and values. Each output uses the corresponding local
prefix and query. There is no token-by-token recurrence inside `ssm_chunk`.
The input length is `tiles * n`: exactly `tiles` chunks of size `n`.
The layer and transformer-stack equivalence theorems cover arbitrary rational
parameters and equal-size chunks, including empty sequences. The layer theorem
takes QKV directly. Use
`transformer (chunkwise_ssm_layer tiles n α) blocks input` for the chunkwise mixer.
This is an executable mathematical reference, not an optimized kernel: the current
index-based carry function can recompute earlier chunk states across output queries.

The version before this simplification is in
[backup/pre-simplification/](backup/pre-simplification/).
The original Python/parser/Triu prototype remains in
[backup/python-lean-prototype/](backup/python-lean-prototype/).

The integer version before normalized rational attention is preserved in
[backup/pre-rationals/Tensors.lean](backup/pre-rationals/Tensors.lean).
