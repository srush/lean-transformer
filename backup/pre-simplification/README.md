# Minimal integer tensors

Read **[Tensors.lean](Tensors.lean)** from top to bottom. It is a single Lean
file with native Verso module documentation (`set_option doc.verso true`),
checked examples, and proofs. No external Verso package is needed.

The walkthrough introduces fixed-length integer tensors, pointwise addition
and multiplication, distributivity, and then `n × m` matrices with matrix
multiplication. Use `a.matmul b` to multiply an `n × m` matrix by an `m × p`
matrix. Matching dimensions are enforced by the types; there is no broadcasting.
Checked examples cover a rectangular product and a zero inner dimension.

`a.row_split h` returns top and bottom matrices; `a.column_split h` returns left
and right matrices. Both require a proof `h` that the dimension being split is
even (`n % 2 = 0` for rows, `m % 2 = 0` for columns). Both parts are exactly
half-sized. Select them with `.1` and `.2`; use `by decide` for the evenness
proof when the size is concrete. Odd split dimensions are not accepted.

`a.tensor_parallel b h` splits the shared dimension: columns of `a` and rows
of `b`. It performs the two smaller matrix multiplications independently and
adds their outputs pointwise. `Matrix.tensor_parallel_correct` proves that
this equals `a.matmul b` for every input and every even shared dimension.
This is an exact-integer model of the computation, not a parallel runtime.

`layer input weight bias` computes an affine layer: multiply a `batch × hidden`
input by a `hidden × hidden` weight matrix, then explicitly add the length-`hidden`
bias to each row. The output has shape `batch × hidden`; no activation is applied.

`loss score matrix` sums `score i (matrix i)` over the batch. The scorer receives
the original batch index and the full hidden vector. For even batches,
`data_parallel_loss score matrix h` scores two row shards independently and adds
their losses. `data_parallel_loss_correct` proves equality with the full loss
for any such scorer, preserving the original indices in both shards.

`Sequence seq batch hidden` adds a sequence dimension. `sequence_layer` applies
the same weights and bias independently at every sequence position.
`sequence_loss` sums `score s i (output s i)` over sequence and batch indices,
passing each full hidden vector to the scorer. `sequence_layer_loss` composes
the layer and loss into one function.

`relu z` is integer `max z 0`. `discrete_softmax z` implements exactly
`relu(zᵢ) - Σⱼ relu(zⱼ)`, including the current entry in the sum. Its weights
are nonpositive rather than probabilities; a singleton vector produces zero.

`attention_layer input wq wk wv bq bk bv` computes separate affine Q/K/V
projections, applies `discrete_softmax` to each row of `Q Kᵀ`, then multiplies
by `V`. The row sum runs over key positions within one batch item. It returns
`seq × batch × hidden`, ready for `sequence_loss`, with no scaling, mask, or
output projection. The permutation proof includes the new score transformation.

`transformer blocks input` alternates an affine layer and attention for each
`TransformerBlock`; `transformer_loss blocks score input` then applies
`sequence_loss`. The empty block list leaves the input unchanged.

`PositionPermutation` describes any reordering of the sequence positions,
certified to retain every index exactly once. `input.permute π` reads old
position `π.index s` at new position `s`. The theorem
`transformer_loss_position_invariant` proves that permuting both the input and
its scores (`fun s => score (π.index s)`) leaves the final loss unchanged.
Intermediate outputs follow the permutation. This holds for arbitrary block
lists and scorers in this model, which has no positional encoding or mask.
A two-block example checks a three-position cycle and shows that moving only
the inputs need not preserve the loss.

`swa radius` is sliding-window attention: each query uses keys and values at
positions within `radius` steps on either side, including itself, without
wrapping at sequence boundaries. This variant retains raw dot-product weights.
`swa_transformer` alternates affine layers
and SWA using the existing block parameters.

`swa_transformer_local` proves by induction that `d` blocks have receptive-field
radius at most `d * radius`. `swa_loss_receptive_field` applies that result to
`swa_position_loss`: changing inputs outside that window cannot affect the
chosen output's score. This holds for arbitrary weights, biases, and scorers,
and other batch items cannot affect that score either. The total sequence loss
can still change. Unlike full attention, fixed windows do not preserve arbitrary
sequence permutations. Checked examples cover the receptive-field boundary,
radius zero, and a window covering the full sequence.

`ssm_layer input K` scans each batch item from a zero initial state using
`state' = state + Kᵀ x`, returning the updated state at each sequence position.
`ssm_transformer_loss` places this SSM before the transformer and final loss.
`ssm_breaks_position_invariance` proves a counterexample with one transformer
block: swapping inputs `[1, 2]` and moving their scores together changes the
loss from -42 to -84. Thus the earlier invariance guarantee does not extend to
an arbitrary SSM prefix. Some parameter choices can still be invariant.

`bidirectional_ssm_layer input K` sums forward and backward states using the
same projection. Both scans include the current position, so the result is
the total projected input plus the current projection. The file proves this
identity, then proves that the output follows sequence permutations and that
`bidirectional_ssm_transformer_loss` is invariant when inputs and scorers are
permuted together. A checked example gives loss -560 in both orders.

`alpha_bidirectional_ssm_layer α input K` uses `state' = α * state + Kᵀ x`
in both directions, with integer `α`. At `α = 1` it provably equals the
unweighted bidirectional layer. For `α = 2`, a three-position counterexample
in `alpha_ssm_breaks_position_invariance` gives losses -106848 and -133632 after
swapping inputs and scores together. The factors now depend on position
distance. This disproves arbitrary-permutation invariance in general;
special cases such as `α = 0` or `α = 1` remain equivariant.

`parallel_ssm_layer α input projection` implements `(Q Kᵀ ⊙ M) V` for the
current scalar-weight bidirectional SSM. `V` contains projected inputs, `Q`
and the attention key matrix are columns of ones, and `M` has entries
`α^distance` off the diagonal and 2 on the diagonal. The diagonal counts the
current input in both scans. The attention key matrix is distinct from the
learned SSM projection. `parallel_ssm_layer_eq` proves equality with
`alpha_bidirectional_ssm_layer` for all dimensions, inputs, projections, and
integer alpha values by unrolling both scans into weighted sums.

The FlashAttention exercise asks for a tiled version of the current discrete
attention. `flash_attention n positive input wq wk wv bq bk bv` visits at most
`n` keys at a time for each query, allowing a shorter final tile. It accumulates
`R = Σ relu(z)`, `P = Σ relu(z) * V`, and `U = Σ V`, then returns `P - R * U`.
The global score sum is never replaced by a separate normalization per tile.
`flash_attention_eq` proves equality with `attention_layer` for all inputs,
parameters, and positive tile sizes. This is a mathematical tiling model for
our discrete rule, not a GPU implementation or a performance guarantee.

```sh
lake build
```

Open `Tensors.lean` in a Lean editor to read the explanation beside the code
and inspect the proof state. This uses Lean's built-in Verso documentation
format; it does not set up a separate HTML book generator.

The previous Python parser, tensor interpreter, Triu proofs, widgets, and
project configuration are preserved in
[`backup/python-lean-prototype/`](backup/python-lean-prototype/). To build the
old version, run `lake build` from that directory.
