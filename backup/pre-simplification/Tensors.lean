import Std

set_option doc.verso true

/-!
# Integer tensors, one step at a time

We start with fixed-length integer tensors, pointwise operations, and one
proof. Then we introduce matrices and matrix multiplication. Dimensions are
part of the types, and there is no broadcasting.

# 1. A tensor is a function from positions to integers

{lit}`Fin n` is the type of positions from zero to {lit}`n - 1`.
A {lit}`Tensor n` will give an integer for each such position. Its length is
part of its type, so there is no out-of-bounds position to handle.
-/

def Tensor (n : Nat) := Fin n → Int

/-!
# 2. Make a small tensor

The function below describes the three entries 1, 2, and 3. The expression
{lit}`i.val` reads the natural-number position, and {lit}`(i.val : Int)`
converts it to an integer. Applying the tensor to a position reads an entry.
-/

def x : Tensor 3 := fun i => (i.val : Int) + 1

example : x 0 = 1 := by rfl
example : x 2 = 3 := by rfl

/-!
# 3. Add matching entries

At position {lit}`i`, addition returns {lit}`a i + b i`.
The {name}`Add` instance tells Lean what the notation {lit}`a + b` means for
tensors. Both inputs have type {lit}`Tensor n`, so their lengths must match.
The result is a new function; neither input changes.
-/

instance : Add (Tensor n) where
  add a b := fun i => a i + b i

example : (x + x) 2 = 6 := by rfl

/-!
# 4. Multiply matching entries

Multiplication follows the same pattern: multiply the integers at the same
position. This is pointwise multiplication, with no summation or broadcasting.
-/

instance : Mul (Tensor n) where
  mul a b := fun i => a i * b i

example : (x * x) 2 = 9 := by rfl

/-!
# 5. Prove distributivity

For any three tensors of the same length, multiplication distributes over
addition. To prove two tensor functions equal, {lit}`funext i` asks us to prove
their entries equal at an arbitrary position. The definitions reduce that
entrywise goal to integer distributivity, which {name}`Int.mul_add` proves.
This works for every length, including zero.
-/

theorem Tensor.mul_add (a b c : Tensor n) : a * (b + c) = a * b + a * c := by
  funext i
  exact Int.mul_add (a i) (b i) (c i)

/-!
# 6. A matrix is a tensor for each row

A {lit}`Matrix n m` has {lit}`n` rows and {lit}`m` columns. Giving it a row
position returns a {lit}`Tensor m`; giving it a column position then returns
an integer. We read an entry with {lit}`a i j`.
-/

def Matrix (n m : Nat) := Fin n → Tensor m

/-!
# 7. Multiply a row by a column

Multiplying an {lit}`n × m` matrix by an {lit}`m × p` matrix produces an
{lit}`n × p` matrix. At output position {lit}`i, j`, we sum
{lit}`a i k * b k j` over all {lit}`m` inner positions.

{name}`List.ofFn` collects one product for each {lit}`k : Fin m`, and
{name}`List.sum` adds those products. Both inputs share the same inner
dimension in the signature, so incompatible shapes cannot be passed here.
We write {lit}`a.matmul b` to distinguish this operation from pointwise
{lit}`*`. When the inner dimension is zero, the empty sum is zero.
-/

def Matrix.matmul (a : Matrix n m) (b : Matrix m p) : Matrix n p :=
  fun i j => (List.ofFn fun k : Fin m => a i k * b k j).sum

/-!
# 8. Check a rectangular example

The formulas below give {lit}`a = [[1, 2, 3], [4, 5, 6]]` and
{lit}`b = [[1, 2], [3, 4], [5, 6]]`. Their product is
{lit}`[[22, 28], [49, 64]]`. For instance, its first entry is
{lit}`1 * 1 + 2 * 3 + 3 * 5 = 22`.

Each {lit}`rfl` checks an entry by evaluating the definition. These are
concrete checks of matrix multiplication, rather than a general algebraic
theorem about it.
-/

def a : Matrix 2 3 := fun i j => 3 * (i.val : Int) + (j.val : Int) + 1
def b : Matrix 3 2 := fun i j => 2 * (i.val : Int) + (j.val : Int) + 1

example : (a.matmul b) 0 0 = 22 := by rfl
example : (a.matmul b) 0 1 = 28 := by rfl
example : (a.matmul b) 1 0 = 49 := by rfl
example : (a.matmul b) 1 1 = 64 := by rfl

example (a : Matrix n 0) (b : Matrix 0 p) (i : Fin n) (j : Fin p) :
    (a.matmul b) i j = 0 := by rfl

/-!
# 9. Split the rows

For natural numbers, Lean's {lit}`n / 2` is Python's {lit}`n // 2`.
Splitting rows requires a proof of {lit}`n % 2 = 0`: the row count is even.
Both results then have exactly {lit}`n / 2` rows, and no row is lost.

The half-size type is simply {lit}`Matrix (n / 2) m`. Its row indices have
type {lit}`Fin (n / 2)`, which carries the bound {lit}`i < n / 2`.
This specializes our existing matrix type; it is not a separate Lean
{name}`Subtype` wrapper around a full-size matrix.

Each result reads from the original matrix. The first uses the same row
position; the second adds the split offset. In {lit}`⟨position, proof⟩`, the
proof certifies that the position is in bounds. {lit}`i.isLt` supplies the
smaller matrix's bound, and {lit}`omega` proves the required arithmetic.
-/

def Matrix.row_split (a : Matrix n m) (h : n % 2 = 0) :
    Matrix (n / 2) m × Matrix (n / 2) m := by
  have halves : n / 2 + n / 2 = n := by omega
  exact ⟨fun i j => a ⟨i.val, by have := i.isLt; omega⟩ j,
         fun i j => a ⟨n / 2 + i.val, by have := i.isLt; omega⟩ j⟩

/-!
# 10. Split the columns

The same construction on column positions gives left and right matrices.
For either split, {lit}`.1` selects the first matrix and {lit}`.2` the second.
Here the column count must be even. Only the dimension being split needs this
proof: a matrix with two rows and three columns can be split by rows, but not
by columns. Zero is even and splits into two empty parts.
-/

def Matrix.column_split (a : Matrix n m) (h : m % 2 = 0) :
    Matrix n (m / 2) × Matrix n (m / 2) := by
  have halves : m / 2 + m / 2 = m := by omega
  exact ⟨fun i j => a i ⟨j.val, by have := j.isLt; omega⟩,
         fun i j => a i ⟨m / 2 + j.val, by have := j.isLt; omega⟩⟩

/-!
# 11. Check the split positions

Our {name}`a` has two rows, so its row split gives one row in each part.
Our {name}`b` has two columns, so its column split gives one column in each
part. For these concrete dimensions, {lit}`by decide` proves evenness by
computation. It cannot prove the required condition for an odd dimension.
-/

example : (a.row_split (by decide)).1 0 2 = 3 := by rfl
example : (a.row_split (by decide)).2 0 2 = 6 := by rfl
example : (b.column_split (by decide)).1 2 0 = 5 := by rfl
example : (b.column_split (by decide)).2 2 0 = 6 := by rfl

/-!
# 12. Compute two partial matrix products

Split the shared dimension of {lit}`a : Matrix n m` and
{lit}`b : Matrix m p`, assuming {lit}`m` is even. Splitting the columns of
{lit}`a` gives {lit}`a₁, a₂`; splitting the rows of {lit}`b` gives
{lit}`b₁, b₂`. Each pair can be multiplied independently.

Both partial results have shape {lit}`n × p`. Combine them by adding matching
entries: {lit}`a₁.matmul b₁ + a₂.matmul b₂`. This is the sum reduction used
in tensor parallelism along the shared dimension. The definition describes
the computation mathematically; it does not launch parallel workers.
-/

def Matrix.tensor_parallel (a : Matrix n m) (b : Matrix m p) (h : m % 2 = 0) :
    Matrix n p :=
  let (a₁, a₂) := a.column_split h
  let (b₁, b₂) := b.row_split h
  let c₁ := a₁.matmul b₁
  let c₂ := a₂.matmul b₂
  fun i j => c₁ i j + c₂ i j

/-!
# 13. Prove equality with the full product

The theorem states equality of the full matrix product and the computation on
the two pairs of halves, for every input matrix and every even shared size.
At each output position, the full dot product is a sum of {lit}`m` products.
{name}`List.ofFn_add` splits that list into two halves, and
{name}`List.sum_append` says that their sums add to the original sum.
-/

theorem Matrix.tensor_parallel_correct
    (a : Matrix n m) (b : Matrix m p) (h : m % 2 = 0) :
    a.tensor_parallel b h = a.matmul b := by
  obtain ⟨k, rfl⟩ : ∃ k, m = k + k := ⟨m / 2, by omega⟩
  have half : (k + k) / 2 = k := by omega
  -- Compare one output entry, then split its full dot-product sum.
  funext i j
  simp [Matrix.tensor_parallel, Matrix.column_split, Matrix.row_split,
    Matrix.matmul, half, List.ofFn_add, List.sum_append,
    Fin.castLE, Fin.addNat, Nat.add_comm]
  congr 1 <;> apply congrArg List.sum <;> apply List.ext_getElem
  all_goals simp [half]

/-!
For a concrete check, multiply {name}`b` by {name}`a`: the shared dimension
is two. The first output entry combines partial results {lit}`1 * 1` and
{lit}`2 * 4`, giving nine. The theorem above covers every entry, every integer
input, and every even shared dimension, including zero.
-/

example : (b.tensor_parallel a (by decide)) 0 0 = 9 := by rfl
example : (b.tensor_parallel a (by decide)) 2 2 = 51 := by rfl

/-!
# 14. A layer with a bias

The input has shape {lit}`batch × hidden`, the weight matrix has shape
{lit}`hidden × hidden`, and the bias is a {lit}`Tensor hidden`.
The output has shape {lit}`batch × hidden` and computes
{lit}`output i j = (input.matmul weight) i j + bias j`.

Each batch row uses the same weights and bias. We explicitly read
{lit}`bias j` for each output entry; no broadcasting rule is needed.
This is an affine layer over integers, with no activation function.
-/

def layer (input : Matrix batch hidden) (weight : Matrix hidden hidden)
    (bias : Tensor hidden) : Matrix batch hidden :=
  let product := input.matmul weight
  fun i j => product i j + bias j

/-!
For example, use {name}`b` as a three-row, two-feature input. The weight
matrix below is {lit}`[[1, 2], [3, 4]]` and the bias is {lit}`[10, -1]`.
The first output row is {lit}`[17, 9]`, and the last is {lit}`[33, 33]`.
-/

example :
    let weight : Matrix 2 2 := fun i j => 2 * (i.val : Int) + (j.val : Int) + 1
    let bias : Tensor 2 := fun j => if j.val = 0 then 10 else -1
    let output := layer b weight bias
    output 0 0 = 17 ∧ output 0 1 = 9 ∧ output 2 0 = 33 ∧ output 2 1 = 33 := by
  decide

/-!
# 15. Score each example and sum the loss

The scoring function takes an example's batch index and its entire hidden
vector. It can inspect any or all hidden entries and returns one integer.
The loss sums {lit}`score i (matrix i)` over the batch. This is a sum, not
an average; an empty batch has loss zero.
-/

def loss (score : Fin batch → Tensor hidden → Int)
    (matrix : Matrix batch hidden) : Int :=
  (List.ofFn fun i => score i (matrix i)).sum

/-!
# 16. Compute the loss on two batch shards

For an even batch, split the rows and compute a loss on each half independently.
Each example keeps its full hidden vector. Add the two scalar losses to get
the combined result.

The scoring function must still receive the original batch index, since it
may use that index to look up a label. The first half keeps its indices;
the second adds {lit}`batch / 2`. Restarting both halves at index zero could
change the loss.
-/

def data_parallel_loss (score : Fin batch → Tensor hidden → Int)
    (matrix : Matrix batch hidden) (h : batch % 2 = 0) : Int :=
  let (first, second) := matrix.row_split h
  loss (fun i row => score ⟨i.val, by have := i.isLt; omega⟩ row) first +
  loss (fun i row => score ⟨batch / 2 + i.val, by have := i.isLt; omega⟩ row) second

/-!
# 17. Prove data-parallel loss is unchanged

This holds for any scoring function with the stated signature, including
nonlinear functions of the full hidden vector. The proof splits the list of
per-example scores into two halves and uses additivity of the sum. There is
no assumption that scoring is linear, only that it scores each row separately.
-/

theorem data_parallel_loss_correct (score : Fin batch → Tensor hidden → Int)
    (matrix : Matrix batch hidden) (h : batch % 2 = 0) :
    data_parallel_loss score matrix h = loss score matrix := by
  obtain ⟨k, rfl⟩ : ∃ k, batch = k + k := ⟨batch / 2, by omega⟩
  have half : (k + k) / 2 = k := by omega
  simp [data_parallel_loss, loss, Matrix.row_split, half,
    List.ofFn_add, List.sum_append, Fin.castLE, Fin.addNat, Nat.add_comm]
  congr 1 <;> apply congrArg List.sum <;> apply List.ext_getElem
  all_goals simp [half]

/-!
For {name}`a`, score each row using its index and the product of its first
and last entries. The scores are {lit}`0 + 1 * 3 = 3` and
{lit}`1 + 4 * 6 = 25`. Both computations must return 28; incorrectly using
index zero for the second shard would give 27.
-/

example :
    let score : Fin 2 → Tensor 3 → Int := fun i row => (i.val : Int) + row 0 * row 2
    loss score a = 28 ∧ data_parallel_loss score a (by decide) = 28 := by
  decide

/-!
# 18. Add a sequence dimension

A {lit}`Sequence seq batch hidden` gives a {lit}`batch × hidden` matrix at
each sequence position. Read one hidden vector with {lit}`input s i`, where
{lit}`s` is the sequence position and {lit}`i` is the batch index.
-/

def Sequence (seq batch hidden : Nat) := Fin seq → Matrix batch hidden

/-!
# 19. Apply the layer at every position

Use the same weights and bias at every sequence position. The existing
{name}`layer` already handles the batch dimension, so we only need to apply
it to each {lit}`input s`. Positions are processed independently; there is
no recurrence or interaction between sequence positions.
-/

def sequence_layer (input : Sequence seq batch hidden)
    (weight : Matrix hidden hidden) (bias : Tensor hidden) :
    Sequence seq batch hidden :=
  fun s => layer (input s) weight bias

/-!
# 20. Score the outputs at every position

The scorer now receives both indices and the full output hidden vector:
{lit}`score s i (output s i)`. At each sequence position, reuse {name}`loss`
to sum over the batch, then sum over the sequence. Empty sequences or batches
give zero loss. This is a sum over all positions, not a mean.
-/

def sequence_loss (score : Fin seq → Fin batch → Tensor hidden → Int)
    (output : Sequence seq batch hidden) : Int :=
  (List.ofFn fun s => loss (score s) (output s)).sum

def sequence_layer_loss (score : Fin seq → Fin batch → Tensor hidden → Int)
    (input : Sequence seq batch hidden) (weight : Matrix hidden hidden)
    (bias : Tensor hidden) : Int :=
  sequence_loss score (sequence_layer input weight bias)

/-!
Here is a sequence of length two with batch size three and hidden size two.
Position zero uses {name}`b`; position one adds ten to each input entry.
The score uses both indices and both hidden entries, so the check exercises
both dimensions as well as applying the layer before scoring.
-/

example :
    let input : Sequence 2 3 2 := fun s i j => b i j + 10 * (s.val : Int)
    let weight : Matrix 2 2 := fun i j => 2 * (i.val : Int) + (j.val : Int) + 1
    let bias : Tensor 2 := fun j => if j.val = 0 then 10 else -1
    let score : Fin 2 → Fin 3 → Tensor 2 → Int :=
      fun s i row => 100 * (s.val : Int) + 10 * (i.val : Int) + row 0 + 2 * row 1
    let output := sequence_layer input weight bias
    output 0 0 0 = 17 ∧ output 1 2 1 = 93 ∧
      sequence_layer_loss score input weight bias = 1242 := by
  decide

/-!
# 21. Transpose a matrix

Attention compares queries with keys using a transposed key matrix.
Transposing exchanges row and column positions: {lit}`a.transpose i j = a j i`.
-/

def Matrix.transpose (a : Matrix n m) : Matrix m n := fun i j => a j i

/-!
# 21a. Integer ReLU and a discrete score transformation

ReLU keeps positive integers and replaces negative integers by zero.
For a vector of scores, use exactly {lit}`relu(zᵢ) - Σⱼ relu(zⱼ)`.
The sum includes the current entry. This is our discrete softmax rule;
it is not the exponential softmax and need not produce probabilities.
Its weights are nonpositive, and a singleton vector produces zero.
-/

def relu (z : Int) : Int := max z 0

def discrete_softmax (z : Tensor n) : Tensor n :=
  let total := (List.ofFn fun j => relu (z j)).sum
  fun i => relu (z i) - total

example : relu (-3) = 0 ∧ relu 0 = 0 ∧ relu 4 = 4 := by decide

example :
    let z : Tensor 3 := fun i => if i.val = 0 then -2 else if i.val = 1 then 3 else 1
    discrete_softmax z 0 = -4 ∧ discrete_softmax z 1 = -1 ∧
      discrete_softmax z 2 = -3 ∧ discrete_softmax (fun _ : Fin 1 => 5) 0 = 0 := by
  decide

/-!
# 22. Attention across the sequence

First compute three independent affine projections of the input: queries
{lit}`Q`, keys {lit}`K`, and values {lit}`V`. Each has shape
{lit}`seq × batch × hidden`, using its own weights and bias.

For each batch item, view these as {lit}`seq × hidden` matrices. The product
{lit}`Q Kᵀ` has shape {lit}`seq × seq`: entry {lit}`s, t` is the dot product
of the query at position {lit}`s` and the key at position {lit}`t`.
Apply {name}`discrete_softmax` to each row of these logits, then multiply by
{lit}`V` to combine values from all sequence positions. The row sum is over
key positions for one query, never over the batch or hidden dimension.

This is single-head, bidirectional self-attention with our discrete score rule
and no scaling.
Batch items never interact. The result has the original shape and can be
passed directly to {name}`sequence_loss`. We include only the Q/K/V attention
computation here, without an output projection, residual, or normalization.
-/

def attention_layer (input : Sequence seq batch hidden)
    (wq wk wv : Matrix hidden hidden) (bq bk bv : Tensor hidden) :
    Sequence seq batch hidden :=
  let q := sequence_layer input wq bq
  let k := sequence_layer input wk bk
  let v := sequence_layer input wv bv
  fun s i =>
    let queries : Matrix seq hidden := fun t j => q t i j
    let keys : Matrix seq hidden := fun t j => k t i j
    let values : Matrix seq hidden := fun t j => v t i j
    let logits := queries.matmul keys.transpose
    let weights : Matrix seq seq := fun t => discrete_softmax (logits t)
    (weights.matmul values) s

/-!
# 23. Check attention on two positions

This example uses two sequence positions and three distinct batch items.
The Q, K, and V weights and biases differ, so the check exercises all three
projections as well as the sequence-axis matrix products.
-/

example :
    let input : Sequence 2 3 2 :=
      fun s i j => 10 * (i.val : Int) + 2 * (s.val : Int) + (j.val : Int) + 1
    let wq : Matrix 2 2 := fun i j =>
      if i.val = j.val then 1 else if i.val = 0 then 2 else 0
    let wk : Matrix 2 2 := fun i j => if i.val = j.val then 0 else 1
    let wv : Matrix 2 2 := fun i j => if i.val = 0 ∧ j.val = 1 then 0 else 1
    let bq : Tensor 2 := fun j => if j.val = 0 then 1 else 0
    let bk : Tensor 2 := fun j => if j.val = 0 then 0 else -1
    let bv : Tensor 2 := fun j => if j.val = 0 then -1 else 2
    let output := attention_layer input wq wk wv bq bk bv
    output 0 0 0 = -56 ∧ output 1 0 1 = -192 ∧
      output 0 1 1 = -15808 ∧ output 1 2 0 = -177560 := by
  decide

/-!
# 24. Alternate a layer and attention

One block contains the parameters of an affine layer and an attention layer.
A list of blocks describes {lit}`layer → attention → layer → attention → …`.
Every block keeps the same hidden size. An empty list leaves the input alone.
-/

structure TransformerBlock (hidden : Nat) where
  weight : Matrix hidden hidden
  bias : Tensor hidden
  wq : Matrix hidden hidden
  wk : Matrix hidden hidden
  wv : Matrix hidden hidden
  bq : Tensor hidden
  bk : Tensor hidden
  bv : Tensor hidden

def transformer (blocks : List (TransformerBlock hidden))
    (input : Sequence seq batch hidden) : Sequence seq batch hidden :=
  match blocks with
  | [] => input
  | block :: rest =>
      let output := sequence_layer input block.weight block.bias
      transformer rest
        (attention_layer output block.wq block.wk block.wv block.bq block.bk block.bv)

def transformer_loss (blocks : List (TransformerBlock hidden))
    (score : Fin seq → Fin batch → Tensor hidden → Int)
    (input : Sequence seq batch hidden) : Int :=
  sequence_loss score (transformer blocks input)

/-!
# 25. Describe a permutation of positions

{lit}`π.index s` specifies which old position belongs at new position {lit}`s`.
Its certificate says that listing these indices gives a permutation of all
the original indices: none is lost or duplicated. This permits every sequence
permutation, not just reversal or swapping adjacent positions.

Reorder the input with {lit}`input (π.index s)` and its associated scorer with
{lit}`score (π.index s)`. The batch and hidden dimensions stay unchanged.
-/

structure PositionPermutation (seq : Nat) where
  index : Fin seq → Fin seq
  valid : (List.ofFn index).Perm (List.ofFn fun i : Fin seq => i)

def Sequence.permute (π : PositionPermutation seq) (input : Sequence seq batch hidden) :
    Sequence seq batch hidden :=
  fun s => input (π.index s)

/-!
# 26. Reordering an integer sum preserves its value

Mapping any integer-valued function over permuted indices gives the same
values in a different order. Integer addition allows that reordering.
We will use this fact for both attention and the final loss.
-/

theorem PositionPermutation.sum (π : PositionPermutation seq) (f : Fin seq → Int) :
    (List.ofFn fun s => f (π.index s)).sum = (List.ofFn f).sum := by
  have hp := π.valid.map f
  simp only [List.map_ofFn] at hp
  exact hp.foldr_eq' (fun x _ y _ z => Int.add_left_comm y x z) 0

theorem discrete_softmax_permute (π : PositionPermutation n) (z : Tensor n) :
    discrete_softmax (fun s => z (π.index s)) =
      (fun s => discrete_softmax z (π.index s)) := by
  funext s
  exact congrArg (fun total => relu (z (π.index s)) - total)
    (π.sum (fun t => relu (z t)))

/-!
# 27. The layers follow the permutation

An affine layer uses the same parameters at every position, so reordering
its input simply reorders its output. Attention also follows the permutation:
the query at the new position is the corresponding old query, and the key/value
pairs are reordered together. ReLU follows the permutation, and the row sum
in the discrete softmax is unchanged. The final weighted sum is also unchanged.
This property is called permutation equivariance.
-/

theorem sequence_layer_permute (π : PositionPermutation seq)
    (input : Sequence seq batch hidden) (weight : Matrix hidden hidden)
    (bias : Tensor hidden) :
    sequence_layer (input.permute π) weight bias =
      (sequence_layer input weight bias).permute π := by
  rfl

theorem attention_layer_permute (π : PositionPermutation seq)
    (input : Sequence seq batch hidden) (wq wk wv : Matrix hidden hidden)
    (bq bk bv : Tensor hidden) :
    attention_layer (input.permute π) wq wk wv bq bk bv =
      (attention_layer input wq wk wv bq bk bv).permute π := by
  let q := sequence_layer input wq bq
  let k := sequence_layer input wk bk
  let v := sequence_layer input wv bv
  funext s i j
  let logits : Tensor seq := fun t =>
    (List.ofFn fun d => q (π.index s) i d * k t i d).sum
  change (List.ofFn fun t => discrete_softmax (fun u => logits (π.index u)) t *
      v (π.index t) i j).sum =
    (List.ofFn fun t => discrete_softmax logits t * v t i j).sum
  rw [discrete_softmax_permute]
  exact π.sum (fun t => discrete_softmax logits t * v t i j)

/-!
# 28. The transformer follows the permutation

Induct over the blocks. Each affine layer and attention layer follows the
permutation, and the remaining blocks preserve that property in turn.
-/

theorem transformer_permute (π : PositionPermutation seq)
    (blocks : List (TransformerBlock hidden)) (input : Sequence seq batch hidden) :
    transformer blocks (input.permute π) = (transformer blocks input).permute π := by
  induction blocks generalizing input with
  | nil => rfl
  | cons block rest ih =>
      simp only [transformer, sequence_layer_permute, attention_layer_permute, ih]

/-!
# 29. The final loss is position invariant

Permute the input and its position-specific scoring functions together.
The transformer outputs follow that permutation, so the final loss contains
exactly the same per-position scores in a different order.

The theorem holds for any number of blocks, weights, biases, inputs, and
scoring functions. It relies on the model defined here: parameters are shared
across positions, and there is no positional encoding or position-dependent
attention mask. The outputs are permuted; the final scalar loss is unchanged.
-/

theorem transformer_loss_position_invariant (π : PositionPermutation seq)
    (blocks : List (TransformerBlock hidden))
    (score : Fin seq → Fin batch → Tensor hidden → Int)
    (input : Sequence seq batch hidden) :
    transformer_loss blocks (fun s => score (π.index s)) (input.permute π) =
      transformer_loss blocks score input := by
  unfold transformer_loss
  rw [transformer_permute]
  exact π.sum (fun s => loss (score s) (transformer blocks input s))

/-!
As a concrete check, rotate three positions from {lit}`[0, 1, 2]` to
{lit}`[1, 2, 0]` and run two blocks. Moving the position-specific scores with
the inputs preserves the loss. Moving only the inputs changes it.
-/

example :
    let rotate : PositionPermutation 3 := {
      index := fun s => ⟨(s.val + 1) % 3, by omega⟩
      valid := by decide
    }
    let one : Matrix 1 1 := fun _ _ => 1
    let zero : Tensor 1 := fun _ => 0
    let block : TransformerBlock 1 := {
      weight := one, bias := fun _ => 1
      wq := one, wk := one, wv := one
      bq := zero, bk := zero, bv := zero
    }
    let input : Sequence 3 1 1 := fun s _ _ => (s.val : Int) + 1
    let score : Fin 3 → Fin 1 → Tensor 1 → Int :=
      fun s _ row => ((s.val : Int) + 1) * row 0
    transformer_loss [block, block] score input = 143459228 ∧
      transformer_loss [block, block] (fun s => score (rotate.index s))
        (input.permute rotate) = 143459228 ∧
      transformer_loss [block, block] score (input.permute rotate) = 121815476 := by
  decide

/-!
# 30. Sliding-window attention

At position {lit}`s`, allow keys and values only from positions within
{lit}`radius` steps on either side, including {lit}`s` itself. The window is
clipped by the sequence boundaries; it does not wrap around. Radius zero
allows only the current position.

{lit}`InWindow radius s t` expresses this without subtraction on natural
numbers. {lit}`swa` uses the same Q/K/V projections as attention, but sets
contributions from outside this window to zero. This variant retains raw
dot-product weights, without the discrete softmax used by full attention.
-/

abbrev InWindow (radius : Nat) (s t : Fin seq) : Prop :=
  s.val ≤ t.val + radius ∧ t.val ≤ s.val + radius

def swa (radius : Nat) (input : Sequence seq batch hidden)
    (wq wk wv : Matrix hidden hidden) (bq bk bv : Tensor hidden) :
    Sequence seq batch hidden :=
  let q := sequence_layer input wq bq
  let k := sequence_layer input wk bk
  let v := sequence_layer input wv bv
  fun s i j => (List.ofFn fun t =>
    if InWindow radius s t then
      (List.ofFn fun d => q s i d * k t i d).sum * v t i j
    else 0).sum

/-!
# 31. One block only reads its window

An affine layer at {lit}`s, i` depends only on that input hidden vector.
One SWA layer at {lit}`s, i` depends only on vectors in its window, in the
same batch item {lit}`i`. We express independence by comparing any two inputs
that agree there; all other entries may differ arbitrarily.
-/

theorem sequence_layer_local (input other : Sequence seq batch hidden)
    (weight : Matrix hidden hidden) (bias : Tensor hidden)
    (s : Fin seq) (i : Fin batch) (agree : input s i = other s i) :
    sequence_layer input weight bias s i = sequence_layer other weight bias s i := by
  funext j
  simp only [sequence_layer, layer, Matrix.matmul, agree]

theorem swa_local (radius : Nat) (input other : Sequence seq batch hidden)
    (wq wk wv : Matrix hidden hidden) (bq bk bv : Tensor hidden)
    (s : Fin seq) (i : Fin batch)
    (agree : ∀ t, InWindow radius s t → input t i = other t i) :
    swa radius input wq wk wv bq bk bv s i =
      swa radius other wq wk wv bq bk bv s i := by
  have center : InWindow radius s s := by constructor <;> omega
  have projection (w : Matrix hidden hidden) (b : Tensor hidden)
      (t : Fin seq) (ht : InWindow radius s t) :=
    sequence_layer_local input other w b t i (agree t ht)
  funext j
  dsimp only [swa]
  apply congrArg List.sum
  apply congrArg List.ofFn
  funext t
  by_cases ht : InWindow radius s t
  · simp only [projection wq bq s center,
      projection wk bk t ht, projection wv bv t ht]
  · simp only [ht, if_false]

/-!
# 32. Stack the windowed blocks

Use the same block parameters as before, replacing full attention by SWA.
Each block adds at most {lit}`radius` to the receptive-field radius; the
pointwise affine layer adds nothing.
-/

def swa_transformer (radius : Nat) (blocks : List (TransformerBlock hidden))
    (input : Sequence seq batch hidden) : Sequence seq batch hidden :=
  match blocks with
  | [] => input
  | block :: rest =>
      let output := sequence_layer input block.weight block.bias
      swa_transformer radius rest
        (swa radius output block.wq block.wk block.wv block.bq block.bk block.bv)

/-!
# 33. Prove the receptive field by induction

For {lit}`d` blocks the radius is at most {lit}`d * radius`. The base case
reads only the original position. In the inductive step, the remaining blocks
read a window of radius {lit}`(d - 1) * radius`; each of those intermediate
positions reads at most another {lit}`radius` steps from the original input.
Adding the distances keeps every dependency within {lit}`d * radius`.
-/

theorem swa_transformer_local (radius : Nat) (blocks : List (TransformerBlock hidden))
    (input other : Sequence seq batch hidden) (s : Fin seq) (i : Fin batch)
    (agree : ∀ t, InWindow (blocks.length * radius) s t → input t i = other t i) :
    swa_transformer radius blocks input s i =
      swa_transformer radius blocks other s i := by
  induction blocks generalizing input other with
  | nil => exact agree s (by simp [InWindow])
  | cons block rest ih =>
      apply ih
      intro t ht
      apply swa_local
      intro u hu
      apply sequence_layer_local
      apply agree
      simp only [List.length_cons, Nat.succ_mul]
      dsimp [InWindow] at ht hu ⊢
      omega

/-!
# 34. The loss at one output has the same receptive field

The score receives only the hidden vector at the chosen output position.
Therefore equal output vectors give equal scores, even for a nonlinear score.
Any changes strictly farther than {lit}`blocks.length * radius` from that
position cannot affect its loss. Changes exactly at the boundary may matter.

This is a statement about one output's loss, not the sum over all positions:
changing a distant input can still change the losses at other positions.
Unlike full attention, the fixed window also depends on position distances,
so the earlier arbitrary-permutation theorem does not apply to this variant.
-/

def swa_position_loss (radius : Nat) (blocks : List (TransformerBlock hidden))
    (score : Fin seq → Fin batch → Tensor hidden → Int)
    (input : Sequence seq batch hidden) (s : Fin seq) (i : Fin batch) : Int :=
  score s i (swa_transformer radius blocks input s i)

theorem swa_loss_receptive_field (radius : Nat) (blocks : List (TransformerBlock hidden))
    (score : Fin seq → Fin batch → Tensor hidden → Int)
    (input other : Sequence seq batch hidden) (s : Fin seq) (i : Fin batch)
    (agree : ∀ t, InWindow (blocks.length * radius) s t → input t i = other t i) :
    swa_position_loss radius blocks score input s i =
      swa_position_loss radius blocks score other s i := by
  exact congrArg (score s i) (swa_transformer_local radius blocks input other s i agree)

/-!
For two radius-one blocks, output position zero can depend on positions zero,
one, and two. Changing position three leaves its loss at 26; changing position
two changes it to 80. Radius zero uses only the current position, and a window
large enough to cover the sequence includes every position.
-/

example :
    let one : Matrix 1 1 := fun _ _ => 1
    let zero : Tensor 1 := fun _ => 0
    let block : TransformerBlock 1 := {
      weight := one, bias := zero
      wq := one, wk := one, wv := one
      bq := zero, bk := zero, bv := zero
    }
    let input : Sequence 5 1 1 := fun _ _ _ => 1
    let far : Sequence 5 1 1 := fun s _ _ => if s.val = 3 then 2 else 1
    let boundary : Sequence 5 1 1 := fun s _ _ => if s.val = 2 then 2 else 1
    let score : Fin 5 → Fin 1 → Tensor 1 → Int := fun _ _ row => row 0
    swa_position_loss 1 [block, block] score input 0 0 = 26 ∧
      swa_position_loss 1 [block, block] score far 0 0 = 26 ∧
      swa_position_loss 1 [block, block] score boundary 0 0 = 80 ∧
      swa_position_loss 0 [block, block] score far 0 0 = 1 ∧
      swa 5 input one one one zero zero zero 0 0 0 = 5 := by
  decide

/-!
# 35. One state-space update

Treat each hidden vector as a column vector. Multiplication by {lit}`Kᵀ`
reads the columns of {lit}`K`. Add that projected input to the previous state:
{lit}`state' = state + Kᵀ x`, with integer entries.
-/

def Matrix.matvec (a : Matrix n m) (x : Tensor m) : Tensor n :=
  fun i => (List.ofFn fun j => a i j * x j).sum

def ssm_step (K : Matrix hidden hidden) (state x : Tensor hidden) : Tensor hidden :=
  state + K.transpose.matvec x

example :
    let K : Matrix 2 2 := fun i j => 2 * (i.val : Int) + (j.val : Int) + 1
    let state : Tensor 2 := fun i => (i.val : Int) + 1
    let x : Tensor 2 := fun i => (i.val : Int) + 3
    ssm_step K state x 0 = 16 ∧ ssm_step K state x 1 = 24 := by
  decide

/-!
# 36. Scan from a zero initial state

For output position {lit}`s`, visit input positions zero through {lit}`s` in
order, updating the state each time. {name}`Fin.foldl` performs this ordered
scan. The state before the first input is zero, and each batch item has its
own state. Output position {lit}`s` contains the state after reading that input.

This definition computes a prefix on demand for each output position. It is a
small mathematical implementation, without a cache of the intermediate states.
-/

def ssm_layer (input : Sequence seq batch hidden) (K : Matrix hidden hidden) :
    Sequence seq batch hidden :=
  fun s i => Fin.foldl (s.val + 1)
    (fun state t => ssm_step K state
      (input ⟨t.val, by have := t.isLt; have := s.isLt; omega⟩ i))
    (fun _ => 0)

def ssm_transformer_loss (blocks : List (TransformerBlock hidden))
    (score : Fin seq → Fin batch → Tensor hidden → Int)
    (input : Sequence seq batch hidden) (K : Matrix hidden hidden) : Int :=
  transformer_loss blocks score (ssm_layer input K)

/-!
# 37. A counterexample to position invariance

Set {lit}`K = [1]` and start with inputs {lit}`[1, 2]`.
The SSM produces cumulative states {lit}`[1, 3]`. Swapping the inputs gives
{lit}`[2, 1]` and states {lit}`[2, 3]`, not a swap of the original states.

Now apply one ordinary transformer block: the affine layer is the identity,
and Q/K/V are identity projections with zero biases. Its outputs are
{lit}`[-6, -18]` in the original order and {lit}`[-24, -36]` after the input swap.
Score the first original position with weight one and the second with weight
two. Moving these scorers with the inputs gives losses -42 and -84 respectively.

The existential theorem below supplies a genuine permutation, inputs,
parameters, a nonempty block list, and scores for which invariance fails.
It disproves the general invariance property; it does not claim that every
SSM parameter choice or scorer is sensitive to order.
-/

theorem ssm_breaks_position_invariance :
    ∃ (π : PositionPermutation 2) (input : Sequence 2 1 1)
      (K : Matrix 1 1) (blocks : List (TransformerBlock 1))
      (score : Fin 2 → Fin 1 → Tensor 1 → Int),
      blocks ≠ [] ∧
      ssm_transformer_loss blocks score input K = -42 ∧
      ssm_transformer_loss blocks (fun s => score (π.index s))
        (input.permute π) K = -84 := by
  let swap : PositionPermutation 2 := {
    index := fun s => ⟨1 - s.val, by omega⟩
    valid := by decide
  }
  let input : Sequence 2 1 1 := fun s _ _ => (s.val : Int) + 1
  let one : Matrix 1 1 := fun _ _ => 1
  let zero : Tensor 1 := fun _ => 0
  let block : TransformerBlock 1 := {
    weight := one, bias := zero
    wq := one, wk := one, wv := one
    bq := zero, bk := zero, bv := zero
  }
  let score : Fin 2 → Fin 1 → Tensor 1 → Int :=
    fun s _ row => ((s.val : Int) + 1) * row 0
  refine ⟨swap, input, one, [block], score, ?_⟩
  decide

/-!
# 38. Sum the forward and backward states

Use the same {lit}`K` and zero initial state in both directions. At position
{lit}`s`, the forward state sums projected inputs from zero through {lit}`s`;
the backward state sums from the end down through {lit}`s`. Both include the
current input, so their sum is the total projected input plus one extra copy
of the current projection. The output is therefore permutation equivariant.

We express the scans as prefix and reversed-suffix sums. This is possible
because the update simply adds the projected input to the state.
-/

def bidirectional_ssm_layer (input : Sequence seq batch hidden)
    (K : Matrix hidden hidden) : Sequence seq batch hidden :=
  fun s i j =>
    let values := List.ofFn fun t => K.transpose.matvec (input t i) j
    (values.take (s.val + 1)).sum + (values.drop s.val).reverse.sum

theorem bidirectional_ssm_layer_eq (input : Sequence seq batch hidden)
    (K : Matrix hidden hidden) (s : Fin seq) (i : Fin batch) (j : Fin hidden) :
    bidirectional_ssm_layer input K s i j =
      (List.ofFn fun t => K.transpose.matvec (input t i) j).sum +
        K.transpose.matvec (input s i) j := by
  let values := List.ofFn fun t => K.transpose.matvec (input t i) j
  have bound : s.val < values.length := by simp [values]
  have split := congrArg List.sum (List.take_append_drop s.val values)
  simp only [List.sum_append] at split
  change (values.take (s.val + 1)).sum + (values.drop s.val).reverse.sum =
    values.sum + K.transpose.matvec (input s i) j
  rw [List.sum_reverse_int, List.take_succ_eq_append_getElem bound, List.sum_append]
  simp only [List.sum_cons, List.sum_nil, Int.add_zero]
  have current : values[s.val] = K.transpose.matvec (input s i) j := by simp [values]
  rw [current]
  omega

/-!
# 39. Recover permutation invariance of the loss

The total projection is a permutation-invariant sum. The extra current
projection follows the current input. Thus the outputs follow any permutation,
and the existing transformer/loss theorem applies again when scorers move with
their inputs. This uses shared projection weights in the two directions.
-/

theorem bidirectional_ssm_layer_permute (π : PositionPermutation seq)
    (input : Sequence seq batch hidden) (K : Matrix hidden hidden) :
    bidirectional_ssm_layer (input.permute π) K =
      (bidirectional_ssm_layer input K).permute π := by
  funext s i j
  simp only [Sequence.permute, bidirectional_ssm_layer_eq]
  exact congrArg (fun total => total + K.transpose.matvec (input (π.index s) i) j)
    (π.sum (fun t => K.transpose.matvec (input t i) j))

def bidirectional_ssm_transformer_loss (blocks : List (TransformerBlock hidden))
    (score : Fin seq → Fin batch → Tensor hidden → Int)
    (input : Sequence seq batch hidden) (K : Matrix hidden hidden) : Int :=
  transformer_loss blocks score (bidirectional_ssm_layer input K)

theorem bidirectional_ssm_loss_position_invariant (π : PositionPermutation seq)
    (blocks : List (TransformerBlock hidden))
    (score : Fin seq → Fin batch → Tensor hidden → Int)
    (input : Sequence seq batch hidden) (K : Matrix hidden hidden) :
    bidirectional_ssm_transformer_loss blocks (fun s => score (π.index s))
        (input.permute π) K =
      bidirectional_ssm_transformer_loss blocks score input K := by
  unfold bidirectional_ssm_transformer_loss
  rw [bidirectional_ssm_layer_permute]
  exact transformer_loss_position_invariant π blocks score (bidirectional_ssm_layer input K)

/-!
With {lit}`K = [1]`, inputs {lit}`[1, 2]` give outputs {lit}`[4, 5]`.
Swapping the inputs gives {lit}`[5, 4]`. After one identity-projection
transformer block, moving the scores with the inputs gives loss -560 both ways.
-/

example :
    let swap : PositionPermutation 2 := {
      index := fun s => ⟨1 - s.val, by omega⟩
      valid := by decide
    }
    let input : Sequence 2 1 1 := fun s _ _ => (s.val : Int) + 1
    let one : Matrix 1 1 := fun _ _ => 1
    let zero : Tensor 1 := fun _ => 0
    let block : TransformerBlock 1 := {
      weight := one, bias := zero
      wq := one, wk := one, wv := one
      bq := zero, bk := zero, bv := zero
    }
    let score : Fin 2 → Fin 1 → Tensor 1 → Int :=
      fun s _ row => ((s.val : Int) + 1) * row 0
    bidirectional_ssm_layer input one 0 0 0 = 4 ∧
      bidirectional_ssm_layer input one 1 0 0 = 5 ∧
      bidirectional_ssm_transformer_loss [block] score input one = -560 ∧
      bidirectional_ssm_transformer_loss [block] (fun s => score (swap.index s))
        (input.permute swap) one = -560 := by
  decide

/-!
# 40. Weight the previous state by alpha

Now use {lit}`state' = α * state + Kᵀ x` in both directions. A projected
input at distance {lit}`d` contributes with weight {lit}`α^d`; the current
input still appears in both scans. Distances can change under a permutation,
so the previous invariance guarantee no longer holds in general.

Here {lit}`α : Int` keeps the model integer-valued. We use two in the
counterexample, which amplifies older contributions. At one, this is exactly
the earlier bidirectional layer. At zero, each scan keeps only the current
projection, so that special case is also permutation equivariant.
-/

def alpha_bidirectional_ssm_layer (α : Int) (input : Sequence seq batch hidden)
    (K : Matrix hidden hidden) : Sequence seq batch hidden :=
  fun s i j =>
    let values := List.ofFn fun t => K.transpose.matvec (input t i) j
    let update := fun state x => α * state + x
    (values.take (s.val + 1)).foldl update 0 +
      (values.drop s.val).reverse.foldl update 0

theorem alpha_bidirectional_ssm_one (input : Sequence seq batch hidden)
    (K : Matrix hidden hidden) :
    alpha_bidirectional_ssm_layer 1 input K = bidirectional_ssm_layer input K := by
  funext s i j
  simp [alpha_bidirectional_ssm_layer, bidirectional_ssm_layer, List.sum_eq_foldl]

def alpha_ssm_transformer_loss (α : Int) (blocks : List (TransformerBlock hidden))
    (score : Fin seq → Fin batch → Tensor hidden → Int)
    (input : Sequence seq batch hidden) (K : Matrix hidden hidden) : Int :=
  transformer_loss blocks score (alpha_bidirectional_ssm_layer α input K)

/-!
# 41. A distance-sensitive counterexample

Use {lit}`α = 2`, {lit}`K = [1]`, and three positions. Inputs {lit}`[1, 2, 3]`
give bidirectional outputs {lit}`[18, 12, 14]`. Swapping the first two inputs
gives {lit}`[2, 1, 3]` and outputs {lit}`[18, 12, 16]`, which are not the
corresponding output permutation.

After one identity-projection transformer block, the losses are -106848 and
-133632, even with scores moved together with their inputs. We use three
positions because swapping two positions is reversal, which preserves all
distances and still respects this symmetric bidirectional operation.
-/

theorem alpha_ssm_breaks_position_invariance :
    ∃ (π : PositionPermutation 3) (input : Sequence 3 1 1)
      (K : Matrix 1 1) (blocks : List (TransformerBlock 1))
      (score : Fin 3 → Fin 1 → Tensor 1 → Int),
      blocks ≠ [] ∧
      alpha_ssm_transformer_loss 2 blocks score input K = -106848 ∧
      alpha_ssm_transformer_loss 2 blocks (fun s => score (π.index s))
        (input.permute π) K = -133632 := by
  let swap : PositionPermutation 3 := {
    index := fun s => if s.val < 2 then ⟨1 - s.val, by omega⟩ else s
    valid := by decide
  }
  let input : Sequence 3 1 1 := fun s _ _ => (s.val : Int) + 1
  let one : Matrix 1 1 := fun _ _ => 1
  let zero : Tensor 1 := fun _ => 0
  let block : TransformerBlock 1 := {
    weight := one, bias := zero
    wq := one, wk := one, wv := one
    bq := zero, bk := zero, bv := zero
  }
  let score : Fin 3 → Fin 1 → Tensor 1 → Int :=
    fun s _ row => ((s.val : Int) + 1) * row 0
  refine ⟨swap, input, one, [block], score, ?_⟩
  decide

/-!
# 42. Build the parallel mask

Unroll the recurrence: an input contributes a power of alpha for each step
between its position and the output. The forward mask is lower triangular;
the backward mask is upper triangular. Adding them gives {lit}`ssm_mask`.

Off the diagonal its entries are {lit}`α^distance`. On the diagonal the entry
is two because both scans include the current position. For length three,
the rows are {lit}`[2, α, α²]`, {lit}`[α, 2, α]`, and {lit}`[α², α, 2]`.
{lit}`Matrix.hadamard` below implements entrywise multiplication, written
{lit}`⊙` in the formula.
-/

def Matrix.hadamard (a b : Matrix n m) : Matrix n m := fun i j => a i j * b i j

def ssm_mask (α : Int) : Matrix seq seq := fun s t =>
  (if t.val ≤ s.val then α ^ (s.val - t.val) else 0) +
  (if s.val ≤ t.val then α ^ (t.val - s.val) else 0)

/-!
# 43. Compute O = (Q Kᵀ ⊙ M) V

For each batch item, {lit}`V` contains the projected inputs, with shape
{lit}`seq × hidden`. The scalar recurrence has no content-dependent queries
or keys: choose {lit}`Q` and {lit}`K` as {lit}`seq × 1` matrices of ones.
Then {lit}`Q Kᵀ` is all ones and {lit}`M = ssm_mask α` carries the weights.

Here the attention key matrix {lit}`K` is distinct from the SSM's learned
projection, named {lit}`projection` below. This is the requested formula as a
special case for our current scalar-state-weight recurrence. General learned
queries and keys would describe a different recurrence.

Every output is a matrix-product entry, with no dependence on earlier output
states. This is a mathematical parallel form, not a parallel execution runtime.
-/

def parallel_ssm_layer (α : Int) (input : Sequence seq batch hidden)
    (projection : Matrix hidden hidden) : Sequence seq batch hidden :=
  let Q : Matrix seq 1 := fun _ _ => 1
  let K : Matrix seq 1 := fun _ _ => 1
  let weights := (Q.matmul K.transpose).hadamard (ssm_mask α)
  fun s i =>
    let V : Matrix seq hidden := fun t => projection.transpose.matvec (input t i)
    (weights.matmul V) s

/-!
# 44. Unroll the two scans

These small induction lemmas turn the sequential updates into weighted sums.
An initial state is multiplied by {lit}`α^length`. Each prefix contribution
has coefficient {lit}`α^(s-t)`; each reversed-suffix contribution has
coefficient {lit}`α^(t-s)`. Summing the two lists combines their coefficients
into the mask. All identities hold for arbitrary integer alpha and inputs.
-/

private theorem scan_initial (α : Int) (xs : List Int) (z : Int) :
    xs.foldl (fun state x => α * state + x) z =
      α ^ xs.length * z + xs.foldl (fun state x => α * state + x) 0 := by
  induction xs generalizing z with
  | nil => simp
  | cons x xs ih =>
      simp only [List.foldl_cons, List.length_cons, Int.mul_zero, Int.zero_add]
      rw [ih (α * z + x), ih x]
      simp [Int.pow_succ, Int.mul_add, Int.mul_assoc, Int.add_assoc]

private theorem sum_mapIdx_zero (xs : List Int) :
    (xs.mapIdx fun _ _ => (0 : Int)).sum = 0 := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simpa using ih

private theorem sum_mapIdx_scale (α : Int) (xs : List Int) (f : Nat → Int → Int) :
    (xs.mapIdx fun i x => α * f i x).sum = α * (xs.mapIdx f).sum := by
  induction xs generalizing f with
  | nil => simp
  | cons x xs ih => simp [ih, Int.mul_add]

private theorem prefix_scan (α : Int) (xs : List Int) (s : Nat) (h : s < xs.length) :
    (xs.take (s + 1)).foldl (fun state x => α * state + x) 0 =
      (xs.mapIdx fun t x => (if t ≤ s then α ^ (s - t) else 0) * x).sum := by
  induction xs generalizing s with
  | nil => simp at h
  | cons x xs ih =>
      cases s with
      | zero => simp [sum_mapIdx_zero]
      | succ s =>
          have hs : s < xs.length := by simpa using h
          simp only [List.take_succ_cons, List.foldl_cons, Int.mul_zero, Int.zero_add]
          rw [scan_initial, ih s hs]
          simp [List.length_take, Nat.min_eq_left (by omega : s + 1 ≤ xs.length)]

private theorem reverse_scan (α : Int) (xs : List Int) :
    xs.reverse.foldl (fun state x => α * state + x) 0 =
      (xs.mapIdx fun t x => α ^ t * x).sum := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
      simp only [List.reverse_cons, List.foldl_append, List.foldl_cons, List.foldl_nil, ih,
        List.mapIdx_cons, List.sum_cons, Int.pow_zero, Int.one_mul]
      simp only [Int.pow_succ', Int.mul_assoc, sum_mapIdx_scale]
      exact Int.add_comm _ _

private theorem suffix_scan (α : Int) (xs : List Int) (s : Nat) :
    (xs.drop s).reverse.foldl (fun state x => α * state + x) 0 =
      (xs.mapIdx fun t x => (if s ≤ t then α ^ (t - s) else 0) * x).sum := by
  induction xs generalizing s with
  | nil => simp
  | cons x xs ih =>
      cases s with
      | zero => simpa using reverse_scan α (x :: xs)
      | succ s => simpa using ih s

private theorem sum_mapIdx_add (xs : List Int) (f g : Nat → Int → Int) :
    (xs.mapIdx f).sum + (xs.mapIdx g).sum =
      (xs.mapIdx fun i x => f i x + g i x).sum := by
  induction xs generalizing f g with
  | nil => rfl
  | cons x xs ih => simp [← ih, Int.add_assoc, Int.add_left_comm]

/-!
# 45. Prove the matrix product equals the recurrence

At an arbitrary sequence, batch, and hidden index, expand the matrix product.
The scan lemmas replace the forward and backward folds with sums. The mask's
entry is exactly the sum of the two coefficients, so every term agrees.

This is equality of the complete layers, for every shape, projection, input,
and integer alpha. The parallel layer may therefore replace the recurrence
before any of our transformer blocks or loss functions.
-/

theorem parallel_ssm_layer_eq (α : Int) (input : Sequence seq batch hidden)
    (projection : Matrix hidden hidden) :
    parallel_ssm_layer α input projection =
      alpha_bidirectional_ssm_layer α input projection := by
  funext s i j
  let values := List.ofFn fun t => projection.transpose.matvec (input t i) j
  simp only [parallel_ssm_layer, Matrix.hadamard, Matrix.matmul, Matrix.transpose,
    List.ofFn_succ, List.ofFn_zero, List.sum_cons, List.sum_nil, Int.one_mul, Int.add_zero]
  change (List.ofFn fun t => ssm_mask α s t * projection.transpose.matvec (input t i) j).sum =
    (values.take (s.val + 1)).foldl (fun state x => α * state + x) 0 +
      (values.drop s.val).reverse.foldl (fun state x => α * state + x) 0
  rw [prefix_scan α values s.val (by simp [values]), suffix_scan, sum_mapIdx_add]
  apply congrArg List.sum
  apply List.ext_getElem
  · simp [values]
  · intro t h₁ h₂
    simp [values, ssm_mask, Int.add_mul]

/-!
Check a non-symmetric projection with multiple batch and hidden entries.
Also check alpha zero and negative one: zero keeps two copies of the current
projection, while negative alpha gives alternating signs with distance.
-/

example :
    let input : Sequence 3 2 2 :=
      fun s i j => 10 * (i.val : Int) + 2 * (s.val : Int) + (j.val : Int) + 1
    let projection : Matrix 2 2 :=
      fun i j => 2 * (i.val : Int) + (j.val : Int) + 1
    parallel_ssm_layer 2 input projection 0 0 0 = 136 ∧
      parallel_ssm_layer 2 input projection 2 1 1 = 632 := by
  decide

example :
    let input : Sequence 3 1 1 := fun s _ _ => (s.val : Int) + 1
    let projection : Matrix 1 1 := fun _ _ => 1
    parallel_ssm_layer 0 input projection 1 0 0 = 4 ∧
      parallel_ssm_layer (-1) input projection 0 0 0 = 3 ∧
      parallel_ssm_layer (-1) input projection 2 0 0 = 5 := by
  decide

/-!
# 46. Exercise: attention a tile at a time

Can we compute the same output as {name}`attention_layer` while visiting only
{lit}`n` key positions at a time, without constructing a full row of logits
or transformed attention weights? Assume {lit}`0 < n`, and allow a shorter
last tile. Prove equality for every input, parameter choice, and tile size.

For our discrete softmax, write {lit}`rₜ = relu(q · kₜ)` and
{lit}`R = Σₜ rₜ`. The output is {lit}`Σₜ (rₜ - R) vₜ`.
Which running totals are sufficient to compute it? The total {lit}`R` spans
all keys: applying the discrete softmax independently inside each tile would
give a different answer.

The following implementation and proof solve this exercise. It models a
FlashAttention-style tiled computation for our discrete rule, not an
exponential-softmax GPU kernel.
-/

/-!
# 47. Visit the keys in tiles

Take up to {lit}`n` keys, update the running state, and continue with the rest.
The positive-size hypothesis ensures that recursion makes progress. No padding
is used, so a short final tile contributes exactly its actual keys.
The proof says that grouping visits into tiles gives the same fold.
-/

def tile_fold (n : Nat) (positive : 0 < n) (step : σ → α → σ)
    (xs : List α) (state : σ) : σ :=
  match xs with
  | [] => state
  | x :: rest => tile_fold n positive step ((x :: rest).drop n)
      (((x :: rest).take n).foldl step state)
termination_by xs.length
decreasing_by simp_wf; omega

theorem tile_fold_eq (n : Nat) (positive : 0 < n) (step : σ → α → σ)
    (xs : List α) (state : σ) :
    tile_fold n positive step xs state = xs.foldl step state := by
  cases xs with
  | nil => simp [tile_fold]
  | cons x rest =>
      rw [tile_fold, tile_fold_eq]
      rw [← List.foldl_append, List.take_append_drop]
termination_by xs.length
decreasing_by simp_wf; omega


/-!
# 48. Accumulate three totals per query

Keep the scalar {lit}`R = Σ rₜ`, and two hidden vectors:
{lit}`P = Σ rₜ vₜ` and {lit}`U = Σ vₜ`. Each key updates them once.
The final answer is {lit}`P - R * U`, entry by entry.

The query/key dot product and ReLU are computed as the key is visited.
There is no full score-row or score-matrix construction. The implementation
tracks mathematical values; we do not claim a measured memory or speed bound
for Lean's function-based tensor representation.
-/

structure FlashStats (hidden : Nat) where
  scoreSum : Int
  weighted : Tensor hidden
  values : Tensor hidden

def flash_step (r : α → Int) (v : α → Tensor hidden)
    (state : FlashStats hidden) (t : α) : FlashStats hidden :=
  let score := r t
  { scoreSum := state.scoreSum + score
    weighted := fun j => state.weighted j + score * v t j
    values := fun j => state.values j + v t j }


def flash_attention (n : Nat) (positive : 0 < n) (input : Sequence seq batch hidden)
    (wq wk wv : Matrix hidden hidden) (bq bk bv : Tensor hidden) :
    Sequence seq batch hidden :=
  let q := sequence_layer input wq bq
  let k := sequence_layer input wk bk
  let v := sequence_layer input wv bv
  fun s i =>
    let r := fun t => relu ((List.ofFn fun d => q s i d * k t i d).sum)
    let initial : FlashStats hidden := ⟨0, fun _ => 0, fun _ => 0⟩
    let state := tile_fold n positive (flash_step r (fun t => v t i))
      (List.ofFn fun t : Fin seq => t) initial
    fun j => state.weighted j - state.scoreSum * state.values j


/-!
# 49. Prove tiled attention is regular attention

First prove that the accumulated fields equal the three sums. Then distribute
the subtraction in {lit}`Σ (rₜ - R) vₜ`. Combined with the tile-fold theorem,
these identities prove equality of the whole attention layers, including
empty sequences and every positive tile size.
-/

private theorem flash_fold (xs : List α) (r : α → Int) (v : α → Tensor hidden)
    (state : FlashStats hidden) :
    xs.foldl (flash_step r v) state =
      { scoreSum := state.scoreSum + (xs.map r).sum
        weighted := fun j => state.weighted j + (xs.map fun t => r t * v t j).sum
        values := fun j => state.values j + (xs.map fun t => v t j).sum } := by
  induction xs generalizing state with
  | nil => cases state; simp
  | cons x xs ih => simp [List.foldl_cons, ih, flash_step, Int.add_assoc]

private theorem sum_attention (xs : List α) (r v : α → Int) (total : Int) :
    (xs.map fun t => (r t - total) * v t).sum =
      (xs.map fun t => r t * v t).sum - total * (xs.map v).sum := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
      simp only [List.map_cons, List.sum_cons]
      rw [ih]
      simp only [Int.sub_mul, Int.mul_add]
      omega


theorem flash_attention_eq (n : Nat) (positive : 0 < n)
    (input : Sequence seq batch hidden) (wq wk wv : Matrix hidden hidden)
    (bq bk bv : Tensor hidden) :
    flash_attention n positive input wq wk wv bq bk bv =
      attention_layer input wq wk wv bq bk bv := by
  funext s i j
  simp only [flash_attention, tile_fold_eq, flash_fold, List.map_ofFn,
    Int.zero_add]
  let q := sequence_layer input wq bq
  let k := sequence_layer input wk bk
  let v := sequence_layer input wv bv
  let r := fun t => relu ((List.ofFn fun d => q s i d * k t i d).sum)
  simpa only [List.map_ofFn, Function.comp_def, attention_layer, discrete_softmax, Matrix.matmul,
    Matrix.transpose, q, k, v, r] using
    (sum_attention (List.ofFn fun t : Fin seq => t) r (fun t => v t i j)
      (List.ofFn r).sum).symm

/-!
Check single-key tiles, a size-two tile with a final singleton, and a tile
larger than the sequence. The logits include both positive and negative
entries, and the batch items have different inputs.
-/

example :
    let input : Sequence 3 2 2 :=
      fun s i j => 2 * (s.val : Int) + 3 * (i.val : Int) + (j.val : Int) - 2
    let identity : Matrix 2 2 := fun i j => if i.val = j.val then 1 else 0
    let swap : Matrix 2 2 := fun i j => if i.val = j.val then 0 else 1
    let zero : Tensor 2 := fun _ => 0
    let bias : Tensor 2 := fun j => if j.val = 0 then 1 else -1
    let run := fun n h => flash_attention n h input identity swap identity zero zero bias
    run 1 (by decide) 2 1 1 = -596 ∧
      run 2 (by decide) 2 1 1 = -596 ∧
      run 4 (by decide) 2 1 1 = -596 ∧
      run 2 (by decide) 0 0 0 = -16 := by
  simp only [flash_attention, List.ofFn_succ, List.ofFn_zero, tile_fold,
    List.take, List.drop]
  decide
