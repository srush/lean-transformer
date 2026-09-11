import Std

set_option doc.verso true

/-!
# Integer tensors: a small verified model

Read this file from top to bottom: tensors → matrices → layers → sequence
models → exercises. Each operation is a pure function on fixed-size integer
arrays. There is no mutation, broadcasting, floating point, or runtime
parallelism. Shapes are checked by Lean; equalities describe exact arithmetic.

The exercises have complete solutions below their statements. Try replacing a
proof with a tactic hole in your editor before reading the solution.
-/

/-!
# 1. Tensors and pointwise arithmetic

A tensor of length {lit}`n` is a function from {lit}`Fin n` to integers.
{lit}`Fin n` contains precisely the indices from zero through {lit}`n - 1`;
an out-of-bounds index cannot be supplied.

Addition and multiplication act independently at each index. To prove two
tensors equal, {lit}`funext i` reduces the goal to equality at an arbitrary
index. Our first proof is just integer distributivity at that index.
-/

def Tensor (n : Nat) := Fin n → Int

instance : Add (Tensor n) where
  add a b := fun i => a i + b i

instance : Mul (Tensor n) where
  mul a b := fun i => a i * b i

theorem Tensor.mul_add (a b c : Tensor n) : a * (b + c) = a * b + a * c := by
  funext i
  exact Int.mul_add (a i) (b i) (c i)

example :
    let x : Tensor 3 := fun i => (i.val : Int) + 1
    (x + x) 2 = 6 ∧ (x * x) 2 = 9 := by decide

/-!
# 2. Matrices and reductions

A matrix is a tensor-valued function: one hidden vector per row.
{lit}`List.ofFn f` enumerates {lit}`f 0, f 1, …` over its finite domain;
{lit}`.sum` adds those entries, returning zero for an empty domain.

Matrix multiplication sums over the shared dimension. Transpose swaps indices,
matrix-vector multiplication computes one dot product per row, and Hadamard
multiplication is pointwise. We define these operations once here and reuse
them throughout the sequence models.
-/

def Matrix (n m : Nat) := Fin n → Tensor m

def Matrix.matmul (a : Matrix n m) (b : Matrix m p) : Matrix n p :=
  fun i j => (List.ofFn fun k : Fin m => a i k * b k j).sum

def Matrix.transpose (a : Matrix n m) : Matrix m n := fun i j => a j i

def Matrix.matvec (a : Matrix n m) (x : Tensor m) : Tensor n :=
  fun i => (List.ofFn fun j => a i j * x j).sum

def Matrix.hadamard (a b : Matrix n m) : Matrix n m := fun i j => a i j * b i j

namespace Examples

def a : Matrix 2 3 := fun i j => 3 * (i.val : Int) + (j.val : Int) + 1
def b : Matrix 3 2 := fun i j => 2 * (i.val : Int) + (j.val : Int) + 1

end Examples

example : Examples.a.matmul Examples.b 0 0 = 22 ∧
    Examples.a.matmul Examples.b 1 1 = 64 := by decide

example (a : Matrix n 0) (b : Matrix 0 p) (i : Fin n) (j : Fin p) :
    (a.matmul b) i j = 0 := by rfl

/-!
# 3. Exercise: tensor-parallel matrix multiplication

Split only an even dimension: the argument {lit}`h` certifies evenness,
and the result types guarantee two equally sized halves. The pair's
{lit}`.1` and {lit}`.2` fields select the first and second halves.

To parallelize {lit}`A B`, split the columns of {lit}`A` and the rows of
{lit}`B`, compute the two products separately, then add them. This is a
split of the reduction dimension, not a split-and-rejoin identity.

Exercise: prove that the result equals ordinary matrix multiplication.
The solution rewrites the shared dimension as two equal halves and uses
the fact that the sum of an appended list is the sum of its two parts.
-/

def Matrix.row_split (a : Matrix n m) (h : n % 2 = 0) :
    Matrix (n / 2) m × Matrix (n / 2) m := by
  have halves : n / 2 + n / 2 = n := by omega
  exact ⟨fun i j => a ⟨i.val, by have := i.isLt; omega⟩ j,
         fun i j => a ⟨n / 2 + i.val, by have := i.isLt; omega⟩ j⟩

def Matrix.column_split (a : Matrix n m) (h : m % 2 = 0) :
    Matrix n (m / 2) × Matrix n (m / 2) := by
  have halves : m / 2 + m / 2 = m := by omega
  exact ⟨fun i j => a i ⟨j.val, by have := j.isLt; omega⟩,
         fun i j => a i ⟨m / 2 + j.val, by have := j.isLt; omega⟩⟩

def Matrix.tensor_parallel (a : Matrix n m) (b : Matrix m p) (h : m % 2 = 0) :
    Matrix n p :=
  let (a₁, a₂) := a.column_split h
  let (b₁, b₂) := b.row_split h
  let c₁ := a₁.matmul b₁
  let c₂ := a₂.matmul b₂
  fun i j => c₁ i j + c₂ i j

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

example : (Examples.b.tensor_parallel Examples.a (by decide)) 0 0 = 9 := by rfl

/-!
# 4. Layers, loss, and data parallelism

An affine layer takes a {lit}`batch × hidden` input, a
{lit}`hidden × hidden` weight matrix, and a length-{lit}`hidden` bias.
It applies the same matrix and bias to each row; it has no activation.

The loss sums {lit}`score i (matrix i)`. The scorer receives the original
batch index and the full hidden vector, so it may depend on either.

Exercise: split an even batch, score both shards, and add their losses.
The second shard must translate its local indices back to original batch
indices. The proof again reduces to splitting a finite sum.
-/

def layer (input : Matrix batch hidden) (weight : Matrix hidden hidden)
    (bias : Tensor hidden) : Matrix batch hidden :=
  let product := input.matmul weight
  fun i j => product i j + bias j

def loss (score : Fin batch → Tensor hidden → Int)
    (matrix : Matrix batch hidden) : Int :=
  (List.ofFn fun i => score i (matrix i)).sum

def data_parallel_loss (score : Fin batch → Tensor hidden → Int)
    (matrix : Matrix batch hidden) (h : batch % 2 = 0) : Int :=
  let (first, second) := matrix.row_split h
  loss (fun i row => score ⟨i.val, by have := i.isLt; omega⟩ row) first +
  loss (fun i row => score ⟨batch / 2 + i.val, by have := i.isLt; omega⟩ row) second

theorem data_parallel_loss_correct (score : Fin batch → Tensor hidden → Int)
    (matrix : Matrix batch hidden) (h : batch % 2 = 0) :
    data_parallel_loss score matrix h = loss score matrix := by
  obtain ⟨k, rfl⟩ : ∃ k, batch = k + k := ⟨batch / 2, by omega⟩
  have half : (k + k) / 2 = k := by omega
  simp [data_parallel_loss, loss, Matrix.row_split, half,
    List.ofFn_add, List.sum_append, Fin.castLE, Fin.addNat, Nat.add_comm]
  congr 1 <;> apply congrArg List.sum <;> apply List.ext_getElem
  all_goals simp [half]

example :
    let score : Fin 2 → Tensor 3 → Int := fun i row => (i.val : Int) + row 0 * row 2
    loss score Examples.a = 28 ∧ data_parallel_loss score Examples.a (by decide) = 28 := by
  decide

/-!
# 5. Sequences and discrete attention

A sequence has shape {lit}`seq × batch × hidden`. The affine layer
acts independently at every position. At the end, sum the hidden vectors over
sequence positions, score that pooled vector once per example, and sum over
the batch. The scorer may be nonlinear: pooling happens before scoring. Compose these functions directly instead of introducing a
new loss wrapper for each kind of layer.
-/

def Sequence (seq batch hidden : Nat) := Fin seq → Matrix batch hidden

def sequence_layer (input : Sequence seq batch hidden)
    (weight : Matrix hidden hidden) (bias : Tensor hidden) :
    Sequence seq batch hidden :=
  fun s => layer (input s) weight bias

def Sequence.sum_seq (output : Sequence seq batch hidden) : Matrix batch hidden :=
  fun i j => (List.ofFn fun s => output s i j).sum

def sequence_loss (score : Fin batch → Tensor hidden → Int)
    (output : Sequence seq batch hidden) : Int :=
  loss score output.sum_seq

example :
    let input : Sequence 2 2 1 := fun s i _ => (s.val : Int) + (i.val : Int)
    let output := sequence_layer input (fun _ _ => 2) (fun _ => 1)
    sequence_loss (fun _ row => row 0 * row 0) output = 80 := by decide

/-!
# Rectified scores and Q/K/V

Integer ReLU clips negative values to zero. Our deliberately nonstandard
“discrete softmax” is {lit}`relu(zᵢ) - Σⱼ relu(zⱼ)`, with the current
entry included in the sum. These weights are nonpositive, not probabilities;
a singleton row becomes zero.

Attention projects the input to Q, K, and V with three affine layers.
Within each batch item it computes {lit}`Q Kᵀ`, transforms each row with
the discrete rule, and multiplies by V. There is no positional encoding,
mask, scaling, or output projection.
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
# 6. Stacking blocks

Each block performs an affine layer followed by a sequence-mixing layer.
A single traversal handles the stack: ordinary attention is one choice of
mixer, and sliding-window attention will be another. An empty stack is the
identity. Only the final output is passed to the sequence loss.
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

def run_blocks
    (mix : TransformerBlock hidden → Sequence seq batch hidden → Sequence seq batch hidden)
    (blocks : List (TransformerBlock hidden)) (input : Sequence seq batch hidden) :
    Sequence seq batch hidden :=
  match blocks with
  | [] => input
  | block :: rest =>
      run_blocks mix rest (mix block (sequence_layer input block.weight block.bias))

def transformer (blocks : List (TransformerBlock hidden))
    (input : Sequence seq batch hidden) : Sequence seq batch hidden :=
  run_blocks (fun block output =>
    attention_layer output block.wq block.wk block.wv block.bq block.bk block.bv) blocks input

def transformer_loss (blocks : List (TransformerBlock hidden))
    (score : Fin batch → Tensor hidden → Int)
    (input : Sequence seq batch hidden) : Int :=
  sequence_loss score (transformer blocks input)

/-!
# 7. Exercise: permutation invariance

A position permutation reorders every sequence index exactly once.
{lit}`input.permute π` reads old position {lit}`π.index s` at new
position {lit}`s`.

Exercise: show that a transformer follows this reordering, and that its
loss is unchanged with the same scorer: summing the sequence removes its order. The key lemma says
that a permutation does not change a finite sum. Apply it first to the
discrete score transformation, then to attention, and finally induct over
blocks. Outputs are equivariant (reordered); the scalar loss is invariant.
-/

structure PositionPermutation (seq : Nat) where
  index : Fin seq → Fin seq
  valid : (List.ofFn index).Perm (List.ofFn fun i : Fin seq => i)

def Sequence.permute (π : PositionPermutation seq) (input : Sequence seq batch hidden) :
    Sequence seq batch hidden :=
  fun s => input (π.index s)

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

theorem transformer_permute (π : PositionPermutation seq)
    (blocks : List (TransformerBlock hidden)) (input : Sequence seq batch hidden) :
    transformer blocks (input.permute π) = (transformer blocks input).permute π := by
  induction blocks generalizing input with
  | nil => rfl
  | cons block rest ih =>
      change transformer rest
        (attention_layer (sequence_layer (input.permute π) block.weight block.bias)
          block.wq block.wk block.wv block.bq block.bk block.bv) = _
      rw [sequence_layer_permute, attention_layer_permute, ih]
      rfl

theorem transformer_loss_position_invariant (π : PositionPermutation seq)
    (blocks : List (TransformerBlock hidden))
    (score : Fin batch → Tensor hidden → Int)
    (input : Sequence seq batch hidden) :
    transformer_loss blocks score (input.permute π) =
      transformer_loss blocks score input := by
  unfold transformer_loss sequence_loss
  rw [transformer_permute]
  congr 1
  funext i j
  exact π.sum (fun s => transformer blocks input s i j)

/-!
# Exercise: batch invariance

Running an example alongside other examples must not change its output.
Select any batch indices and run the same transformer: the result is the
same as selecting those indices after running the full batch. This works
because neither affine layers nor attention mix different batch items.

Consequently a batch of size {lit}`n + 1` has the same loss as its first
{lit}`n` examples plus its last example, each run separately through the
whole model. This is the requested {lit}`N = (N - 1) + 1` split, written
without a subtraction or positivity proof. It includes a singleton batch:
the first shard is then empty. Unlike the earlier equal-halves exercise,
there is no evenness restriction.

Scores retain their original batch indices, and may be arbitrary nonlinear
functions of the sequence-summed hidden vector.
-/

def Sequence.select_batch (pick : Fin small → Fin batch)
    (input : Sequence seq batch hidden) : Sequence seq small hidden :=
  fun s i => input s (pick i)

theorem transformer_select_batch (pick : Fin small → Fin batch)
    (blocks : List (TransformerBlock hidden)) (input : Sequence seq batch hidden) :
    transformer blocks (input.select_batch pick) =
      (transformer blocks input).select_batch pick := by
  induction blocks generalizing input with
  | nil => rfl
  | cons block rest ih =>
      change transformer rest
        ((attention_layer (sequence_layer input block.weight block.bias)
          block.wq block.wk block.wv block.bq block.bk block.bv).select_batch pick) = _
      exact ih _

theorem transformer_loss_batch_split (blocks : List (TransformerBlock hidden))
    (score : Fin (n + 1) → Tensor hidden → Int)
    (input : Sequence seq (n + 1) hidden) :
    transformer_loss blocks score input =
      transformer_loss blocks (fun i => score i.castSucc)
        (input.select_batch fun i : Fin n => i.castSucc) +
      transformer_loss blocks (fun _ : Fin 1 => score (Fin.last n))
        (input.select_batch fun _ : Fin 1 => Fin.last n) := by
  unfold transformer_loss
  rw [transformer_select_batch, transformer_select_batch]
  simp only [sequence_loss, loss,
    List.ofFn_succ_last, List.sum_append, List.sum_cons, List.sum_nil,
    List.ofFn_zero, Int.add_zero, Int.zero_add]
  rfl

/-!
# Shared small examples

The remaining checks reuse scalar projections, a zero-bias block, and a
ramp input. The scorer receives a sequence-summed hidden vector; it no longer
has a sequence-position argument.
-/

namespace Examples

def one : Matrix 1 1 := fun _ _ => 1
def zero : Tensor 1 := fun _ => 0
def block : TransformerBlock 1 :=
  ⟨one, zero, one, one, one, zero, zero, zero⟩
def ramp (seq : Nat) : Sequence seq 1 1 := fun s _ _ => (s.val : Int) + 1
def score : Fin 1 → Tensor 1 → Int := fun _ row => row 0
def swap2 : PositionPermutation 2 :=
  ⟨fun s => ⟨1 - s.val, by omega⟩, by decide⟩
def swap3 : PositionPermutation 3 :=
  ⟨fun s => if s.val < 2 then ⟨1 - s.val, by omega⟩ else s, by decide⟩
def rotate3 : PositionPermutation 3 :=
  ⟨fun s => ⟨(s.val + 1) % 3, by omega⟩, by decide⟩

end Examples

-- Nonlinear scoring after pooling, with original batch indices on both shards.
example :
    let input : Sequence 2 3 1 := fun s i _ => (s.val : Int) + (i.val : Int) + 1
    let score : Fin 3 → Tensor 1 → Int := fun i row => row 0 * row 0 + (i.val : Int)
    transformer_loss [Examples.block] score input = 31971 ∧
      transformer_loss [Examples.block] (fun i => score i.castSucc)
        (input.select_batch fun i : Fin 2 => i.castSucc) = 3745 ∧
      transformer_loss [Examples.block] (fun _ : Fin 1 => score 2)
        (input.select_batch fun _ : Fin 1 => 2) = 28226 := by decide

-- Empty sequences pool to zero, but an arbitrary scorer need not score zero as zero.
example :
    let input : Sequence 0 3 1 := fun _ _ _ => 0
    transformer_loss [Examples.block] (fun _ row => row 0 + 1) input = 3 := by decide

example :
    let block := { Examples.block with bias := fun _ => 1 }
    let input := Examples.ramp 3
    transformer_loss [block, block] Examples.score input =
      transformer_loss [block, block] Examples.score
        (input.permute Examples.rotate3) := by decide

/-!
# 8. Exercise: a sliding-window receptive field

A window includes positions at most {lit}`radius` steps to either side,
without wrapping at the sequence boundaries. This SWA variant uses raw
Q/K dot products, not the discrete softmax defined above.

Exercise: prove that after {lit}`d` blocks, an output can depend only on
inputs within radius {lit}`d * radius` in the same batch item.
First prove that affine layers are pointwise and one SWA layer is local;
then induct over the shared block traversal. Each block adds one radius.

The locality theorem concerns one output hidden vector, before pooling.
The final loss pools all positions, so its receptive field is generally the
whole sequence; there is no longer a separate loss at each position.
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

def swa_transformer (radius : Nat) (blocks : List (TransformerBlock hidden))
    (input : Sequence seq batch hidden) : Sequence seq batch hidden :=
  run_blocks (fun block output =>
    swa radius output block.wq block.wk block.wv block.bq block.bk block.bv) blocks input

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

example :
    let one := Examples.one
    let zero := Examples.zero
    let block := Examples.block
    let input : Sequence 5 1 1 := fun _ _ _ => 1
    let far : Sequence 5 1 1 := fun s _ _ => if s.val = 3 then 2 else 1
    let boundary : Sequence 5 1 1 := fun s _ _ => if s.val = 2 then 2 else 1
    swa_transformer 1 [block, block] input 0 0 0 = 26 ∧
      swa_transformer 1 [block, block] far 0 0 0 = 26 ∧
      swa_transformer 1 [block, block] boundary 0 0 0 = 80 ∧
      swa_transformer 0 [block, block] far 0 0 0 = 1 ∧
      swa 5 input one one one zero zero zero 0 0 0 = 5 := by
  decide

/-!
# 9. One recurrence, two scan directions

An SSM uses {lit}`state' = α * state + Kᵀ x`, starting from zero.
The scalar scan below is shared by both layers and all choices of integer
{lit}`α`. The forward layer scans each prefix. The bidirectional layer
adds that prefix scan to a reverse suffix scan.

Both directions include the current input, so it is counted twice.
There is no second implementation for the unweighted case: set {lit}`α = 1`.
We use lists to describe the recurrence mathematically, not to promise an
efficient implementation of every prefix.
-/

def ssm_scan (α : Int) (values : List Int) : Int :=
  values.foldl (fun state x => α * state + x) 0

def ssm_layer (α : Int) (input : Sequence seq batch hidden)
    (K : Matrix hidden hidden) : Sequence seq batch hidden :=
  fun s i j =>
    let values := List.ofFn fun t => K.transpose.matvec (input t i) j
    ssm_scan α (values.take (s.val + 1))

def bidirectional_ssm_layer (α : Int) (input : Sequence seq batch hidden)
    (K : Matrix hidden hidden) : Sequence seq batch hidden :=
  fun s i j =>
    let values := List.ofFn fun t => K.transpose.matvec (input t i) j
    ssm_scan α (values.take (s.val + 1)) + ssm_scan α (values.drop s.val).reverse

/-!
# When does order matter?

The forward scan depends on which inputs came before a position, even
when {lit}`α = 1`. The first counterexample places it before a nonempty
transformer and compares the final pooled losses.

For bidirectional {lit}`α = 1`, every output is the total projected
input plus the current projection. Prove this identity, then reuse the
permutation-sum lemma and the transformer theorem to obtain loss invariance.

For {lit}`α = 2`, distance-dependent weights break that guarantee.
These are counterexamples to invariance in general, not claims that every
input or parameter choice changes the loss.
-/

theorem ssm_breaks_position_invariance :
    let input := Examples.ramp 2
    let π := Examples.swap2
    transformer_loss [Examples.block] Examples.score (ssm_layer 1 input Examples.one) ≠
      transformer_loss [Examples.block] Examples.score
        (ssm_layer 1 (input.permute π) Examples.one) := by decide

theorem bidirectional_ssm_layer_eq (input : Sequence seq batch hidden)
    (K : Matrix hidden hidden) (s : Fin seq) (i : Fin batch) (j : Fin hidden) :
    bidirectional_ssm_layer 1 input K s i j =
      (List.ofFn fun t => K.transpose.matvec (input t i) j).sum +
        K.transpose.matvec (input s i) j := by
  let values := List.ofFn fun t => K.transpose.matvec (input t i) j
  have bound : s.val < values.length := by simp [values]
  have split := congrArg List.sum (List.take_append_drop s.val values)
  simp only [List.sum_append] at split
  simp only [bidirectional_ssm_layer, ssm_scan, Int.one_mul, ← List.sum_eq_foldl]
  change (values.take (s.val + 1)).sum + (values.drop s.val).reverse.sum =
    values.sum + K.transpose.matvec (input s i) j
  rw [List.sum_reverse_int, List.take_succ_eq_append_getElem bound, List.sum_append]
  simp only [List.sum_cons, List.sum_nil, Int.add_zero]
  have current : values[s.val] = K.transpose.matvec (input s i) j := by simp [values]
  rw [current]
  omega

theorem bidirectional_ssm_layer_permute (π : PositionPermutation seq)
    (input : Sequence seq batch hidden) (K : Matrix hidden hidden) :
    bidirectional_ssm_layer 1 (input.permute π) K =
      (bidirectional_ssm_layer 1 input K).permute π := by
  funext s i j
  simp only [Sequence.permute, bidirectional_ssm_layer_eq]
  exact congrArg (fun total => total + K.transpose.matvec (input (π.index s) i) j)
    (π.sum (fun t => K.transpose.matvec (input t i) j))

theorem bidirectional_ssm_loss_position_invariant (π : PositionPermutation seq)
    (blocks : List (TransformerBlock hidden))
    (score : Fin batch → Tensor hidden → Int)
    (input : Sequence seq batch hidden) (K : Matrix hidden hidden) :
    transformer_loss blocks score
        (bidirectional_ssm_layer 1 (input.permute π) K) =
      transformer_loss blocks score (bidirectional_ssm_layer 1 input K) := by
  rw [bidirectional_ssm_layer_permute]
  exact transformer_loss_position_invariant π blocks score (bidirectional_ssm_layer 1 input K)

example : bidirectional_ssm_layer 1 (Examples.ramp 2) Examples.one 0 0 0 = 4 ∧
    bidirectional_ssm_layer 1 (Examples.ramp 2) Examples.one 1 0 0 = 5 := by decide

theorem alpha_ssm_breaks_position_invariance :
    let input := Examples.ramp 3
    let π := Examples.swap3
    transformer_loss [Examples.block] Examples.score
        (bidirectional_ssm_layer 2 input Examples.one) ≠
      transformer_loss [Examples.block] Examples.score
        (bidirectional_ssm_layer 2 (input.permute π) Examples.one) := by decide

/-!
# 10. Exercise: the parallel form of the SSM

Exercise: express the bidirectional recurrence as {lit}`(Q Kᵀ ⊙ M) V`.
Here Q and the attention key matrix are columns of ones; V contains the
projected inputs. The mask is {lit}`α^distance` off the diagonal and two
on the diagonal because both scans include the current position. The
attention key matrix is distinct from the learned SSM projection.

The proof unrolls a scan into a weighted sum, does the same for the reversed
suffix, and adds the two sums. The private list lemmas below carry out that
algebra; the public equality at the end is the exercise's result.
It holds for every integer {lit}`α`, including zero and negative values.
-/

def ssm_mask (α : Int) : Matrix seq seq := fun s t =>
  (if t.val ≤ s.val then α ^ (s.val - t.val) else 0) +
  (if s.val ≤ t.val then α ^ (t.val - s.val) else 0)

def parallel_ssm_layer (α : Int) (input : Sequence seq batch hidden)
    (projection : Matrix hidden hidden) : Sequence seq batch hidden :=
  let Q : Matrix seq 1 := fun _ _ => 1
  let K : Matrix seq 1 := fun _ _ => 1
  let weights := (Q.matmul K.transpose).hadamard (ssm_mask α)
  fun s i =>
    let V : Matrix seq hidden := fun t => projection.transpose.matvec (input t i)
    (weights.matmul V) s

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

theorem parallel_ssm_layer_eq (α : Int) (input : Sequence seq batch hidden)
    (projection : Matrix hidden hidden) :
    parallel_ssm_layer α input projection =
      bidirectional_ssm_layer α input projection := by
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
# 11. Final exercise: tiled attention

Compute the same discrete attention while visiting at most {lit}`n` keys
at a time per query. Require {lit}`n > 0`; the final tile may be shorter.

Write {lit}`rₜ = relu(q · kₜ)`. Instead of storing all attention weights,
accumulate three quantities across tiles: {lit}`R = Σ rₜ`,
{lit}`P = Σ rₜ vₜ`, and {lit}`U = Σ vₜ`. The answer is
{lit}`P - R * U`. In particular, do not replace the global score sum with
a separate normalization inside each tile.

The solution has three steps: show that tiling preserves a fold; establish
the accumulator's finite-sum invariant; distribute subtraction to recover
ordinary attention. This is a FlashAttention-style algebraic exercise for
our discrete rule, not exponential softmax or a GPU performance claim.
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
# Check tile boundaries

The last check covers tiles of size one, a shorter final tile, and a tile
larger than the sequence. It also uses mixed-sign logits and separate batch
items. The theorem above covers all positive tile sizes, not just these
examples.
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
