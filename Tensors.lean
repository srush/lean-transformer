import Std
import MatrixSimps

set_option doc.verso true

namespace TensorPuzzles

attribute [local simp] Rat.add_zero Rat.zero_add

/-!
# Rational vectors: a small verified model
-/

/-!
# 1. Vectors and pointwise arithmetic
-/

abbrev fori {α : Type u} {n : Nat} (f : Fin n → α) : List α := List.ofFn f

def Vector (n : Nat) := Fin n → Rat

instance : Repr (Vector n) where
  reprPrec v p := reprPrec (fori v) p

instance : Add (Vector n) where
  add a b := fun i => a i + b i

instance : Mul (Vector n) where
  mul a b := fun i => a i * b i

def Vector.dot_product (a b : Vector n) : Rat :=
  (fori fun i => a i * b i).sum

theorem Vector.mul_add (a b c : Vector n) : a * (b + c) = a * b + a * c := by
  funext i
  exact Rat.mul_add (a i) (b i) (c i)

#eval
  let x : Vector 3 := fun i => (i.val : Rat) + 1
  (x + x, x * x, x.dot_product x)

#eval (show Vector 0 from fun _ => 0)

/-!
# 2. Matrices and reductions
-/

def Matrix (n m : Nat) := Fin n → Vector m

instance : Repr (Matrix n m) where
  reprPrec a p := reprPrec (fori a) p

instance : Add (Matrix n m) where
  add a b := fun i j => a i j + b i j

instance : Mul (Matrix n m) where
  mul a b := fun i j => a i j * b i j

@[simp] theorem Matrix.add_apply (a b : Matrix n m) (i : Fin n) (j : Fin m) :
    (a + b) i j = a i j + b i j := rfl

@[simp] theorem Matrix.mul_apply (a b : Matrix n m) (i : Fin n) (j : Fin m) :
    (a * b) i j = a i j * b i j := rfl

def Matrix.col (a : Matrix n m) (j : Fin m) : Vector n :=
  fun i => a i j

def Matrix.matmul (a : Matrix n m) (b : Matrix m p) : Matrix n p :=
  fun i j => (a i).dot_product (b.col j)

/-!
![Row equivariance: swapping the rows before matmul gives the same result as swapping the output rows.](site/diagrams/row-equivariance.svg)
-/

theorem Matrix.matmul_row_equivariance (pick : Fin rows → Fin n)
    (a : Matrix n m) (b : Matrix m p) :
    Matrix.matmul (fun i j => a (pick i) j : Matrix rows m) b =
      (fun i j => a.matmul b (pick i) j) := by
  rfl

theorem Matrix.matmul_column_equivariance (pick : Fin cols → Fin p)
    (a : Matrix n m) (b : Matrix m p) :
    a.matmul (fun i j => b i (pick j) : Matrix m cols) =
      (fun i j => a.matmul b i (pick j)) := by
  rfl

def Matrix.transpose (a : Matrix n m) : Matrix m n := fun i j => a j i

def Matrix.matvec (a : Matrix n m) (x : Vector m) : Vector n :=
  fun i => (a i).dot_product x

namespace Examples

def mat_ex1 : Matrix 2 3 := fun i j => 3 * (i.val : Rat) + (j.val : Rat) + 1

def mat_ex2 : Matrix 3 2 := fun i j => 2 * (i.val : Rat) + (j.val : Rat) + 1

end Examples

#eval Examples.mat_ex1.matmul Examples.mat_ex2

#eval
  let swap : Fin 2 → Fin 2 := fun i => ⟨1 - i.val, by omega⟩
  let a := Examples.mat_ex1
  let b := Examples.mat_ex2
  (Matrix.matmul (fun i j => a (swap i) j : Matrix 2 3) b,
    a.matmul (fun i j => b i (swap j) : Matrix 3 2))

#eval (Examples.mat_ex1 + Examples.mat_ex1, Examples.mat_ex1 * Examples.mat_ex1)

theorem Matrix.matmul_zero_inner (a : Matrix n 0) (b : Matrix 0 p) (i : Fin n) (j : Fin p) :
    (a.matmul b) i j = 0 := by rfl

#eval (show Matrix 2 0 from fun _ _ => 0).matmul
  (show Matrix 0 3 from fun _ _ => 0)

/-!
# 3. Exercise: tensor-parallel matrix multiplication

![Tensor parallelism: split A by columns and B by rows, multiply each pair independently, and add the results.](site/diagrams/tensor-parallel.svg)
-/

def Matrix.row_split (a : Matrix (k + k) m) :
    Matrix k m × Matrix k m :=
  ⟨fun i j => a (i.castAdd k) j,
   fun i j => a (i.natAdd k) j⟩

#eval Examples.mat_ex1.row_split (k := 1)

def Matrix.tensor_parallel (a : Matrix n (k + k)) (b : Matrix (k + k) p) :
    Matrix n p :=
  let (a₁, a₂) := a.transpose.row_split
  let (b₁, b₂) := b.row_split
  let c₁ := a₁.transpose.matmul b₁
  let c₂ := a₂.transpose.matmul b₂
  c₁ + c₂

attribute [matrix_simps]
  Matrix.tensor_parallel Matrix.transpose Matrix.row_split Matrix.matmul Matrix.col
  Vector.dot_product List.ofFn_add List.sum_append

theorem Matrix.tensor_parallel_correct
    (a : Matrix n (k + k)) (b : Matrix (k + k) p) :
    a.tensor_parallel b = a.matmul b := by
  funext i j
  simp [matrix_simps, Fin.castAdd, Fin.natAdd, Fin.castLE]

#eval (Examples.mat_ex2.tensor_parallel (k := 1) Examples.mat_ex1,
  Examples.mat_ex2.matmul Examples.mat_ex1)

/-!
# 4a. Layers, ReLU, and neural networks
-/

def layer (input : Matrix batch hidden) (weight : Matrix hidden hidden) :
    Matrix batch hidden :=
  input.matmul weight

def relu (z : Rat) : Rat := max z 0

#eval [relu (-3), relu 0, relu 4]

structure NeuralLayer (hidden : Nat) where
  weight : Matrix hidden hidden

def neural_network (layers : List (NeuralLayer hidden))
    (input : Matrix batch hidden) : Matrix batch hidden :=
  match layers with
  | [] => input
  | next :: rest =>
      let output := layer input next.weight
      neural_network rest (fun b j => relu (output b j))

namespace Examples

def nn_layers : List (NeuralLayer 3) :=
  let identity : Matrix 3 3 := fun i j => if i = j then 1 else 0
  [⟨identity⟩, ⟨identity⟩]

end Examples

#eval neural_network Examples.nn_layers Examples.mat_ex1

theorem nn_batch_independence (pick : Fin small → Fin batch)
    (layers : List (NeuralLayer hidden)) (input : Matrix batch hidden) :
    neural_network layers (fun b j => input (pick b) j) =
      (fun b => neural_network layers input (pick b)) := by
  induction layers generalizing input with
  | nil => rfl
  | cons next rest ih =>
      exact ih (fun b j => relu (layer input next.weight b j))

def loss (point_loss : Fin batch → Vector hidden → Rat)
    (matrix : Matrix batch hidden) : Rat :=
  (fori fun b => point_loss b (matrix b)).sum

/-!
# 4b. Exercise: data-parallel neural networks

![Data parallelism: apply the same network to each batch half, sum point losses using their original batch indices, and add the two losses.](site/diagrams/data-parallel.svg)
-/

def data_parallel_loss (layers : List (NeuralLayer hidden))
    (point_loss : Fin (k + k) → Vector hidden → Rat)
    (input : Matrix (k + k) hidden) : Rat :=
  let (first, second) := input.row_split
  loss (fun b row => point_loss (b.castAdd k) row)
    (neural_network layers first) +
  loss (fun b row => point_loss (b.natAdd k) row)
    (neural_network layers second)

theorem data_parallel_loss_correct (layers : List (NeuralLayer hidden))
    (point_loss : Fin (k + k) → Vector hidden → Rat)
    (input : Matrix (k + k) hidden) :
    data_parallel_loss layers point_loss input =
      loss point_loss (neural_network layers input) := by
  simp [data_parallel_loss, loss, Matrix.row_split, nn_batch_independence,
    List.ofFn_add, List.sum_append, Fin.castAdd, Fin.natAdd, Fin.castLE]

#eval
  let point_loss : Fin 2 → Vector 3 → Rat := fun b row => (b.val : Rat) + row 0 * row 2
  (loss point_loss (neural_network Examples.nn_layers Examples.mat_ex1),
    data_parallel_loss (k := 1) Examples.nn_layers point_loss Examples.mat_ex1)

/-!
# 5. Sequences and discrete attention
-/

def Sequence (seq batch hidden : Nat) := Fin seq → Matrix batch hidden

abbrev Sequence.hidden (input : Sequence seq batch h) (s : Fin seq)
    (b : Fin batch) : Vector h :=
  input s b

structure PositionPermutation (seq : Nat) where
  index : Fin seq → Fin seq
  valid : (fori index).Perm (fori fun i : Fin seq => i)

def Sequence.permute {hidden : Nat} (π : PositionPermutation seq) (input : Sequence seq batch hidden) :
    Sequence seq batch hidden :=
  fun s => input (π.index s)

def Sequence.select_batch {hidden : Nat} (pick : Fin small → Fin batch)
    (input : Sequence seq batch hidden) : Sequence seq small hidden :=
  fun s b => input s (pick b)

instance : Repr (Sequence seq batch hidden) where
  reprPrec x p := reprPrec (fori x) p

def sequence_layer (input : Sequence seq batch hidden)
    (weight : Matrix hidden hidden) :
    Sequence seq batch hidden :=
  fun s => layer (input s) weight

theorem sequence_layer_local (input other : Sequence seq batch hidden)
    (weight : Matrix hidden hidden)
    (s : Fin seq) (b : Fin batch) (agree : input.hidden s b = other.hidden s b) :
    (sequence_layer input weight).hidden s b = (sequence_layer other weight).hidden s b := by
  funext j
  simpa only [Sequence.hidden, sequence_layer, layer, Matrix.matmul, Vector.dot_product]
    using congrArg (fun row => row.dot_product (weight.col j)) agree

def Sequence.sum_seq {hidden : Nat} (output : Sequence seq batch hidden) : Matrix batch hidden :=
  fun b j => (fori fun s => output.hidden s b j).sum

def sequence_loss (point_loss : Fin batch → Vector hidden → Rat)
    (output : Sequence seq batch hidden) : Rat :=
  loss point_loss output.sum_seq

#eval
  let input : Sequence 2 2 1 := fun s b _ => (s.val : Rat) + (b.val : Rat)
  let output := sequence_layer input (fun _ _ => 2)
  (output, sequence_loss (fun _ row => row 0 * row 0) output)

/-!
# Rectified scores and Q/K/V
-/

theorem PositionPermutation.sum (π : PositionPermutation seq) (f : Fin seq → Rat) :
    (fori fun s => f (π.index s)).sum = (fori f).sum := by
  have hp := π.valid.map f
  simp only [List.map_ofFn] at hp
  exact hp.foldr_eq' (fun x _ y _ z => Rat.add_left_comm y x z) 0

def softmax_like (z : Vector n) : Vector n :=
  let total := (fori fun j => 1 + relu (z j)).sum
  fun i => (1 + relu (z i)) / total

theorem softmax_like_permute (π : PositionPermutation n) (z : Vector n) :
    softmax_like (fun s => z (π.index s)) =
      (fun s => softmax_like z (π.index s)) := by
  funext s
  exact congrArg (fun total => (1 + relu (z (π.index s))) / total)
    (π.sum (fun t => 1 + relu (z t)))

#eval
  let z : Vector 3 := fun i => if i.val = 0 then -2 else if i.val = 1 then 3 else 1
  (softmax_like z, softmax_like (fun _ : Fin 1 => 5))

structure QKV (seq batch hidden : Nat) where
  q : Sequence seq batch hidden
  k : Sequence seq batch hidden
  v : Sequence seq batch hidden

abbrev project_qkv (input : Sequence seq batch hidden)
    (wq wk wv : Matrix hidden hidden) : QKV seq batch hidden :=
  ⟨sequence_layer input wq, sequence_layer input wk, sequence_layer input wv⟩

def base_attention (input : QKV seq batch hidden)
    (weights : Matrix seq seq → Matrix seq seq) : Sequence seq batch hidden :=
  let ⟨q, k, v⟩ := input
  fun s b =>
    let queries : Matrix seq hidden := fun t => q.hidden t b
    let keys : Matrix seq hidden := fun t => k.hidden t b
    let values : Matrix seq hidden := fun t => v.hidden t b
    ((weights (queries.matmul keys.transpose)).matmul values) s

def linear_attention (input : QKV seq batch hidden)
    (mask : Matrix seq seq := fun _ _ => 1) : Sequence seq batch hidden :=
  base_attention input (fun logits => logits * mask)

def attention_layer (input : QKV seq batch hidden) :
    Sequence seq batch hidden :=
  base_attention input (fun logits t => softmax_like (logits t))

#eval
  let input : Sequence 2 3 2 :=
    fun s b j => 10 * (b.val : Rat) + 2 * (s.val : Rat) + (j.val : Rat) + 1
  let wq : Matrix 2 2 := fun b j =>
    if b.val = j.val then 1 else if b.val = 0 then 2 else 0
  let wk : Matrix 2 2 := fun b j => if b.val = j.val then 0 else 1
  let wv : Matrix 2 2 := fun b j => if b.val = 0 ∧ j.val = 1 then 0 else 1
  let output := attention_layer (project_qkv input wq wk wv)
  output

/-!
# 6. Stacking blocks
-/

structure TransformerBlock (hidden : Nat) where
  weight : Matrix hidden hidden
  wq : Matrix hidden hidden
  wk : Matrix hidden hidden
  wv : Matrix hidden hidden

abbrev Params (hidden : Nat) := List (TransformerBlock hidden)

abbrev Mixer (seq batch hidden : Nat) :=
  QKV seq batch hidden → Sequence seq batch hidden

def transformer (mixer : Mixer seq batch hidden)
    (blocks : Params hidden) (input : Sequence seq batch hidden) :
    Sequence seq batch hidden :=
  match blocks with
  | [] => input
  | block :: rest =>
      let output := sequence_layer input block.weight
      let qkv := project_qkv output block.wq block.wk block.wv
      transformer mixer rest (mixer qkv)

def transformer_loss (mixer : Mixer seq batch hidden)
    (blocks : Params hidden)
    (point_loss : Fin batch → Vector hidden → Rat)
    (input : Sequence seq batch hidden) : Rat :=
  sequence_loss point_loss (transformer mixer blocks input)

/-!
# 7. Exercise: permutation invariance
-/

theorem attention_layer_permute (π : PositionPermutation seq)
    (input : Sequence seq batch hidden) (wq wk wv : Matrix hidden hidden) :
    attention_layer (project_qkv (input.permute π) wq wk wv) =
      (attention_layer (project_qkv input wq wk wv)).permute π := by
  let q := sequence_layer input wq
  let k := sequence_layer input wk
  let v := sequence_layer input wv
  funext s b j
  let logits : Vector seq := fun t =>
    (fori fun d => q (π.index s) b d * k t b d).sum
  change (fori fun t => softmax_like (fun u => logits (π.index u)) t *
      v (π.index t) b j).sum =
    (fori fun t => softmax_like logits t * v t b j).sum
  rw [softmax_like_permute]
  exact π.sum (fun t => softmax_like logits t * v t b j)

theorem transformer_permute (π : PositionPermutation seq)
    (blocks : Params hidden) (input : Sequence seq batch hidden) :
    transformer attention_layer blocks (input.permute π) = (transformer attention_layer blocks input).permute π := by
  induction blocks generalizing input with
  | nil => rfl
  | cons block rest ih =>
      change transformer attention_layer rest
        (attention_layer (project_qkv ((sequence_layer input block.weight).permute π) block.wq block.wk block.wv)) = _
      rw [attention_layer_permute, ih]
      rfl

theorem transformer_loss_position_invariant (π : PositionPermutation seq)
    (blocks : Params hidden)
    (point_loss : Fin batch → Vector hidden → Rat)
    (input : Sequence seq batch hidden) :
    transformer_loss attention_layer blocks point_loss (input.permute π) =
      transformer_loss attention_layer blocks point_loss input := by
  unfold transformer_loss sequence_loss
  rw [transformer_permute]
  congr 1
  funext b j
  exact π.sum (fun s => transformer attention_layer blocks input s b j)

/-!
# Exercise: batch invariance
-/

theorem transformer_select_batch (pick : Fin small → Fin batch)
    (blocks : Params hidden) (input : Sequence seq batch hidden) :
    transformer attention_layer blocks (input.select_batch pick) =
      (transformer attention_layer blocks input).select_batch pick := by
  induction blocks generalizing input with
  | nil => rfl
  | cons block rest ih =>
      change transformer attention_layer rest
        ((attention_layer (project_qkv (sequence_layer input block.weight) block.wq block.wk block.wv)).select_batch pick) = _
      exact ih _

theorem transformer_loss_batch_split (blocks : Params hidden)
    (point_loss : Fin (n + 1) → Vector hidden → Rat)
    (input : Sequence seq (n + 1) hidden) :
    transformer_loss attention_layer blocks point_loss input =
      transformer_loss attention_layer blocks (fun b => point_loss b.castSucc)
        (input.select_batch fun b : Fin n => b.castSucc) +
      transformer_loss attention_layer blocks (fun _ : Fin 1 => point_loss (Fin.last n))
        (input.select_batch fun _ : Fin 1 => Fin.last n) := by
  unfold transformer_loss
  rw [transformer_select_batch, transformer_select_batch]
  simp only [sequence_loss, loss,
    List.ofFn_succ_last, List.sum_append, List.sum_cons, List.sum_nil,
    List.ofFn_zero, Rat.add_zero, Rat.zero_add]
  rfl

/-!
# Exercise: quadratic positional features
-/

def position_features (i : Rat) : Vector 3 :=
  fun d => if d.val = 0 then i else if d.val = 1 then i * i else 1

def position_query (i : Rat) : Vector 3 :=
  let p := position_features i
  fun d => if d.val = 0 then 2 * p 0 else if d.val = 1 then -p 1 else p 2

def position_key (j : Rat) : Vector 3 :=
  let p := position_features j
  fun d => if d.val = 0 then p 0 else if d.val = 1 then p 2 else -p 1

theorem position_dot_product (i j : Rat) :
    (position_query i).dot_product (position_key j) = -(i - j) * (i - j) := by
  simp [Vector.dot_product, position_query, position_key, position_features,
    List.ofFn_succ]
  grind

def Sequence.add_positions (input : Sequence seq batch 3) : Sequence seq batch 3 :=
  fun s b => input s b + position_features (s.val : Rat)

theorem positional_transformer_breaks_permutation_invariance :
    let identity : Matrix 3 3 := fun i j => if i = j then 1 else 0
    let block : TransformerBlock 3 := ⟨identity, identity, identity, identity⟩
    let input : Sequence 2 1 3 := fun s _ d => if d.val = 0 then (s.val : Rat) else 0
    let swap : PositionPermutation 2 :=
      ⟨fun s => ⟨1 - s.val, by omega⟩, by decide⟩
    let point_loss : Fin 1 → Vector 3 → Rat := fun _ row => row 0
    transformer_loss attention_layer [block] point_loss (input.permute swap).add_positions ≠
      transformer_loss attention_layer [block] point_loss input.add_positions := by
  decide +kernel

#eval
  let q : Matrix 4 3 := fun i => position_query (i.val : Rat)
  let k : Matrix 4 3 := fun j => position_key (j.val : Rat)
  q.matmul k.transpose

/-!
# 8. Exercise: a sliding-window receptive field
-/

abbrev InWindow (radius : Nat) (s t : Fin seq) : Prop :=
  s.val ≤ t.val + radius ∧ t.val ≤ s.val + radius

def swa (radius : Nat) (input : QKV seq batch hidden) :
    Sequence seq batch hidden :=
  let ⟨q, k, v⟩ := input
  fun s b j => (fori fun t =>
    if InWindow radius s t then
      (fori fun d => q s b d * k t b d).sum * v t b j
    else 0).sum

theorem swa_local (radius : Nat) (input other : Sequence seq batch hidden)
    (wq wk wv : Matrix hidden hidden)
    (s : Fin seq) (b : Fin batch)
    (agree : ∀ t, InWindow radius s t → input.hidden t b = other.hidden t b) :
    (swa radius (project_qkv input wq wk wv)).hidden s b =
      (swa radius (project_qkv other wq wk wv)).hidden s b := by
  have center : InWindow radius s s := by constructor <;> omega
  have projection (w : Matrix hidden hidden)
      (t : Fin seq) (ht : InWindow radius s t) :=
    sequence_layer_local input other w t b (agree t ht)
  funext j
  dsimp only [swa]
  apply congrArg List.sum
  apply congrArg fori
  funext t
  by_cases ht : InWindow radius s t
  · simp only [projection wq s center,
      projection wk t ht, projection wv t ht]
  · simp only [ht, if_false]

theorem swa_transformer_local (radius : Nat) (blocks : Params hidden)
    (input other : Sequence seq batch hidden) (s : Fin seq) (b : Fin batch)
    (agree : ∀ t, InWindow (blocks.length * radius) s t → input.hidden t b = other.hidden t b) :
    (transformer (swa radius) blocks input).hidden s b =
      (transformer (swa radius) blocks other).hidden s b := by
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

namespace Examples

def one : Matrix 1 1 := fun _ _ => 1

def block : TransformerBlock 1 :=
  ⟨one, one, one, one⟩

end Examples

#eval
  let one := Examples.one
  let block := Examples.block
  let input : Sequence 5 1 1 := fun _ _ _ => 1
  let far : Sequence 5 1 1 := fun s _ _ => if s.val = 3 then 2 else 1
  let boundary : Sequence 5 1 1 := fun s _ _ => if s.val = 2 then 2 else 1
  (transformer (swa 1) [block, block] input 0 0,
    transformer (swa 1) [block, block] far 0 0,
    transformer (swa 1) [block, block] boundary 0 0,
    transformer (swa 0) [block, block] far 0 0,
    swa 5 (project_qkv input one one one) 0 0)

/-!
# 9. Exercise: tiled attention
-/

def tile (n c : Nat) (xs : List α) : List α :=
  (xs.drop (c * n)).take n

def tile_fold (tiles n : Nat) (step : σ → α → σ)
    (xs : List α) (shape : xs.length = tiles * n) (state : σ) : σ :=
  match tiles with
  | 0 => state
  | tiles + 1 =>
      tile_fold tiles n step (xs.drop n)
        (by simp [List.length_drop, shape, Nat.succ_mul])
        ((tile n 0 xs).foldl step state)

theorem tile_fold_eq (tiles n : Nat) (step : σ → α → σ)
    (xs : List α) (shape : xs.length = tiles * n) (state : σ) :
    tile_fold tiles n step xs shape state = xs.foldl step state := by
  induction tiles generalizing xs state with
  | zero =>
      have empty : xs = [] := by simpa using shape
      subst xs
      rfl
  | succ tiles ih =>
      simp only [tile_fold, tile, Nat.zero_mul, List.drop_zero]
      rw [ih, ← List.foldl_append, List.take_append_drop]

structure FlashStats (hidden : Nat) where
  scoreSum : Rat
  weighted : Vector hidden

def flash_step (r : α → Rat) (v : α → Vector hidden)
    (state : FlashStats hidden) (t : α) : FlashStats hidden :=
  let score := r t
  { scoreSum := state.scoreSum + score
    weighted := fun j => state.weighted j + score * v t j }

def flash_attention (tiles n : Nat) (input : QKV (tiles * n) batch hidden) :
    Sequence (tiles * n) batch hidden :=
  let ⟨q, k, v⟩ := input
  fun s b =>
    let r := fun t => 1 + relu ((fori fun d => q s b d * k t b d).sum)
    let initial : FlashStats hidden := ⟨0, fun _ => 0⟩
    let state := tile_fold tiles n (flash_step r (fun t => v t b))
      (fori fun t : Fin (tiles * n) => t) (by simp [fori]) initial
    fun j => state.weighted j / state.scoreSum

private theorem flash_fold (xs : List α) (r : α → Rat) (v : α → Vector hidden)
    (state : FlashStats hidden) :
    xs.foldl (flash_step r v) state =
      { scoreSum := state.scoreSum + (xs.map r).sum
        weighted := fun j => state.weighted j + (xs.map fun t => r t * v t j).sum } := by
  induction xs generalizing state with
  | nil => cases state; simp
  | cons x xs ih => simp [List.foldl_cons, ih, flash_step, Rat.add_assoc]

private theorem sum_attention (xs : List α) (r v : α → Rat) (total : Rat) :
    (xs.map fun t => (r t / total) * v t).sum =
      (xs.map fun t => r t * v t).sum / total := by
  induction xs with
  | nil => simp; grind
  | cons x xs ih =>
      simp only [List.map_cons, List.sum_cons, ih]
      grind

theorem flash_attention_eq (tiles n : Nat)
    (input : QKV (tiles * n) batch hidden) :
    flash_attention tiles n input = attention_layer input := by
  rcases input with ⟨q, k, v⟩
  funext s b j
  simp only [flash_attention, tile_fold_eq, flash_fold, List.map_ofFn,
    Rat.zero_add]
  let r := fun t => 1 + relu ((fori fun d => q s b d * k t b d).sum)
  simpa only [List.map_ofFn, Function.comp_def, attention_layer, base_attention, softmax_like, Matrix.matmul, Matrix.col, Vector.dot_product,
    Matrix.transpose, r] using
    (sum_attention (fori fun t : Fin (tiles * n) => t) r (fun t => v t b j)
      (fori r).sum).symm

/-!
# Compare equal-size tiles
-/

#eval
  let input : Sequence 4 2 2 :=
    fun s b j => 2 * (s.val : Rat) + 3 * (b.val : Rat) + (j.val : Rat) - 2
  let identity : Matrix 2 2 := fun b j => if b.val = j.val then 1 else 0
  let swap : Matrix 2 2 := fun b j => if b.val = j.val then 0 else 1
  [flash_attention 4 1 (project_qkv input identity swap identity),
    flash_attention 2 2 (project_qkv input identity swap identity),
    flash_attention 1 4 (project_qkv input identity swap identity)]

/-!
# 10. One recurrence, two scan directions
-/

def ssm_scan (α : Rat) (values : List Rat) : Rat :=
  values.foldl (fun state x => α * state + x) 0

def ssm_state (α : Rat) (updates : List (Matrix hidden hidden)) : Matrix hidden hidden :=
  fun d j => ssm_scan α (updates.map fun update => update d j)

theorem ssm_state_step (α : Rat) (updates : List (Matrix hidden hidden))
    (update : Matrix hidden hidden) :
    ssm_state α (updates ++ [update]) =
      (fun d j => α * ssm_state α updates d j + update d j) := by
  funext d j
  simp [ssm_state, ssm_scan, List.foldl_append]

def ssm_layer (α : Rat) (input : QKV seq batch hidden) : Sequence seq batch hidden :=
  let ⟨q, k, v⟩ := input
  fun s b =>
    let updates := fori fun t => (fun d j => k t b d * v t b j : Matrix hidden hidden)
    let state := ssm_state α (updates.take (s.val + 1))
    state.transpose.matvec (q.hidden s b)

def bidirectional_ssm_layer (α : Rat) (input : QKV seq batch hidden) : Sequence seq batch hidden :=
  let ⟨q, k, v⟩ := input
  fun s b =>
    let updates := fori fun t => (fun d j => k t b d * v t b j : Matrix hidden hidden)
    let state := ssm_state α (updates.take (s.val + 1)) +
      ssm_state α (updates.drop s.val).reverse
    state.transpose.matvec (q.hidden s b)

/-!
# When does order matter?
-/

namespace Examples

def ramp (seq : Nat) : Sequence seq 1 1 := fun s _ _ => (s.val : Rat) + 1

def point_loss : Fin 1 → Vector 1 → Rat := fun _ row => row 0

def swap2 : PositionPermutation 2 :=
  ⟨fun s => ⟨1 - s.val, by omega⟩, by decide⟩

def swap3 : PositionPermutation 3 :=
  ⟨fun s => if s.val < 2 then ⟨1 - s.val, by omega⟩ else s, by decide⟩

end Examples

theorem ssm_breaks_position_invariance :
    let input := Examples.ramp 2
    transformer_loss (ssm_layer 1) [Examples.block] Examples.point_loss
        (input.permute Examples.swap2) ≠
      transformer_loss (ssm_layer 1) [Examples.block] Examples.point_loss input := by
  decide +kernel

theorem alpha_ssm_breaks_position_invariance :
    let input := Examples.ramp 3
    transformer_loss (bidirectional_ssm_layer 2) [Examples.block] Examples.point_loss
        (input.permute Examples.swap3) ≠
      transformer_loss (bidirectional_ssm_layer 2) [Examples.block] Examples.point_loss input := by
  decide +kernel

private theorem bidirectional_scan_one (f : Fin n → Rat) (s : Fin n) :
    ssm_scan 1 ((fori f).take (s.val + 1)) +
      ssm_scan 1 ((fori f).drop s.val).reverse = (fori f).sum + f s := by
  let values := fori f
  have bound : s.val < values.length := by simp [values, fori]
  have split := congrArg List.sum (List.take_append_drop s.val values)
  simp only [List.sum_append] at split
  simp only [ssm_scan, Rat.one_mul, ← List.sum_eq_foldl]
  change (values.take (s.val + 1)).sum + (values.drop s.val).reverse.sum =
    values.sum + f s
  rw [List.sum_reverse, List.take_succ_eq_append_getElem bound, List.sum_append]
  simp only [List.sum_cons, List.sum_nil, Rat.add_zero]
  have current : values[s.val] = f s := by simp [values, fori]
  rw [current]
  grind

set_option backward.isDefEq.respectTransparency false in
theorem bidirectional_ssm_layer_eq (input : Sequence seq batch hidden)
    (wq wk wv : Matrix hidden hidden) (s : Fin seq) (b : Fin batch) (j : Fin hidden) :
    bidirectional_ssm_layer 1 (project_qkv input wq wk wv) s b j =
      (fori fun d =>
        ((fori fun t => sequence_layer input wk t b d * sequence_layer input wv t b j).sum +
          sequence_layer input wk s b d * sequence_layer input wv s b j) *
        sequence_layer input wq s b d).sum := by
  simp only [bidirectional_ssm_layer, Matrix.matvec, Matrix.transpose,
    Matrix.add_apply, Vector.dot_product, ssm_state, Sequence.hidden]
  simp only [List.map_take, List.map_reverse, List.map_drop, List.map_ofFn, Function.comp_def]
  simp only [bidirectional_scan_one]

theorem bidirectional_ssm_layer_permute (π : PositionPermutation seq)
    (input : Sequence seq batch hidden) (wq wk wv : Matrix hidden hidden) :
    bidirectional_ssm_layer 1 (project_qkv (input.permute π) wq wk wv) =
      (bidirectional_ssm_layer 1 (project_qkv input wq wk wv)).permute π := by
  funext s b j
  simp only [Sequence.permute, bidirectional_ssm_layer_eq]
  apply congrArg List.sum
  apply congrArg fori
  funext d
  exact congrArg (fun total =>
    (total + sequence_layer input wk (π.index s) b d * sequence_layer input wv (π.index s) b j) *
      sequence_layer input wq (π.index s) b d)
    (π.sum (fun t => sequence_layer input wk t b d * sequence_layer input wv t b j))

theorem bidirectional_ssm_loss_position_invariant (π : PositionPermutation seq)
    (blocks : Params hidden) (point_loss : Fin batch → Vector hidden → Rat)
    (input : Sequence seq batch hidden) :
    transformer_loss (bidirectional_ssm_layer 1) blocks point_loss (input.permute π) =
      transformer_loss (bidirectional_ssm_layer 1) blocks point_loss input := by
  have equiv (blocks : Params hidden) (input : Sequence seq batch hidden) :
      transformer (bidirectional_ssm_layer 1) blocks (input.permute π) =
        (transformer (bidirectional_ssm_layer 1) blocks input).permute π := by
    induction blocks generalizing input with
    | nil => rfl
    | cons block rest ih =>
        change transformer (bidirectional_ssm_layer 1) rest
          (bidirectional_ssm_layer 1 (project_qkv ((sequence_layer input block.weight).permute π) block.wq block.wk block.wv)) = _
        rw [bidirectional_ssm_layer_permute, ih]
        rfl
  unfold transformer_loss sequence_loss
  rw [equiv]
  congr 1
  funext b j
  exact π.sum (fun s => transformer (bidirectional_ssm_layer 1) blocks input s b j)

/-!
# 11. Exercise: chunkwise SSM
-/

def ssm_chunk (α : Rat) (updates : List (Matrix hidden hidden))
    (incoming : Matrix hidden hidden) : Matrix hidden hidden :=
  fun d j => α ^ updates.length * incoming d j +
    (updates.mapIdx fun t update => α ^ (updates.length - 1 - t) * update d j).sum

def ssm_chunk_carry (n : Nat) (α : Rat) (updates : List (Matrix hidden hidden)) :
    Nat → Matrix hidden hidden
  | 0 => fun _ _ => 0
  | c + 1 => ssm_chunk α (tile n c updates)
      (ssm_chunk_carry n α updates c)

def chunkwise_ssm_layer (tiles n : Nat) (α : Rat)
    (input : QKV (tiles * n) batch hidden) :
    Sequence (tiles * n) batch hidden :=
  let ⟨q, k, v⟩ := input
  fun s b =>
    let updates := fori fun t => (fun d j => k t b d * v t b j : Matrix hidden hidden)
    let c := s.val / n
    let incoming := ssm_chunk_carry n α updates c
    let local_updates := (updates.drop (c * n)).take (s.val % n + 1)
    (ssm_chunk α local_updates incoming).transpose.matvec (q.hidden s b)

private theorem scan_initial (α : Rat) (xs : List Rat) (z : Rat) :
    xs.foldl (fun state x => α * state + x) z =
      α ^ xs.length * z + xs.foldl (fun state x => α * state + x) 0 := by
  induction xs generalizing z with
  | nil => simp
  | cons x xs ih =>
      simp only [List.foldl_cons, List.length_cons, Rat.mul_zero, Rat.zero_add]
      rw [ih (α * z + x), ih x]
      simp [Rat.pow_succ, Rat.mul_add, Rat.mul_assoc, Rat.add_assoc]

private theorem sum_mapIdx_zero (xs : List Rat) :
    (xs.mapIdx fun _ _ => (0 : Rat)).sum = 0 := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simpa using ih

private theorem prefix_scan (α : Rat) (xs : List Rat) (s : Nat) (h : s < xs.length) :
    (xs.take (s + 1)).foldl (fun state x => α * state + x) 0 =
      (xs.mapIdx fun t x => (if t ≤ s then α ^ (s - t) else 0) * x).sum := by
  induction xs generalizing s with
  | nil => simp at h
  | cons x xs ih =>
      cases s with
      | zero => simp [sum_mapIdx_zero]
      | succ s =>
          have hs : s < xs.length := by simpa using h
          simp only [List.take_succ_cons, List.foldl_cons, Rat.mul_zero, Rat.zero_add]
          rw [scan_initial, ih s hs]
          simp [List.length_take, Nat.min_eq_left (by omega : s + 1 ≤ xs.length)]

private theorem scan_all_weighted (α : Rat) (xs : List Rat) :
    ssm_scan α xs = (xs.mapIdx fun t x => α ^ (xs.length - 1 - t) * x).sum := by
  cases xs with
  | nil => rfl
  | cons x xs =>
      have h := prefix_scan α (x :: xs) xs.length (by simp)
      simp only [List.length_cons, Nat.add_sub_cancel] at h ⊢
      calc
        ssm_scan α (x :: xs) =
            ((x :: xs).mapIdx fun t x => (if t ≤ xs.length then α ^ (xs.length - t) else 0) * x).sum := by
          have take : (x :: xs).take (xs.length + 1) = x :: xs :=
            List.take_length (l := x :: xs)
          simpa only [ssm_scan, take] using h
        _ = ((x :: xs).mapIdx fun t x => α ^ (xs.length - t) * x).sum := by
          apply congrArg List.sum
          apply List.ext_getElem
          · simp
          · intro t ht ht'
            have bound : t ≤ xs.length := by simp only [List.length_mapIdx, List.length_cons] at ht; omega
            simp only [List.getElem_mapIdx, if_pos bound]

set_option backward.isDefEq.respectTransparency false in
theorem ssm_chunk_eq (α : Rat) (updates : List (Matrix hidden hidden))
    (incoming : Matrix hidden hidden) (d j : Fin hidden) :
    ssm_chunk α updates incoming d j =
      (updates.map fun update => update d j).foldl (fun state x => α * state + x)
        (incoming d j) := by
  rw [scan_initial]
  simp only [ssm_chunk, List.length_map]
  congr 1
  rw [← ssm_scan, scan_all_weighted]
  congr 1
  apply List.ext_getElem
  · simp
  · intro t ht ht'
    simp

private theorem take_split (xs : List a) (start count : Nat) :
    xs.take (start + count) = xs.take start ++ (xs.drop start).take count := by
  induction start generalizing xs with
  | zero => simp
  | succ start ih =>
      cases xs with
      | nil => simp
      | cons x xs => simpa [Nat.succ_add] using congrArg (List.cons x) (ih xs)

set_option backward.isDefEq.respectTransparency false in
theorem ssm_chunk_resume (α : Rat) (updates : List (Matrix hidden hidden))
    (start count : Nat) :
    ssm_chunk α ((updates.drop start).take count) (ssm_state α (updates.take start)) =
      ssm_state α (updates.take (start + count)) := by
  funext d j
  rw [ssm_chunk_eq, take_split]
  simp only [ssm_state, ssm_scan, List.map_append, List.foldl_append]

theorem ssm_chunk_carry_eq (n : Nat) (α : Rat) (updates : List (Matrix hidden hidden))
    (c : Nat) :
    ssm_chunk_carry n α updates c = ssm_state α (updates.take (c * n)) := by
  induction c with
  | zero =>
      funext d j
      simp [ssm_chunk_carry, ssm_state, ssm_scan]
  | succ c ih =>
      rw [ssm_chunk_carry, tile, ih, ssm_chunk_resume, Nat.succ_mul]

set_option backward.isDefEq.respectTransparency false in
theorem chunkwise_ssm_layer_eq (tiles n : Nat) (α : Rat)
    (input : QKV (tiles * n) batch hidden) :
    chunkwise_ssm_layer tiles n α input = ssm_layer α input := by
  funext s b
  simp only [chunkwise_ssm_layer, ssm_chunk_carry_eq, ssm_layer]
  rw [ssm_chunk_resume]
  have offset : s.val / n * n + (s.val % n + 1) = s.val + 1 := by
    have := Nat.div_add_mod' s.val n
    omega
  rw [offset]

theorem chunkwise_transformer_eq (tiles n : Nat) (α : Rat)
    (blocks : Params hidden) (input : Sequence (tiles * n) batch hidden) :
    transformer (chunkwise_ssm_layer tiles n α) blocks input =
      transformer (ssm_layer α) blocks input := by
  induction blocks generalizing input with
  | nil => rfl
  | cons block rest ih =>
      change transformer (chunkwise_ssm_layer tiles n α) rest
        (chunkwise_ssm_layer tiles n α (project_qkv (sequence_layer input block.weight) block.wq block.wk block.wv)) = _
      rw [chunkwise_ssm_layer_eq, ih]
      rfl

#eval
  let input := project_qkv (Examples.ramp 6) Examples.one Examples.one Examples.one
  [chunkwise_ssm_layer 6 1 (1 / 2) input,
    chunkwise_ssm_layer 3 2 (1 / 2) input,
    chunkwise_ssm_layer 1 6 (1 / 2) input]

#eval
  let input : Sequence 6 2 2 :=
    fun s b j => (s.val : Rat) / 2 + (b.val : Rat) - (j.val : Rat)
  let wq : Matrix 2 2 := fun d j => if d = j then 1 else -1
  let wk : Matrix 2 2 := fun d j => (d.val : Rat) + (j.val : Rat) + 1
  let wv : Matrix 2 2 := fun d j => if d = j then 2 else 1
  let entries := fun (x : Sequence 6 2 2) => fori fun s => fori fun b => fori (x.hidden s b)
  [-1, 0, 1 / 2, 1, 2].map fun α =>
    let expected := entries (ssm_layer α (project_qkv input wq wk wv))
    [entries (chunkwise_ssm_layer 6 1 α (project_qkv input wq wk wv)) == expected,
      entries (chunkwise_ssm_layer 3 2 α (project_qkv input wq wk wv)) == expected,
      entries (chunkwise_ssm_layer 1 6 α (project_qkv input wq wk wv)) == expected]

end TensorPuzzles
