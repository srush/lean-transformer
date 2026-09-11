import Std
import MatrixSimps

set_option doc.verso true

namespace TensorPuzzles

/-!
# Integer vectors: a small verified model
-/

/-!
# 1. Vectors and pointwise arithmetic
-/

abbrev fori {α : Type u} {n : Nat} (f : Fin n → α) : List α := List.ofFn f

def Vector (n : Nat) := Fin n → Int

instance : Repr (Vector n) where
  reprPrec v p := reprPrec (fori v) p

instance : Add (Vector n) where
  add a b := fun i => a i + b i

instance : Mul (Vector n) where
  mul a b := fun i => a i * b i

def Vector.dot_product (a b : Vector n) : Int :=
  (fori fun i => a i * b i).sum

theorem Vector.mul_add (a b c : Vector n) : a * (b + c) = a * b + a * c := by
  funext i
  exact Int.mul_add (a i) (b i) (c i)

#eval
  let x : Vector 3 := fun i => (i.val : Int) + 1
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

def mat_ex1 : Matrix 2 3 := fun i j => 3 * (i.val : Int) + (j.val : Int) + 1

def mat_ex2 : Matrix 3 2 := fun i j => 2 * (i.val : Int) + (j.val : Int) + 1

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

def relu (z : Int) : Int := max z 0

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

def loss (point_loss : Fin batch → Vector hidden → Int)
    (matrix : Matrix batch hidden) : Int :=
  (fori fun b => point_loss b (matrix b)).sum

/-!
# 4b. Exercise: data-parallel neural networks
-/

def data_parallel_loss (layers : List (NeuralLayer hidden))
    (point_loss : Fin (k + k) → Vector hidden → Int)
    (input : Matrix (k + k) hidden) : Int :=
  let (first, second) := input.row_split
  loss (fun b row => point_loss (b.castAdd k) row)
    (neural_network layers first) +
  loss (fun b row => point_loss (b.natAdd k) row)
    (neural_network layers second)

theorem data_parallel_loss_correct (layers : List (NeuralLayer hidden))
    (point_loss : Fin (k + k) → Vector hidden → Int)
    (input : Matrix (k + k) hidden) :
    data_parallel_loss layers point_loss input =
      loss point_loss (neural_network layers input) := by
  simp [data_parallel_loss, loss, Matrix.row_split, nn_batch_independence,
    List.ofFn_add, List.sum_append, Fin.castAdd, Fin.natAdd, Fin.castLE]

#eval
  let point_loss : Fin 2 → Vector 3 → Int := fun b row => (b.val : Int) + row 0 * row 2
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

def sequence_loss (point_loss : Fin batch → Vector hidden → Int)
    (output : Sequence seq batch hidden) : Int :=
  loss point_loss output.sum_seq

#eval
  let input : Sequence 2 2 1 := fun s b _ => (s.val : Int) + (b.val : Int)
  let output := sequence_layer input (fun _ _ => 2)
  (output, sequence_loss (fun _ row => row 0 * row 0) output)

/-!
# Rectified scores and Q/K/V
-/

theorem PositionPermutation.sum (π : PositionPermutation seq) (f : Fin seq → Int) :
    (fori fun s => f (π.index s)).sum = (fori f).sum := by
  have hp := π.valid.map f
  simp only [List.map_ofFn] at hp
  exact hp.foldr_eq' (fun x _ y _ z => Int.add_left_comm y x z) 0

def discrete_softmax (z : Vector n) : Vector n :=
  let total := (fori fun j => relu (z j)).sum
  fun i => relu (z i) - total

theorem discrete_softmax_permute (π : PositionPermutation n) (z : Vector n) :
    discrete_softmax (fun s => z (π.index s)) =
      (fun s => discrete_softmax z (π.index s)) := by
  funext s
  exact congrArg (fun total => relu (z (π.index s)) - total)
    (π.sum (fun t => relu (z t)))

#eval
  let z : Vector 3 := fun i => if i.val = 0 then -2 else if i.val = 1 then 3 else 1
  (discrete_softmax z, discrete_softmax (fun _ : Fin 1 => 5))

def attention_layer (input : Sequence seq batch hidden)
    (wq wk wv : Matrix hidden hidden) :
    Sequence seq batch hidden :=
  let q := sequence_layer input wq
  let k := sequence_layer input wk
  let v := sequence_layer input wv
  fun s b =>
    let queries : Matrix seq hidden := fun t j => q t b j
    let keys : Matrix seq hidden := fun t j => k t b j
    let values : Matrix seq hidden := fun t j => v t b j
    let logits := queries.matmul keys.transpose
    let weights : Matrix seq seq := fun t => discrete_softmax (logits t)
    (weights.matmul values) s

#eval
  let input : Sequence 2 3 2 :=
    fun s b j => 10 * (b.val : Int) + 2 * (s.val : Int) + (j.val : Int) + 1
  let wq : Matrix 2 2 := fun b j =>
    if b.val = j.val then 1 else if b.val = 0 then 2 else 0
  let wk : Matrix 2 2 := fun b j => if b.val = j.val then 0 else 1
  let wv : Matrix 2 2 := fun b j => if b.val = 0 ∧ j.val = 1 then 0 else 1
  let output := attention_layer input wq wk wv
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

class Mixer (kind : Type) where
  mix : {seq batch hidden : Nat} → kind → Sequence seq batch hidden →
    Matrix hidden hidden → Matrix hidden hidden → Matrix hidden hidden →
    Sequence seq batch hidden

structure Attention where

instance : Mixer Attention where
  mix _ := attention_layer

def transformer {kind : Type} [Mixer kind] (mixer : kind)
    (blocks : Params hidden) (input : Sequence seq batch hidden) :
    Sequence seq batch hidden :=
  match blocks with
  | [] => input
  | block :: rest =>
      transformer mixer rest
        (Mixer.mix mixer (sequence_layer input block.weight) block.wq block.wk block.wv)

def transformer_loss {kind : Type} [Mixer kind] (mixer : kind)
    (blocks : Params hidden)
    (point_loss : Fin batch → Vector hidden → Int)
    (input : Sequence seq batch hidden) : Int :=
  sequence_loss point_loss (transformer mixer blocks input)

/-!
# 7. Exercise: permutation invariance
-/

theorem attention_layer_permute (π : PositionPermutation seq)
    (input : Sequence seq batch hidden) (wq wk wv : Matrix hidden hidden) :
    attention_layer (input.permute π) wq wk wv =
      (attention_layer input wq wk wv).permute π := by
  let q := sequence_layer input wq
  let k := sequence_layer input wk
  let v := sequence_layer input wv
  funext s b j
  let logits : Vector seq := fun t =>
    (fori fun d => q (π.index s) b d * k t b d).sum
  change (fori fun t => discrete_softmax (fun u => logits (π.index u)) t *
      v (π.index t) b j).sum =
    (fori fun t => discrete_softmax logits t * v t b j).sum
  rw [discrete_softmax_permute]
  exact π.sum (fun t => discrete_softmax logits t * v t b j)

theorem transformer_permute (π : PositionPermutation seq)
    (blocks : Params hidden) (input : Sequence seq batch hidden) :
    transformer (Attention.mk) blocks (input.permute π) = (transformer (Attention.mk) blocks input).permute π := by
  induction blocks generalizing input with
  | nil => rfl
  | cons block rest ih =>
      change transformer (Attention.mk) rest
        (attention_layer ((sequence_layer input block.weight).permute π)
          block.wq block.wk block.wv) = _
      rw [attention_layer_permute, ih]
      rfl

theorem transformer_loss_position_invariant (π : PositionPermutation seq)
    (blocks : Params hidden)
    (point_loss : Fin batch → Vector hidden → Int)
    (input : Sequence seq batch hidden) :
    transformer_loss (Attention.mk) blocks point_loss (input.permute π) =
      transformer_loss (Attention.mk) blocks point_loss input := by
  unfold transformer_loss sequence_loss
  rw [transformer_permute]
  congr 1
  funext b j
  exact π.sum (fun s => transformer (Attention.mk) blocks input s b j)

/-!
# Exercise: batch invariance
-/

theorem transformer_select_batch (pick : Fin small → Fin batch)
    (blocks : Params hidden) (input : Sequence seq batch hidden) :
    transformer (Attention.mk) blocks (input.select_batch pick) =
      (transformer (Attention.mk) blocks input).select_batch pick := by
  induction blocks generalizing input with
  | nil => rfl
  | cons block rest ih =>
      change transformer (Attention.mk) rest
        ((attention_layer (sequence_layer input block.weight)
          block.wq block.wk block.wv).select_batch pick) = _
      exact ih _

theorem transformer_loss_batch_split (blocks : Params hidden)
    (point_loss : Fin (n + 1) → Vector hidden → Int)
    (input : Sequence seq (n + 1) hidden) :
    transformer_loss (Attention.mk) blocks point_loss input =
      transformer_loss (Attention.mk) blocks (fun b => point_loss b.castSucc)
        (input.select_batch fun b : Fin n => b.castSucc) +
      transformer_loss (Attention.mk) blocks (fun _ : Fin 1 => point_loss (Fin.last n))
        (input.select_batch fun _ : Fin 1 => Fin.last n) := by
  unfold transformer_loss
  rw [transformer_select_batch, transformer_select_batch]
  simp only [sequence_loss, loss,
    List.ofFn_succ_last, List.sum_append, List.sum_cons, List.sum_nil,
    List.ofFn_zero, Int.add_zero, Int.zero_add]
  rfl

/-!
# Exercise: quadratic positional features
-/

def position_features (i : Int) : Vector 3 :=
  fun d => if d.val = 0 then i else if d.val = 1 then i * i else 1

def position_query (i : Int) : Vector 3 :=
  let p := position_features i
  fun d => if d.val = 0 then 2 * p 0 else if d.val = 1 then -p 1 else p 2

def position_key (j : Int) : Vector 3 :=
  let p := position_features j
  fun d => if d.val = 0 then p 0 else if d.val = 1 then p 2 else -p 1

theorem position_dot_product (i j : Int) :
    (position_query i).dot_product (position_key j) = -(i - j) * (i - j) := by
  simp [Vector.dot_product, position_query, position_key, position_features,
    List.ofFn_succ]
  grind

def Sequence.add_positions (input : Sequence seq batch 3) : Sequence seq batch 3 :=
  fun s b => input s b + position_features (s.val : Int)

theorem positional_transformer_breaks_permutation_invariance :
    let identity : Matrix 3 3 := fun i j => if i = j then 1 else 0
    let block : TransformerBlock 3 := ⟨identity, identity, identity, identity⟩
    let input : Sequence 2 1 3 := fun s _ d => if d.val = 0 then (s.val : Int) else 0
    let swap : PositionPermutation 2 :=
      ⟨fun s => ⟨1 - s.val, by omega⟩, by decide⟩
    let point_loss : Fin 1 → Vector 3 → Int := fun _ row => row 0
    transformer_loss (Attention.mk) [block] point_loss (input.permute swap).add_positions ≠
      transformer_loss (Attention.mk) [block] point_loss input.add_positions := by
  decide

#eval
  let q : Matrix 4 3 := fun i => position_query (i.val : Int)
  let k : Matrix 4 3 := fun j => position_key (j.val : Int)
  q.matmul k.transpose

/-!
# 8. Exercise: a sliding-window receptive field
-/

abbrev InWindow (radius : Nat) (s t : Fin seq) : Prop :=
  s.val ≤ t.val + radius ∧ t.val ≤ s.val + radius

def swa (radius : Nat) (input : Sequence seq batch hidden)
    (wq wk wv : Matrix hidden hidden) :
    Sequence seq batch hidden :=
  let q := sequence_layer input wq
  let k := sequence_layer input wk
  let v := sequence_layer input wv
  fun s b j => (fori fun t =>
    if InWindow radius s t then
      (fori fun d => q s b d * k t b d).sum * v t b j
    else 0).sum

structure SWA where
  radius : Nat

instance : Mixer SWA where
  mix config := swa config.radius

theorem swa_local (radius : Nat) (input other : Sequence seq batch hidden)
    (wq wk wv : Matrix hidden hidden)
    (s : Fin seq) (b : Fin batch)
    (agree : ∀ t, InWindow radius s t → input.hidden t b = other.hidden t b) :
    (swa radius input wq wk wv).hidden s b =
      (swa radius other wq wk wv).hidden s b := by
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
    (transformer (SWA.mk radius) blocks input).hidden s b =
      (transformer (SWA.mk radius) blocks other).hidden s b := by
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
  (transformer (SWA.mk 1) [block, block] input 0 0,
    transformer (SWA.mk 1) [block, block] far 0 0,
    transformer (SWA.mk 1) [block, block] boundary 0 0,
    transformer (SWA.mk 0) [block, block] far 0 0,
    swa 5 input one one one 0 0)

/-!
# 9. Exercise: tiled attention
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
  weighted : Vector hidden
  values : Vector hidden

def flash_step (r : α → Int) (v : α → Vector hidden)
    (state : FlashStats hidden) (t : α) : FlashStats hidden :=
  let score := r t
  { scoreSum := state.scoreSum + score
    weighted := fun j => state.weighted j + score * v t j
    values := fun j => state.values j + v t j }

def flash_attention (n : Nat) (positive : 0 < n) (input : Sequence seq batch hidden)
    (wq wk wv : Matrix hidden hidden) :
    Sequence seq batch hidden :=
  let q := sequence_layer input wq
  let k := sequence_layer input wk
  let v := sequence_layer input wv
  fun s b =>
    let r := fun t => relu ((fori fun d => q s b d * k t b d).sum)
    let initial : FlashStats hidden := ⟨0, fun _ => 0, fun _ => 0⟩
    let state := tile_fold n positive (flash_step r (fun t => v t b))
      (fori fun t : Fin seq => t) initial
    fun j => state.weighted j - state.scoreSum * state.values j

private theorem flash_fold (xs : List α) (r : α → Int) (v : α → Vector hidden)
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
    (input : Sequence seq batch hidden) (wq wk wv : Matrix hidden hidden) :
    flash_attention n positive input wq wk wv =
      attention_layer input wq wk wv := by
  funext s b j
  simp only [flash_attention, tile_fold_eq, flash_fold, List.map_ofFn,
    Int.zero_add]
  let q := sequence_layer input wq
  let k := sequence_layer input wk
  let v := sequence_layer input wv
  let r := fun t => relu ((fori fun d => q s b d * k t b d).sum)
  simpa only [List.map_ofFn, Function.comp_def, attention_layer, discrete_softmax, Matrix.matmul, Matrix.col, Vector.dot_product,
    Matrix.transpose, q, k, v, r] using
    (sum_attention (fori fun t : Fin seq => t) r (fun t => v t b j)
      (fori r).sum).symm

/-!
# Check tile boundaries
-/

#eval
  let input : Sequence 3 2 2 :=
    fun s b j => 2 * (s.val : Int) + 3 * (b.val : Int) + (j.val : Int) - 2
  let identity : Matrix 2 2 := fun b j => if b.val = j.val then 1 else 0
  let swap : Matrix 2 2 := fun b j => if b.val = j.val then 0 else 1
  let run := fun n h => flash_attention n h input identity swap identity
  [run 1 (by decide), run 2 (by decide), run 4 (by decide)]

/-!
# 10. One recurrence, two scan directions
-/

def ssm_scan (α : Int) (values : List Int) : Int :=
  values.foldl (fun state x => α * state + x) 0

def ssm_state (α : Int) (updates : List (Matrix hidden hidden)) : Matrix hidden hidden :=
  fun d j => ssm_scan α (updates.map fun update => update d j)

theorem ssm_state_step (α : Int) (updates : List (Matrix hidden hidden))
    (update : Matrix hidden hidden) :
    ssm_state α (updates ++ [update]) =
      (fun d j => α * ssm_state α updates d j + update d j) := by
  funext d j
  simp [ssm_state, ssm_scan, List.foldl_append]

def ssm_layer (α : Int) (input : Sequence seq batch hidden)
    (wq wk wv : Matrix hidden hidden) : Sequence seq batch hidden :=
  let q := sequence_layer input wq
  let k := sequence_layer input wk
  let v := sequence_layer input wv
  fun s b =>
    let updates := fori fun t => (fun d j => k t b d * v t b j : Matrix hidden hidden)
    let state := ssm_state α (updates.take (s.val + 1))
    state.transpose.matvec (q.hidden s b)

def bidirectional_ssm_layer (α : Int) (input : Sequence seq batch hidden)
    (wq wk wv : Matrix hidden hidden) : Sequence seq batch hidden :=
  let q := sequence_layer input wq
  let k := sequence_layer input wk
  let v := sequence_layer input wv
  fun s b =>
    let updates := fori fun t => (fun d j => k t b d * v t b j : Matrix hidden hidden)
    let state := ssm_state α (updates.take (s.val + 1)) +
      ssm_state α (updates.drop s.val).reverse
    state.transpose.matvec (q.hidden s b)

structure SSM where
  α : Int
  bidirectional : Bool := false

instance : Mixer SSM where
  mix config := if config.bidirectional then bidirectional_ssm_layer config.α
    else ssm_layer config.α

/-!
# When does order matter?
-/

namespace Examples

def ramp (seq : Nat) : Sequence seq 1 1 := fun s _ _ => (s.val : Int) + 1

def point_loss : Fin 1 → Vector 1 → Int := fun _ row => row 0

def swap2 : PositionPermutation 2 :=
  ⟨fun s => ⟨1 - s.val, by omega⟩, by decide⟩

def swap3 : PositionPermutation 3 :=
  ⟨fun s => if s.val < 2 then ⟨1 - s.val, by omega⟩ else s, by decide⟩

end Examples

theorem ssm_breaks_position_invariance :
    let input := Examples.ramp 2
    transformer_loss (SSM.mk 1 false) [Examples.block] Examples.point_loss
        (input.permute Examples.swap2) ≠
      transformer_loss (SSM.mk 1 false) [Examples.block] Examples.point_loss input := by
  decide

theorem alpha_ssm_breaks_position_invariance :
    let input := Examples.ramp 3
    transformer_loss (SSM.mk 2 true) [Examples.block] Examples.point_loss
        (input.permute Examples.swap3) ≠
      transformer_loss (SSM.mk 2 true) [Examples.block] Examples.point_loss input := by
  decide

private theorem bidirectional_scan_one (f : Fin n → Int) (s : Fin n) :
    ssm_scan 1 ((fori f).take (s.val + 1)) +
      ssm_scan 1 ((fori f).drop s.val).reverse = (fori f).sum + f s := by
  let values := fori f
  have bound : s.val < values.length := by simp [values, fori]
  have split := congrArg List.sum (List.take_append_drop s.val values)
  simp only [List.sum_append] at split
  simp only [ssm_scan, Int.one_mul, ← List.sum_eq_foldl]
  change (values.take (s.val + 1)).sum + (values.drop s.val).reverse.sum =
    values.sum + f s
  rw [List.sum_reverse_int, List.take_succ_eq_append_getElem bound, List.sum_append]
  simp only [List.sum_cons, List.sum_nil, Int.add_zero]
  have current : values[s.val] = f s := by simp [values, fori]
  rw [current]
  omega

set_option backward.isDefEq.respectTransparency false in
theorem bidirectional_ssm_layer_eq (input : Sequence seq batch hidden)
    (wq wk wv : Matrix hidden hidden) (s : Fin seq) (b : Fin batch) (j : Fin hidden) :
    bidirectional_ssm_layer 1 input wq wk wv s b j =
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
    bidirectional_ssm_layer 1 (input.permute π) wq wk wv =
      (bidirectional_ssm_layer 1 input wq wk wv).permute π := by
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
    (blocks : Params hidden) (point_loss : Fin batch → Vector hidden → Int)
    (input : Sequence seq batch hidden) :
    transformer_loss (SSM.mk 1 true) blocks point_loss (input.permute π) =
      transformer_loss (SSM.mk 1 true) blocks point_loss input := by
  have equiv (blocks : Params hidden) (input : Sequence seq batch hidden) :
      transformer (SSM.mk 1 true) blocks (input.permute π) =
        (transformer (SSM.mk 1 true) blocks input).permute π := by
    induction blocks generalizing input with
    | nil => rfl
    | cons block rest ih =>
        change transformer (SSM.mk 1 true) rest
          (bidirectional_ssm_layer 1 ((sequence_layer input block.weight).permute π)
            block.wq block.wk block.wv) = _
        rw [bidirectional_ssm_layer_permute, ih]
        rfl
  unfold transformer_loss sequence_loss
  rw [equiv]
  congr 1
  funext b j
  exact π.sum (fun s => transformer (SSM.mk 1 true) blocks input s b j)

/-!
# 11. Exercise: the parallel form of the SSM
-/

def ssm_mask (α : Int) : Matrix seq seq := fun s t =>
  (if t.val ≤ s.val then α ^ (s.val - t.val) else 0) +
  (if s.val ≤ t.val then α ^ (t.val - s.val) else 0)

def parallel_ssm_layer (α : Int) (input : Sequence seq batch hidden)
    (wq wk wv : Matrix hidden hidden) : Sequence seq batch hidden :=
  let q := sequence_layer input wq
  let k := sequence_layer input wk
  let v := sequence_layer input wv
  fun s b =>
    let Q : Matrix seq hidden := fun t => q.hidden t b
    let K : Matrix seq hidden := fun t => k.hidden t b
    let V : Matrix seq hidden := fun t => v.hidden t b
    ((Q.matmul K.transpose * ssm_mask α).matmul V) s

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

private theorem bidirectional_scan_weighted (α : Int) (f : Fin n → Int) (s : Fin n) :
    ssm_scan α ((fori f).take (s.val + 1)) +
      ssm_scan α ((fori f).drop s.val).reverse =
        (fori fun t => ssm_mask α s t * f t).sum := by
  unfold ssm_scan
  rw [prefix_scan α (fori f) s.val (by simp [fori]), suffix_scan, sum_mapIdx_add]
  apply congrArg List.sum
  apply List.ext_getElem
  · simp [fori]
  · intro t h₁ h₂
    simp [fori, ssm_mask, Int.add_mul]

private theorem sum_scale (xs : List a) (f : a → Int) (c : Int) :
    (xs.map fun x => c * f x).sum = c * (xs.map f).sum := by
  induction xs with
  | nil => simp
  | cons x xs ih => simp [ih, Int.mul_add]

private theorem sum_add (xs : List a) (f g : a → Int) :
    (xs.map fun x => f x + g x).sum = (xs.map f).sum + (xs.map g).sum := by
  induction xs with
  | nil => simp
  | cons x xs ih => simp [ih, Int.add_assoc, Int.add_left_comm]

private theorem sum_swap (xs : List a) (ys : List b) (f : a → b → Int) :
    (xs.map fun x => (ys.map fun y => f x y).sum).sum =
      (ys.map fun y => (xs.map fun x => f x y).sum).sum := by
  induction xs with
  | nil =>
      induction ys with
      | nil => rfl
      | cons y ys ih => simpa using ih
  | cons x xs ih => simp [sum_add, ih]

set_option backward.isDefEq.respectTransparency false in
theorem parallel_ssm_layer_eq (α : Int) (input : Sequence seq batch hidden)
    (wq wk wv : Matrix hidden hidden) :
    parallel_ssm_layer α input wq wk wv =
      bidirectional_ssm_layer α input wq wk wv := by
  funext s b j
  let q := sequence_layer input wq
  let k := sequence_layer input wk
  let v := sequence_layer input wv
  simp only [parallel_ssm_layer, bidirectional_ssm_layer, Matrix.matmul, Matrix.col,
    Matrix.matvec, Matrix.transpose, Matrix.mul_apply, Matrix.add_apply,
    Vector.dot_product, ssm_state, Sequence.hidden]
  simp only [List.map_take, List.map_reverse, List.map_drop,
    List.map_ofFn, Function.comp_def]
  change (fori fun t => (fori fun d => q s b d * k t b d).sum *
      ssm_mask α s t * v t b j).sum =
    (fori fun d => (ssm_scan α ((fori fun t => k t b d * v t b j).take (s.val + 1)) +
      ssm_scan α ((fori fun t => k t b d * v t b j).drop s.val).reverse) * q s b d).sum
  simp only [bidirectional_scan_weighted]
  have scale {m : Nat} (f : Fin m → Int) (c : Int) :
      (fori fun x => c * f x).sum = c * (fori f).sum := by
    simpa only [List.map_ofFn, Function.comp_def] using
      sum_scale (fori fun x : Fin m => x) f c
  have swap (f : Fin seq → Fin hidden → Int) :
      (fori fun t => (fori fun d => f t d).sum).sum =
        (fori fun d => (fori fun t => f t d).sum).sum := by
    simpa only [List.map_ofFn, Function.comp_def] using
      sum_swap (fori fun t : Fin seq => t) (fori fun d : Fin hidden => d) f
  simp only [Int.mul_comm, ← scale]
  rw [swap]
  apply congrArg List.sum
  apply congrArg fori
  funext d
  apply congrArg List.sum
  apply congrArg fori
  funext t
  grind

#eval
  let input := Examples.ramp 3
  (transformer (SSM.mk 2 true) [Examples.block] input,
    parallel_ssm_layer 2 input Examples.one Examples.one Examples.one)

#eval
  let input : Sequence 3 2 2 :=
    fun s b j => (s.val : Int) + 2 * (b.val : Int) - (j.val : Int)
  let wq : Matrix 2 2 := fun d j => if d = j then 1 else -1
  let wk : Matrix 2 2 := fun d j => (d.val : Int) + (j.val : Int) + 1
  let wv : Matrix 2 2 := fun d j => if d = j then 2 else 1
  ([-1, 0, 1, 2].map fun α =>
    (bidirectional_ssm_layer α input wq wk wv,
      parallel_ssm_layer α input wq wk wv))

end TensorPuzzles
