import Std

set_option doc.verso true

namespace TensorPuzzles


/-!
# Lean for Transformers: Invariants

This post explores writing formally verified ML code in Lean.
Since the cost of proofs is declining rapidly and the amount of code generated is skyrocketing,
the value of verified code seems likely to climb.
While understanding proofs remains challenging, collaborating with AI to get proofs of easy to understand
properties seems like a natural middle ground.

The goal of this post is to verify foundational properties of Transformers.
These are crtical properties that are used for
parallelization and optimization, including tensor parallelism, data parallelism,
batch invariance, permutation invariance, correctness of tiling, and locality of sparse attention models.
The text, comments, and structure of the blog are all-human written;
the proofs are all written by AI. Hopefully it can also serve as an advanced intro to Lean.

This project is inspired by [TorchLean](https://arxiv.org/abs/2602.22631), [Verified Deep Learning with Lean 4](https://lean.brettkoonce.com/blueprint/), and the [Dex Programming Language](https://github.com/google-research/dex-lang)

-/

/-!

# Invariance and Equivariance

Different neural network architectures retain different properties of their input.
We generally classify these properties in terms of equivariance and invariance.
These allow researchers to reason about what they can learn, and
implementers to optimize computation while maintaining equivalence.
Our goal will be to prove equivariances and invariances for specific architectures.


![Equivariance transforms the output along with the input; invariance leaves the output unchanged.](site/diagrams/equivariance-invariance.svg)

Notationally, ML definitions often assume the same functions can work on different input shapes, e.g. batch sizes.
For this reason our Lean definition will be a bit complex to allow for functions that are polymorphic over the shape.

-/

def Equivariant
    -- Arguments with { } are implicit
    {Shape : Type u}
    {Input : Shape → Type v} {Output : Shape → Type w}
    -- Arguments with ( ) are explicit
    (f : {shape : Shape} → Input shape → Output shape)
    {source target : Shape}
    (T : Input source → Input target)
    (S : Output source → Output target)
    -- : gives the return type. Here it is a property.
    : Prop :=
  ∀ x, f (T x) = S (f x)

def Invariant
    {Shape : Type u}
    {Input : Shape → Type v} {Output : Type w}
    (f : {shape : Shape} → Input shape → Output)
    {source target : Shape}
    (T : Input source → Input target) : Prop :=
  ∀ x, f (T x) = f x

/-!
# Vectors, Matrices, and Neural Networks

We begin by building a simple neural network library in Lean.

The `relu` function takes in a number and returns its non-negative part.
Along with the definition, we prove it does what we claim.

-/

def relu (z : Rat) : Rat := max z 0

theorem relu_non_negative
   -- For all z
   (z : Rat) :
   -- relu is ≥ 0
   relu z ≥ 0
   := by
   -- Do a short (grind) search
   grind [relu]
/-!


Following the style of [Jax](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html), we lift scalar functions to operate on vectors.
Vectors (and tensors) are represented as higher-order functions mapping
indices to rational numbers. This makes our proofs easier since we do not have to care
about storage or efficiency.

-/

-- Vector type. Maps a finite set of {0,...,n-1} to a rational.
abbrev Vector (n : Nat) := Fin n → Rat

-- Examples
-- [10, 10, 10, 10, 10]
def vector_of_tens_example: Vector 5 := fun _ => 10
-- [0, 1, 2, 3]
def arange (n: Nat) : Vector n := fun i => i

-- Greek letters are types.
variable {α : Type u} {β : Type v} {δ : Type w}

-- vmap on 1-arg functions.
def vmap (fn: α -> β) {n : Nat} :
   ((Fin n -> α) -> (Fin n -> β)) :=
  fun a => fun i => fn (a i)

-- Example: vector vmap.
def vector_relu (z: Vector n) : Vector n :=
   (vmap relu) z

-- vmap on 2-arg functions
def vmap2 (fn: α -> β -> δ) : ((Fin n -> α) -> (Fin n -> β) -> (Fin n -> δ)) :=
  fun a b => vmap (fun i => fn (a i) (b i)) id

-- Add two vectors as + overload
instance : Add (Vector n) where
  add  := vmap2 (fun a b => a + b)

-- Mul two vectors with * overload
instance : Mul (Vector n) where
  mul := vmap2 (fun a b => a * b)
/-!


For aggregations we define a vector scan.  Since we are using rationals for simplicity we do not have an
exponential, so define a "softmax-like" non-linear normalization instead.


-/
-- Fold over vectors.

abbrev fori {α : Type u} {n : Nat} (f : Fin n → α) : List α := List.ofFn f

def scan (step : σ → α → σ) (xs : Fin n → α) (initial : σ) : σ :=
  Fin.foldl n (fun state i => step state (xs i)) initial

-- Sum is a fold
def Vector.sum (a : Vector n) : Rat :=
  --alternative: scan (fun a b => a + b) a 0
  (fori (fun i => a i)).sum

def softmax_like (z : Vector n) : Vector n :=
  let weights : Vector n := vmap (fun x => 1 + relu x) z
  let total := weights.sum
  vmap (fun w => w / total) weights

def Vector.dot_product (a b : Vector n) : Rat :=
  (a * b).sum
/-!


As an exercise, let's look at a simple vector theorem. Click the square □
next to each line of the proof and it will show you the current proof state.
The proof state divides the context from the goal ⊢. Each step will transform
these terms until we can construct the goal.

-/

-- Theorem: Multiplication distributes.
theorem Vector.mul_add
   -- Given vectors a, b, c, of length n
   (a b c : Vector n) :
   -- then
   a * (b + c) = a * b + a * c
   := by
  -- Strategy: show equiv for all indices i of the output vector
  funext i
  -- Apply the rational property to the numbers at position i.
  exact Rat.mul_add (a i) (b i) (c i)

/-!

Matrices are defined similarly. We are basically just stacking
`vmap`'s to get our core operations. Note the implementation of `matmul` in particular
which will be the target of future proofs.

-/

abbrev Matrix (n m : Nat) := Fin n → Vector m

instance : Add (Matrix n m) where
  add := vmap2 (fun a b => a + b)

instance : Mul (Matrix n m) where
  mul := vmap2 (fun a b => a * b)

def Matrix.transpose (a : Matrix n m) : Matrix m n := fun i j => a j i

def Matrix.matvec (a : Matrix n m) (x : Vector m) : Vector n :=
  vmap (fun row => row.dot_product x) a

def Matrix.matmul (a : Matrix n m) (b : Matrix m p) : Matrix n p :=
  -- Functions can be called directly or with .transpose when the type is clear.
  Matrix.transpose (vmap a.matvec b.transpose)

/-!

We now have the full machinery of deep learning.
A neural network is just stacking layers and applying a simple loss function.

-/

def forward (layer: Matrix hidden hidden) {batch : Nat} (input: Matrix batch hidden) :
   Matrix batch hidden :=
   (vmap (vmap relu)) (input.matmul layer)

abbrev Layer {Shape : Type v} (State : Shape → Type u) :=
  {shape : Shape} → State shape → State shape

def neural_network {Shape : Type v} {State : Shape → Type u}
    (layers : List (Layer State)) : Layer State :=
  fun input => layers.foldl (fun state layer => layer state) input

def loss (point_loss : Fin batch → Vector hidden → Rat)
    (matrix : Matrix batch hidden) : Rat :=
  Vector.sum (vmap2 point_loss id matrix)
/-!


# Properties of Neural Networks

Now let us return to our goal of proving network equivariances.
Our strategy will be to first show that in general equivariances compose,
and then show that they propagate through a neural network.

-/

-- Equivariances compose
theorem Equivariant.comp
    -- Boilerplate
    {Shape : Type u} {A : Shape → Type v} {B : Shape → Type w} {C : Shape → Type z}
    {first : {shape : Shape} → A shape → B shape}
    {next : {shape : Shape} → B shape → C shape}
    {source target : Shape}
    {T : A source → A target} {S : B source → B target} {U : C source → C target}

    -- If f(T x) = S f(x)
    (hfirst : Equivariant (Input := A) (Output := B) first T S)
    -- and g(S x) = U g(x)
    (hnext : Equivariant (Input := B) (Output := C) next S U) :
    -- then g(f(T x )) = U (g (f (x)))
    Equivariant (Input := A) (Output := C) (fun input => next (first input)) T U := by
  intro input
  exact (congrArg next (hfirst input)).trans (hnext (first input))

-- Equivariances flow through tuples
theorem Equivariant.prod
    -- Boilerplate
    {Shape : Type u}
    {Input₁ : Shape → Type u₁} {Input₂ : Shape → Type u₂}
    {Output₁ : Shape → Type v₁} {Output₂ : Shape → Type v₂}
    {f : {shape : Shape} → Input₁ shape → Output₁ shape}
    {g : {shape : Shape} → Input₂ shape → Output₂ shape}
    {source target : Shape}
    {T₁ : Input₁ source → Input₁ target} {S₁ : Output₁ source → Output₁ target}
    {T₂ : Input₂ source → Input₂ target} {S₂ : Output₂ source → Output₂ target}

    -- If f(T1 x) = S1 f(x)
    (hf : Equivariant (Input := Input₁) (Output := Output₁) f T₁ S₁)
    -- and g(T2 x) = S2 g(x)
    (hg : Equivariant (Input := Input₂) (Output := Output₂) g T₂ S₂) :
    -- Then <f,g> <T1 x, T2 y> = <S1 f( x), S2 g( y)>
    Equivariant (Input := fun shape => Input₁ shape × Input₂ shape)
      (Output := fun shape => Output₁ shape × Output₂ shape)
      (fun input => Prod.map f g input) (Prod.map T₁ T₂) (Prod.map S₁ S₂) := by
  intro x
  exact Prod.ext (hf x.1) (hg x.2)


-- Equivariants flow through neural networks.
theorem neural_network_equivariant
    {Shape : Type v} {State : Shape → Type u} {source target : Shape}
    (layers : List (Layer State))
    (transform : State source → State target)
    -- If all layers preserve equivariance
    (equivariant : ∀ layer ∈ layers,
      Equivariant (Input := State) (Output := State) layer transform transform) :
    -- Then the neural network itself preserves it.
    Equivariant (Input := State) (Output := State) (neural_network layers) transform transform := by

  -- Proof is by induction over layers.
  induction layers with
  | nil => intro input; rfl -- trivial
  | cons layer rest ih =>
      -- Utilize the fact that all layers are equivariant.
      have composed := Equivariant.comp (equivariant layer (by simp))
        (ih (fun layer member => equivariant layer (by simp [member])))
      exact composed
/-!

We can use these properties to show that our neural network
is selection equivariant, roughly that each individual result should be the same no matter
how batches are built or ordered.

![Selecting, reordering, and repeating positions commutes with a selection-equivariant function.](site/diagrams/selection-equivariance.svg)
-/

-- Select m arbitrary elements of a set of n elements.
def select (selection : Fin m → Fin n) (a : Fin n → α) : Fin m → α :=
  fun i => a (selection i)

-- Slice out a fixed-size group.
def slice (start count : Nat) (h : start + count ≤ n) (xs : Fin n → α) : Fin count → α :=
  select (fun i => ⟨i.val + start, by omega⟩) xs


-- Equivariance under every selection, including selection across lengths.
def SelectionEquivariant (op : {n : Nat} → (Fin n → α) → (Fin n → β)) : Prop :=
  ∀ {n m} (selection : Fin m → Fin n),
    Equivariant (Input := fun n => Fin n → α) (Output := fun n => Fin n → β)
      op (select selection) (select selection)

-- Under vmap selection doesn't matter.
theorem vmap_selection_equivariant (fn : α → β) :
    SelectionEquivariant (vmap fn) := by
  intro n m selection a
  rfl

@[simp] theorem vmap2_apply (fn : α → β → δ) (a : Fin n → α)
    (b : Fin n → β) (i : Fin n) :
    vmap2 fn a b i = fn (a i) (b i) := rfl

/-!

From these individual results, we directly build up to our first main result.
A simple neural network does not depend on the order or content of its batch.

-/
theorem Matrix.matmul_row_equivariant
    (b : Matrix m p) :
    SelectionEquivariant (fun (a : Matrix _ m) => a.matmul b) :=
  vmap_selection_equivariant (fun (row : Vector m) => vmap row.dot_product b.transpose)

-- Transpose to expose columns as the position axis.
theorem Matrix.matmul_column_equivariant (a : Matrix n m) :
    SelectionEquivariant (fun (bt : Matrix _ m) =>
      (a.matmul bt.transpose).transpose) :=
  vmap_selection_equivariant a.matvec

-- MLPs are selection equivariant on batch.
theorem forward_selection_equivariant (layer : Matrix hidden hidden) :
    SelectionEquivariant (forward layer) := by
    intro n m selection input
    rfl

theorem neural_network_selection_equivariant
    (layers : List (Layer (fun n => Fin n → α)))
    (equivariant : ∀ layer ∈ layers, SelectionEquivariant layer) :
    SelectionEquivariant (neural_network layers) := by
  intro n m selection
  exact neural_network_equivariant layers (select selection)
    (fun layer member => equivariant layer member selection)

/-!


# System Optimization


While these properties so far seem basic, they are essential for
 designing large-scale LLMs. These properties provide the mean for parallelizing
 and optimizing these systems. They also are properties that are commonly broken when
 new low-level optimization are introduced. Let's look at a couple of these in more detail.


## Batch Invariance

Batch invariance ensures that the final loss of the system is independent of the size of the batch used. This property can
ensure replicability across systems. See [Horace He's](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) beautifully
described blog about why batch invariance is useful and how it is often sacrificed under different optimizations.

Here we prove that selection equivariance implies a simple form of batch invariance. Basically, you get the same loss
independent of the batch.

![Batch invariance: selecting an example before or after the same network gives the same output.](site/diagrams/batch-invariance.svg)

-/

-- The same example has the same scalar loss in a batch or on its own.
theorem nn_batch_invariant
    {nn : {batch : Nat} → (Fin batch → α) → (Fin batch → β)}
    (equivariant : SelectionEquivariant nn) (point_loss : β → Rat)
    (input : Fin batch → α) (b : Fin batch) :
    point_loss (nn (select (fun _ : Fin 1 => b) input) 0) =
      point_loss (nn input b) := by
  exact congrArg point_loss (congrFun (equivariant (fun _ : Fin 1 => b) input) 0)


/-!
## Tensor Parallel

Tensor parallelism is a common optimization for distributed neural networks.
It's a fancy way of saying that instead of doing a matrix multiplication on one
host, you can instead split it into two or more parts, do those multiplications separately,
and then merge them.


![Tensor parallelism: split A by columns and B by rows, multiply each pair independently, and add the results.](site/diagrams/tensor-parallel.svg)
-/

def Matrix.row_split (a : Matrix (k + k) m) :
    Matrix k m × Matrix k m :=
  -- Note here that i ∈ {0..k-1} but to index row need i ∈ {0..2 k-1}.
  -- These functions handle that cast.
  ⟨select (fun i => i.castAdd k) a,
   select (fun i => i.natAdd k) a⟩

-- sum is splittable
theorem Vector.sum_split (values : Vector (m + n)) :
    Vector.sum (select (fun i : Fin m => i.castAdd n) values) +
      Vector.sum (select (fun i : Fin n => i.natAdd m) values) = values.sum := by
  simp only [Vector.sum, fori, select, List.ofFn_add, List.sum_append,
    Fin.castAdd, Fin.castLE]


def Matrix.tensor_parallel (a : Matrix n (k + k)) (b : Matrix (k + k) p) :
    Matrix n p :=
  let (a₁, a₂) := a.transpose.row_split
  let (b₁, b₂) := b.row_split
  let c₁ := a₁.transpose.matmul b₁
  let c₂ := a₂.transpose.matmul b₂
  c₁ + c₂

theorem Matrix.tensor_parallel_correct
    -- For any splittable a, b
    (a : Matrix n (k + k)) (b : Matrix (k + k) p) :
    -- running tensor_parallel gives the same result as matmul
    a.tensor_parallel b = a.matmul b := by
  -- Show each final i, j ends up the same.
  funext i j
  exact Vector.sum_split (fun t => a i t * b t j)
/-!


## Data Parallel

Data parallelism says that, in training, we can split the data into
different groups, run the full neural network and loss on different machines,
and then combine. We need to ensure that we get the same result
by running things separately as together.


![Data parallelism: apply the same network to each batch half, sum point losses using their original batch indices, and add the two losses.](site/diagrams/data-parallel.svg)
-/

theorem loss_row_split (point_loss : Fin (k + k) → Vector hidden → Rat)
    (output : Matrix (k + k) hidden) :
    loss (fun b row => point_loss (b.castAdd k) row) output.row_split.1 +
      loss (fun b row => point_loss (b.natAdd k) row) output.row_split.2 =
      loss point_loss output :=
  Vector.sum_split (fun b => point_loss b (output b))

-- Split data points in half and run on separate machines.
def data_parallel_loss
    (layers : List (Layer (fun batch => Fin batch → Vector hidden)))
    (point_loss : Fin (k + k) → Vector hidden → Rat)
    (input : Matrix (k + k) hidden) : Rat :=
  let (first, second) := input.row_split
  loss (fun b row => point_loss (b.castAdd k) row)
    (neural_network layers first) +
  loss (fun b row => point_loss (b.natAdd k) row)
    (neural_network layers second)

theorem data_parallel_loss_correct
    (layers : List (Layer (fun batch => Fin batch → Vector hidden)))
    (equivariant : ∀ layer ∈ layers, SelectionEquivariant layer)
    (point_loss : Fin (k + k) → Vector hidden → Rat)
    (input : Matrix (k + k) hidden) :
    data_parallel_loss layers point_loss input =
      loss point_loss (neural_network layers input) := by
  dsimp only [Matrix] at input
  have equiv := neural_network_selection_equivariant layers equivariant (n := k + k) (m := k)
  unfold Equivariant at equiv
  simpa only [data_parallel_loss, Matrix.row_split, equiv] using
    loss_row_split point_loss (neural_network layers input)
/-!

# Transformers and attention

We next study a simple bidirectional Transformer with attention.
The sequence becomes an additional dimension of our tensor.
We first define attention.

-/


-- A mixer takes <<q,k>,v> as an arg and returns the result.
abbrev Mixer (seq hidden : Nat) :=
  (Matrix seq hidden × Matrix seq hidden) × Matrix seq hidden → Matrix seq hidden

-- The famed softmax(Q K^T) V formula.
def base_attention (s : Matrix seq seq → Matrix seq seq) : Mixer seq hidden :=
  fun ((q,k), v) => Matrix.matmul (s (q.matmul k.transpose)) v

def attention_layer : Mixer seq hidden :=
  fun input => base_attention (vmap softmax_like) input
/-!

Attention is included in the main network through a parameterized Transformer block.

-/
structure TransformerBlock (hidden : Nat) where
  weight : Matrix hidden hidden
  wq : Matrix hidden hidden
  wk : Matrix hidden hidden
  wv : Matrix hidden hidden

abbrev Params (hidden : Nat) := List (TransformerBlock hidden)

abbrev project_qkv (input : Matrix seq hidden)
    (wq wk wv : Matrix hidden hidden) : (Matrix seq hidden × Matrix seq hidden) × Matrix seq hidden :=
  ((input.matmul wq, input.matmul wk), input.matmul wv)

def transformer_block (mixer : Mixer seq hidden)
    (block : TransformerBlock hidden) (input : Matrix seq hidden) :
    Matrix seq hidden :=
  let output := (vmap (vmap relu)) (Matrix.matmul input block.weight)
  mixer (project_qkv output block.wq block.wk block.wv)

/-!

One of the more surprising properties of the vanilla Transformer
is that it is a set-based model, i.e. it is permutation equivariant
in its input. Let's define first what that means generally.

-/
structure PositionPermutation (n : Nat) where
  index : Fin n → Fin n
  valid : (fori index).Perm (fori fun i : Fin n => i)

def permute (π : PositionPermutation n) (a : Fin n → α) : Fin n → α :=
  select π.index a

def permute_both (π : PositionPermutation n) (a : Fin n → Fin n → α) :
    Fin n → Fin n → α :=
  permute π (vmap (permute π) a)

def permute_qkv {hidden : Nat} (π : PositionPermutation seq) :=
  Prod.map (Prod.map (permute (α := Vector hidden) π) (permute (α := Vector hidden) π))
    (permute (α := Vector hidden) π)


def PermuteEquivariant (op : α → β)
    -- If permutation is applied to our input,
    (inputAction : PositionPermutation n → α → α := by exact permute)
    -- The same permutation applied somehow to output yields the same result.
    (outputAction : PositionPermutation n → β → β := by exact permute) : Prop :=
  ∀ π : PositionPermutation n,
    Equivariant (Input := fun _ : Unit => α) (Output := fun _ : Unit => β)
       op  (source := ()) (target := ()) (inputAction π) (outputAction π)
/-!

Most of the core operations we have defined have the necessary equivariance.
The main additional property we need is for our softmax, which follows directly from
addition.

![Row equivariance: swapping the rows before matmul gives the same result as swapping the output rows.](site/diagrams/row-equivariance.svg)


-/

-- SelectionEquivariance (vmap) implies permutation invariance.
theorem SelectionEquivariant.permute
    {op : {n : Nat} → (Fin n → α) → (Fin n → β)}
    (equivariant : SelectionEquivariant op) : PermuteEquivariant (@op n) := by
  intro π input
  exact equivariant π.index input

theorem vmap_permute_both (fn : (Fin n → α) → (Fin n → β))
    (equivariant : PermuteEquivariant fn) :
    PermuteEquivariant (vmap fn) permute_both permute_both := by
  intro π input
  calc
    _ = permute π (vmap fn (vmap (permute π) input)) :=
      vmap_selection_equivariant fn π.index _
    _ = _ := congrArg (permute π) (funext (fun s => equivariant π (input s)))


theorem sum_rat {xs ys : List Rat} (h : xs.Perm ys) : xs.sum = ys.sum :=
   h.foldr_eq' (fun x _ y _ z => Rat.add_left_comm y x z) 0

theorem permute_sum (π : PositionPermutation seq) (f : Fin seq → Rat) :
     (Vector.sum (fun s => f (π.index s))) = (Vector.sum f) := by
   simpa [Vector.sum, permute, select,
     List.map_ofFn, Function.comp_def] using
    sum_rat (π.valid.map f)

theorem softmax_like_permute_equivariant :
    PermuteEquivariant (n := n) softmax_like := by
  intro π z
  funext s
  exact congrArg (fun total => (1 + relu (z (π.index s))) / total)
    ((permute_sum π) (fun t => 1 + relu (z t)))

def PermutationInvariant (op : (Fin n → α) → β) : Prop :=
  ∀ π : PositionPermutation n,
    Invariant (Input := fun _ : Unit => Fin n → α)
      (fun input => op input) (source := ()) (target := ()) (permute π)

theorem Matrix.matmul_transpose_permute :
    PermuteEquivariant
      (fun input : Matrix n hidden × Matrix n hidden => input.1.matmul input.2.transpose)
      (fun π => Prod.map (permute π) (permute π)) permute_both := by
  intro π input
  rfl

theorem Matrix.matmul_permute :
    PermuteEquivariant
      (fun input : Matrix n n × Matrix n hidden => input.1.matmul input.2)
      (fun π input => (permute_both π input.1, permute π input.2)) := by
  intro π input
  funext s j
  exact permute_sum π (fun t => input.1 (π.index s) t * input.2 t j)
/-!

Now we can show that the vanilla Transformer is permutation equivariant.
-/

theorem attention_layer_permute :
    PermuteEquivariant (attention_layer (seq := seq) (hidden := hidden)) permute_qkv := by
  intro π
  have logits := Matrix.matmul_transpose_permute (n := seq) (hidden := hidden) π
  have normalized := logits.comp (vmap_permute_both softmax_like softmax_like_permute_equivariant π)
  exact (normalized.prod (SelectionEquivariant.permute (vmap_selection_equivariant id) π)).comp
    (Matrix.matmul_permute π)

theorem projected_attention_permute (wq wk wv : Matrix hidden hidden) :
    PermuteEquivariant (n := seq) (fun input : Matrix seq hidden =>
      attention_layer (project_qkv input wq wk wv)) := by
  have projected : PermuteEquivariant
      (fun input : Matrix seq hidden => project_qkv input wq wk wv) permute permute_qkv := by
    intro π input
    rfl
  intro π
  exact Equivariant.comp (projected π) (attention_layer_permute π)

theorem transformer_block_permute (block : TransformerBlock hidden) :
    PermuteEquivariant (n := seq) (transformer_block attention_layer block) := by
  intro π
  unfold transformer_block
  exact Equivariant.comp
    (SelectionEquivariant.permute (forward_selection_equivariant block.weight) π)
    (projected_attention_permute block.wq block.wk block.wv π)

theorem transformer_permute (blocks : Params hidden) :
    PermuteEquivariant (n := seq)
      (neural_network (State := fun _ : Unit => _) (shape := ())
          (blocks.map (fun block => transformer_block attention_layer block))) := by
  intro π
  apply neural_network_equivariant
  intro layer member
  obtain ⟨block, _, rfl⟩ := List.mem_map.mp member
  exact transformer_block_permute block π

/-!

Of course in practice we add additional information that breaks this property.
The simplest way is through the use of positional features. We can show that even
simple positional features break equivariance with a direct counterexample.

-/

def position_features (i : Rat) : Vector 3 :=
  fun d => if d.val = 0 then i else if d.val = 1 then i * i else 1

def Matrix.add_positions (input : Matrix seq 3) : Matrix seq 3 :=
  fun s => input s + position_features (s.val : Rat)

theorem positional_transformer_breaks_permutation_equivariance :
    let identity : Matrix 3 3 := fun i j => if i = j then 1 else 0
    let block : TransformerBlock 3 := ⟨identity, identity, identity, identity⟩
    let layers : List (Layer (fun seq => Matrix seq 3)) :=
      [@Matrix.add_positions, fun input => transformer_block attention_layer block input]
    ¬ PermuteEquivariant (n := 2) (neural_network (shape := 2) layers) := by
  dsimp only
  intro equivariant
  let input : Matrix 2 3 := fun s d => if d.val = 0 then (s.val : Rat) else 0
  let swap : PositionPermutation 2 :=
    ⟨fun s => ⟨1 - s.val, by omega⟩, by decide⟩
  have same := congrArg (fun output => output 0 0) (equivariant swap input)
  revert same
  decide +kernel

/-!


# Sparse Attention


An alternative to full attention over the sequence is a sparse
attention over a local region. Sliding window attention
only looks at the surrounding window. We implement this with a
windowed mask.

-/

abbrev InWindow (radius : Nat) (s t : Fin seq) : Prop :=
  s.val ≤ t.val + radius ∧ t.val ≤ s.val + radius

def window_mask (radius : Nat) : Matrix seq seq :=
  fun s t => if InWindow radius s t then 1 else 0

def matrix_mask (radius: Nat) (a: Matrix m m) : Matrix m m :=
  let mask: Matrix m m := window_mask radius
  a * mask

def swa (radius : Nat) : Mixer seq hidden :=
  base_attention (matrix_mask radius)


/-!


The sliding window has the property that each position is only impacted by
the region around it. We formalize this idea below.


![Inputs agreeing within a window give the same output at its center, even when outside values differ.](site/diagrams/region-invariance.svg)

-/

-- Value is not impacted outside a region of a given radius.
def RegionInvariant (radius : Nat)
    (op : (Fin seq → α) → Fin seq → β) : Prop :=
  ∀ (input other : Fin seq → α) (s : Fin seq),
    (∀ t, InWindow radius s t → input t = other t ) →
      op input s = op other s

-- Selection equivariant (MLP layers) have radius 0
theorem SelectionEquivariant.region_invariant
    {op : {n : Nat} → (Fin n → α) → (Fin n → β)}
    (equivariant : SelectionEquivariant op) : RegionInvariant (seq := seq) 0 op := by
  intro a other i agree
  have same := agree i (by constructor <;> omega)
  calc
    op a i = op (fun _ : Fin 1 => a i) 0 :=
      (congrFun (equivariant (fun _ : Fin 1 => i) a) 0).symm
    _ = op (fun _ : Fin 1 => other i) 0 := by rw [same]
    _ = op other i := congrFun (equivariant (fun _ : Fin 1 => i) other) 0

-- vmap has radius 0
theorem vmap_region_invariant (fn : α → β) :
    RegionInvariant (seq := seq) 0 (vmap fn) :=
  SelectionEquivariant.region_invariant (vmap_selection_equivariant fn)

/-!

The interesting new aspect of region invariance will be how it composes.
Region invariance is additive under composition.

-/

-- Composing sliding windows increases the radius.
theorem RegionInvariant.comp
    {first : (Fin seq → α) → Fin seq → β}
    {second : (Fin seq → β) → Fin seq → δ}
    (hfirst : RegionInvariant r first) (hsecond : RegionInvariant t second) :
    RegionInvariant (r + t) (fun input => second (first input)) := by
  intro input other s agree
  apply hsecond
  intro u hu
  apply hfirst
  intro v hv
  apply agree
  dsimp [InWindow] at hu hv ⊢
  constructor <;> omega

-- NNs with same layers have multiplicative radius.
theorem neural_network_region_invariant (radius : Nat)
    (layers : List (Layer (fun seq => Fin seq → α)))
    (invariant : ∀ layer ∈ layers, RegionInvariant radius (@layer seq)) :
    RegionInvariant (seq := seq) (layers.length * radius) (neural_network layers) := by
  induction layers with
  | nil =>
      intro input other s agree
      exact agree s (by constructor <;> omega)
  | cons layer rest ih =>
      have composed := RegionInvariant.comp (invariant @layer (by simp))
        (ih (fun layer member => invariant @layer (by simp [member])))
      simpa only [RegionInvariant, List.length_cons, Nat.add_mul, Nat.one_mul, Nat.add_comm,
        neural_network, List.foldl_cons] using composed


/-!

Finally we need to show the radius of SWA. We first show that our implementation of masking
leads to a given radius and then apply this to the sparse attention implementation.
-/
attribute [local simp] Rat.add_zero Rat.zero_add Rat.zero_mul Rat.mul_zero


theorem Matrix.masked_matmul_region_invariant (radius : Nat)
    (weights : Matrix seq seq) :
    RegionInvariant radius (fun values : Matrix seq hidden =>
      (matrix_mask radius weights).matmul values) := by
  intro input other s agree
  funext j
  apply congrArg Vector.sum
  funext t
  change weights s t * window_mask radius s t * input t j =
    weights s t * window_mask radius s t * other t j
  by_cases ht : InWindow radius s t
  · rw [agree t ht]
  · simp [window_mask, ht]



theorem swa_region_invariant (radius : Nat) (wq wk wv : Matrix hidden hidden) :
    RegionInvariant (seq := seq) radius
      (fun input => swa radius (project_qkv input wq wk wv)) := by
  intro input other s agree
  have center := agree s (by constructor <;> omega)
  let contribution (query row : Vector hidden) : Vector hidden := fun j =>
    Vector.dot_product (fun d => query.dot_product (wq.transpose d))
      (fun d => row.dot_product (wk.transpose d)) * row.dot_product (wv.transpose j)
  have masked (x : Matrix seq hidden) :
      swa radius (project_qkv x wq wk wv) s =
        (matrix_mask radius (fun _ _ => 1)).matmul (vmap (contribution (x s)) x) s := by
    funext j
    apply congrArg Vector.sum
    funext t
    exact (show ∀ a b c : Rat, a * b * c = (1 * b) * (a * c) by
      intros; grind) _ _ _
  have same := Matrix.masked_matmul_region_invariant radius (fun _ _ => 1)
    (vmap (contribution (input s)) input) (vmap (contribution (input s)) other) s
    (fun t ht => congrArg (contribution (input s)) (agree t ht))
  exact (masked input).trans (same.trans (by simpa only [center] using (masked other).symm))


/-!
# Flash Attention

Once we have standard attention implemented, we can begin to consider optimizations.
Flash attention uses a tiling approach to split the full sequence into groups which
can be computed separately to reduce memory. We define a typed tiling.


-/

-- Split into n groups of size tiles.
def tile (n : Nat) (xs : Fin (tiles * n) → α) (c : Fin tiles) : Fin n → α :=
  -- The proof here shows that it is legal to make this slicing.
  slice (c.val * n) n (by
    have := Nat.mul_le_mul_right n c.isLt
    simpa [Nat.succ_mul] using this) xs

-- Apply an accumulator across each of these.
def tile_fold (tiles n : Nat) (step : σ → α → σ)
    (xs : Fin (tiles * n) → α) (initial : σ) : σ :=
  scan (fun state chunk => scan step chunk state) (tile n xs) initial



/-!

Once we have this primitive we can compute and aggregate a value for each tile.

![Process equal-size key/value tiles, carry the two accumulators, and normalize once at the end.](site/diagrams/flash-attention.svg)

-/

structure FlashAcc (hidden : Nat) where
  scoreSum : Rat
  weighted : Vector hidden

def flash_step (r : α → Rat) (v : α → Vector hidden)
    (state : FlashAcc hidden) (t : α) : FlashAcc hidden :=
  let score := r t
  { scoreSum := state.scoreSum + score
    weighted := fun j => state.weighted j + score * v t j }

def flash_attention (tiles n : Nat) (input : (Matrix (tiles * n) hidden × Matrix (tiles * n) hidden) × Matrix (tiles * n) hidden) :
    Matrix (tiles * n) hidden :=
  let ⟨⟨q, k⟩, v⟩ := input
  fun s =>
    let r := fun t => 1 + relu (Vector.sum (fun d => q s d * k t d))
    let initial : FlashAcc hidden := ⟨0, fun _ => 0⟩
    let state := tile_fold tiles n (flash_step r v)
      (fun t : Fin (tiles * n) => t) initial
    fun j => state.weighted j / state.scoreSum

/-!

For our equivalence proof, we need to define some basic properties of scans and sums.
These are not unique to Flash attention, but would be in a core mathematics library.

-/


theorem scan_slice (step : σ → α → σ) (xs : Fin n → α)
    (start count : Nat) (h : start + count ≤ n) (initial : σ) :
    scan step (slice start count h xs)
      (scan step (slice 0 start (by omega) xs) initial) =
    scan step (slice 0 (start + count) (by omega) xs) initial := by
  simp only [scan, Fin.foldl_add, slice, select, Nat.add_zero, Fin.val_castLE,
    Fin.val_natAdd, Nat.add_comm]

theorem scan_full (step : σ → α → σ) (xs : Fin n → α) (initial : σ)
    (h : count = n) : scan step (slice 0 count (by omega) xs) initial = scan step xs initial := by
  subst count
  rfl

theorem Vector.sum_succ (f : Vector (n + 1)) :
    f.sum = f 0 + Vector.sum (fun i : Fin n => f i.succ) := by
  simp [Vector.sum, fori, List.ofFn_succ]

theorem Vector.sum_mul (f : Vector n) (c : Rat) :
    f.sum * c = Vector.sum (fun i => f i * c) := by
  induction n with
  | zero => simp [Vector.sum, fori, Rat.zero_mul]
  | succ n ih => rw [Vector.sum_succ, Rat.add_mul, ih, Vector.sum_succ]

theorem Vector.sum_last (f : Vector (n + 1)) :
    f.sum = Vector.sum (fun i : Fin n => f i.castSucc) + f (Fin.last n) := by
  simp only [Vector.sum, fori, List.ofFn_succ_last, List.sum_append, List.sum_cons,
    List.sum_nil, Rat.add_zero]

theorem Vector.sum_rev (f : Vector n) : Vector.sum (fun t => f t.rev) = f.sum := by
  induction n with
  | zero => rfl
  | succ n ih =>
      rw [Vector.sum_succ, Vector.sum_last f]
      have h := ih (fun i => f i.castSucc)
      simpa [Fin.rev_succ, Rat.add_comm] using congrArg (fun z => f (Fin.last n) + z) h


theorem tile_fold_eq (tiles n : Nat) (step : σ → α → σ)
    (xs : Fin (tiles * n) → α) (initial : σ) :
    tile_fold tiles n step xs initial = scan step xs initial := by
  induction tiles generalizing initial with
  | zero => simp [tile_fold, scan]
  | succ tiles ih =>
      have shape : (tiles + 1) * n = tiles * n + n := Nat.succ_mul tiles n
      rw [tile_fold, scan, Fin.foldl_succ_last]
      change scan step (slice (tiles * n) n (by omega) xs)
        (tile_fold tiles n step (slice 0 (tiles * n) (by omega) xs) initial) = _
      rw [ih, scan_slice]
      exact scan_full _ _ _ shape.symm

/-!

The main proof is that flash attention is equivalent to our original attention.
This is done by showing a lemma over the internal fold that we are accumulating the correct values.

-/


private theorem flash_fold (xs : Fin n → α) (r : α → Rat) (v : α → Vector hidden)
    (state : FlashAcc hidden) :
    scan (flash_step r v) xs state =
      { scoreSum := state.scoreSum + Vector.sum (fun i => r (xs i))
        weighted := fun j => state.weighted j + Vector.sum (fun i => r (xs i) * v (xs i) j) } := by
  induction n generalizing state with
  | zero => cases state; simp [scan, Vector.sum, fori]
  | succ n ih =>
      simp only [scan, Fin.foldl_succ]
      change scan (flash_step r v) (fun i => xs i.succ) (flash_step r v state (xs 0)) = _
      rw [ih]
      simp [Vector.sum_succ, flash_step, Rat.add_assoc]

theorem flash_attention_eq (tiles n : Nat)
    (input : (Matrix (tiles * n) hidden × Matrix (tiles * n) hidden) × Matrix (tiles * n) hidden) :
    flash_attention tiles n input = attention_layer input := by
  rcases input with ⟨⟨q, k⟩, v⟩
  funext s j
  simp only [flash_attention, tile_fold_eq, flash_fold,
    Rat.zero_add]
  let r := fun t => 1 + relu (Vector.sum (fun d => q s d * k t d))
  change Vector.sum (fun t => r t * v t j) / Vector.sum r =
    Vector.sum (fun t => (r t / Vector.sum r) * v t j)
  simp only [Rat.div_def, Vector.sum_mul]
  apply congrArg Vector.sum
  funext t
  grind

/-!

# State Space Models

Another alternative to standard attention is to use a state space model or linear attention approach.
These can be defined by the following recurrence.



![The recurrent, masked linear-attention, and chunkwise forms compute the same decayed SSM.](site/diagrams/ssm.svg)

-/

def ssm_scan (a : Rat) (values : Vector n) (initial : Rat := 0) : Rat :=
  scan (fun state x => a * state + x) values initial

def ssm_state (a : Rat) (updates : Fin n → Matrix hidden hidden)
    (incoming : Matrix hidden hidden := fun _ _ => 0) : Matrix hidden hidden :=
  fun d j => ssm_scan a (fun t => updates t d j) (incoming d j)

def ssm_layer (a : Rat) : Mixer seq hidden :=
  fun input =>
    let ⟨⟨q, k⟩, v⟩ := input
    let updates := fun t => (fun d j => k t d * v t j : Matrix hidden hidden)
    fun s => (ssm_state a (slice 0 (s.val + 1) (by omega) updates)).transpose.matvec (q s)


-- By induction, show how each term is weighted.
theorem scan_weighted (a : Rat) (values : Vector n) (z : Rat) :
    ssm_scan a values z =
      a ^ n * z + Vector.sum (fun t => a ^ (n - 1 - t.val) * values t) := by
  induction n generalizing z with
  | zero => simp [ssm_scan, scan, Vector.sum, fori]
  | succ n ih =>
      simp only [ssm_scan, scan, Fin.foldl_succ]
      change ssm_scan a (fun i : Fin n => values i.succ) (a * z + values 0) = _
      rw [ih, Vector.sum_succ]
      simp [Nat.sub_sub, Nat.add_comm, Rat.pow_succ,
        Rat.mul_add, Rat.mul_assoc, Rat.add_assoc]

/-!


Unlike vanilla attention, state space models clearly induce an ordering on the sequence
in the multiplicative term `a`. However in the special case where that term is 1 and
we use a bidirectional SSM, we can show that the permutation equivariance remains.

-/

def bidirectional_ssm_scan (a : Rat) (values : Vector n) : Vector n :=
  fun s =>
    let suffix := slice s.val (n - s.val) (by omega) values
    ssm_scan a (slice 0 (s.val + 1) (by omega) values) +
      ssm_scan a (fun t => suffix t.rev)

def bidirectional_ssm_layer (α : Rat) : Mixer seq hidden :=
  fun input =>
    let ⟨⟨q, k⟩, v⟩ := input
    fun s =>
      let state : Matrix hidden hidden := fun d j =>
        bidirectional_ssm_scan α (fun t => k t d * v t j) s
      state.transpose.matvec (q s)

theorem bidirectional_scan_one (f : Vector n) (s : Fin n) :
    bidirectional_ssm_scan 1 f s = f.sum + f s := by
  dsimp [Vector] at f
  have one_pow (m : Nat) : (1 : Rat) ^ m = 1 := by
    induction m <;> simp_all [Rat.pow_succ]
  simp only [bidirectional_ssm_scan, scan_weighted, one_pow,
    Rat.one_mul, Rat.mul_zero, Rat.zero_add]
  rw [Vector.sum_rev]
  rw [show Vector.sum (slice 0 (s.val + 1) (by omega) f) =
    Vector.sum (slice 0 s.val (by omega) f) + f s by
      exact Vector.sum_last (slice 0 (s.val + 1) (by omega) f)]
  have split := Vector.sum_split (values := fun t : Fin (s.val + (n - s.val)) => f ⟨t.val, by omega⟩)
  have total : Vector.sum (fun t : Fin (s.val + (n - s.val)) => f ⟨t.val, by omega⟩) = Vector.sum f := by
    have full (m : Nat) (h : m = n) : Vector.sum (slice 0 m (by omega) f) = Vector.sum f := by
      subst m
      rfl
    exact full _ (by omega)
  rw [total] at split
  have split' : Vector.sum (slice 0 s.val (by omega) f) +
      Vector.sum (slice s.val (n - s.val) (by omega) f) = Vector.sum f := by
    simpa [Vector.sum, slice, select, Nat.add_comm] using split
  grind

theorem bidirectional_ssm_layer_permute :
    PermuteEquivariant (bidirectional_ssm_layer (seq := seq) (hidden := hidden) 1)
      permute_qkv := by
  intro π input
  rcases input with ⟨⟨q, k⟩, v⟩
  funext s j
  apply congrArg Vector.sum
  funext d
  change bidirectional_ssm_scan 1 (fun t => k (π.index t) d * v (π.index t) j) s *
      q (π.index s) d =
    bidirectional_ssm_scan 1 (fun t => k t d * v t j) (π.index s) * q (π.index s) d
  rw [bidirectional_scan_one, bidirectional_scan_one]
  exact congrArg (fun total => (total + k (π.index s) d * v (π.index s) j) * q (π.index s) d)
    (permute_sum π (fun t => k t d * v t j))

/-!


These models also have the property that we can chunk them into groups which can be computed separately.
Here we consider a simplified version of chunking in order to distribute across machines.

![Equal-size chunks pass a carry between boundaries; each position combines the decayed incoming state with its local weighted updates.](site/diagrams/chunkwise-ssm.svg)

-/



def ssm_chunk (a : Rat) (values : Vector n) (incoming : Rat) : Rat :=
  a ^ n * incoming + Vector.sum (fun t => a ^ (n - 1 - t.val) * values t)

def ssm_chunk_carry (tiles n : Nat) (a : Rat) (values : Vector (tiles * n)) : Rat :=
  scan (fun state values => ssm_chunk a values state) (tile n values) 0

def chunkwise_ssm_layer (tiles n : Nat) (a : Rat) : Mixer (tiles * n) hidden :=
  fun input =>
    let ⟨⟨q, k⟩, v⟩ := input
    fun s =>
      let start := s.val / n * n
      let state : Matrix hidden hidden := fun d j =>
        let updates : Vector (tiles * n) := fun t => k t d * v t j
        let incoming := ssm_chunk_carry (s.val / n) n a (slice 0 start (by
          have := Nat.div_add_mod' s.val n
          omega) updates)
        ssm_chunk a (slice start (s.val % n + 1) (by
          have := Nat.div_add_mod' s.val n
          omega) updates) incoming
      state.transpose.matvec (q s)

/-!

We can prove that this yields the same result as our simple implementation.

-/

theorem ssm_chunk_carry_eq (tiles n : Nat) (a : Rat) (values : Vector (tiles * n)) :
    ssm_chunk_carry tiles n a values = ssm_scan a values := by
  simp only [ssm_chunk_carry, ssm_chunk, ← scan_weighted, ssm_scan]
  exact tile_fold_eq tiles n _ values 0

theorem chunkwise_ssm_layer_eq (tiles n : Nat) (a : Rat)
    (input : (Matrix (tiles * n) hidden × Matrix (tiles * n) hidden) × Matrix (tiles * n) hidden) :
    chunkwise_ssm_layer tiles n a input = ssm_layer a input := by
  rcases input with ⟨⟨q, k⟩, v⟩
  funext s j
  apply congrArg Vector.sum
  funext d
  let updates : Vector (tiles * n) := fun t => k t d * v t j
  have bound : s.val / n * n + (s.val % n + 1) ≤ tiles * n := by
    have := Nat.div_add_mod' s.val n
    omega
  change ssm_chunk a (slice (s.val / n * n) (s.val % n + 1) bound updates)
    (ssm_chunk_carry (s.val / n) n a (slice 0 (s.val / n * n) (by omega) updates)) * q s d =
      ssm_scan a (slice 0 (s.val + 1) (by omega) updates) * q s d
  apply congrArg (fun state => state * q s d)
  rw [ssm_chunk_carry_eq]
  change (a ^ _ * _ + _) = _
  rw [← scan_weighted]
  change scan (fun state x => a * state + x) _ (scan (fun state x => a * state + x) _ 0) = _
  rw [scan_slice]
  have offset : s.val / n * n + (s.val % n + 1) = s.val + 1 := by
    have := Nat.div_add_mod' s.val n
    omega
  exact scan_full _ (slice 0 (s.val + 1) (by omega) updates) 0 offset


/-!


# Conclusion


This blog considered the use of Lean as a method to prove some elementary properties about Transformers and related models.
There are many additional things one might consider here including bounding errors introduced from quantization, aggregation, backpropagation,
and training-inference mismatch. Additionally, there are likely many ways to simplify these proofs or develop libraries to make them more minimal.

At a high level though, the main change is not the machinery for proving these properties, it is the ease with which a person can specify "what" they want to be
proven and receive a certificate that a property is true. Understanding how that interface should work and how it can be used will be an extremely interesting challenge over the next year.


-/


end TensorPuzzles
