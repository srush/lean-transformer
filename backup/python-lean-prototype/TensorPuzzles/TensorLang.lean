import Std

namespace TensorPuzzles

/-! A tiny, pure tensor language. The frontend produces `Expr` values; this
module owns shape checking and execution. Runtime tensors are functions over
indices, which keeps every primitive mutation-free. -/

inductive Dim where
  | lit (value : Nat)
  | param (name : String)
  deriving Repr, BEq, DecidableEq

abbrev SymShape := List Dim

inductive DType where
  | num
  | bool
  deriving Repr, BEq, DecidableEq

inductive BinOp where
  | add
  | sub
  | mul
  | div
  | matmul
  deriving Repr, BEq, DecidableEq

inductive CmpOp where
  | eq
  | ne
  | lt
  | le
  | gt
  | ge
  deriving Repr, BEq, DecidableEq

inductive Expr where
  | input (name : String)
  | scalar (value : Rat)
  | arange (size : Dim)
  | binary (op : BinOp) (left right : Expr)
  | compare (op : CmpOp) (left right : Expr)
  | where_ (condition ifTrue ifFalse : Expr)
  | unsqueeze (value : Expr)
  deriving Repr, BEq, DecidableEq

structure Program where
  dimParams : List String
  tensorParams : List (String × SymShape)
  body : Expr
  deriving Repr, BEq, DecidableEq

def lookupAssoc [BEq α] (key : α) : List (α × β) → Option β
  | [] => none
  | (candidate, value) :: rest =>
      if candidate == key then some value else lookupAssoc key rest

private def isOne : Dim → Bool
  | .lit 1 => true
  | _ => false

def broadcastSymDim (left right : Dim) : Except String Dim :=
  if left == right then
    pure left
  else if isOne left then
    pure right
  else if isOne right then
    pure left
  else
    throw s!"cannot broadcast symbolic dimensions {repr left} and {repr right}"

private def broadcastSymRev : SymShape → SymShape → Except String SymShape
  | [], right => pure right
  | left, [] => pure left
  | left :: leftRest, right :: rightRest => do
      let head ← broadcastSymDim left right
      pure (head :: (← broadcastSymRev leftRest rightRest))

def broadcastSymShape (left right : SymShape) : Except String SymShape := do
  pure (← broadcastSymRev left.reverse right.reverse).reverse

private def expectType
    (expected : DType) (actual : DType × SymShape) : Except String SymShape :=
  if actual.1 == expected then
    pure actual.2
  else
    throw s!"expected {repr expected}, found {repr actual.1}"

private def inferMatmul
    (left right : SymShape) : Except String SymShape :=
  match left, right with
  | [m, k], [k', n] =>
      if k == k' then pure [m, n]
      else throw s!"matmul inner dimensions differ: {repr k} and {repr k'}"
  | _, _ => throw "minimal matmul currently requires two rank-2 tensors"

def inferExpr
    (dims : List String)
    (tensors : List (String × SymShape)) : Expr → Except String (DType × SymShape)
  | .input name =>
      match lookupAssoc name tensors with
      | some shape => pure (.num, shape)
      | none => throw s!"unknown tensor input '{name}'"
  | .scalar _ => pure (.num, [])
  | .arange (.lit n) => pure (.num, [.lit n])
  | .arange dim@(.param name) =>
      if dims.contains name then pure (.num, [dim])
      else throw s!"unknown dimension parameter '{name}'"
  | .binary op left right => do
      let leftShape ← expectType .num (← inferExpr dims tensors left)
      let rightShape ← expectType .num (← inferExpr dims tensors right)
      let shape ←
        if op == .matmul then inferMatmul leftShape rightShape
        else broadcastSymShape leftShape rightShape
      pure (.num, shape)
  | .compare _ left right => do
      let leftShape ← expectType .num (← inferExpr dims tensors left)
      let rightShape ← expectType .num (← inferExpr dims tensors right)
      pure (.bool, ← broadcastSymShape leftShape rightShape)
  | .where_ condition ifTrue ifFalse => do
      let conditionShape ← expectType .bool (← inferExpr dims tensors condition)
      let trueShape ← expectType .num (← inferExpr dims tensors ifTrue)
      let falseShape ← expectType .num (← inferExpr dims tensors ifFalse)
      let branchShape ← broadcastSymShape trueShape falseShape
      pure (.num, ← broadcastSymShape conditionShape branchShape)
  | .unsqueeze value => do
      let (dtype, shape) ← inferExpr dims tensors value
      pure (dtype, shape ++ [.lit 1])

def inferProgram (program : Program) : Except String (DType × SymShape) :=
  inferExpr program.dimParams program.tensorParams program.body

def inferProgramChecked
    (program : Program) (expected : DType × SymShape) : Bool :=
  match inferProgram program with
  | .ok actual => decide (actual = expected)
  | .error _ => false

theorem inferProgramChecked_sound
    {program : Program} {expected : DType × SymShape}
    (checked : inferProgramChecked program expected = true) :
    inferProgram program = .ok expected := by
  unfold inferProgramChecked at checked
  split at checked
  next actual inferred =>
    simp only [decide_eq_true_eq] at checked
    subst actual
    exact inferred
  next message inferred =>
    simp at checked

structure Tensor (α : Type) where
  shape : List Nat
  get : List Nat → α

namespace Tensor

def scalar (value : α) : Tensor α where
  shape := []
  get := fun _ => value

def indices : List Nat → List (List Nat)
  | [] => [[]]
  | size :: rest =>
      (List.range size).flatMap fun i =>
        (indices rest).map fun tail => i :: tail

def values (tensor : Tensor α) : List α :=
  (indices tensor.shape).map tensor.get

def unsqueeze (tensor : Tensor α) : Tensor α where
  shape := tensor.shape ++ [1]
  get := fun index => tensor.get index.dropLast

def broadcastRev : List Nat → List Nat → Except String (List Nat)
  | [], right => pure right
  | left, [] => pure left
  | left :: leftRest, right :: rightRest => do
      let head ←
        if left == right then pure left
        else if left == 1 then pure right
        else if right == 1 then pure left
        else throw s!"cannot broadcast dimensions {left} and {right}"
      pure (head :: (← broadcastRev leftRest rightRest))

def broadcastShape (left right : List Nat) : Except String (List Nat) := do
  pure (← broadcastRev left.reverse right.reverse).reverse

def broadcastIndex
    (sourceShape outputShape outputIndex : List Nat) : List Nat :=
  let padding := outputShape.length - sourceShape.length
  let paddedShape := List.replicate padding 1 ++ sourceShape
  let paddedIndex := List.zipWith
    (fun sourceDim index => if sourceDim == 1 then 0 else index)
    paddedShape outputIndex
  paddedIndex.drop padding

def map2
    (operation : α → β → γ)
    (left : Tensor α)
    (right : Tensor β) : Except String (Tensor γ) := do
  let outputShape ← broadcastShape left.shape right.shape
  pure {
    shape := outputShape
    get := fun index => operation
      (left.get (broadcastIndex left.shape outputShape index))
      (right.get (broadcastIndex right.shape outputShape index))
  }

def where_
    (condition : Tensor Bool)
    (ifTrue ifFalse : Tensor Rat) : Except String (Tensor Rat) := do
  let branchShape ← broadcastShape ifTrue.shape ifFalse.shape
  let outputShape ← broadcastShape condition.shape branchShape
  pure {
    shape := outputShape
    get := fun index =>
      let conditionIndex := broadcastIndex condition.shape outputShape index
      let trueIndex := broadcastIndex ifTrue.shape outputShape index
      let falseIndex := broadcastIndex ifFalse.shape outputShape index
      if condition.get conditionIndex then
        ifTrue.get trueIndex
      else
        ifFalse.get falseIndex
  }

def matmul (left right : Tensor Rat) : Except String (Tensor Rat) := do
  match left.shape, right.shape with
  | [m, k], [k', n] =>
      if k != k' then
        throw s!"matmul inner dimensions differ: {k} and {k'}"
      pure {
        shape := [m, n]
        get := fun index =>
          match index with
          | [i, j] => (List.range k).foldl
              (fun total p => total + left.get [i, p] * right.get [p, j]) 0
          | _ => 0
      }
  | _, _ => throw "minimal matmul currently requires two rank-2 tensors"

end Tensor

inductive Value where
  | num (tensor : Tensor Rat)
  | bool (tensor : Tensor Bool)

abbrev DimEnv := List (String × Nat)
abbrev TensorEnv := List (String × Tensor Rat)

def resolveDim (dims : DimEnv) : Dim → Except String Nat
  | .lit value => pure value
  | .param name =>
      match lookupAssoc name dims with
      | some value => pure value
      | none => throw s!"missing dimension argument '{name}'"

def expectNumValue : Value → Except String (Tensor Rat)
  | .num tensor => pure tensor
  | .bool _ => throw "expected a numeric tensor"

def expectBoolValue : Value → Except String (Tensor Bool)
  | .bool tensor => pure tensor
  | .num _ => throw "expected a boolean tensor"

private def applyBinOp
    (op : BinOp) (left right : Tensor Rat) : Except String (Tensor Rat) :=
  match op with
  | .add => Tensor.map2 (· + ·) left right
  | .sub => Tensor.map2 (· - ·) left right
  | .mul => Tensor.map2 (· * ·) left right
  | .div => do
      let shape ← Tensor.broadcastShape left.shape right.shape
      let candidate ← Tensor.map2 (· / ·) left right
      if (Tensor.indices shape).any fun index =>
          let padding := shape.length - right.shape.length
          let paddedShape := List.replicate padding 1 ++ right.shape
          let paddedIndex := List.zipWith
            (fun sourceDim i => if sourceDim == 1 then 0 else i)
            paddedShape index
          right.get (paddedIndex.drop padding) == 0
      then throw "division by zero"
      else pure candidate
  | .matmul => Tensor.matmul left right

def applyCmpOp (op : CmpOp) (left right : Rat) : Bool :=
  match op with
  | .eq => left == right
  | .ne => left != right
  | .lt => decide (left < right)
  | .le => decide (left ≤ right)
  | .gt => decide (left > right)
  | .ge => decide (left ≥ right)

def evalExpr
    (dims : DimEnv) (tensors : TensorEnv) : Expr → Except String Value
  | .input name =>
      match lookupAssoc name tensors with
      | some tensor => pure (.num tensor)
      | none => throw s!"missing tensor argument '{name}'"
  | .scalar value => pure (.num (Tensor.scalar value))
  | .arange size => do
      let n ← resolveDim dims size
      pure (.num {
        shape := [n]
        get := fun index =>
          match index with
          | i :: _ => i
          | _ => 0
      })
  | .binary op left right => do
      let left ← expectNumValue (← evalExpr dims tensors left)
      let right ← expectNumValue (← evalExpr dims tensors right)
      pure (.num (← applyBinOp op left right))
  | .compare op left right => do
      let left ← expectNumValue (← evalExpr dims tensors left)
      let right ← expectNumValue (← evalExpr dims tensors right)
      pure (.bool (← Tensor.map2 (applyCmpOp op) left right))
  | .where_ condition ifTrue ifFalse => do
      let condition ← expectBoolValue (← evalExpr dims tensors condition)
      let ifTrue ← expectNumValue (← evalExpr dims tensors ifTrue)
      let ifFalse ← expectNumValue (← evalExpr dims tensors ifFalse)
      pure (.num (← Tensor.where_ condition ifTrue ifFalse))
  | .unsqueeze value => do
      match ← evalExpr dims tensors value with
      | .num tensor => pure (.num tensor.unsqueeze)
      | .bool tensor => pure (.bool tensor.unsqueeze)

private def resolveShape (dims : DimEnv) (shape : SymShape) : Except String (List Nat) :=
  shape.mapM (resolveDim dims)

def checkTensorArguments
    (dims : DimEnv)
    (expected : List (String × SymShape))
    (actual : TensorEnv) : Except String Unit := do
  for (name, symbolicShape) in expected do
    let expectedShape ← resolveShape dims symbolicShape
    let tensor ←
      match lookupAssoc name actual with
      | some tensor => pure tensor
      | none => throw s!"missing tensor argument '{name}'"
    if tensor.shape != expectedShape then
      throw s!"tensor '{name}' has shape {tensor.shape}, expected {expectedShape}"

def Program.eval
    (program : Program) (dims : DimEnv) (tensors : TensorEnv) : Except String Value := do
  let _ ← inferProgram program
  checkTensorArguments dims program.tensorParams tensors
  evalExpr dims tensors program.body

structure TensorData (α : Type) where
  shape : List Nat
  values : List α
  deriving Repr, BEq, DecidableEq

def Value.toData : Value → Sum (TensorData Rat) (TensorData Bool)
  | .num tensor => .inl { shape := tensor.shape, values := tensor.values }
  | .bool tensor => .inr { shape := tensor.shape, values := tensor.values }

def Program.evalData
    (program : Program) (dims : DimEnv) (tensors : TensorEnv) :
    Except String (Sum (TensorData Rat) (TensorData Bool)) := do
  pure (← program.eval dims tensors).toData

end TensorPuzzles
