import TensorPuzzles.Generated.Triu

namespace TensorPuzzles.Specs

open TensorPuzzles TensorPuzzles.Generated

/-- An n-by-n matrix with 1 on or above the diagonal and 0 below it.
Only valid indices are constrained by the specification. -/
def TriuSpec (n : Nat) (output : Tensor Rat) : Prop :=
  output.shape = [n, n] ∧
    ∀ i j, i < n → j < n →
      output.get [i, j] = if i ≤ j then 1 else 0

/-- For n > 1, executing the Triu AST succeeds and satisfies the spec.
The assumption removes the size-one broadcasting case. -/
theorem triu_correct (n : Nat) (hn : 1 < n) :
    ∃ output,
      triuProgram.eval [("n", n)] [] = .ok (.num output) ∧
      TriuSpec n output := by
  tensor_ast triuProgram
  have h : n ≠ 1 := Nat.ne_of_gt hn
  -- Evaluate the AST: broadcast row and column indices, then select 1 or 0.
  unfold Program.eval
  rw [triu_inferred]
  simp [triuProgram, evalExpr, Tensor.unsqueeze, Tensor.scalar, Tensor.map2,
    Tensor.where_, Tensor.broadcastShape, Tensor.broadcastRev,
    Tensor.broadcastIndex, resolveDim, lookupAssoc, expectNumValue,
    expectBoolValue, applyCmpOp, checkTensorArguments, TriuSpec, Ne.symm h]
  -- Use the computed tensor; its shape is correct by definition.
  refine ⟨_, rfl, rfl, ?_⟩
  intro i j _ _
  simp [h, Rat.natCast_le_natCast]

end TensorPuzzles.Specs
