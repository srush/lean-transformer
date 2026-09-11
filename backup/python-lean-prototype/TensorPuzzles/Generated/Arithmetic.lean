import TensorPuzzles.AstWidget

namespace TensorPuzzles.Generated

open TensorPuzzles

/- Generated from examples/arithmetic.py. The theorem below makes Lean independently
   check the frontend's claimed dtype and symbolic output shape. -/

def arithmeticProgram : Program := {
  dimParams := ["n"]
  tensorParams := [("a", [.param "n", .param "n"]), ("b", [.param "n", .param "n"])]
  body := .binary .matmul (.binary .div (.binary .mul (.binary .add (.input "a") (.input "b")) (.binary .sub (.input "a") (.input "b"))) (.scalar 2)) (.input "b")
}

def arithmeticClaimedShape : SymShape := [.param "n", .param "n"]

theorem arithmetic_shape_checked :
    inferProgramChecked arithmeticProgram
      (.num, arithmeticClaimedShape) = true := by
  decide

theorem arithmetic_inferred :
    inferProgram arithmeticProgram =
      .ok (.num, arithmeticClaimedShape) := by
  tensor_ast arithmeticProgram
  exact inferProgramChecked_sound arithmetic_shape_checked

end TensorPuzzles.Generated
