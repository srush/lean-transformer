import TensorPuzzles.AstWidget

namespace TensorPuzzles.Generated

open TensorPuzzles

/- Generated from examples/triu.py. The theorem below makes Lean independently
   check the frontend's claimed dtype and symbolic output shape. -/

def triuProgram : Program := {
  dimParams := ["n"]
  tensorParams := []
  body := .where_ (.compare .le (.unsqueeze (.arange (.param "n"))) (.arange (.param "n"))) (.scalar 1) (.scalar 0)
}

def triuClaimedShape : SymShape := [.param "n", .param "n"]

theorem triu_shape_checked :
    inferProgramChecked triuProgram
      (.num, triuClaimedShape) = true := by
  decide

theorem triu_inferred :
    inferProgram triuProgram =
      .ok (.num, triuClaimedShape) := by
  tensor_ast triuProgram
  exact inferProgramChecked_sound triu_shape_checked

end TensorPuzzles.Generated
