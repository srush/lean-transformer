import TensorPuzzles.Generated.Triu

namespace TensorPuzzles.Examples

open TensorPuzzles
open TensorPuzzles.Generated

-- Put the cursor on this command to see the AST in the Lean Infoview.
#tensor_ast triuProgram

-- The tactic form shows the same widget beside the live proof state and does
-- not alter the goal.
example : True := by
  tensor_ast triuProgram
  trivial

end TensorPuzzles.Examples
