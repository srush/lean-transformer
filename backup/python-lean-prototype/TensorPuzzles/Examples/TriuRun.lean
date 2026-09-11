import TensorPuzzles.Generated.Triu

namespace TensorPuzzles.Examples

open TensorPuzzles
open TensorPuzzles.Generated

-- The executor is pure: this evaluates to a 4x4 upper-triangular tensor.
#eval triuProgram.evalData [("n", 4)] []

end TensorPuzzles.Examples
