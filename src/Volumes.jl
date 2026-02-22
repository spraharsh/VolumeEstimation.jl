module Volumes

using Pigeons
using CommonSolve
using Random
using LinearAlgebra
import ForwardDiff

include("problem.jl")
include("find_kmax.jl")
include("log_potential.jl")
include("solve.jl")

export VolumeProblem, VolumeSolution

end # module Volumes
