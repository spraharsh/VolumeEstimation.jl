module Volumes

using Pigeons
using CommonSolve
using Random
using LinearAlgebra
import ForwardDiff
using PyMBAR

include("problem.jl")
include("find_kmax.jl")
include("log_potential.jl")
include("thin_traces.jl")
include("mbar.jl")
include("solve.jl")

export VolumeProblem, VolumeSolution
export extract_u_kn, mbar_log_ratio, thin_traces

end # module Volumes
