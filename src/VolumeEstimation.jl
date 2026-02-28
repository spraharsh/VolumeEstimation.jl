module VolumeEstimation

using Pigeons
using CommonSolve
using Random
using LinearAlgebra
using PyMBAR

include("problem.jl")
include("find_kmax.jl")
include("log_potential.jl")
include("thin_traces.jl")
include("mbar.jl")
include("dos.jl")
include("solve.jl")

export VolumeProblem, VolumeSolution
export extract_u_kn, mbar_log_ratio, thin_traces
export compute_dos, shape_factor, produce_analysis

end # module VolumeEstimation
