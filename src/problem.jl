"""
    VolumeProblem(membership, dim; x0=zeros(dim))

Define a high-dimensional volume estimation problem.

# Arguments
- `membership`: A function `f(x::AbstractVector) -> Bool` returning `true` if `x` is
  inside the region whose volume is to be estimated.
- `dim::Int`: Dimensionality of the space.

# Keyword Arguments
- `x0::AbstractVector`: A point known to be inside the region, used for MCMC initialization.
  Default: `zeros(dim)`.
"""
struct VolumeProblem{F}
    membership::F
    dim::Int
    x0::Vector{Float64}

    function VolumeProblem(
        membership::F, dim::Int;
        x0::Union{Nothing, AbstractVector{<:Real}} = nothing
    ) where {F}
        x_init = x0 === nothing ? zeros(dim) : Vector{Float64}(x0)
        if !membership(x_init)
            throw(ArgumentError(
                "Initial point x0 must be inside the region (membership(x0) must return true). " *
                "Got membership(x0) = false for x0 = $x_init"
            ))
        end
        new{F}(membership, dim, x_init)
    end
end
