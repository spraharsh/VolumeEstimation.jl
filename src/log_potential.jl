"""
Internal log-potential wrapping a VolumeProblem for use with Pigeons.jl.

Target distribution: uniform on region S.
- log-density = 0 if x ∈ S
- log-density = -Inf if x ∉ S

Reference: N(x0, σ²I) truncated to S via CenteredGaussianLogPotential.
"""
struct VolumeLogPotential{F}
    membership::F
    dim::Int
    sigma::Float64
    x0::Vector{Float64}
end

function VolumeLogPotential(prob::VolumeProblem)
    sigma = prob.sigma === nothing ? 1.0 : prob.sigma
    VolumeLogPotential(prob.membership, prob.dim, sigma, prob.x0)
end

(lp::VolumeLogPotential)(x) = lp.membership(x) ? 0.0 : -Inf

Pigeons.LogDensityProblems.logdensity(lp::VolumeLogPotential, x) = lp(x)
Pigeons.LogDensityProblems.dimension(lp::VolumeLogPotential) = lp.dim
Pigeons.LogDensityProblems.capabilities(::Type{<:VolumeLogPotential}) = Pigeons.LogDensityProblems.LogDensityOrder{0}()

function Pigeons.initialization(lp::VolumeLogPotential, ::AbstractRNG, ::Int)
    copy(lp.x0)
end

function Pigeons.default_reference(lp::VolumeLogPotential)
    precision = 1.0 / (lp.sigma^2)
    CenteredGaussianLogPotential(lp.membership, precision, lp.dim, lp.x0)
end

function Pigeons.default_explorer(::VolumeLogPotential)
    Pigeons.AutoMALA()
end

"""
Gaussian log-potential centered at x0, truncated to region S.
- log-density = -(precision/2)|x - x0|² if x ∈ S
- log-density = -Inf if x ∉ S

Partition function: Z = (2π/precision)^(dim/2) * p_acc,
where p_acc is the acceptance fraction of the full Gaussian inside S.
"""
struct CenteredGaussianLogPotential{F}
    membership::F
    precision::Float64
    dim::Int
    x0::Vector{Float64}
end

function (lp::CenteredGaussianLogPotential)(x)
    if !lp.membership(x)
        return -Inf
    end
    s = 0.0
    @inbounds for i in 1:lp.dim
        d = x[i] - lp.x0[i]
        s += d * d
    end
    return -0.5 * lp.precision * s
end

Pigeons.LogDensityProblems.logdensity(lp::CenteredGaussianLogPotential, x) = lp(x)
Pigeons.LogDensityProblems.dimension(lp::CenteredGaussianLogPotential) = lp.dim
Pigeons.LogDensityProblems.capabilities(::Type{<:CenteredGaussianLogPotential}) = Pigeons.LogDensityProblems.LogDensityOrder{0}()

function Pigeons.sample_iid!(lp::CenteredGaussianLogPotential, replica, shared)
    σ = 1.0 / sqrt(lp.precision)
    while true
        for i in 1:lp.dim
            replica.state[i] = lp.x0[i] + σ * randn(replica.rng)
        end
        if lp.membership(replica.state)
            return
        end
    end
end
