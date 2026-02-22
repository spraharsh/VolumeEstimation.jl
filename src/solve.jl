"""
    VolumeSolution

Result of solving a `VolumeProblem`.

# Fields
- `log_volume::Float64`: Estimated log-volume (primary result; use this in high dimensions).
- `volume::Float64`: Estimated volume `exp(log_volume)`. May underflow to 0 in high dimensions.
- `pt`: The Pigeons `PT` object for diagnostics (swap rates, round trips, etc.).
"""
struct VolumeSolution
    log_volume::Float64
    volume::Float64
    pt
end

"""
    solve(prob::VolumeProblem; n_rounds=10, n_chains=10, kwargs...)

Estimate the volume of a region defined by `prob.membership` using parallel tempering
via Pigeons.jl.

# Keyword Arguments
- `n_rounds::Int`: Number of PT adaptation rounds (samples double each round). Default: `10`.
- `n_chains::Int`: Number of tempering chains. Default: `10`.
- `kwargs...`: Additional keyword arguments passed to `Pigeons.pigeons()`.

# Mathematical Details
The volume is recovered from the normalizing constant ratio:

    log(Volume) = stepping_stone(pt) + (dim/2) * log(2π * σ²) + log(p_acc)

where `stepping_stone(pt) = log(Z_target / Z_ref)` and
`Z_ref = (2πσ²)^(dim/2) * p_acc` is the partition function of the Gaussian
truncated to the membership region.
"""
function CommonSolve.solve(prob::VolumeProblem;
        n_rounds::Int = 10,
        n_chains::Int = 10,
        kwargs...)

    # Estimate sigma and acceptance fraction via kmax
    (; kmax, acceptance) = find_kmax(prob.membership, prob.x0)
    sigma = 1 / sqrt(kmax)
    p_acc = acceptance

    target = VolumeLogPotential(prob.membership, prob.dim, sigma, prob.x0)

    pt = Pigeons.pigeons(;
        target = target,
        n_rounds = n_rounds,
        n_chains = n_chains,
        kwargs...
    )

    log_ratio = Pigeons.stepping_stone(pt)
    log_ref = (prob.dim / 2) * log(2 * π * sigma^2) + log(p_acc)
    log_vol = log_ratio + log_ref

    VolumeSolution(log_vol, exp(log_vol), pt)
end
