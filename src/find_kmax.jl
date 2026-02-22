"""
    find_kmax(membership, x0; kwargs...)

Find the maximum spring constant `kmax` such that sampling from `N(x0, (1/k)I)`
keeps approximately `target` fraction of samples inside the membership region.

This sets the natural scale for the Gaussian reference distribution: `σ = 1/√kmax`.

# Algorithm
Adaptive bisection on `k`: at each step, draw `n_samples` from `N(x0, (1/k)I)`,
compute the acceptance fraction, and adjust `k` upward (if acceptance > target)
or downward (if acceptance < target).

# Keyword Arguments
- `target::Real`: target acceptance fraction. Default: `0.95`.
- `n_samples::Integer`: number of samples per iteration. Default: `1000`.
- `k_start::Real`: initial spring constant. Default: `1.0`.
- `tol::Real`: convergence tolerance on acceptance fraction. Default: `0.025`.
- `max_iter::Integer`: maximum number of bisection iterations. Default: `100`.
- `rng::AbstractRNG`: random number generator. Default: `Random.default_rng()`.

# Returns
A named tuple `(kmax, acceptance)`.
"""
function find_kmax(membership, x0::AbstractVector{<:Real};
        target::Real = 0.95,
        n_samples::Integer = 1000,
        k_start::Real = 1.0,
        tol::Real = 0.025,
        max_iter::Integer = 100,
        rng::AbstractRNG = Random.default_rng())

    k = float(k_start)
    acceptance = zero(k)
    x = similar(x0, typeof(k))

    for _ in 1:max_iter
        sigma = 1 / sqrt(k)
        n_accept = 0
        for _ in 1:n_samples
            randn!(rng, x)
            @. x = x0 + sigma * x
            if membership(x)
                n_accept += 1
            end
        end
        acceptance = n_accept / n_samples

        if abs(acceptance - target) < tol
            return (; kmax = k, acceptance)
        end

        correction = 1 + (target - acceptance) / (target + acceptance)
        k *= correction * correction
    end

    return (; kmax = k, acceptance)
end
