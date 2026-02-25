"""
    VolumeSolution

Result of solving a `VolumeProblem`.

# Fields
- `log_volume::Float64`: Estimated log-volume from the chosen estimator.
- `volume::Float64`: `exp(log_volume)`. May underflow to 0 in high dimensions.
- `stepping_stone_log_volume::Float64`: Log-volume via stepping-stone (always computed).
- `pt`: The Pigeons `PT` object for diagnostics (swap rates, round trips, etc.).
"""
struct VolumeSolution
    log_volume::Float64
    volume::Float64
    stepping_stone_log_volume::Float64
    pt
end

"""
    solve(prob::VolumeProblem; n_rounds=10, n_chains=10, estimator=:stepping_stone, kwargs...)

Estimate the volume of a region defined by `prob.membership` using parallel tempering
via Pigeons.jl.

# Keyword Arguments
- `n_rounds::Int`: Number of PT adaptation rounds (samples double each round). Default: `10`.
- `n_chains::Int`: Number of tempering chains. Default: `10`.
- `estimator::Symbol`: Method for the normalizing-constant ratio.
  `:stepping_stone` (default) or `:mbar` (via PyMBAR.jl).
- `decorrelate::Bool`: When `estimator=:mbar`, decorrelate samples via
  `pymbar.timeseries` before running MBAR. Default: `false`.
- `kmax_options::NamedTuple`: Keyword arguments forwarded to `find_kmax` (e.g.
  `kmax_options=(; target=0.9, n_samples=2000)`). Default: `(;)`.
- `kwargs...`: Additional keyword arguments passed to `Pigeons.pigeons()`.

# Mathematical Details
The volume is recovered from the normalizing constant ratio:

    log(Volume) = log(Z_target / Z_ref) + (dim/2) * log(2π * σ²) + log(p_acc)

where the ratio is estimated by stepping stone or MBAR, and
`Z_ref = (2πσ²)^(dim/2) * p_acc` is the partition function of the Gaussian
truncated to the membership region.
"""
function CommonSolve.solve(prob::VolumeProblem;
        n_rounds::Int = 10,
        n_chains::Int = 10,
        estimator::Symbol = :stepping_stone,
        decorrelate::Bool = false,
        kmax_options::NamedTuple = (;),
        kwargs...)

    estimator in (:stepping_stone, :mbar) ||
        throw(ArgumentError("estimator must be :stepping_stone or :mbar, got :$estimator"))

    decorrelate && estimator != :mbar &&
        @warn "decorrelate=true has no effect with estimator=:$estimator"

    # Estimate sigma and acceptance fraction via kmax
    (; kmax, acceptance) = find_kmax(prob.membership, prob.x0; kmax_options...)
    sigma = 1 / sqrt(kmax)
    p_acc = acceptance

    target = VolumeLogPotential(prob.membership, prob.dim, sigma, prob.x0)

    # When not using stepping stone, add a (possibly thinned) traces recorder
    pigeons_kw = Dict{Symbol, Any}(kwargs)
    if estimator != :stepping_stone
        record = Any[get(pigeons_kw, :record, Pigeons.record_default())...]
        if !any(r -> Symbol(r) == :traces, record)
            trace_thin = get(pigeons_kw, :trace_thin, 1)
            push!(record, trace_thin > 1 ? thin_traces(trace_thin) : Pigeons.traces)
        end
        delete!(pigeons_kw, :trace_thin)
        pigeons_kw[:record] = record
        pigeons_kw[:extended_traces] = true
    end

    pt = Pigeons.pigeons(;
        target = target,
        n_rounds = n_rounds,
        n_chains = n_chains,
        pigeons_kw...
    )

    log_ref = (prob.dim / 2) * log(2 * π * sigma^2) + log(p_acc)

    ss_log_vol = Pigeons.stepping_stone(pt) + log_ref

    if estimator == :stepping_stone
        log_vol = ss_log_vol
    elseif estimator == :mbar
        log_ratio = mbar_log_ratio(pt; decorrelate)
        log_vol = log_ratio + log_ref
    end

    VolumeSolution(log_vol, exp(log_vol), ss_log_vol, pt)
end
