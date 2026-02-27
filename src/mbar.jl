"""
    extract_u_kn(pt; decorrelate=false)

Extract the reduced potential energy matrix from a Pigeons PT result.

Requires the PT run to have been configured with traces recording
and `extended_traces = true` so that samples from all chains are available.
Thinning is handled at the recorder level via `thin_traces()`.

# Keyword Arguments
- `decorrelate::Bool`: If `true`, use `pymbar.timeseries` to detect equilibration
  and subsample correlated data per chain. Default: `true`.

# Returns
A named tuple `(u_kn, N_k, r_by_chain)` where:
- `u_kn::Matrix{Float64}`: K × N_total reduced potential matrix.
  `u_kn[k, n] = -log_potential_k(x_n)`.
- `N_k::Vector{Int}`: sample counts per chain.
- `r_by_chain::Vector{Vector{Float64}}`: radial distances `|x - x0|` for each
  sample, grouped by chain. `r_by_chain[k]` has length `N_k[k]`.
"""
function extract_u_kn(pt; decorrelate::Bool = true)
    traces = pt.reduced_recorders.traces
    log_potentials = pt.shared.tempering.log_potentials
    K = length(log_potentials)

    # Collect samples per chain, sorted by scan index for time-series order.
    # Each trace entry is [x_1, ..., x_d, log_density] (d+1 vector).
    chain_scans = Dict{Int, Vector{Pair{Int, Vector{Float64}}}}()
    for ((chain, scan), contents) in traces
        if !haskey(chain_scans, chain)
            chain_scans[chain] = Pair{Int, Vector{Float64}}[]
        end
        d = length(contents) - 1          # strip appended log_density
        push!(chain_scans[chain], scan => contents[1:d])
    end

    # Sort each chain's samples by scan index
    for v in values(chain_scans)
        sort!(v; by = first)
    end

    # Optional decorrelation via pymbar.timeseries
    chain_samples = Dict{Int, Vector{Vector{Float64}}}()
    if decorrelate
        timeseries = PyMBAR.pymbar.timeseries
        for k in 1:K
            pairs = get(chain_scans, k, Pair{Int, Vector{Float64}}[])
            xs = [p.second for p in pairs]
            if length(xs) < 3
                chain_samples[k] = xs
                continue
            end
            # Use the reference log potential as the decorrelation observable.
            # The chain's own log potential is degenerate (constant 0) for
            # near-target chains, making autocorrelation undetectable.
            # The reference potential -(precision/2)|x-x0|² varies across
            # all of S regardless of chain.
            A_t = Float64[log_potentials[1](x) for x in xs]
            t0, g, _ = timeseries.detect_equilibration(A_t)
            t0 = Int(t0) + 1  # Python 0-indexed → Julia 1-indexed
            indices = timeseries.subsample_correlated_data(A_t[t0:end], g)
            indices = Int.(indices) .+ 1
            chain_samples[k] = xs[t0:end][indices]
        end
    else
        for k in 1:K
            pairs = get(chain_scans, k, Pair{Int, Vector{Float64}}[])
            chain_samples[k] = [p.second for p in pairs]
        end
    end

    # Extract center point for radial distance computation
    x0 = pt.shared.tempering.path.ref.x0

    # Build N_k, flatten samples, and compute radial distances per chain
    N_k = zeros(Int, K)
    all_samples = Vector{Float64}[]
    r_by_chain = Vector{Vector{Float64}}(undef, K)
    for k in 1:K
        xs = chain_samples[k]
        N_k[k] = length(xs)
        append!(all_samples, xs)
        r_by_chain[k] = [norm(x - x0) for x in xs]
    end
    N_total = sum(N_k)

    # Build u_kn: cross-evaluate all log potentials at all samples
    u_kn = Matrix{Float64}(undef, K, N_total)
    for n in 1:N_total
        x_n = all_samples[n]
        for k in 1:K
            lp = log_potentials[k](x_n)
            u_kn[k, n] = isinf(lp) && lp < 0 ? Inf : -lp
        end
    end

    return (; u_kn, N_k, r_by_chain)
end


"""
    _stepping_stone_f_k(pt)

Extract per-state dimensionless free energy estimates from stepping stone,
suitable as `initial_f_k` for MBAR. Uses the sandwich estimator (average of
forward and backward) for each adjacent pair.

Returns a `Vector{Float64}` of length K with `f_k[1] = 0`.
"""
function _stepping_stone_f_k(pt)
    log_sum_ratios = pt.reduced_recorders.log_sum_ratio
    K = length(pt.shared.tempering.log_potentials)

    f_k = zeros(K)
    for k in 2:K
        fwd_key = (k-1, k)
        bwd_key = (k, k-1)
        has_fwd = haskey(log_sum_ratios.value, fwd_key)
        has_bwd = haskey(log_sum_ratios.value, bwd_key)

        if has_fwd && has_bwd
            fwd_ls = log_sum_ratios[fwd_key]
            bwd_ls = log_sum_ratios[bwd_key]
            fwd_est = fwd_ls.value - log(fwd_ls.n)
            bwd_est = -(bwd_ls.value - log(bwd_ls.n))
            log_ratio_k = (fwd_est + bwd_est) / 2.0
        elseif has_fwd
            fwd_ls = log_sum_ratios[fwd_key]
            log_ratio_k = fwd_ls.value - log(fwd_ls.n)
        elseif has_bwd
            bwd_ls = log_sum_ratios[bwd_key]
            log_ratio_k = -(bwd_ls.value - log(bwd_ls.n))
        else
            log_ratio_k = 0.0
        end

        # f_k = -log(Z_k / Z_1), so subtract each log(Z_k / Z_{k-1})
        f_k[k] = f_k[k-1] - log_ratio_k
    end

    return f_k
end


"""
    mbar_log_ratio(pt; decorrelate=true)

Compute `log(Z_target / Z_ref)` from a Pigeons PT result using MBAR.

This is the MBAR analog of `Pigeons.stepping_stone(pt)`. It can be called
post-hoc on any PT result that was run with traces enabled for all chains.
MBAR is initialized with stepping stone free energy estimates for faster
convergence.

# Returns
`log_ratio::Float64`: estimated `log(Z_target / Z_ref)`.
"""
function mbar_log_ratio(pt; decorrelate::Bool = true)
    (; u_kn, N_k) = extract_u_kn(pt; decorrelate)

    initial_f_k = _stepping_stone_f_k(pt)
    solver_protocol = (
        Dict("method" => "L-BFGS-B", "tol" => 1e-8, "options" => Dict("maxiter" => 10000)),
    )
    mbar_obj = PyMBAR.pymbar.MBAR(u_kn, N_k, initial_f_k=initial_f_k, solver_protocol=solver_protocol)
    results = mbar_obj.compute_free_energy_differences()
    Delta_f = results["Delta_f"]

    # Delta_f[i,j] = f_j - f_i = ln(Z_i / Z_j)
    # We want ln(Z_target / Z_ref) = ln(Z_K / Z_1) = -Delta_f[1, K]
    K = length(N_k)
    log_ratio = -Delta_f[1, K]

    return log_ratio
end
