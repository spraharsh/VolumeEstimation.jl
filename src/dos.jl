"""
    dos_from_offsets(visits, log_dos_all, offsets; nodata_value=0.0)

Combine per-chain log-DOS estimates into a single estimate using MBAR offsets.

Computes a visit-weighted average of offset-corrected log-DOS across chains:

    ldos[j] = Σ_k (log_dos[k,j] + w_k) * visits[k,j] / Σ_k visits[k,j]

# Arguments
- `visits::Matrix{Float64}`: K × nbins density-normalized histograms per chain.
- `log_dos_all::Matrix{Float64}`: K × nbins per-chain log-DOS (before offset correction).
- `offsets::Vector{Float64}`: length-K MBAR free energy offsets (`-Δf[1, :]`).

# Returns
`Vector{Float64}` of length nbins: combined log-DOS.
"""
function dos_from_offsets(visits::Matrix{Float64}, log_dos_all::Matrix{Float64},
                          offsets::Vector{Float64}; nodata_value::Float64 = 0.0)
    shifted = log_dos_all .+ offsets  # K-vector broadcasts along dim=2
    numerator = vec(sum(shifted .* visits; dims=1))
    norm = vec(sum(visits; dims=1))
    return [n > 0 ? num / n : nodata_value for (num, n) in zip(numerator, norm)]
end


"""
    compute_dos(pt; decorrelate=true, nbins=200)

Compute the density of states g(r) from a Pigeons PT result using MBAR.

Requires the PT run to have been configured with `extended_traces = true` and
a traces recorder (as done by `solve(...; estimator=:mbar)`).

# Keyword Arguments
- `decorrelate::Bool`: Decorrelate samples via pymbar.timeseries. Default: `true`.
- `nbins::Int`: Number of histogram bins for radial distance. Default: `200`.

# Returns
A named tuple with fields:
- `bin_centers::Vector{Float64}`: radial distance at bin centers.
- `log_dos::Vector{Float64}`: combined log(g(r)) estimate.
- `log_dos_per_chain::Matrix{Float64}`: K × nbins per-chain log(g(r)) after
  offset correction (for diagnostic plots).
- `hist_visits::Matrix{Float64}`: K × nbins density-normalized histograms.
- `betas::Vector{Float64}`: beta schedule from PT.
- `k_eff::Vector{Float64}`: effective spring constants `(1-β_k) * precision`.
- `dim::Int`: problem dimensionality.
- `r_by_chain::Vector{Vector{Float64}}`: raw radial distances per chain.
- `log_ratio::Float64`: `log(Z_target / Z_ref)` from MBAR.
"""
function compute_dos(pt; decorrelate::Bool = true, nbins::Int = 200)
    # 1. Extract samples and radial distances
    (; u_kn, N_k, r_by_chain) = extract_u_kn(pt; decorrelate)
    K = length(N_k)

    # 2. Access PT metadata
    betas = collect(Float64, pt.shared.tempering.schedule.grids)
    precision = pt.shared.tempering.path.ref.precision
    dim = pt.shared.tempering.path.ref.dim
    k_eff = [(1 - beta) * precision for beta in betas]

    # 3. Determine bin edges from pooled r values
    all_r = vcat(r_by_chain...)
    r_min = minimum(all_r)
    r_max = maximum(all_r)
    bin_edges = range(r_min, r_max; length=nbins + 1)
    bin_width = step(bin_edges)
    bin_centers = collect(bin_edges[1:end-1]) .+ bin_width / 2

    # 4. Build density-normalized histograms per chain
    hist_visits = zeros(K, nbins)
    for k in 1:K
        rs = r_by_chain[k]
        isempty(rs) && continue
        for r in rs
            j = clamp(floor(Int, (r - r_min) / bin_width) + 1, 1, nbins)
            hist_visits[k, j] += 1.0
        end
        hist_visits[k, :] ./= (N_k[k] * bin_width)
    end

    # 5. Unbias histograms: u_k(r) = (1 - β_k) * (precision/2) * r²
    hist_unbiased = zeros(K, nbins)
    for k in 1:K
        for j in 1:nbins
            hist_unbiased[k, j] = 0.5 * k_eff[k] * bin_centers[j]^2
        end
    end

    # 6. Per-chain log-DOS: log(hist) + u_k(r)
    log_dos_raw = zeros(K, nbins)
    for k in 1:K, j in 1:nbins
        if hist_visits[k, j] > 0
            log_dos_raw[k, j] = log(hist_visits[k, j]) + hist_unbiased[k, j]
        end
    end

    # 7. MBAR: get free energy offsets
    initial_f_k = _stepping_stone_f_k(pt)
    solver_protocol = (
        Dict("method" => "L-BFGS-B", "tol" => 1e-8,
             "options" => Dict("maxiter" => 10000)),
    )
    mbar_obj = PyMBAR.pymbar.MBAR(u_kn, N_k;
        initial_f_k=initial_f_k, solver_protocol=solver_protocol)
    results = mbar_obj.compute_free_energy_differences()
    Delta_f = results["Delta_f"]
    offsets = Float64[-Delta_f[1, k] for k in 1:K]

    # log(Z_target / Z_ref) = -Delta_f[1, K]
    log_ratio = -Delta_f[1, K]

    # 8. Combine via dos_from_offsets
    ldos = dos_from_offsets(hist_visits, log_dos_raw, offsets)

    # 9. Per-chain log-DOS with offsets (for diagnostic overlay plot)
    log_dos_per_chain = log_dos_raw .+ offsets

    return (;
        bin_centers,
        log_dos = ldos,
        log_dos_per_chain,
        hist_visits,
        betas,
        k_eff,
        dim,
        r_by_chain,
        log_ratio,
    )
end


"""
    shape_factor(dos_result)

Compute the shape factor `g(r)/r^(d-1)` from a `compute_dos` result.

Removes the radial Jacobian from the density of states, revealing the
intrinsic geometry of the membership region. For a spherical region,
the shape factor is constant.

# Returns
A named tuple `(; bin_centers, log_shape, shape)`.
"""
function shape_factor(dos_result)
    (; bin_centers, log_dos, dim) = dos_result
    nbins = length(bin_centers)

    log_shape = copy(log_dos)
    for j in 1:nbins
        if bin_centers[j] > 0 && isfinite(log_dos[j]) && log_dos[j] != 0.0
            log_shape[j] -= (dim - 1) * log(bin_centers[j])
        end
    end

    # Normalize: shift so mean of first few finite values is 0
    finite_idx = findall(v -> isfinite(v) && v != 0.0, log_shape)
    if !isempty(finite_idx)
        n_norm = min(3, length(finite_idx))
        shift = sum(log_shape[finite_idx[1:n_norm]]) / n_norm
        log_shape .-= shift
    end

    shape = [isfinite(v) && v != 0.0 ? exp(v) : 0.0 for v in log_shape]

    return (; bin_centers, log_shape, shape)
end


"""
    produce_analysis(pt, outdir; decorrelate=true, nbins=200)

Generate all density-of-states diagnostic plots from a Pigeons PT result.

Requires the `Plots` and `Statistics` packages to be loaded (package extension).
"""
function produce_analysis end
