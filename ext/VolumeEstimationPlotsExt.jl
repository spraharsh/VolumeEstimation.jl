module VolumeEstimationPlotsExt

using VolumeEstimation
import Plots
import Statistics

"""
    produce_analysis(pt, outdir; decorrelate=true, nbins=200)

Generate all density-of-states diagnostic plots from a Pigeons PT result.

Produces the same plots as the Python `mbar_compute_volume`:
- `histograms.png`: per-chain density histograms of radial distance r
- `raw_log_dos.png`: per-chain unbiased log-DOS overlaid (should collapse)
- `dos.png`: combined density of states g(r)
- `log_dos.png`: log ξ(r) and log(ξ(r)/r^(d-1)) (shape factor)
- `ratio_g.png`: shape factor g(r)/r^(d-1) in linear scale
- `ratio_g_loglog.png`: log(g(r)/r^(d-1)) with log r axis
- `hist_var_k.png`: variance of r vs effective spring constant

# Arguments
- `pt`: Pigeons PT result (with traces enabled via `estimator=:mbar`).
- `outdir::AbstractString`: directory to save plots (created if needed).

# Keyword Arguments
- `decorrelate::Bool`: decorrelate samples before analysis. Default: `true`.
- `nbins::Int`: number of histogram bins. Default: `200`.

# Returns
The `compute_dos` named tuple (for further programmatic use).
"""
function VolumeEstimation.produce_analysis(pt, outdir::AbstractString;
                                  decorrelate::Bool = true, nbins::Int = 200)
    mkpath(outdir)

    dos = VolumeEstimation.compute_dos(pt; decorrelate, nbins)
    sf = VolumeEstimation.shape_factor(dos)

    K = length(dos.betas)
    finite = isfinite.(dos.log_dos) .& (dos.log_dos .!= 0.0)
    bc_f = dos.bin_centers[finite]

    # --- Histograms ---
    p = Plots.plot(; title="Histograms", xlabel="r", ylabel="density")
    for k in 1:K
        mask = dos.hist_visits[k, :] .> 0
        any(mask) || continue
        Plots.plot!(p, dos.bin_centers[mask], dos.hist_visits[k, mask];
            label="k_eff=$(round(dos.k_eff[k]; digits=1))", linewidth=1.5)
    end
    Plots.savefig(p, joinpath(outdir, "histograms.png"))

    # --- Raw log-DOS per chain ---
    p = Plots.plot(; title="Raw log-DOS per chain", xlabel="r", ylabel="log g(r) + offset")
    for k in 1:K
        mask = dos.hist_visits[k, :] .> 0
        any(mask) || continue
        Plots.plot!(p, dos.bin_centers[mask], dos.log_dos_per_chain[k, mask];
            label="β=$(round(dos.betas[k]; digits=2))", linewidth=1.5)
    end
    Plots.savefig(p, joinpath(outdir, "raw_log_dos.png"))

    # --- Combined DOS g(r) ---
    log_dos_f = dos.log_dos[finite]
    log_dos_shifted = log_dos_f .- maximum(log_dos_f)
    g = exp.(log_dos_shifted)
    bin_width = length(bc_f) > 1 ? bc_f[2] - bc_f[1] : 1.0
    g ./= sum(g) * bin_width  # normalize to integrate to 1
    p = Plots.plot(bc_f, g; title="Density of states", xlabel="r", ylabel="g(r)",
        label="g(r)", linewidth=2)
    Plots.savefig(p, joinpath(outdir, "dos.png"))

    # --- Log DOS with shape factor ---
    p = Plots.plot(; title="Log DOS", xlabel="r")
    Plots.plot!(p, bc_f, log_dos_f .- maximum(log_dos_f);
        label="log ξ(r)", linewidth=2)
    Plots.plot!(p, bc_f, sf.log_shape[finite];
        label="log(ξ(r)/r^(d-1))", linewidth=2, linestyle=:dash)
    Plots.savefig(p, joinpath(outdir, "log_dos.png"))

    # --- Shape factor linear ---
    p = Plots.plot(sf.bin_centers[finite], sf.shape[finite];
        title="Shape factor", xlabel="r", ylabel="g(r)/r^(d-1)",
        label="", linewidth=2)
    Plots.savefig(p, joinpath(outdir, "ratio_g.png"))

    # --- Shape factor log-log ---
    p = Plots.plot(sf.bin_centers[finite], sf.log_shape[finite];
        title="Shape factor (log-log)", xlabel="r", ylabel="log(g(r)/r^(d-1))",
        xscale=:log10, label="", linewidth=2)
    Plots.savefig(p, joinpath(outdir, "ratio_g_loglog.png"))

    # --- Variance vs k_eff ---
    vars = [isempty(rs) ? 0.0 : Statistics.var(rs) for rs in dos.r_by_chain]
    p = Plots.plot(dos.k_eff, vars; title="Variance vs k_eff", xlabel="k_eff",
        ylabel="var(r)", seriestype=:scatter, label="", markersize=4)
    Plots.savefig(p, joinpath(outdir, "hist_var_k.png"))

    @info "Saved 7 plots to $outdir"
    return dos
end

end # module VolumeEstimationPlotsExt
