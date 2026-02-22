using Volumes
using CommonSolve: solve
using Plots; gr()
using Pigeons
using Statistics
using Printf
using Test

outdir = joinpath(@__DIR__, "plots")
mkpath(outdir)

# ── Geometry definitions ─────────────────────────────────────────────
cube_membership(x) = all(abs.(x) .<= 0.5)
cube_true_logvol(d) = 0.0  # log(1^d) = 0

# Elongated cuboid: side 4 in first 2 dims, side 1 in rest
function cuboid_membership(x)
    for i in eachindex(x)
        half = i <= 2 ? 2.0 : 0.5
        abs(x[i]) > half && return false
    end
    return true
end
cuboid_true_logvol(d) = 2 * log(4.0)  # 4^2 * 1^(d-2)

# ── Configuration ────────────────────────────────────────────────────
dims   = [2, 5, 10, 20, 50, 100]
n_reps = 5   # independent runs per (geometry, dimension)

geometries = [
    ("cube",   cube_membership,   cube_true_logvol),
    ("cuboid", cuboid_membership, cuboid_true_logvol),
]

# ── Storage ──────────────────────────────────────────────────────────
struct RunResult
    dim::Int
    logvols::Vector{Float64}      # one per rep
    true_logvol::Float64
    barriers::Vector{Float64}     # global barrier Λ per rep
    round_trips::Vector{Int}      # round trips per rep
    restarts::Vector{Int}         # tempered restarts per rep
end

all_results = Dict{String, Vector{RunResult}}()

# ── Main loop ────────────────────────────────────────────────────────
for (name, membership, true_lv_fn) in geometries
    all_results[name] = RunResult[]

    for d in dims
        true_lv = true_lv_fn(d)
        logvols   = Float64[]
        barriers  = Float64[]
        trips     = Int[]
        restarts  = Int[]
        last_pt   = nothing

        for rep in 1:n_reps
            @info "── $name  d=$d  rep=$rep/$n_reps ──"

            prob = VolumeProblem(membership, d)
            sol = solve(prob;
                n_rounds     = 15,
                n_chains     = 10,
                multithreaded = true,
                record       = [Pigeons.index_process, round_trip],
            )
            pt = sol.pt

            lv  = sol.log_volume
            Λ   = Pigeons.global_barrier(pt)
            nrt = n_round_trips(pt)
            nrs = n_tempered_restarts(pt)

            push!(logvols,  lv)
            push!(barriers, Λ)
            push!(trips,    nrt)
            push!(restarts, nrs)

            @info @sprintf("  logV̂=%.3f  true=%.3f  err=%+.3f  Λ=%.2f  round_trips=%d  restarts=%d",
                           lv, true_lv, lv - true_lv, Λ, nrt, nrs)

            last_pt = pt
        end

        push!(all_results[name], RunResult(d, logvols, true_lv, barriers, trips, restarts))

        # ── Pigeons diagnostic plots (from last rep) ─────────
        p1 = plot(last_pt.shared.tempering.communication_barriers.localbarrier,
                  title="Local Barrier — $name d=$d")
        savefig(p1, joinpath(outdir, "$(name)_d$(d)_barrier.png"))

        p2 = plot(last_pt.reduced_recorders.index_process,
                  title="Index Process — $name d=$d")
        savefig(p2, joinpath(outdir, "$(name)_d$(d)_index.png"))
    end
end

# ── Summary table ────────────────────────────────────────────────────
println("\n" * "="^100)
println("  VOLUME ESTIMATION RESULTS  ($n_reps independent runs per configuration)")
println("="^100)

for (name, _...) in geometries
    println("\n── $name ──")
    @printf("  %5s  %10s  %10s  %10s  %10s  %8s  %10s  %8s\n",
            "dim", "true logV", "mean logV̂", "std err", "95% CI±", "mean Λ", "mean trips", "covered?")
    println("  " * "-"^90)

    for r in all_results[name]
        μ   = mean(r.logvols)
        se  = std(r.logvols) / sqrt(length(r.logvols))
        ci  = 1.96 * se
        covered = abs(μ - r.true_logvol) < ci ? "YES" : "no"
        mΛ  = mean(r.barriers)
        mrt = mean(r.round_trips)

        @printf("  %5d  %10.3f  %10.3f  %10.4f  %10.4f  %8.2f  %10.1f  %8s\n",
                r.dim, r.true_logvol, μ, se, ci, mΛ, mrt, covered)

        @test abs(μ - r.true_logvol) < max(ci * 3, 1.0)  # generous: within 3σ or 1 nat
    end
end

# ── Summary plots ────────────────────────────────────────────────────

# 1) log-volume: mean ± CI vs dimension, with true value
for (name, _...) in geometries
    rs = all_results[name]
    ds    = [r.dim for r in rs]
    means = [mean(r.logvols) for r in rs]
    cis   = [1.96 * std(r.logvols) / sqrt(length(r.logvols)) for r in rs]
    trues = [r.true_logvol for r in rs]

    p = plot(ds, means; ribbon=cis,
        xlabel="dimension", ylabel="log volume",
        title="$name: estimated vs true log-volume",
        label="estimated (mean ± 95% CI)", marker=:circle, linewidth=2, fillalpha=0.3)
    plot!(p, ds, trues; label="true", linestyle=:dash, linewidth=2, color=:red)
    savefig(p, joinpath(outdir, "$(name)_logvol_vs_dim.png"))
end

# 2) Error plot: mean error ± CI
for (name, _...) in geometries
    rs = all_results[name]
    ds   = [r.dim for r in rs]
    errs = [mean(r.logvols) - r.true_logvol for r in rs]
    cis  = [1.96 * std(r.logvols) / sqrt(length(r.logvols)) for r in rs]

    p = plot(ds, errs; ribbon=cis,
        xlabel="dimension", ylabel="log-volume error",
        title="$name: estimation error (mean ± 95% CI)",
        label="error", marker=:circle, linewidth=2, fillalpha=0.3)
    hline!(p, [0.0]; linestyle=:dash, color=:gray, label="zero")
    savefig(p, joinpath(outdir, "$(name)_error_vs_dim.png"))
end

# 3) Global barrier Λ vs dimension
for (name, _...) in geometries
    rs = all_results[name]
    ds = [r.dim for r in rs]
    mΛ = [mean(r.barriers) for r in rs]

    p = plot(ds, mΛ; marker=:circle, linewidth=2, legend=false,
        xlabel="dimension", ylabel="global barrier Λ",
        title="$name: communication barrier vs dimension")
    savefig(p, joinpath(outdir, "$(name)_barrier_vs_dim.png"))
end

# 4) Round trips vs dimension
for (name, _...) in geometries
    rs = all_results[name]
    ds  = [r.dim for r in rs]
    mrt = [mean(r.round_trips) for r in rs]

    p = plot(ds, mrt; marker=:circle, linewidth=2, legend=false,
        xlabel="dimension", ylabel="mean round trips",
        title="$name: round trips vs dimension")
    savefig(p, joinpath(outdir, "$(name)_roundtrips_vs_dim.png"))
end

# 5) Combined error comparison
p = plot(title="log-volume error: cube vs cuboid",
         xlabel="dimension", ylabel="log-volume error",
         linewidth=2, marker=:circle)
for (name, _...) in geometries
    rs = all_results[name]
    ds   = [r.dim for r in rs]
    errs = [mean(r.logvols) - r.true_logvol for r in rs]
    cis  = [1.96 * std(r.logvols) / sqrt(length(r.logvols)) for r in rs]
    plot!(p, ds, errs; ribbon=cis, label=name, fillalpha=0.2)
end
hline!(p, [0.0]; linestyle=:dash, color=:gray, label="")
savefig(p, joinpath(outdir, "combined_error.png"))

# 6) Individual run scatter
for (name, _...) in geometries
    rs = all_results[name]
    p = plot(xlabel="dimension", ylabel="log-volume error",
             title="$name: all individual runs",
             legend=false)
    for r in rs
        errs = r.logvols .- r.true_logvol
        scatter!(p, fill(r.dim, length(errs)), errs; alpha=0.6, markersize=5)
    end
    hline!(p, [0.0]; linestyle=:dash, color=:gray)
    savefig(p, joinpath(outdir, "$(name)_scatter.png"))
end

@info "All plots saved to $outdir"
