using Test
using VolumeEstimation
using CommonSolve: solve
using LinearAlgebra: norm

@testset "VolumeEstimation.jl" begin
    @testset "Unit Hypercube (auto sigma) d=$d" for d in [2, 3, 5]
        membership(x) = all(abs.(x) .<= 0.5)
        prob = VolumeProblem(membership, d)  # sigma estimated via kmax
        sol = solve(prob; n_rounds=15, n_chains=10)

        @test sol.volume > 0
        @test isapprox(sol.log_volume, 0.0, rtol=0.03, atol=0.05) # log(1.0) = 0
        @info "d=$d: estimated volume=$(sol.volume), log_volume=$(sol.log_volume)"
    end

    @testset "Scaled Hypercube (auto sigma)" begin
        d = 3
        L = 2.0
        membership(x) = all(abs.(x) .<= L / 2)
        prob = VolumeProblem(membership, d)  # sigma estimated via kmax
        sol = solve(prob; n_rounds=15, n_chains=10)

        true_log_volume = d * log(L)
        @test isapprox(sol.log_volume, true_log_volume, rtol=0.03, atol=0.05)
        @info "Scaled cube: estimated=$(sol.volume), true=$(L^d)"
    end

    @testset "Standard Simplex d=$d" for d in [2, 3, 5]
        # n-simplex: x_i >= 0 for all i, sum(x) <= 1
        # Volume = 1/d!, so log(Volume) = -log(d!)
        membership(x) = all(x .>= 0) && sum(x) <= 1
        x0 = fill(1.0 / (d + 1), d)  # centroid of simplex
        prob = VolumeProblem(membership, d; x0=x0)
        sol = solve(prob; n_rounds=15, n_chains=10)

        true_log_volume = -sum(log.(1:d))
        @test isapprox(sol.log_volume, true_log_volume, rtol=0.03, atol=0.05)
        @info "Simplex d=$d: estimated=$(sol.log_volume), true=$(true_log_volume)"
    end

    @testset "Cross Polytope d=$d" for d in [2, 3, 5]
        # n-cross-polytope: sum(|x_i|) <= 1
        # Volume = 2^d / d!, so log(Volume) = d*log(2) - log(d!)
        membership(x) = sum(abs.(x)) <= 1
        prob = VolumeProblem(membership, d)
        sol = solve(prob; n_rounds=15, n_chains=10)

        true_log_volume = d * log(2) - sum(log.(1:d))
        @test isapprox(sol.log_volume, true_log_volume, rtol=0.03, atol=0.05)
        @info "Cross polytope d=$d: estimated=$(sol.log_volume), true=$(true_log_volume)"
    end

    @testset "VolumeProblem validation" begin
        bad_membership(x) = false
        @test_throws ArgumentError VolumeProblem(bad_membership, 2)
    end

    @testset "MBAR estimator" begin
        @testset "MBAR Unit Hypercube d=$d" for d in [2, 3]
            membership(x) = all(abs.(x) .<= 0.5)
            prob = VolumeProblem(membership, d)
            sol = solve(prob; n_rounds=15, n_chains=10, estimator=:mbar)

            @test sol.volume > 0
            @test isapprox(sol.log_volume, 0.0, atol=0.1)
            @info "MBAR d=$d: estimated volume=$(sol.volume), log_volume=$(sol.log_volume)"
        end

        @testset "MBAR with thinned traces" begin
            membership(x) = all(abs.(x) .<= 0.5)
            prob = VolumeProblem(membership, 3)
            sol = solve(prob; n_rounds=15, n_chains=10, estimator=:mbar, trace_thin=2)

            @test sol.volume > 0
            @test isapprox(sol.log_volume, 0.0, atol=0.1)
        end

        @testset "MBAR invalid estimator" begin
            membership(x) = all(abs.(x) .<= 0.5)
            prob = VolumeProblem(membership, 2)
            @test_throws ArgumentError solve(prob; estimator=:invalid)
        end
    end

    @testset "DOS computation" begin
        @testset "dos_from_offsets" begin
            visits = [1.0 2.0 0.0;
                      0.0 1.0 3.0]
            log_dos = [0.5 1.0 0.0;
                       0.0 0.8 1.2]
            offsets = [0.0, 0.1]

            ldos = VolumeEstimation.dos_from_offsets(visits, log_dos, offsets)

            @test length(ldos) == 3
            # Bin 1: only chain 1 → (0.5 + 0.0) * 1.0 / 1.0 = 0.5
            @test ldos[1] ≈ 0.5
            # Bin 3: only chain 2 → (1.2 + 0.1) * 3.0 / 3.0 = 1.3
            @test ldos[3] ≈ 1.3
            # Bin 2: both → ((1.0+0.0)*2.0 + (0.8+0.1)*1.0) / 3.0
            @test ldos[2] ≈ (2.0 + 0.9) / 3.0
        end

        @testset "extract_u_kn returns r_by_chain" begin
            membership(x) = all(abs.(x) .<= 0.5)
            prob = VolumeProblem(membership, 3)
            sol = solve(prob; n_rounds=10, n_chains=5, estimator=:mbar)

            result = extract_u_kn(sol.pt; decorrelate=false)

            @test hasproperty(result, :r_by_chain)
            @test length(result.r_by_chain) == 5
            for k in 1:5
                @test all(r >= 0 for r in result.r_by_chain[k])
                @test length(result.r_by_chain[k]) == result.N_k[k]
            end

            # Backward compatibility
            (; u_kn, N_k) = extract_u_kn(sol.pt; decorrelate=false)
            @test size(u_kn, 1) == 5
        end

        @testset "r_by_chain correctness" begin
            membership(x) = all(abs.(x) .<= 0.5)
            prob = VolumeProblem(membership, 2)
            sol = solve(prob; n_rounds=8, n_chains=5, estimator=:mbar)
            pt = sol.pt

            result = extract_u_kn(pt; decorrelate=false)
            x0 = pt.shared.tempering.path.ref.x0

            # Manually recompute r from traces
            traces = pt.reduced_recorders.traces
            chain_scans = Dict{Int, Vector{Pair{Int, Vector{Float64}}}}()
            for ((chain, scan), contents) in traces
                d = length(contents) - 1
                v = get!(chain_scans, chain, Pair{Int, Vector{Float64}}[])
                push!(v, scan => contents[1:d])
            end
            for v in values(chain_scans)
                sort!(v; by=first)
            end

            for k in 1:5
                xs = [p.second for p in chain_scans[k]]
                expected_r = [norm(x - x0) for x in xs]
                @test result.r_by_chain[k] ≈ expected_r
            end
        end

        @testset "compute_dos structure" begin
            membership(x) = all(abs.(x) .<= 0.5)
            prob = VolumeProblem(membership, 3)
            sol = solve(prob; n_rounds=12, n_chains=8, estimator=:mbar)

            dos = compute_dos(sol.pt; decorrelate=true, nbins=100)

            @test length(dos.bin_centers) == 100
            @test length(dos.log_dos) == 100
            @test size(dos.hist_visits) == (8, 100)
            @test size(dos.log_dos_per_chain) == (8, 100)
            @test length(dos.betas) == 8
            @test length(dos.k_eff) == 8
            @test dos.dim == 3
            @test issorted(dos.betas)
            @test issorted(dos.bin_centers)
            @test all(dos.hist_visits .>= 0)

            sf = shape_factor(dos)
            @test length(sf.log_shape) == 100
            @test all(sf.shape .>= 0)
        end
    end
end
