using Test
using Volumes
using CommonSolve: solve

@testset "Volumes.jl" begin
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
end
