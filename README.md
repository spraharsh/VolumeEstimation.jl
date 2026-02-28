# VolumeEstimation.jl

Estimate the volume of arbitrary high-dimensional regions in Julia.

Given a boolean membership function that defines a region, VolumeEstimation.jl estimates its volume using parallel tempering and stepping-stone sampling via [Pigeons.jl](https://github.com/Julia-Tempering/Pigeons.jl).

## Quick Start

```julia
using VolumeEstimation, CommonSolve

# Estimate the volume of a 5D unit hypercube (true volume = 1)
membership(x) = all(abs.(x) .<= 0.5)
prob = VolumeProblem(membership, 5)
sol = solve(prob)

sol.log_volume  # ≈ 0.0 (log of 1)
sol.volume      # ≈ 1.0
```

```julia
# Estimate the volume of a 3D simplex (true volume = 1/6)
membership(x) = all(x .>= 0) && sum(x) <= 1
prob = VolumeProblem(membership, 3; x0=[1/4, 1/4, 1/4])
sol = solve(prob; n_rounds=15, n_chains=10)

sol.volume  # ≈ 0.1667
```

## How It Works

The volume of a region S is the normalizing constant of the uniform distribution on S. VolumeEstimation.jl computes this by constructing a tempered path between:

- **Target**: uniform distribution on S (log-density 0 inside, -Inf outside)
- **Reference**: Gaussian N(x0, sigma^2 I) truncated to S, whose partition function is known analytically. The scale sigma is chosen automatically (see below).

Pigeons.jl runs parallel tempering across this path, and stepping-stone sampling gives the log normalizing constant ratio. The volume is then:

```
log(Volume) = log(Z_target / Z_ref) + log(Z_ref)
```

where `Z_ref = (2*pi*sigma^2)^(dim/2) * p_acc` and `p_acc` is the fraction of the Gaussian that falls inside S.

### Automatic Scale Selection

VolumeEstimation.jl automatically finds the right Gaussian scale by searching for `kmax` -- the maximum spring constant k such that ~95% of samples from N(x0, (1/k)I) land inside the region. The reference standard deviation is then `sigma = 1/sqrt(kmax)`.

## API

### `VolumeProblem(membership, dim; x0=nothing)`

- `membership`: function `f(x) -> Bool`, returns `true` if `x` is inside the region
- `dim`: number of dimensions
- `x0`: a point known to be inside the region (defaults to the origin)

### `solve(prob; n_rounds=10, n_chains=10, kwargs...)`

Runs parallel tempering and returns a `VolumeSolution` with fields:

- `log_volume`: log of the estimated volume (use this in high dimensions to avoid underflow)
- `volume`: `exp(log_volume)`
- `pt`: the Pigeons `PT` object for diagnostics (swap rates, round trips, etc.)

Additional keyword arguments are forwarded to `Pigeons.pigeons()`.
