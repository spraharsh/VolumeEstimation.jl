"""
    ThinTraces <: AbstractDict{Pair{Int,Int}, Any}

A drop-in replacement for Pigeons' `traces` recorder that only stores
every `thin`-th scan. This reduces memory usage by a factor of `thin`
while keeping enough samples for MBAR.

The recorder registers under the `:traces` key so that Pigeons'
`@record_if_requested!(:traces, ...)` finds it transparently.
"""
struct ThinTraces <: AbstractDict{Pair{Int,Int}, Any}
    thin::Int
    dict::Dict{Pair{Int,Int}, Any}
    function ThinTraces(thin::Int, dict::Dict{Pair{Int,Int}, Any} = Dict{Pair{Int,Int}, Any}())
        thin > 0 || throw(ArgumentError("thin must be positive, got $thin"))
        new(thin, dict)
    end
end

# Pigeons record! interface — skip scans that aren't multiples of thin
function Pigeons.record!(recorder::ThinTraces, datum)
    if datum.scan % recorder.thin == 0
        key = datum.chain => datum.scan
        recorder.dict[key] = datum.contents
    end
end

Base.empty!(recorder::ThinTraces) = (empty!(recorder.dict); recorder)
function Base.merge(a::ThinTraces, b::ThinTraces)
    a.thin == b.thin || throw(ArgumentError("cannot merge ThinTraces with different thin values: $(a.thin) vs $(b.thin)"))
    ThinTraces(a.thin, merge(a.dict, b.dict))
end

# AbstractDict interface
Base.iterate(r::ThinTraces, args...) = iterate(r.dict, args...)
Base.length(r::ThinTraces) = length(r.dict)
Base.getindex(r::ThinTraces, k) = r.dict[k]

struct ThinTracesBuilder <: Function
    thin::Int
end

# Pigeons calls recorder_builder() to create the recorder
(b::ThinTracesBuilder)() = ThinTraces(b.thin)

# Pigeons uses Symbol(recorder_builder) as the NamedTuple key — must be :traces
Base.Symbol(::ThinTracesBuilder) = :traces

"""
    thin_traces(thin::Int) -> ThinTracesBuilder

Create a recorder builder for thinned traces. Use in place of
`Pigeons.traces` in the `record` list:

    solve(prob; record=[thin_traces(4), Pigeons.record_default()])

This records every 4th scan, reducing trace memory by `thin`×.
"""
thin_traces(thin::Int) = ThinTracesBuilder(thin)
