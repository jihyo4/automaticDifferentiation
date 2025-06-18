using Random

struct DataLoader{T}
    data::T
    batchsize::Int
    shuffle::Bool
    indices::Vector{Vector{Int}}
end

function DataLoader(data::T; batchsize::Int=1, shuffle::Bool=false) where T
    n = size(data[1], 2)
    @assert all(d -> size(d, 2) == n, data)

    order = shuffle ? randperm(n) : collect(1:n)
    batches = [order[i:min(i + batchsize - 1, n)] for i in 1:batchsize:n]

    return DataLoader{T}(data, batchsize, shuffle, batches)
end
Base.IteratorSize(::Type{<:DataLoader}) = Base.HasLength()
Base.length(dl::DataLoader) = length(dl.indices)

function Base.iterate(dl::DataLoader, state=1)
    state > length(dl.indices) && return nothing

    idx = dl.indices[state]
    batch = map(d -> d[:, idx], dl.data)
    return (batch, state + 1)
end


