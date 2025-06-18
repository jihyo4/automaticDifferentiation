mutable struct Embedding{T} <: AbstractLayer{T}
    weights::Matrix{T}
    grad_weights::Matrix{T}
    input_indices::Matrix{Int}
    mW::Matrix{T}
    vW::Matrix{T}
    in::Integer
    out::Integer

    function Embedding{T}(vocab_size::Int, embedding_dim::Int) where T
        weights = glorot_uniform(T, embedding_dim, vocab_size)
        grad_weights = zeros(T, size(weights)...)
        input_indices = zeros(Int, 1, 1)
        mW = zeros(T, size(weights)...)
        vW = zeros(T, size(weights)...)
        new{T}(weights, grad_weights, input_indices, mW, vW, vocab_size, embedding_dim)
    end
end

function (layer::Embedding)(x::AbstractMatrix{Int})
    layer.input_indices = x
    return reshape(layer.weights[:, vec(x)], size(layer.weights, 1), size(x)...)
end

function backward_layer(layer::Embedding{T}, grad_output::Array{T, 3}) where T
    seq_len, batch_size = size(layer.input_indices)
    for i in 1:seq_len
        for j in 1:batch_size
            idx = layer.input_indices[i, j]
            layer.grad_weights[:, idx] .+= grad_output[:, i, j]
        end
    end
    return nothing
end