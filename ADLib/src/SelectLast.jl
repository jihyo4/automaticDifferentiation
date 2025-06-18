mutable struct SelectLast{T} <: AbstractLayer{T}
    seq_len::Int
    grad_input::Array{T, 3}
    function SelectLast{T}() where T
        new{T}(0, zeros(T, 1, 1, 1))
    end
end

function (layer::SelectLast{T})(x::Array{T, 3}) where T
    layer.seq_len = size(x, 2)
    return @view x[:, end, :]
end

function backward_layer(layer::SelectLast{T}, grad_output::Array{T,2}) where T
    hidden_dim, batch_size = size(grad_output)

    if size(layer.grad_input) != (hidden_dim, layer.seq_len, batch_size)
        layer.grad_input = zeros(T, hidden_dim, layer.seq_len, batch_size)
    else
        fill!(layer.grad_input, zero(T))
    end
    @views layer.grad_input[:, end, :] .= grad_output
    return layer.grad_input
end