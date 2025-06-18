struct Sequence{T}
    layers::Vector{AbstractLayer{T}}

    function Sequence(layers::AbstractLayer{T}...) where T
        for i in 1:length(layers)-1
            curr = layers[i]
            nxt  = layers[i+1]

            if hasfield(typeof(curr), :out) && hasfield(typeof(nxt), :in)
                curr_out = getfield(curr, :out)
                next_in  = getfield(nxt, :in)
                if curr_out != next_in
                    throw(ArgumentError("Layer $i output size ($curr_out) does not match layer $(i+1) input size ($next_in)"))
                end
            end
        end
        return new{T}(collect(layers))
    end
end

function (s::Sequence)(input)
    for layer in s.layers
        input = layer(input)
    end
    return input
end

function backward_pass(s::Sequence, grad_output::AbstractArray)
    for layer in reverse(s.layers)
        grad_output = backward_layer(layer, grad_output)
    end
    return grad_output
end

function update_weights(s::Sequence, optimizer::Optimizer; clip_norm=1.0)
    @inbounds for layer in s.layers
        clip_gradients!(layer, clip_norm=clip_norm)
        update_layer!(layer, optimizer)
    end
end

function reset!(s::Sequence{T}, seq_len::Int, batch_size::Int) where T
    @inbounds for layer in s.layers
        if layer isa Recurrent
            layer.h = zeros(T, layer.out, seq_len + 1, batch_size)
            layer.x = zeros(T, layer.in, seq_len, batch_size)
        end
    end
end

function clip_gradients!(layer; clip_norm::Float64=1.0)
    if layer isa Dense
        grad_norm = sqrt(sum(abs2, layer.grad_weights) + sum(abs2, layer.grad_biases))
        if grad_norm > clip_norm
            scale = clip_norm / grad_norm
            layer.grad_weights .*= scale
            layer.grad_biases .*= scale
        end
        
    elseif layer isa Recurrent
        grad_norm = sqrt(sum(abs2, layer.grad_weights) + 
                        sum(abs2, layer.grad_Wh) + 
                        sum(abs2, layer.grad_biases))
        if grad_norm > clip_norm
            scale = clip_norm / grad_norm
            layer.grad_weights .*= scale
            layer.grad_Wh .*= scale
            layer.grad_biases .*= scale
        end

    elseif layer isa Embedding
        grad_norm = sqrt(sum(abs2, layer.grad_weights))
        if grad_norm > clip_norm
            scale = clip_norm / grad_norm
            layer.grad_weights .*= scale
        end
    end

    nothing
end