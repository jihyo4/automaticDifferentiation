abstract type AbstractLayer{T} end

mutable struct Dense{T} <: AbstractLayer{T}
    weights::Matrix{T}
    biases::Vector{T}
    grad_weights::Matrix{T}
    grad_biases::Vector{T}
    activation::Activation
    activations::Matrix{T}
    inputs::Matrix{T}
    grad_input::Matrix{T}
    in::Integer
    out::Integer
    mW::Matrix{T}
    vW::Matrix{T} 
    mb::Vector{T}
    vb::Vector{T}

    function Dense{T}(activation::Activation, in::Integer, out::Integer) where T
        weights = glorot_uniform(T, out, in)
        biases = zeros(T, out)
        grad_weights = zeros(T, out, in)
        grad_biases = zeros(T, out)
        activations = zeros(T, out, 1)
        inputs = zeros(T, in, 1)
        grad_input = zeros(T, in, 1)
        mW = zeros(size(weights))
        vW = zeros(size(weights))
        mb = zeros(size(biases))
        vb = zeros(size(biases))
        new{T}(weights, biases, grad_weights, grad_biases, activation, activations, inputs, grad_input, in, out, mW, vW, mb, vb)      
    end
end

function (layer::Dense)(x::AbstractMatrix{T}) where T
    layer.activations = forward(layer.activation, muladd(layer.weights, x, layer.biases))
    layer.inputs = x
    return layer.activations
end

function backward_layer(layer::Dense, grad_output::AbstractMatrix{T}) where T
    d_activation = backward(layer.activation, layer.activations, grad_output)
    @views begin
        d_weights = d_activation * layer.inputs'
        d_biases = sum(d_activation, dims=2)
        layer.grad_input = layer.weights' * d_activation
    end
    layer.grad_weights .+= d_weights
    layer.grad_biases .+= vec(d_biases)
    return layer.grad_input
end