mutable struct Recurrent{T} <: AbstractLayer{T}
    weights::Matrix{T}
    Wh::Matrix{T}
    biases::Vector{T}
    activation::Activation

    grad_weights::Matrix{T}
    grad_Wh::Matrix{T}
    grad_biases::Vector{T}

    mW::Matrix{T}
    vW::Matrix{T}
    mWh::Matrix{T}
    vWh::Matrix{T}
    mb::Vector{T}
    vb::Vector{T}

    h::Array{T, 3}
    x::Array{T, 3}
    buffer::Matrix{T}

    in::Int
    out::Int

    function Recurrent{T}(activation::Activation, in::Int, out::Int) where T
        weights = glorot_uniform(T, out, in)
        Wh = glorot_uniform(T, out, out)
        biases = zeros(T, out)

        grad_weights = zeros(T, out, in)
        grad_Wh = zeros(T, out, out)
        grad_biases = zeros(T, out)

        mW = zeros(T, out, in)
        vW = zeros(T, out, in)
        mWh = zeros(T, out, out)
        vWh = zeros(T, out, out)
        mb = zeros(T, out)
        vb = zeros(T, out)

        h = zeros(T, out, 1, 1)
        x = zeros(T, in, 1, 1)
        buffer = zeros(T, out, 1)

        new{T}(weights, Wh, biases, activation, grad_weights, grad_Wh, grad_biases,
               mW, vW, mWh, vWh, mb, vb, h, x, buffer, in, out)
    end
end

function (rnn::Recurrent)(x::Array{T, 3}) where T
    seq_len = size(x, 2)
    batch_size = size(x, 3)
    
    if size(rnn.h, 2) != seq_len + 1 || size(rnn.h, 3) != batch_size
        rnn.h = zeros(T, rnn.out, seq_len + 1, batch_size)
    else
        fill!(rnn.h, zero(T))
    end

    rnn.x = x

    @inbounds for t in 1:seq_len
        xt = @view x[:, t, :]
        prev_h = @view rnn.h[:, t, :]
        z = rnn.weights * xt .+ rnn.Wh * prev_h .+ rnn.biases
        rnn.h[:, t + 1, :] .= forward(rnn.activation, z)
    end

    return rnn.h[:, 2:end, :]
end

function backward_layer(rnn::Recurrent, grad_output::Array{T, 3}) where T
    x = rnn.x
    h = rnn.h
    seq_len = size(x, 2)
    batch_size = size(x, 3)

    fill!(rnn.grad_weights, zero(T))
    fill!(rnn.grad_Wh, zero(T))
    fill!(rnn.grad_biases, zero(T))

    grad_input = zeros(T, rnn.in, seq_len, batch_size)
    grad_next = zeros(T, rnn.out, batch_size)

    @inbounds for t in seq_len:-1:1
        xt = @view x[:, t, :]
        ht = @view h[:, t + 1, :]
        ht_prev = @view h[:, t, :]

        d_act = backward(rnn.activation, ht, grad_output[:, t, :] .+ grad_next)

        rnn.grad_biases .+= vec(sum(d_act, dims=2))
        rnn.grad_weights .+= d_act * xt'
        rnn.grad_Wh .+= d_act * ht_prev'

        grad_input[:, t, :] .= rnn.weights' * d_act
        grad_next .= rnn.Wh' * d_act
    end

    return grad_input
end