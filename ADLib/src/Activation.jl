abstract type Activation end

function forward(::Activation, x::T) where T
    throw(ArgumentError("forward not implemented for $(typeof(x))"))
end

function backward(::Activation, x::T, grad_output::T) where T
    throw(ArgumentError("backward not implemented for $(typeof(x))"))
end


struct ReLU <: Activation end

@inline forward(::ReLU, x::T) where T = max(zero(T), x)
@inline forward(::ReLU, x::AbstractArray{T}) where T = max.(zero(T), x)

@inline backward(::ReLU, x::T, grad_output::T) where T = x > zero(T) ? grad_output : zero(T)
@inline backward(::ReLU, x::AbstractArray{T}, grad_output::AbstractArray{T}) where T = grad_output .* (x .> zero(T))


struct Sigmoid <: Activation end

function forward(::Sigmoid, x::T) where T
    return one(T) / (one(T) + exp(-x))
end

function forward(::Sigmoid, x::AbstractArray{T}) where T
    return one(T) ./ (one(T) .+ exp.(-x))
end

function backward(::Sigmoid, x::T, grad_output::T) where T
    y = forward(Sigmoid(), x)
    return grad_output * y * (one(T) - y)
end

function backward(::Sigmoid, x::AbstractArray{T}, grad_output::AbstractArray{T}) where T
    y = forward(Sigmoid(), x)
    return grad_output .* y .* (one(T) .- y)
end

struct Identity <: Activation end

function forward(::Identity, x::T) where T
    return x
end

function forward(::Identity, x::AbstractArray{T}) where T
    return x
end

function backward(::Identity, x::T, grad_output::T) where T
    return grad_output
end

function backward(::Identity, x::AbstractArray{T}, grad_output::AbstractArray{T}) where T
    return grad_output
end