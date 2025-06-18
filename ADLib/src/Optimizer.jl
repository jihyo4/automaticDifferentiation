abstract type Optimizer end

mutable struct Adam <: Optimizer 
    t::Int
    β1::Float64
    β2::Float64
    ϵ::Float64
    lr::Float64
    function Adam(; β1=0.9, β2=0.999, ϵ=1e-8, lr=0.001)
        new(0, β1, β2, ϵ, lr)
    end
end

function update_layer!(layer::AbstractLayer, opt::Adam; lr=0.001, β1=0.9, β2=0.999, ϵ=1e-8)
    opt.t += 1

    if hasfield(typeof(layer), :weights)
        layer.mW .= β1 .* layer.mW .+ (1 - β1) .* layer.grad_weights
        layer.vW .= β2 .* layer.vW .+ (1 - β2) .* (layer.grad_weights .^ 2)
        mW_hat = layer.mW ./ (1 - β1^opt.t)
        vW_hat = layer.vW ./ (1 - β2^opt.t)
        layer.weights .-= lr .* mW_hat ./ (sqrt.(vW_hat) .+ ϵ)
    end

    if hasfield(typeof(layer), :Wh)
        layer.mWh .= β1 .* layer.mWh .+ (1 - β1) .* layer.Wh
        layer.vWh .= β2 .* layer.vWh .+ (1 - β2) .* (layer.grad_Wh .^ 2)
        mWh_hat = layer.mWh ./ (1 - β1^opt.t)
        vWh_hat = layer.vWh ./ (1 - β2^opt.t)
        layer.Wh .-= lr .* mWh_hat ./ (sqrt.(vWh_hat) .+ ϵ)
    end

    if hasfield(typeof(layer), :biases)
        layer.mb .= β1 .* layer.mb .+ (1 - β1) .* layer.grad_biases
        layer.vb .= β2 .* layer.vb .+ (1 - β2) .* (layer.grad_biases .^ 2)
        mb_hat = layer.mb ./ (1 - β1^opt.t)
        vb_hat = layer.vb ./ (1 - β2^opt.t)
        layer.biases .-= lr .* mb_hat ./ (sqrt.(vb_hat) .+ ϵ)
    end
end

mutable struct RMSProp <: Optimizer
    decay::Float64
    ϵ::Float64
    lr::Float64

    function RMSProp(; decay=0.9, ϵ=1e-8, lr=0.001)
        new(decay, ϵ, lr)
    end
end

function update_layer!(layer::AbstractLayer, opt::RMSProp)
    decay, ϵ, lr = opt.decay, opt.ϵ, opt.lr

    if hasfield(typeof(layer), :weights)
        layer.vW .= decay .* layer.vW .+ (1 - decay) .* (layer.grad_weights .^ 2)
        layer.weights .-= lr .* layer.grad_weights ./ (sqrt.(layer.vW) .+ ϵ)
        fill!(layer.grad_weights, 0)
    end

    if hasfield(typeof(layer), :Wh)
        layer.vWh .= decay .* layer.vWh .+ (1 - decay) .* (layer.grad_Wh .^ 2)
        layer.Wh .-= lr .* layer.grad_Wh ./ (sqrt.(layer.vWh) .+ ϵ)
        fill!(layer.grad_Wh, 0)
    end

    if hasfield(typeof(layer), :biases)
        layer.vb .= decay .* layer.vb .+ (1 - decay) .* (layer.grad_biases .^ 2)
        layer.biases .-= lr .* layer.grad_biases ./ (sqrt.(layer.vb) .+ ϵ)
        fill!(layer.grad_biases, 0)
    end
end

function adam_step!(param::AbstractArray{T}, grad::AbstractArray{T}, 
                     m::AbstractArray{T}, v::AbstractArray{T}, 
                     opt::Adam) where T
    β1 = T(opt.β1)
    β2 = T(opt.β2)
    ϵ = T(opt.ϵ)
    β1_t = β1^opt.t
    β2_t = β2^opt.t
    inv_1_β1_t = 1 / (1 - β1_t)
    inv_1_β2_t = 1 / (1 - β2_t)
    
    @inbounds for i in eachindex(param, grad, m, v)
        m_i = β1 * m[i] + (1 - β1) * grad[i]
        v_i = β2 * v[i] + (1 - β2) * grad[i] * grad[i]
        
        m[i] = m_i
        v[i] = v_i
        
        m̂ = m_i * inv_1_β1_t
        v̂ = v_i * inv_1_β2_t
        param[i] -= opt.lr * m̂ / (sqrt(v̂) + ϵ)
    end
end

function update_layer!(layer::Dense, opt::Adam; lr=0.001, β1=0.9, β2=0.999, ϵ=1e-8)
    opt.t += 1
    adam_step!(layer.weights, layer.grad_weights, layer.mW, layer.vW, opt)
    adam_step!(layer.biases,  layer.grad_biases,  layer.mb, layer.vb, opt)
    fill!(layer.grad_weights, 0)
    fill!(layer.grad_biases, 0)
end

function update_layer!(layer::Recurrent, opt::Adam; lr=0.001, β1=0.9, β2=0.999, ϵ=1e-8)
    opt.t += 1
    adam_step!(layer.weights, layer.grad_weights, layer.mW, layer.vW, opt)
    adam_step!(layer.Wh,      layer.grad_Wh,      layer.mWh, layer.vWh, opt)
    adam_step!(layer.biases,  layer.grad_biases,  layer.mb,  layer.vb,  opt)
    fill!(layer.grad_weights, 0)
    fill!(layer.grad_Wh, 0)
    fill!(layer.grad_biases, 0)
end

function update_layer!(layer::Embedding, opt::Adam; lr=0.001, β1=0.9, β2=0.999, ϵ=1e-8)
    opt.t += 1
    adam_step!(layer.weights, layer.grad_weights, layer.mW, layer.vW, opt)
    fill!(layer.grad_weights, 0)
end

function update_layer!(::SelectLast, ::Adam; kwargs...)
    return
end

function rmsprop_step!(param::AbstractArray{T}, grad::AbstractArray{T},
                        v::AbstractArray{T}, opt::RMSProp) where T
    decay = T(opt.decay)
    ϵ = T(opt.ϵ)
    lr = T(opt.lr)
    one_minus_decay = 1 - decay
    
    @inbounds for i in eachindex(param, grad, v)
        v_i = decay * v[i] + one_minus_decay * grad[i] * grad[i]
        v[i] = v_i
        param[i] -= lr * grad[i] / (sqrt(v_i) + ϵ)
    end
end

function update_layer!(layer::Dense, opt::RMSProp)
    rmsprop_step!(layer.weights, layer.grad_weights, layer.vW, opt)
    rmsprop_step!(layer.biases,  layer.grad_biases,  layer.vb, opt)
    fill!(layer.grad_weights, 0)
    fill!(layer.grad_biases, 0)
end

function update_layer!(layer::Recurrent, opt::RMSProp)
    rmsprop_step!(layer.weights, layer.grad_weights, layer.vW, opt)
    rmsprop_step!(layer.Wh,      layer.grad_Wh,      layer.vWh, opt)
    rmsprop_step!(layer.biases,  layer.grad_biases,  layer.vb,  opt)
    fill!(layer.grad_weights, 0)
    fill!(layer.grad_Wh, 0)
    fill!(layer.grad_biases, 0)
end

function update_layer!(layer::Embedding, opt::RMSProp)
    rmsprop_step!(layer.weights, layer.grad_weights, layer.vW, opt)
    fill!(layer.grad_weights, 0)
end

function update_layer!(::SelectLast, ::RMSProp)
    return
end