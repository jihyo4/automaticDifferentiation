using Plots, Printf
using Statistics: mean

function glorot_uniform(::Type{T}, input_size::Integer, output_size::Integer; gain::Real = 1.0) where T
    limit = gain * sqrt(T(6) / (input_size + output_size))
    return rand(T, input_size, output_size) .* (2limit) .- limit
end

function binary_crossentropy(y::AbstractArray, ŷ::AbstractArray)
    y = Float64.(y)
    ϵ = eps(eltype(ŷ))
    ŷ_clipped = clamp.(ŷ, ϵ, 1 - ϵ)
    return mean(@. -y * log(ŷ + ϵ) - (1 - y) * log(1 - ŷ + ϵ))
end

function binary_accuracy(y::AbstractArray, logits::AbstractArray; threshold=0.5)
    ŷ = @. 1 / (1 + exp(-logits))
    y_pred = ŷ .>= threshold
    y_true = y .>= 0.5
    return mean(y_pred .== y_true)
end

function binary_crossentropy_gradient(y::AbstractArray, ŷ::AbstractArray)
    return ŷ .- convert.(eltype(ŷ), y)
end

function binary_crossentropy_logits(y::AbstractArray, z::AbstractArray)
    return mean(@. max(0, z) - z .* y + log1p(exp(-abs(z))))
end

function binary_crossentropy_logits_gradient(y, z)
    sig = @. 1 / (1 + exp(-z))
    return sig - y
end

function train(
    model::Sequence,
    optimizer::Optimizer,
    dataset,
    X_val::AbstractArray,
    y_val::AbstractArray,
    loss_function::Function,
    accuracy_function::Function,
    gradient_function::Function;
    epochs::Int=12,
    verbose::Bool=true,
    dtype::DataType=Float32
)
    history = Dict{String, Vector{dtype}}(
        "train_loss" => dtype[],
        "train_acc" => dtype[],
        "val_loss" => dtype[],
        "val_acc" => dtype[],
        "time" => dtype[],
        "total_samples" => Int[]
    )

    for epoch in 1:epochs
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_samples = 0
        
        stats = @timed begin
            for (x, y) in dataset
                current_batch_size = size(x, 3)
                ADLib.reset!(model, size(x, 2), current_batch_size)
                output = model(x)

                batch_loss = loss_function(y, output)
                batch_accuracy = accuracy_function(y, output)
                
                epoch_loss += batch_loss * current_batch_size
                epoch_accuracy += batch_accuracy * current_batch_size
                epoch_samples += current_batch_size
                
                gradient = gradient_function(y, output)
                gradient ./= current_batch_size
                ADLib.backward_pass(model, gradient)
                ADLib.update_weights(model, optimizer)
            end
            
            train_loss = epoch_loss / epoch_samples
            train_acc = epoch_accuracy / epoch_samples
            
            val_output = model(X_val)
            val_loss = loss_function(y_val, val_output)
            val_acc = accuracy_function(y_val, val_output)
        end
        
        gc_pct = 100 * stats.gctime / stats.time
        compile_pct = 100 * (stats.compile_time / stats.time)
        push!(history["train_loss"], train_loss)
        push!(history["train_acc"], train_acc)
        push!(history["val_loss"], val_loss)
        push!(history["val_acc"], val_acc)
        push!(history["total_samples"], epoch_samples)
        push!(history["time"], stats.time)

        if verbose
            println(@sprintf("Epoch: %d/%d (%.2fs) \tTrain: (loss: %.2f, acc: %.2f) \tval: (loss: %.2f, acc: %.2f) (allocations: %.2f GiB, %.1f%% gc, %.1f%% compilation)", 
                epoch, epochs, stats.time, 
                train_loss, train_acc, 
                val_loss, val_acc, stats.bytes/1e9, 
                gc_pct, compile_pct))
        end
    end
    
    return history
end

function plot_training_history(history)    
    p1 = plot(history["train_loss"], label="Train Loss", linewidth=2)
    plot!(history["val_loss"], label="val Loss", linewidth=2)
    title!(p1, "Loss")
    xlabel!(p1, "Epoch")
    ylabel!(p1, "Loss Value")
    title!("Training Loss")
    
    p2 = plot(history["train_acc"].*100, label="Train Accuracy", linewidth=2)
    plot!(history["val_acc"].*100, label="val Accuracy", linewidth=2)
    title!(p2, "Accuracy")
    xlabel!(p2, "Epoch")
    ylabel!(p2, "Accuracy (%)")
    title!("Training Accuracy")

    p3 = plot(history["time"], label="Training Time", linewidth=2)
    title!(p3, "Time")
    xlabel!(p3, "Epoch")
    ylabel!(p3, "Time (s)")
    title!("Training Duration")
    
    plot(p1, p2, p3, layout=(3,1), size=(600,800), legend=:outertopleft)
end