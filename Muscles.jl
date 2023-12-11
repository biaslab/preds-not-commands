module Muscles

using LinearAlgebra
using Distributions
using RxInfer

export Muscle, MusclePair, update!, emit!, step!, params

mutable struct Muscle
    "Model of a muscle"

    state       ::Float64
    
    prediction  ::UnivariateDistribution
    sensation   ::UnivariateDistribution
    mnoise_sd   ::Float64

    state_lims  ::Tuple{Float64,Float64}
    action_lims ::Tuple{Float64,Float64}

    Δt          ::Float64

    function Muscle(;init_state::Float64=0.0, 
                     state_lims::Tuple{Float64,Float64}=(0.0, 1.0),
                     action_lims::Tuple{Float64,Float64}=(-1.0, 1.0),
                     mnoise_sd::Float64=1.0,
                     Δt::Float64=1.0)

        if (init_state < 0.0) | (init_state > 1.0); error("Initial state has to be in [0,1]"); end
        
        init_prediction = NormalMeanVariance(0.0, 1.0)
        init_sensation  = NormalMeanVariance(init_state, mnoise_sd^2)

        return new(init_state, init_prediction, init_sensation, mnoise_sd, state_lims, action_lims, Δt)
    end
end

mutable struct MusclePair
    "Model of a pair of opposing muscles"

    states      ::Vector{T} where {T <: Real}
    
    prediction  ::Vector{T} where {T <: UnivariateDistribution}
    sensation   ::Vector{T} where {T <: UnivariateDistribution}
    mnoise_sd   ::Float64

    state_lims  ::Tuple{Float64,Float64}
    action_lims ::Tuple{Float64,Float64}

    Δt          ::Float64

    function MusclePair(;init_state::Vector{Float64}=[0.5, 0.5], 
                        state_lims::Tuple{Float64,Float64}=(0.0, 1.0),
                        action_lims::Tuple{Float64,Float64}=(-1.0, 1.0),
                        mnoise_sd::Float64=1.0,
                        Δt::Float64=1.0)

        if any(init_state .< 0.0) | any(init_state .> 1.0); error("Initial states have to be in [0,1]"); end
        
        init_prediction = [NormalMeanVariance(0.5, 1.0), 
                           NormalMeanVariance(0.5, 1.0)]
        init_sensation  = [NormalMeanVariance(init_state[1], mnoise_sd^2), 
                           NormalMeanVariance(init_state[2], mnoise_sd^2)]

        return new(init_state, init_prediction, init_sensation, mnoise_sd, state_lims, action_lims, Δt)
    end
end

function update!(sys::Muscle, prediction::UnivariateDistribution)
    "Evolve state of muscle based on prediction"
    
    # Extract moments of prediction
    m,v = mean_var(prediction)
    
    # Update based on minimizing precision-weighted prediction error
    new_state = sys.state + sys.Δt*(m - sys.state)/v
    
    # Clamp to limits
    sys.state = clamp(new_state, sys.state_lims...)
    
    # Store prediction
    sys.prediction = prediction
end

function update!(sys::MusclePair, prediction::Vector{T}) where {T<:UnivariateDistribution}
    "Evolve state of muscle pair based on prediction"
    
    # Extract moments of prediction
    m = mean.(prediction)
    v = var.( prediction)
    
    # Update based on minimizing precision-weighted prediction error
    new_states = [0.0, 0.0]
    new_states[1] = sys.states[1] + sys.Δt*(m[1] - sys.states[1])/(2v[1]) - sys.Δt*(m[2] - sys.states[2])/(2v[2])
    new_states[2] = sys.states[2] + sys.Δt*(m[2] - sys.states[2])/(2v[2]) - sys.Δt*(m[1] - sys.states[1])/(2v[1])
    
    # Clamp to limits
    sys.states = clamp.(new_states, sys.state_lims...)
    
    # Store prediction
    sys.prediction = prediction
end

function emit!(sys::Muscle)
    sys.sensation = NormalMeanVariance(sys.state, sys.mnoise_sd)
end

function emit!(sys::MusclePair)
    sys.sensation = [NormalMeanVariance(sys.states[1], sys.mnoise_sd),
                     NormalMeanVariance(sys.states[2], sys.mnoise_sd)]
end

function step!(sys::Muscle, prediction::UnivariateDistribution)
    update!(sys, prediction)
    emit!(sys)
end

function step!(sys::MusclePair, prediction::Vector{T}) where {T<:UnivariateDistribution}
    update!(sys, prediction)
    emit!(sys)
end

end