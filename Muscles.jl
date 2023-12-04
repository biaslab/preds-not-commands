module Muscles

using LinearAlgebra
using Distributions
using RxInfer

export Muscle, update!, emit!, step!, params

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

function emit!(sys::Muscle)
    sys.sensation = NormalMeanVariance(sys.state, sys.mnoise_sd)
end

function step!(sys::Muscle, prediction::UnivariateDistribution)
    update!(sys, prediction)
    emit!(sys)
end         

end