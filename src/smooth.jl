# ------ Smoothing functions often used to permit continuous differentiability -----


"""
    abs_smooth(x, delta_x)

Smooth out the absolute value function with a quadratic interval.
delta_x is the half width of the smoothing interval.
Typically usage is with gradient-based optimization.
"""
function abs_smooth(x, delta_x)

    if x < delta_x && x > -delta_x
        return x^2/(2*delta_x) + delta_x/2
    else
        return abs(x)
    end

end

"""
    ksmax(x, hardness=50)

Kreisselmeier–Steinhauser constraint aggregation function.  In the limit as `hardness`
goes to infinity the maximum function is returned. Is mathematically guaranteed to
overestimate the maximum function, i.e. `maximum(x) <= ksmax(x, hardness)`.
"""
function ksmax(x, hardness=50)
    k = maximum(x)
    return 1.0/hardness*log(sum(exp.(hardness*(x.-k)))) .+ k
end

"""
    ksmin(x, hardness=50)

Kreisselmeier–Steinhauser constraint aggregation function.  In the limit as `hardness`
goes to infinity the minimum function is returned. Is mathematically guaranteed to
underestimate the minimum function, i.e. `minimum(x) <= ksmin(x, hardness)`.
"""
ksmin(x, hardness=50) = -ksmax(-x, hardness)

"""
    KSAdaptiveHardness(default=50, tol=1e-6, step=1e-3)

Defines inputs for using the Kreisselmeier-Steinhauser constraint aggregation
function with an adaptive hardness as defined by Poon and Martins in
"An adaptive approach to constraint aggregation using adjoint sensitivity analysis"
"""
@with_kw struct KSAdaptiveHardness{TF}
    default::TF = 50.0
    tol::TF = 1e-6
    step::TF = 1e-3
end

"""
    ksmax(x, hardness::KSAdaptiveHardness)

Kreisselmeier–Steinhauser constraint aggregation function using the adaptive
hardness proposed by Poon and Martins.
"""
function ksmax(x, hardness::KSAdaptiveHardness)
    # unpack original hardness and target slope
    original_hardness = hardness.default # original hardness
    target_derivative = -hardness.tol # minimum target derivative of ksmax wrt hardness
    # compute KS function value and derivative
    original_value = ksmax(x, original_hardness)
    original_derivative = ksmax_h(x, original_hardness)
    # return result if it is within the tolerance
    if original_derivative > target_derivative
        return original_value
    end
    # otherwise compute the derivative of the KS function derivative in the log scale
    perturbed_hardness = original_hardness + hardness.step
    perturbed_derivative = ksmax_h(x, perturbed_hardness)
    tmp = log10(perturbed_derivative/original_derivative)/log10(perturbed_hardness/original_hardness)
    # and apply newton's method once in the log scale to get the new hardness
    new_hardness = log10(original_hardness) - log10(target_derivative/original_derivative)/tmp
    # then take it out of the log scale
    new_hardness = 10^new_hardness
    println(new_hardness)
    # and use the new hardness to compute ksmax
    return ksmax(x, new_hardness)
end

"""
    ksmax_h(x, hardness)


"""
function ksmax_h(x, hardness=25)
    k = maximum(x)
    return 1.0/hardness*(sum((x.-k).*exp.(hardness*(x.-k)))/sum(exp.(hardness*(x.-k))) - 1.0/hardness*log(sum(exp.(hardness*(x.-k)))))
end

"""
    sigmoid(x)

Sigmoid function, implemented with branching to avoid NaNs
"""
function sigmoid(x)
    if x >= zero(x)
        z = exp(-x)
        return one(z) / (one(z) + z)
    else
        z = exp(x)
        return z / (one(z) + z)
    end
end

"""
    sigmoid_blend(f1x, f2x, x, xt, hardness=50)

Smoothly transitions the results of functions f1 and f2 using the sigmoid function,
with the transition between the functions located at `xt`. `hardness` controls the
sharpness of the transition between the two functions.
"""
function sigmoid_blend(f1x, f2x, x, xt, hardness=50)
    sx = sigmoid(hardness*(x-xt))
    return f1x + sx*(f2x-f1x)
end

# TODO AN: add smooth max/min with cubic splines
