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

"""
    cubic_blend(f1x, f2x, x, xt, delta_x)

Smoothly transitions the results of functions f1 and f2 using a cubic polynomial,
with the transition between the functions located at `xt`. delta_x is the half
width of the smoothing interval.  The resulting function is C1 continuous.
"""
function cubic_blend(f1x, f2x, x, xt, delta_x)
    if x <= xt - delta_x
        return f1x
    elseif x >= xt + delta_x
        return f2x
    else
        xp = (x-xt)/(2*delta_x) + 1/2
        sx = -2*xp^3 + 3*xp^2
        return f1x + sx*(f2x-f1x)
    end
end

"""
    quintic_blend(f1x, f2x, x, xt, delta_x)

Smoothly transitions the results of functions f1 and f2 using a quintic polynomial,
with the transition between the functions located at `xt`. delta_x is the half
width of the smoothing interval.  The resulting function is C2 continuous.
"""
function quintic_blend(f1x, f2x, x, xt, delta_x)
    if x <= xt - delta_x
        return f1x
    elseif x >= xt + delta_x
        return f2x
    else
        xp = (x-xt)/(2*delta_x) + 1/2
        sx = 6*xp^5 - 15*xp^4 + 10*xp^3
        return f1x + sx*(f2x-f1x)
    end
end

# TODO AN: add smooth max/min with cubic splines
