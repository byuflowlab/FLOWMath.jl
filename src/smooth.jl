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

# TODO AN: add smooth max/min with cubic splines
