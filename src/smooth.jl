# ------ Smoothing functions often used to permit continuous differentiability -----

"""
    abs_smooth(x, delta_x)

Smooth out the absolute value function with a quadratic interval.
delta_x is the half width of the smoothing interval.
Typically usage is with gradient-based optimization.
"""
function abs_smooth(x, delta_x)
    if real(x) < delta_x && real(x) > -delta_x
        return x^2 / (2 * delta_x) + delta_x / 2
    else
        return abs_cs_safe(x)
    end
end

"""
    ksmax(x, hardness=50)

Kreisselmeier–Steinhauser constraint aggregation function.  In the limit as `hardness`
goes to infinity the maximum function is returned. Is mathematically guaranteed to
overestimate the maximum function, i.e. `maximum(x) <= ksmax(x, hardness)`.
"""
function ksmax(x, hardness=50)
    k = maximum(real(x))
    return 1.0 / hardness * log(sum(exp.(hardness * (x .- k)))) .+ k
end

"""
    ksmin(x, hardness=50)

Kreisselmeier–Steinhauser constraint aggregation function.  In the limit as `hardness`
goes to infinity the minimum function is returned. Is mathematically guaranteed to
underestimate the minimum function, i.e. `minimum(x) <= ksmin(x, hardness)`.
"""
ksmin(x, hardness=50) = -ksmax(-x, hardness)

"""
    ksmax_adaptive(x, hardness=50; tol=1e-6, smoothing_fraction=0.1)

Kreisselmeier–Steinhauser constraint aggregation function using the adaptive
hardness proposed by Poon and Martins in "An adaptive approach to constraint
aggregation using adjoint sensitivity analysis".  This implementation uses
Newton's method rather than the secant method for increasing hardness values.
Some blending is also used to ensure that result is C1 continuous.
`smoothing_fraction` controls the smoothness of this blending.
"""
function ksmax_adaptive(x, hardness=50; tol=1e-6, smoothing_fraction=0.1)
    # check if derivative of the KS function wrt hardness is within the tolerance
    deriv = ksmax_h(x, hardness)
    target_deriv = -tol
    if abs(deriv) > tol * (1 - smoothing_fraction)
        # increase hardness by applying Newton's method once in log-log space
        dderiv = ksmax_hh(x, hardness) * hardness / deriv
        new_hardness = 10^(log10(hardness) + log10(target_deriv / deriv) / dderiv)
        hardness = quintic_blend(
            hardness, new_hardness, -deriv, -target_deriv, smoothing_fraction * tol
        )
    end
    # use the new hardness to compute ksmax
    return ksmax(x, hardness)
end

"""
    ksmin_adaptive(x, hardness=50; tol=1e-6, smoothing_fraction=0.1)

Kreisselmeier–Steinhauser constraint aggregation function using the adaptive
hardness proposed by Poon and Martins in "An adaptive approach to constraint
aggregation using adjoint sensitivity analysis".  This implementation uses
Newton's method rather than the secant method for increasing hardness values.
Some blending is also used to ensure that result is C1 continuous.
`smoothing_fraction` controls the smoothness of this blending.
"""
function ksmin_adaptive(x, hardness=50; tol=1e-6, smoothing_fraction=0.1)
    return -ksmax_adaptive(-x, hardness; tol=tol, smoothing_fraction=smoothing_fraction)
end

"""
    ksmax_h(x, hardness)

Computes the derivative of the Kreisselmeier–Steinhauser constraint aggregation
function with respect to `hardness`.
"""
function ksmax_h(x, hardness)
    k = maximum(real(x))
    tmp1 = exp.(hardness * (x .- k))
    tmp2 = sum((x .- k) .* tmp1)
    tmp3 = sum(tmp1)
    tmp4 = 1.0 / hardness * log(tmp3)
    return 1.0 / hardness * (tmp2 / tmp3 - tmp4)
end

"""
    ksmax_hh(x, hardness)

Computes the second derivative of the Kreisselmeier–Steinhauser constraint aggregation
function with respect to `hardness`.
"""
function ksmax_hh(x, hardness)
    k = maximum(real(x))
    tmp1 = exp.(hardness * (x .- k))
    tmp2 = sum((x .- k) .* tmp1)
    tmp2_h = sum((x .- k) .^ 2 .* tmp1)
    tmp3 = sum(tmp1)
    tmp3_h = sum((x .- k) .* tmp1)
    tmp4 = 1.0 / hardness * log(tmp3)
    tmp4_h = 1.0 / hardness * (tmp2 / tmp3 - tmp4)
    return -1.0 / hardness^2 * (tmp2 / tmp3 - tmp4) +
           1.0 / hardness * (tmp2_h / tmp3 - tmp2 * tmp3_h / tmp3^2 - tmp4_h)
end

"""
    sigmoid(x)

Sigmoid function, implemented with branching to avoid NaNs
"""
function sigmoid(x)
    if real(x) >= zero(real(x))
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
    sx = sigmoid(hardness * (x - xt))
    return f1x + sx * (f2x - f1x)
end

"""
    sigmoid_blend(fx::Tuple, xt::Tuple, x, hardness=50)

Smoothly transitions the results of the functions in `fx` using the sigmoid function,
with the transition between the functions at the locations in `xt`. `hardness` controls
the sharpness of the transition between the functions.
"""
function sigmoid_blend(fx::Tuple, xt::Tuple, x, hardness=50)
    new_fx = sigmoid_blend.(fx[1:(end - 1)], fx[2:end], x, xt, hardness)
    if length(new_fx) == 1
        return only(new_fx)
    else
        new_xt = (xt[1:(end - 1)] .+ xt[2:end]) ./ 2
        return sigmoid_blend(new_fx, new_xt, x, hardness)
    end
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
        xp = (x - xt) / (2 * delta_x) + 1 / 2
        sx = -2 * xp^3 + 3 * xp^2
        return f1x + sx * (f2x - f1x)
    end
end

"""
    cubic_blend(fx::Tuple, xt::Tuple, x, delta_x)

Smoothly transitions the results of the functions in `fx` using cubic polynomials,
with the transition between the functions at the locations in `xt`. `delta_x` is the half
width of the smoothing interval.  The resulting function is C1 continuous
"""
function cubic_blend(fx::Tuple, xt, x, delta_x)
    new_fx = cubic_blend.(fx[1:(end - 1)], fx[2:end], x, xt, delta_x)
    if length(new_fx) == 1
        return only(new_fx)
    else
        new_xt = (xt[1:(end - 1)] .+ xt[2:end]) ./ 2
        return cubic_blend(new_fx, new_xt, x, delta_x)
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
        xp = (x - xt) / (2 * delta_x) + 1 / 2
        sx = 6 * xp^5 - 15 * xp^4 + 10 * xp^3
        return f1x + sx * (f2x - f1x)
    end
end

"""
    quintic_blend(fx::Tuple, xt::Tuple, x, delta_x)

Smoothly transitions the results of the functions in `fx` using quintic polynomials,
with the transition between the functions at the locations in `xt`. `delta_x` is the half
width of the smoothing interval.  The resulting function is C2 continuous
"""
function quintic_blend(fx::Tuple, xt, x, delta_x)
    new_fx = quintic_blend.(fx[1:(end - 1)], fx[2:end], x, xt, delta_x)
    if length(new_fx) == 1
        return only(new_fx)
    else
        new_xt = (xt[1:(end - 1)] .+ xt[2:end]) ./ 2
        return quintic_blend(new_fx, new_xt, x, delta_x)
    end
end

"""
    step_smooth()

Smoothly transition between `a` and `b` when `x` equals `x_step`. The transition starts 
at `x_step`-`dx` and ends at `x_step`+`dx`. The polynomial used in the transition is of 
order 2`N`+1. (https://en.wikipedia.org/wiki/Smoothstep)
"""
function step_smooth(x, x_step, dx, a=zero(x), b=one(x); N=2)
    @assert dx > 0
    edge_left = x_step - dx
    x <= edge_left && return a
    edge_right = x_step + dx
    x >= edge_right && return b
    x = (x - edge_left) / (edge_right - edge_left)
    return _step_smooth(x; N=N) * (b - a) + a
end

function _step_smooth(x; N=2)
    if N == 0
        return x
    elseif N == 1
        return -2*x^3 + 3*x^2
    elseif N == 2
        return 6*x^5 - 15*x^4 + 10*x^3
    end
    S = 0
    for n in 0:N
        S += binomial(-N-1, n) * binomial(2N+1, N-n) * x^(N+n+1)
    end
    return S
end


# TODO AN: add smooth max/min with cubic splines
