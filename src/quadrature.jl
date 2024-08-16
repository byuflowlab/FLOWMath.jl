# ---- Quadrature Methods ---------




"""
    trapz(x, y)

Integrate y w.r.t. x using the trapezoidal method.
"""
function trapz(x, y)

    integral = 0.0
    for i = 1:length(x)-1
        integral += (x[i+1]-x[i])*0.5*(y[i] + y[i+1])
    end
    return integral
end


"""
    cumtrapz(x, y)

Cumulatively integrate `y` w.r.t `x` using the trapezoidal method, returning an array the same size as `x` and `y`.
"""
function cumtrapz(x, y)
    integral = similar(y)

    integral[begin] = 0
    for i in eachindex(x, y, integral)[begin+1:end]
        integral[i] = integral[i-1] + (x[i] - x[i-1])*0.5*(y[i] + y[i-1])
    end

    return integral
end


"""
    cumtrapz!(integral, x, y)

Cumulatively integrate `y` w.r.t `x` using the trapezoidal method, writing the result to `integral`.
"""
function cumtrapz!(integral, x, y)
    integral[begin] = 0
    for i in eachindex(x, y, integral)[begin+1:end]
        integral[i] = integral[i-1] + (x[i] - x[i-1])*0.5*(y[i] + y[i-1])
    end

    return integral
end
