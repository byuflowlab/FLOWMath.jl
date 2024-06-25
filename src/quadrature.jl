# ---- Quadrature Methods ---------

"""
    trapz(x, y)

Integrate y w.r.t. x using the trapezoidal method.
"""
function trapz(x, y)
    integral = 0.0
    for i in 1:(length(x) - 1)
        integral += (x[i + 1] - x[i]) * 0.5 * (y[i] + y[i + 1])
    end
    return integral
end
