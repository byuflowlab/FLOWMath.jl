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

# TODO AN: add smooth max/min with cubic splines