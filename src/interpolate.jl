#=
Interpolation Methods

=#

using OffsetArrays: OffsetVector
using Infiltrator


"""
private function, find index in array where x would be inserted for interpolation
between xvec[i] and xvec[i+1]
"""
function findindex(xvec, x)

    n = length(xvec)
    i = searchsortedlast(xvec, x)

    # this version allows extrapolation
    if i == 0
        i = 1
    elseif i == n
        i = n - 1
    end

    # this version prevents extrapolation
    # if i == 0
    #     throw(DomainError(x, "x falls below range of provided spline data"))
    # elseif i == n 
    #     if x == xvec[n]
    #         i = n - 1
    #     else
    #         throw(DomainError(x, "x falls above range of provided spline data"))
    #     end
    # end

    return i
end

## ------ Akima Interpolation  ---------

# struct used internally
struct Akima{TX, TY, TCoeff}

    xdata::TX
    ydata::TY
    p0::Vector{TCoeff}
    p1::Vector{TCoeff}
    p2::Vector{TCoeff}
    p3::Vector{TCoeff}

end


"""
    Akima(xdata, ydata, delta_x=0.0)

Creates an akima spline at node points: `xdata`, `ydata`.  This is a 1D spline that avoids
overshooting issues common with many other polynomial splines resulting in a more
natural curve.  It also only depends on local points (`i-2`...`i+2`) allow for more efficient
computation.  `delta_x` is the half width of a smoothing interval used for the absolute
value function.  Set `delta_x=0` to recover the original akima spline.  The smoothing
is only useful if you want to differentiate xdata and ydata.  In many case the nodal points
are fixed so this is not needed.  Returns an akima spline object (Akima struct).
This function, only performs construction of the spline, not evaluation.
This is useful if you want to evaluate the same mesh at multiple different conditions.
A convenience method exists below to perform both in one shot.
"""
function Akima(xdata::AbstractVector{TFX}, ydata::AbstractVector{TFY}, delta_x=0.0) where {TFX, TFY}

    # setup
    eps = 1e-30
    n = length(xdata)
    TCoeff = promote_type(TFX, TFY)

    # compute segment slopes
    m = OffsetVector(zeros(TCoeff, n+3), -1:n+1)
    for i = 1:n-1
        m[i] = (ydata[i+1] - ydata[i]) / (xdata[i+1] - xdata[i])
    end

    # estimation for end points
    m[0] = 2.0*m[1] - m[2]
    m[-1] = 2.0*m[0] - m[1]
    m[n] = 2.0*m[n-1] - m[n-2]
    m[n+1] = 2.0*m[n] - m[n-1]

    # slope at points
    t = zeros(TCoeff, n)
    for i = 1:n
        m1 = m[i-2]
        m2 = m[i-1]
        m3 = m[i]
        m4 = m[i+1]
        w1 = abs_smooth(m4 - m3, delta_x)
        w2 = abs_smooth(m2 - m1, delta_x)
        if (w1 < eps && w2 < eps)
            t[i] = 0.5*(m2 + m3)  # special case to avoid divide by zero
        else
            t[i] = (w1*m2 + w2*m3) / (w1 + w2)
        end
    end

    # polynomial cofficients
    p0 = zeros(TCoeff, n-1)
    p1 = zeros(TCoeff, n-1)
    p2 = zeros(TCoeff, n-1)
    p3 = zeros(TCoeff, n-1)
    for i = 1:n-1
        dx = xdata[i+1] - xdata[i]
        t1 = t[i]
        t2 = t[i+1]
        p0[i] = ydata[i]
        p1[i] = t1
        p2[i] = (3.0*m[i] - 2.0*t1 - t2)/dx
        p3[i] = (t1 + t2 - 2.0*m[i])/dx^2
    end

    return Akima(xdata, ydata, p0, p1, p2, p3)
end

function (spline::Akima)(x::Number)

    j = findindex(spline.xdata, x)

    # evaluate polynomial
    dx = x - spline.xdata[j]
    y = spline.p0[j] + spline.p1[j]*dx + spline.p2[j]*dx^2 + spline.p3[j]*dx^3

    return y
end

(spline::Akima)(x::AbstractVector) = spline.(x)

"""
    akima(x, y, xpt)

A convenience method to perform construction and evaluation of the spline in one step.
See docstring for Akima for more details.

**Arguments**
- `x, y::Vector{Float}`: the node points
- `xpt::Vector{Float} or ::Float`: the evaluation point(s)

**Returns**
- `ypt::Vector{Float} or ::Float`: interpolated value(s) at xpt using akima spline.
"""
akima(x, y, xpt) = Akima(x, y)(xpt)

"""
    derivative(spline, x)

Computes the derivative of an Akima spline at x.

**Arguments**
- `spline::Akima}`: an Akima spline
- `x::Float`: the evaluation point(s)

**Returns**
- `dydx::Float`: derivative at x using akima spline.
"""
function derivative(spline::Akima, x)

    j = findindex(spline.xdata, x)

    # evaluate polynomial
    dx = x - spline.xdata[j]
    dydx = spline.p1[j] + 2*spline.p2[j]*dx + 3*spline.p3[j]*dx^2

    return dydx
end

"""
    gradient(spline, x)

Computes the gradient of a Akima spline at x.

**Arguments**
- `spline::Akima}`: an Akima spline
- `x::Vector{Float}`: the evaluation point(s)

**Returns**
- `dydx::Vector{Float}`: gradient at x using akima spline.
"""
gradient(spline::Akima, x) = derivative.(Ref(spline), x)


# ------ Linear Interpolation  ---------

"""
    linear(xdata, ydata, x::Number)

Linear interpolation.

**Arguments**
- `xdata::Vector{Float64}`: x data used in constructing interpolation
- `ydata::Vector{Float64}`: y data used in constructing interpolation
- `x::Float64`: point to evaluate spline at

**Returns**
- `y::Float64`: value at x using linear interpolation
"""
function linear(xdata, ydata, x::Number)

    i = findindex(xdata, x)

    # lienar interpolation
    eta = (x - xdata[i]) / (xdata[i+1] - xdata[i])
    y = ydata[i] + eta*(ydata[i+1] - ydata[i])

    return y
end

"""
    linear(xdata, ydata, x::AbstractVector)
    
Convenience function to perform linear interpolation at multiple points.
"""
linear(xdata, ydata, x::AbstractVector) = linear.(Ref(xdata), Ref(ydata), x)


"""
derivative of linear interpolation at `x::Number`
"""
function derivative(xdata, ydata, x)

    i = findindex(xdata, x)
    dydx = (ydata[i+1] - ydata[i]) / (xdata[i+1] - xdata[i])

    return dydx
end

"""
gradient of linear interpolation at `x::Vector`
"""
gradient(xdata, ydata, x) = derivative.(Ref(xdata), Ref(ydata), x)


# ------- higher order recursive 1D interpolation ------

"""
   interp2d(interp1d, xdata, ydata, fdata, xpt, ypt)

2D interpolation using recursive 1D interpolation.  This approach is likely less efficient than a more direct
2D interpolation method, especially one you can create separate creation from evaluation,
but it is generalizable to any spline approach and any dimension.

**Arguments**
- `interp1d`: any spline function of form: ypt = interp1d(xdata, ydata, xpt) where data are the known
    data(node) points and pt are the points where you want to evaluate the spline at.
- `xdata::Vector{Float}`, `ydata::Vector{Float}`: Define the 2D grid
- `fdata::Matrix{Float}`: where fdata[i, j] is the function value at xdata[i], ydata[j]
- `xpt::Vector{Float}`, `ypt::Vector{Float}`: the locations where you want to evaluate the spline

**Returns**
- `fhat::Matrix{Float}`: where fhat[i, j] is the estimate function value at xpt[i], ypt[j]
"""
function interp2d(interp1d, xdata, ydata, fdata, xpt, ypt)

    ny = length(ydata)
    nxpt = length(xpt)
    nypt = length(ypt)
    yinterp = Array{eltype(xpt)}(undef, ny, nxpt)
    output = Array{eltype(xpt)}(undef, nxpt, nypt)

    for i = 1:ny
        yinterp[i, :] = interp1d(xdata, fdata[:, i], xpt)
    end
    for i = 1:nxpt
        output[i, :] = interp1d(ydata, yinterp[:, i], ypt)
    end


    return output
end

"""
    interp3d(interp1d, xdata, ydata, zdata, fdata, xpt, ypt, zpt)

Same as interp2d, except in three dimension.
"""
function interp3d(interp1d, xdata, ydata, zdata, fdata, xpt, ypt, zpt)

    nz = length(zdata)
    nxpt = length(xpt)
    nypt = length(ypt)
    nzpt = length(zpt)
    zinterp = Array{eltype(xpt)}(undef, nz, nxpt, nypt)
    output = Array{eltype(xpt)}(undef, nxpt, nypt, nzpt)

    for i = 1:nz
        zinterp[i, :, :] = interp2d(interp1d, xdata, ydata, fdata[:, :, i], xpt, ypt)
    end
    for j = 1:nypt
        for i = 1:nxpt
            output[i, j, :] = interp1d(zdata, zinterp[:, i, j], zpt)
        end
    end

    return output
end


"""
    interp4d(interp1d, xdata, ydata, zdata, fdata, xpt, ypt, zpt)

Same as interp3d, except in four dimensions.
"""
function interp4d(interp1d, xdata, ydata, zdata, tdata, fdata, xpt, ypt, zpt, tpt)

    nt = length(tdata)
    nxpt = length(xpt)
    nypt = length(ypt)
    nzpt = length(zpt)
    ntpt = length(tpt)
    tinterp = zeros(nt, nxpt, nypt, nzpt)
    output = zeros(nxpt, nypt, nzpt, ntpt)

    for i = 1:nt
        tinterp[i, :, :, :] = interp3d(interp1d, xdata, ydata, zdata, fdata[:, :, :, i], xpt, ypt, zpt)
    end
    for k = 1:nzpt
        for j = 1:nypt
            for i = 1:nxpt
                output[i, j, k, :] = interp1d(tdata, tinterp[:, i, j, k], tpt)
            end
        end
    end

    return output
end
