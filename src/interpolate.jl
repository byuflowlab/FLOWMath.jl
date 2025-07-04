#=
Interpolation Methods

=#

using OffsetArrays: OffsetVector

"""
private function, find index in array where x would be inserted for interpolation
between xvec[i] and xvec[i+1]
"""
function findindex(xvec, x)
    n = length(xvec)
    i = searchsortedlast(real(xvec), real(x))

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
struct Akima{TX,TY,TCoeff}
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
computation.

`delta_x` is the half width of a smoothing interval used for the absolute
value function.  Set `delta_x=0` to recover the original akima spline.  The smoothing
is only useful if you want to differentiate xdata and ydata.  In many case the nodal points
are fixed so this is not needed.

`eps` is a cutoff used to avoid dividing by zero in the
weighting function. Default is `1e-30` but this could be raised to machine precision in
some cases to improve derivatives. (E.g., when the denominator in line 105 is very small.)

Returns an akima spline object (Akima struct). This function only performs construction
of the spline, not evaluation. This is useful if you want to evaluate the same mesh at
multiple different conditions. A convenience method exists below to perform both in one shot.
"""
function Akima(
    xdata::AbstractVector{TFX}, ydata::AbstractVector{TFY}, delta_x=0.0, eps=1e-30
) where {TFX,TFY}

    # setup
    n = length(xdata)
    TCoeff = promote_type(TFX, TFY)

    # compute segment slopes
    m = OffsetVector(zeros(TCoeff, n + 3), -1:(n + 1))
    for i in 1:(n - 1)
        m[i] = (ydata[i + 1] - ydata[i]) / (xdata[i + 1] - xdata[i])
    end

    # estimation for end points
    m[0] = 2.0 * m[1] - m[2]
    m[-1] = 2.0 * m[0] - m[1]
    m[n] = 2.0 * m[n - 1] - m[n - 2]
    m[n + 1] = 2.0 * m[n] - m[n - 1]

    # slope at points
    t = zeros(TCoeff, n)
    for i in 1:n
        m1 = m[i - 2]
        m2 = m[i - 1]
        m3 = m[i]
        m4 = m[i + 1]
        w1 = abs_smooth(m4 - m3, delta_x)
        w2 = abs_smooth(m2 - m1, delta_x)
        if (real(w1) < eps && real(w2) < eps)
            t[i] = 0.5 * (m2 + m3)  # special case to avoid divide by zero
        else
            t[i] = (w1 * m2 + w2 * m3) / (w1 + w2)
        end
    end

    # polynomial cofficients
    p0 = zeros(TCoeff, n - 1)
    p1 = zeros(TCoeff, n - 1)
    p2 = zeros(TCoeff, n - 1)
    p3 = zeros(TCoeff, n - 1)
    for i in 1:(n - 1)
        dx = xdata[i + 1] - xdata[i]
        t1 = t[i]
        t2 = t[i + 1]
        p0[i] = ydata[i]
        p1[i] = t1
        p2[i] = (3.0 * m[i] - 2.0 * t1 - t2) / dx
        p3[i] = (t1 + t2 - 2.0 * m[i]) / dx^2
    end

    return Akima(xdata, ydata, p0, p1, p2, p3)
end

function (spline::Akima)(x::Number)
    j = findindex(spline.xdata, x)

    # evaluate polynomial
    dx = x - spline.xdata[j]
    y = spline.p0[j] + spline.p1[j] * dx + spline.p2[j] * dx^2 + spline.p3[j] * dx^3

    return y
end

(spline::Akima)(x::AbstractVector) = spline.(x)

"""
    akima(x, y, xpt, delta=0.0, eps=1e-30)

A convenience method to perform construction and evaluation of the spline in one step.
See docstring for Akima for more details.

**Arguments**
- `x, y::Vector{Float}`: the node points
- `xpt::Vector{Float} or ::Float`: the evaluation point(s)
- `delta_x::Float=0.0` : the half width of a smoothing interval used for the absolute
value function.
- `eps::Float=1e-30` : a cutoff used to avoid dividing by zero in the
weighting function. Default is `1e-30` but this could be raised to machine precision in
some cases to improve derivatives.

**Returns**
- `ypt::Vector{Float} or ::Float`: interpolated value(s) at xpt using akima spline.
"""
akima(x, y, xpt, delta=0.0, eps=1e-30) = Akima(x, y, delta, eps)(xpt)

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
    dydx = spline.p1[j] + 2 * spline.p2[j] * dx + 3 * spline.p3[j] * dx^2

    return dydx
end

"""
    second_derivative(spline, x)

Computes the second derivative of an Akima spline at x.

**Arguments**
- `spline::Akima}`: an Akima spline
- `x::Float`: the evaluation point(s)

**Returns**
- `d2ydx2::Float`: second derivative at x using akima spline.
"""
function second_derivative(spline::Akima, x)
    j = findindex(spline.xdata, x)

    # evaluate polynomial
    dx = x - spline.xdata[j]
    d2ydx2 = 2.0 * spline.p2[j] + 6.0 * spline.p3[j] * dx

    return d2ydx2
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
    eta = (x - xdata[i]) / (xdata[i + 1] - xdata[i])
    y = ydata[i] + eta * (ydata[i + 1] - ydata[i])

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
    dydx = (ydata[i + 1] - ydata[i]) / (xdata[i + 1] - xdata[i])

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

    R = promote_type(eltype(xpt), eltype(ypt))

    yinterp = Array{R}(undef, ny, nxpt)
    output = Array{R}(undef, nxpt, nypt)

    for i in 1:ny
        yinterp[i, :] .= interp1d(xdata, fdata[:, i], xpt)
    end
    for i in 1:nxpt
        output[i, :] .= interp1d(ydata, yinterp[:, i], ypt)
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

    R = promote_type(eltype(xpt), eltype(ypt), eltype(zpt))
    zinterp = Array{R}(undef, nz, nxpt, nypt)
    output = Array{R}(undef, nxpt, nypt, nzpt)

    for i in 1:nz
        zinterp[i, :, :] .= interp2d(interp1d, xdata, ydata, fdata[:, :, i], xpt, ypt)
    end
    for j in 1:nypt
        for i in 1:nxpt
            output[i, j, :] .= interp1d(zdata, zinterp[:, i, j], zpt)
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

    R = promote_type(eltype(xpt), eltype(ypt), eltype(zpt), eltype(tpt))
    tinterp = Array{R}(undef, nt, nxpt, nypt, nzpt)
    output = Array{R}(undef, nxpt, nypt, nzpt, ntpt)

    for i in 1:nt
        tinterp[i, :, :, :] .= interp3d(
            interp1d, xdata, ydata, zdata, fdata[:, :, :, i], xpt, ypt, zpt
        )
    end
    for k in 1:nzpt
        for j in 1:nypt
            for i in 1:nxpt
                output[i, j, k, :] .= interp1d(tdata, tinterp[:, i, j, k], tpt)
            end
        end
    end

    return output
end


import LinearAlgebra
import StaticArrays

"""
    CatmullRom(points::Array{Float64, 2})
Creates a Catmull-Rom spline from the given control points. The points should be
a 2D array where each row represents a point in space. The function returns a
CatmullRom object that can be called with a value to evaluate the spline at that point.
"""
struct CatmullRom{T, TF} 
    segments::T
    alpha::TF

    function CatmullRom(points::Array{TF1, 2}; alpha::TF2=0.5) where {TF1, TF2}
        segments = build_catmull_rom(points)
        new{Vector{StaticArrays.SMatrix{4, 2, TF1, 8}}, TF2}(segments, alpha)
    end
end

function CatmullRom(x, y; alpha=0.5)
    points = hcat(x, y)
    return CatmullRom(points; alpha)
end

function (c::CatmullRom)(x)
    return eval_catmull(c.segments, x; alpha=c.alpha)
end

function centripetal_catmull_rom(P, t, alpha=0.5)
    # Compute parameter values based on distance
    function tj(ti, Pi, Pj)
        return ti + LinearAlgebra.norm(Pj .- Pi)^alpha
    end

    P0 = view(P, 1, :)
    P1 = view(P, 2, :)
    P2 = view(P, 3, :)
    P3 = view(P, 4, :)

    t0 = 0.0
    t1 = tj(t0, P0, P1)
    t2 = tj(t1, P1, P2)
    t3 = tj(t2, P2, P3)

    # Map t ∈ [0, 1] to t in [t1, t2]
    t = t1 + t * (t2 - t1)

    # Cubic Lagrange interpolation for centripetal CR spline
    A1 = ((t1 - t)/(t1 - t0)) .* P0 .+ ((t - t0)/(t1 - t0)) .* P1
    A2 = ((t2 - t)/(t2 - t1)) .* P1 .+ ((t - t1)/(t2 - t1)) .* P2
    A3 = ((t3 - t)/(t3 - t2)) .* P2 .+ ((t - t2)/(t3 - t2)) .* P3

    B1 = ((t2 - t)/(t2 - t0)) .* A1 .+ ((t - t0)/(t2 - t0)) .* A2
    B2 = ((t3 - t)/(t3 - t1)) .* A2 .+ ((t - t1)/(t3 - t1)) .* A3

    C  = ((t2 - t)/(t2 - t1)) .* B1 .+ ((t - t1)/(t2 - t1)) .* B2

    return C
end

function build_catmull_rom(control_pts)
    sorted_pts = sortslices(control_pts, dims=1)
    n, m = size(sorted_pts)

    if n < 2
        error("Need at least two control points.")
    end

    # Add phantom endpoints
    first = view(sorted_pts, 1, :) #[1, 2]
    second = view(sorted_pts, 2, :)
    last = view(sorted_pts, n, :)
    penultimate = view(sorted_pts, n-1, :)

    extended = vcat((2 .* first .- second)', sorted_pts, (2 .* last .- penultimate)')

    TF = typeof(extended[1, 1])
    segments = Vector{StaticArrays.SMatrix{4, m, TF, 4m}}(undef, n-1)
    for i in 1:n-1
        segments[i] = StaticArrays.SMatrix{4,m}(view(extended, i:i+3, :))
    end

    return segments
end

function eval_catmull(segments, x; alpha=0.5)
    # Clamp to range
    if x <= segments[1][2, 1] #P1, x
        t = (x - segments[1][2, 1]) / (segments[1][3, 1] - segments[1][2, 1])
        return centripetal_catmull_rom(segments[1], t, alpha)[2]

    elseif x >= segments[end][3, 1]
        t = (x - segments[end][2, 1]) / (segments[end][3, 1] - segments[end][2, 1])
        return centripetal_catmull_rom(segments[end], t, alpha)[2]
    end

    # Find segment x is in
    for seg in segments
        if x >= seg[2, 1] && x <= seg[3, 1]
            t = (x - seg[2, 1]) / (seg[3, 1] - seg[2, 1])
            return centripetal_catmull_rom(seg, t, alpha)[2]
        end
    end
end

