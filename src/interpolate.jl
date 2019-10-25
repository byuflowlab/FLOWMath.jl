#= 
Interpolation Methods

=#

## ------ Akima Interpolation  ---------

# struct used internally
struct Akima{TF, T1<:AbstractVector{TF}, T2<:AbstractVector{TF}}

    xdata::T1
    ydata::T2
    p0::Vector{TF}
    p1::Vector{TF}
    p2::Vector{TF}
    p3::Vector{TF}

end


"""
    akima_setup(xdata, ydata, delta_x=0.1)

Setup akima spline at node points: xdata, ydata.  This is a 1D spline that avoids
overshooting issues common with many other polynomial splines resulting in a more
natural curve.  It also only depends on local points (i-2...i+2) allow for more efficient
computation.  delta_x is the half width of a smoothing interval used for the absolute
value function.  Set delta_x=0 to recover the original akima spline.  The smoothing
is only useful if you want to differentiate xdata and ydata.  In many case the nodal points 
are fixed so this is not needed.  Returns an akima spline object (Akima struct).  
This function, in connection with akima_interp separates the construction from evaluation.  
This is useful if you want to evaluate the same mesh at multiple different conditions.
"""
function akima_setup(xdata, ydata, delta_x=0.0)

    # setup
    eps = 1e-30
    n = length(xdata)

    # compute segment slopes
    m = OffsetVector(zeros(n+3), -1:n+1)
    for i = 1:n-1
        m[i] = (ydata[i+1] - ydata[i]) / (xdata[i+1] - xdata[i])
    end

    # estimation for end points
    m[0] = 2.0*m[1] - m[2]
    m[-1] = 2.0*m[0] - m[1]
    m[n] = 2.0*m[n-1] - m[n-2]
    m[n+1] = 2.0*m[n] - m[n-1]

    # slope at points
    t = zeros(n)
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
    p0 = zeros(n-1)
    p1 = zeros(n-1)
    p2 = zeros(n-1)
    p3 = zeros(n-1)
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


"""
    akima_interp(x, spline)

Evaluate the spline setup by akima_setup.  x are the locations to evaluate the spline at
and spline is the object returned by akima_setup.
"""
function akima_interp(x, spline)
   
    # interpolate at each point
    n = length(x)
    y = zeros(n)

    for i = 1:n

        j = findlast(x[i] .>= spline.xdata[1:end-1])
        if isnothing(j)
            j = 1
        end

        # evaluate polynomial
        dx = x[i] - spline.xdata[j]
        y[i] = spline.p0[j] + spline.p1[j]*dx + spline.p2[j]*dx^2 + spline.p3[j]*dx^3

    end

    return y
end

"""
    akima(xdata, ydata, xpt)

A convenience method to perform both the setup and evaluation of the akima spline in one go.
xpt may be an array.
"""
function akima(xdata, ydata, xpt)
    spline = akima_setup(xdata, ydata)
    return akima_interp(xpt, spline)
end


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
    yinterp = zeros(ny, nxpt)
    output = zeros(nxpt, nypt)

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
    zinterp = zeros(nz, nxpt, nypt)
    output = zeros(nxpt, nypt, nzpt)

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
