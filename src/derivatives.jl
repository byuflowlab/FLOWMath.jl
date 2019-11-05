#=
Functions to estimate derivatives.

While I'd strongly recommend using algorithmic differentiation instead,
these estimates can be useful as checks.
=#


"""
    forwarddiff(func, x; h=1e-6, parallel=false)

Forward finite difference.  Evaluates both function values and the Jacobian.
f and x can both be vector.  Returns Jacobian df[i]/dx[j]

**Arguments**
- `func`: function of type: `f = func(x)`
- `x`: input vector to take derivatives at.
- `h`: default step size.  actual step size is:  s_i = h * (1 + |x_i|) so that step sizing is relative
    unless x_i is very small.
- `parallel`: evaluate in paralell
"""
function forwarddiff(func, x; h=1e-6, parallel=false)

    Nin = length(x)
    step = Vector{Float64}(undef, Nin)
    xp = Vector{Vector{Float64}}(undef, Nin+1)

    # setup a matrix of x values to compute at, with a small step for each x_i
    for i = 1:Nin
        step[i] = h*(1+abs(x[i]))
        xp[i] = copy(x)
        xp[i][i] += step[i]
    end
    xp[Nin+1] = x  # evaluate one more point at original x (no step)

    # do all function calls at same time
    if parallel
        out = pmap(func, xp)
    else
        out = map(func, xp)
    end

    fp = out[1:Nin]
    f = out[Nin+1]

    # compute gradient
    Nout = length(f)
    dfdx = zeros(Nout, Nin)
    for i = 1:Nin
        dfdx[:, i] = (fp[i] - f)/step[i]
    end

    return dfdx    
end




"""
Backward finite difference.
h should be positive.
"""
function backwarddiff(func, x; h=1e-6, parallel=false)
    return forwarddiff(func, x, h=-h, parallel=parallel)
end

"""
Central finite difference.
h should be positive.
"""
function centraldiff(func, x; h=1e-4, parallel=false)

    Nin = length(x)
    step = Vector{Float64}(undef, Nin)
    xp = Vector{Vector{Float64}}(undef, 2*Nin)

   # setup x values
    for i = 1:Nin
        step[i] = h*(1+abs(x[i]))
        xp[i] = copy(x)
        xp[i][i] += step[i]
        xp[Nin+i] = copy(x)
        xp[Nin+i][i] -= step[i]
    end

    # do all function calls at same time
    if parallel
        out = pmap(func, xp, retry_delays=zeros(Float64,3))  # TODO: Taylor what is this?
    else
        out = map(func, xp)
    end

    fp = out[1:Nin]
    fm = out[Nin+1:end]

    # compute gradient
    Nout = length(fp[1])
    dfdx = zeros(Nout, Nin)
    for i = 1:Nin
        dfdx[:, i] = (fp[i] - fm[i])/(2*step[i])
    end

    return dfdx
end


"""
Complex step
"""
function complexstep(func, x; h=1e-30, parallel=false)

    Nin = length(x)
    step = Vector{Float64}(undef, Nin)
    xp = Vector{Vector{Complex{Float64}}}(undef, Nin)

    # setup x values
    for i = 1:Nin
        xp[i] = copy(x)
        xp[i][i] += im*h
    end

    # do all function calls at same time
    if parallel
        fc = pmap(func, xp)
    else
        fc = map(func, xp)
    end

    # compute gradient
    Nout = length(fc[1])
    dfdx = zeros(Nout, Nin)
    for i = 1:Nin
        dfdx[:, i] = imag(fc[i])/h
    end

    return dfdx
end


# """
# check gradients in gcheck against centraldiff
# """
# function check(func, x, gcheck, gtol=1e-6)

#     f, gfd = centraldiff(func, x)

#     nf = length(f)
#     nx = length(x)
#     gerror = zeros(nf, nx)

#     for j = 1:nx
#         for i = 1:nf
#             if gcheck[i, j] <= gtol
#                 gerror[i, j] = abs(gcheck[i, j] - gfd[i, j])
#             else
#                 gerror[i, j] = abs(1 - gfd[i, j]/gcheck[i, j])
#             end
#             if gerror[i, j] > gtol
#                 println("**** gerror(", i, ", ", j, ") = ", gerror[i, j])
#             end
#         end
#     end

#     return gerror
# end


# """
# Forward mode AD
# """
# function forwardad(fun, x)

#     f = fun(x)
#     J = ForwardDiff.jacobian(fun, x)  # clearly a warapper is not needed here, however new AD methods are in development so it may be helpful to retain this wrapper.

#     return f, J
# end


# function reverseadinit(fun, x)

#     f_tape = ReverseDiff.JacobianTape(fun, x)
#     compiled_f_tape = ReverseDiff.compile(f_tape)  # TODO: need to save this outside of function scope if we will be reusing

#     return compiled_f_tape
# end

# """
# Reverse mode AD
# """
# function reversead(fun, x, ftape)

#     f = fun(x)
#     Nin = length(x)
#     Nout = length(f)

#     dfdx = zeros(Nout, Nin)
#     ReverseDiff.jacobian!(dfdx, ftape, x)

#     return f, dfdx
# end

# """
# An implementation of fzero that propgates derivative correctly with AD.
# """
# function fzero(R, lower, upper)

#     xstar = Roots.fzero(R, lower, upper)

#     # a couple iterations of Newton's method to propagate the dual number.
#     for i = 1:2
#         dR = ForwardDiff.derivative(R, xstar)
#         xstar = xstar - R(xstar)/dR
#     end

#     return xstar
# end

# end # module