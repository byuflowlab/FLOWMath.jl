# ------ Root Finding Methods -----




"""
    brent(f, a, b; args=(), xtol=2e-12, rtol=4*eps(), maxiter=100, full_output=false)

1D root finding using Brent's method.  Based off the brentq implementation in scipy.

**Arguments**
- `f`: scalar function, that optionally takes additional arguments
- `a`::Float, `b::Float`: bracketing interval for a root - sign changes sign between: (f(a) * f(b) < 0)
- `args::Tuple`: tuple of additional arguments to pass to f
- `xtol::Float`: absolute tolerance (positive) for root
- `rtol::Float`: relative tolerance for root
- `maxiter::Int`: maximum number of iterations allowed
- `full_output::Bool`: flag to indicate whether you want just the root, or the root with a 
    second argument (tuple) containing the number of iterations, function calls, and a convergence message.
"""
function brent(f, a, b; args=(), xtol=2e-12, rtol=4*eps(), maxiter=100, full_output=false)

    xpre = a; xcur = b
    xblk = 0.0; fblk = 0.0; spre = 0.0; scur = 0.0
    error_num = "INPROGRESS"

    fpre = f(xpre, args...)
    fcur = f(xcur, args...)
    funcalls = 2
    iterations = 0
    
    if fpre*fcur > 0
        error_num = "SIGNERR"
        return _pack_brent_results(0.0, iterations, funcalls, error_num, full_output)
    end
    if fpre == 0
        error_num = "CONVERGED"
        return _pack_brent_results(xpre, iterations, funcalls, error_num, full_output)
    end
    if fcur == 0
        error_num = "CONVERGED"
        return _pack_brent_results(xcur, iterations, funcalls, error_num, full_output)
    end

    for i = 1:maxiter
        iterations += 1
        if fpre*fcur < 0 
            xblk = xpre
            fblk = fpre
            spre = scur = xcur - xpre
        end
        if abs(fblk) < abs(fcur)
            xpre = xcur
            xcur = xblk
            xblk = xpre

            fpre = fcur
            fcur = fblk
            fblk = fpre
        end

        delta = (xtol + rtol*abs(xcur))/2
        sbis = (xblk - xcur)/2
        if fcur == 0 || abs(sbis) < delta
            error_num = "CONVERGED"
            return _pack_brent_results(xcur, iterations, funcalls, error_num, full_output) 
        end

        if abs(spre) > delta && abs(fcur) < abs(fpre)
            if xpre == xblk
                # interpolate
                stry = -fcur*(xcur - xpre)/(fcur - fpre)
            else
                # extrapolate
                dpre = (fpre - fcur)/(xpre - xcur)
                dblk = (fblk - fcur)/(xblk - xcur)
                stry = -fcur*(fblk*dblk - fpre*dpre)/(dblk*dpre*(fblk - fpre))
            end
            if 2*abs(stry) < min(abs(spre), 3*abs(sbis) - delta)
                # good short step
                spre = scur
                scur = stry
            else
                # bisect
                spre = sbis
                scur = sbis
            end
        else 
            # bisect
            spre = sbis
            scur = sbis
        end

        xpre = xcur; fpre = fcur
        if abs(scur) > delta
            xcur += scur
        else
            xcur += (sbis > 0 ? delta : -delta)
        end

        fcur = f(xcur, args...)
        funcalls += 1
    end
    error_num = "CONVERR"
    return _pack_brent_results(xcur, iterations, funcalls, error_num, full_output)
end

# TODO AN: replace w/ newer Brent method and automatic bracketing?


"""
private method

pack up results from brent method
"""
function _pack_brent_results(x, iter, fcalls, flag, full_output)
    if full_output
        return x, (iter=iter, fcalls=fcalls, flag=flag)
    else
        return x
    end
end
