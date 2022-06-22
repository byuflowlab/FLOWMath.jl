var documenterSearchIndex = {"docs":
[{"location":"#FLOWMath.jl-1","page":"Home","title":"FLOWMath.jl","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Examples of the available methods are shown below.  More examples are available in the test suite (/test/runtests.jl)","category":"page"},{"location":"#Quadrature-1","page":"Home","title":"Quadrature","text":"","category":"section"},{"location":"#Trapezoidal-Integration-1","page":"Home","title":"Trapezoidal Integration","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"This is just simple trapezoidal integration using vectors.  Gaussian quadrature is much better, but for times when we need to define a mesh for other purposes and cannot use an adaptive method a simple trapezoidal integration fits the bill.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"using FLOWMath: trapz\n\nx = range(0.0, stop=pi+1e-15, step=pi/100)\ny = sin.(x)\nz = trapz(x, y)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"trapz","category":"page"},{"location":"#FLOWMath.trapz","page":"Home","title":"FLOWMath.trapz","text":"trapz(x, y)\n\nIntegrate y w.r.t. x using the trapezoidal method.\n\n\n\n\n\n","category":"function"},{"location":"#Root-Finding-1","page":"Home","title":"Root Finding","text":"","category":"section"},{"location":"#Brent's-Method-(1D-functions)-1","page":"Home","title":"Brent's Method (1D functions)","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Brent's method is an effective 1D root finding method as it combines bracketing methods (bisection) with fast quadratic interpolation.  Thus, you can get near quadratic convergence but with safeguarding.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"using FLOWMath: brent\n\nf(x) = x^2 - 1.0\nxstar, outputs = brent(f, -2.0, 0)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"The above example shows basic usage.  Additional inputs and outputs are available as described below.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"brent","category":"page"},{"location":"#FLOWMath.brent","page":"Home","title":"FLOWMath.brent","text":"brent(f, a, b; args=(), atol=2e-12, rtol=4*eps(), maxiter=100)\n\n1D root finding using Brent's method.  Based off the brentq implementation in scipy.\n\nArguments\n\nf: scalar function, that optionally takes additional arguments\na::Float, b::Float`: bracketing interval for a root - sign changes sign between: (f(a) * f(b) < 0)\nargs::Tuple: tuple of additional arguments to pass to f\natol::Float: absolute tolerance (positive) for root\nrtol::Float: relative tolerance for root\nmaxiter::Int: maximum number of iterations allowed\n\nReturns\n\nxstar::Float: a root of f\ninfo::Tuple: A named tuple containing:\niter::Int: number of iterations\n'fcalls::Int`: number of function calls\n'flag::String`: a convergence/error message.\n\n\n\n\n\n","category":"function"},{"location":"#Interpolation-1","page":"Home","title":"Interpolation","text":"","category":"section"},{"location":"#Akima-Spline-1","page":"Home","title":"Akima Spline","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"An Akima spline is a 1D spline that avoids overshooting issues common with many other polynomial splines resulting in a more natural curve.  It also only uses local support allowing for more efficient computation.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Interpolation is perhaps clearest through plotting so we'll load a plotting package for this examples.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"using PyPlot\nusing FLOWMath: akima, Akima, derivative, gradient\n\nx = 0:pi/4:2*pi\ny = sin.(x)\nxpt = 0:pi/16:2*pi\n\nypt = akima(x, y, xpt)\n\nfigure()\nplot(x, y, \"o\")\nplot(xpt, ypt)\nsavefig(\"interp.svg\"); nothing # hide","category":"page"},{"location":"#","page":"Home","title":"Home","text":"(Image: )","category":"page"},{"location":"#","page":"Home","title":"Home","text":"or if you plan to evaluate the spline repeatedly","category":"page"},{"location":"#","page":"Home","title":"Home","text":"\nspline = Akima(x, y)\nypt = similar(xpt)\nypt .= spline.(xpt) # ypt change in place\nypt = spline(xpt)\nnothing # hide","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Akima\nakima","category":"page"},{"location":"#FLOWMath.Akima","page":"Home","title":"FLOWMath.Akima","text":"Akima(xdata, ydata, delta_x=0.0)\n\nCreates an akima spline at node points: xdata, ydata.  This is a 1D spline that avoids overshooting issues common with many other polynomial splines resulting in a more natural curve.  It also only depends on local points (i-2...i+2) allow for more efficient computation.  delta_x is the half width of a smoothing interval used for the absolute value function.  Set delta_x=0 to recover the original akima spline.  The smoothing is only useful if you want to differentiate xdata and ydata.  In many case the nodal points are fixed so this is not needed.  Returns an akima spline object (Akima struct). This function, only performs construction of the spline, not evaluation. This is useful if you want to evaluate the same mesh at multiple different conditions. A convenience method exists below to perform both in one shot.\n\n\n\n\n\n","category":"type"},{"location":"#FLOWMath.akima","page":"Home","title":"FLOWMath.akima","text":"akima(x, y, xpt)\n\nA convenience method to perform construction and evaluation of the spline in one step. See docstring for Akima for more details.\n\nArguments\n\nx, y::Vector{Float}: the node points\nxpt::Vector{Float} or ::Float: the evaluation point(s)\n\nReturns\n\nypt::Vector{Float} or ::Float: interpolated value(s) at xpt using akima spline.\n\n\n\n\n\n","category":"function"},{"location":"#","page":"Home","title":"Home","text":"You can also compute the derivative and/or gradient of the spline.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"dydx = derivative(spline, pi/2)\ndydx = gradient(spline, xpt)\nnothing # hide","category":"page"},{"location":"#","page":"Home","title":"Home","text":"derivative\ngradient","category":"page"},{"location":"#FLOWMath.derivative","page":"Home","title":"FLOWMath.derivative","text":"derivative(spline, x)\n\nComputes the derivative of an Akima spline at x.\n\nArguments\n\nspline::Akima}: an Akima spline\nx::Float: the evaluation point(s)\n\nReturns\n\ndydx::Float: derivative at x using akima spline.\n\n\n\n\n\nderivative of linear interpolation at x::Number\n\n\n\n\n\n","category":"function"},{"location":"#FLOWMath.gradient","page":"Home","title":"FLOWMath.gradient","text":"gradient(spline, x)\n\nComputes the gradient of a Akima spline at x.\n\nArguments\n\nspline::Akima}: an Akima spline\nx::Vector{Float}: the evaluation point(s)\n\nReturns\n\ndydx::Vector{Float}: gradient at x using akima spline.\n\n\n\n\n\ngradient of linear interpolation at x::Vector\n\n\n\n\n\n","category":"function"},{"location":"#Linear-Spline-1","page":"Home","title":"Linear Spline","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Linear interpolation is straightforward.  ","category":"page"},{"location":"#","page":"Home","title":"Home","text":"using FLOWMath: linear, derivative, gradient\n\nxvec = [1.0, 2.0, 4.0, 5.0]\nyvec = [2.0, 3.0, 5.0, 8.0]\n\ny = linear(xvec, yvec, 1.5)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"or we can evaluate at multiple points at once.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"y = linear(xvec, yvec, [1.0, 1.5, 3.0, 4.5, 5.0])","category":"page"},{"location":"#","page":"Home","title":"Home","text":"linear","category":"page"},{"location":"#FLOWMath.linear","page":"Home","title":"FLOWMath.linear","text":"linear(xdata, ydata, x::Number)\n\nLinear interpolation.\n\nArguments\n\nxdata::Vector{Float64}: x data used in constructing interpolation\nydata::Vector{Float64}: y data used in constructing interpolation\nx::Float64: point to evaluate spline at\n\nReturns\n\ny::Float64: value at x using linear interpolation\n\n\n\n\n\nlinear(xdata, ydata, x::AbstractVector)\n\nConvenience function to perform linear interpolation at multiple points.\n\n\n\n\n\n","category":"function"},{"location":"#","page":"Home","title":"Home","text":"We can also compute derivatives and gradients just as we can for akima.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"dydx = derivative(xvec, yvec, 1.5)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"dydx = gradient(xvec, yvec, [1.0, 1.5, 3.0, 4.5, 5.0])","category":"page"},{"location":"#D/3D/4D-Interpolation-using-Recursive-1D-Interpolation-1","page":"Home","title":"2D/3D/4D Interpolation using Recursive 1D Interpolation","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"The functions interp2d, interp3d, and interp4d are generic and will accept any method that performs 1D interpolation as the first argument.  In the below examples, akima is used.  These examples are based off of examples from Matlab's interpn documentation.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"2D:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"using FLOWMath: interp2d\n\nx = -5.0:5.0\ny = -5.0:5.0\nz = zeros(11, 11)\nfor i = 1:11\n    for j = 1:11\n        v = sqrt(x[i]^2 + y[j]^2) + 1e-15\n        z[i, j] = sin(v) / v\n    end\nend\n\nxpt = range(-5.0, 5.0, length=100)\nypt = range(-5.0, 5.0, length=100)\n\nzpt = interp2d(akima, x, y, z, xpt, ypt)\n\nfigure()\ncontour(xpt, ypt, zpt)\nsavefig(\"contour.svg\"); nothing # hide","category":"page"},{"location":"#","page":"Home","title":"Home","text":"(Image: )","category":"page"},{"location":"#","page":"Home","title":"Home","text":"4D:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"using FLOWMath: interp4d\n\nx = -1:0.2:1\ny = -1:0.2:1\nz = -1:0.2:1\nt = 0:2:10.0\n\nnx = length(x)\nny = length(y)\nnz = length(z)\nnt = length(t)\n\nf = Array{typeof(x[1])}(undef, nx, ny, nz, nt)\n\nfor i = 1:nx\n    for j = 1:ny\n        for k = 1:nz\n            for l = 1:nt\n                f[i, j, k, l] = t[l]*exp(-x[i]^2 - y[j]^2 - z[k]^2)\n            end\n        end\n    end\nend\n\nxpt = -1:0.05:1\nypt = -1:0.08:1\nzpt = -1:0.05:1\ntpt = 0:0.5:10.0\n\nfpt = interp4d(akima, x, y, z, t, f, xpt, ypt, zpt, tpt)\nnothing # hide","category":"page"},{"location":"#","page":"Home","title":"Home","text":"interp2d\ninterp3d\ninterp4d","category":"page"},{"location":"#FLOWMath.interp2d","page":"Home","title":"FLOWMath.interp2d","text":"interp2d(interp1d, xdata, ydata, fdata, xpt, ypt)\n\n2D interpolation using recursive 1D interpolation.  This approach is likely less efficient than a more direct 2D interpolation method, especially one you can create separate creation from evaluation, but it is generalizable to any spline approach and any dimension.\n\nArguments\n\ninterp1d: any spline function of form: ypt = interp1d(xdata, ydata, xpt) where data are the known   data(node) points and pt are the points where you want to evaluate the spline at.\nxdata::Vector{Float}, ydata::Vector{Float}: Define the 2D grid\nfdata::Matrix{Float}: where fdata[i, j] is the function value at xdata[i], ydata[j]\nxpt::Vector{Float}, ypt::Vector{Float}: the locations where you want to evaluate the spline\n\nReturns\n\nfhat::Matrix{Float}: where fhat[i, j] is the estimate function value at xpt[i], ypt[j]\n\n\n\n\n\n","category":"function"},{"location":"#FLOWMath.interp3d","page":"Home","title":"FLOWMath.interp3d","text":"interp3d(interp1d, xdata, ydata, zdata, fdata, xpt, ypt, zpt)\n\nSame as interp2d, except in three dimension.\n\n\n\n\n\n","category":"function"},{"location":"#FLOWMath.interp4d","page":"Home","title":"FLOWMath.interp4d","text":"interp4d(interp1d, xdata, ydata, zdata, fdata, xpt, ypt, zpt)\n\nSame as interp3d, except in four dimensions.\n\n\n\n\n\n","category":"function"},{"location":"#Smoothing-1","page":"Home","title":"Smoothing","text":"","category":"section"},{"location":"#Absolute-value-1","page":"Home","title":"Absolute value","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"The absolute value function is not differentiable at x = 0.  The below function smoothly adds a small quadratic function in place of the cusp with a half-width given by delta_x.  This small rounding at the bottom can prevent numerical issues with gradient-based optimization.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"using FLOWMath: abs_smooth\n\nx = range(-2.0, 2.0, length=100)\ndelta_x = 0.1\n\ny = abs_smooth.(x, delta_x)\n\nusing PyPlot\nfigure()\nplot(x, y)\nsavefig(\"abs.svg\"); nothing # hide","category":"page"},{"location":"#","page":"Home","title":"Home","text":"(Image: )","category":"page"},{"location":"#","page":"Home","title":"Home","text":"abs_smooth","category":"page"},{"location":"#FLOWMath.abs_smooth","page":"Home","title":"FLOWMath.abs_smooth","text":"abs_smooth(x, delta_x)\n\nSmooth out the absolute value function with a quadratic interval. delta_x is the half width of the smoothing interval. Typically usage is with gradient-based optimization.\n\n\n\n\n\n","category":"function"},{"location":"#Kreisselmeier-Steinhauser-Constraint-Aggregation-Function-1","page":"Home","title":"Kreisselmeier-Steinhauser Constraint Aggregation Function","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"The Kreisselmeier-Steinhauser (KS) function is often used with constrained gradient-based optimization problems to smoothly aggregate an arbitrary number of constraints into a single constraint.  It may also be used as a smooth approximation of the maximum function (or minimum function).  A salient feature of this function is that it is guaranteed to overestimate the maximum function (or underestimate the minimum function).  This feature of the function can be used to ensure that the resulting constraint is conservative.  ","category":"page"},{"location":"#","page":"Home","title":"Home","text":"We provide two implementations of this function: ksmax and ksmin.  ksmax and ksmin may be used to smoothly approximate the maximum and minimum functions, respectively.  Both functions take the optional parameter hardness that controls the smoothness of the resulting function.  As hardness increases the function more and more closely approximates the maximum (or minimum) function.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"using FLOWMath: ksmax, ksmin\n\nx = [1.2, 0.0, 0.5]\nhardness = 100\nmax_x = ksmax(x, hardness)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"min_x = ksmin(x, hardness)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"ksmax\nksmin","category":"page"},{"location":"#FLOWMath.ksmax","page":"Home","title":"FLOWMath.ksmax","text":"ksmax(x, hardness=50)\n\nKreisselmeier–Steinhauser constraint aggregation function.  In the limit as hardness goes to infinity the maximum function is returned. Is mathematically guaranteed to overestimate the maximum function, i.e. maximum(x) <= ksmax(x, hardness).\n\n\n\n\n\n","category":"function"},{"location":"#FLOWMath.ksmin","page":"Home","title":"FLOWMath.ksmin","text":"ksmin(x, hardness=50)\n\nKreisselmeier–Steinhauser constraint aggregation function.  In the limit as hardness goes to infinity the minimum function is returned. Is mathematically guaranteed to underestimate the minimum function, i.e. minimum(x) <= ksmin(x, hardness).\n\n\n\n\n\n","category":"function"},{"location":"#Blending-functions-using-the-sigmoid-function-1","page":"Home","title":"Blending functions using the sigmoid function","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"The sigmoid function may be used to smoothly blend the results of two continuous one-dimensional functions.  The method implemented in this package uses a user-specified transition location (xt) and scales the input of the sigmoid function using the input hardness in order to adjust the smoothness of the transition between the two functions.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"using FLOWMath: sigmoid_blend\n\nx = 0.1\nf1x = x\nf2x = x^2\nxt = 0.0\nhardness = 25\ny = sigmoid_blend(f1x, f2x, x, xt, hardness)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"sigmoid_blend can also be used with vector inputs using broadcasting.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"x = -0.25:0.001:0.25\nf1x = x\nf2x = x.^2\nxt = 0.0\nhardness = 25\ny = sigmoid_blend.(f1x, f2x, x, xt, hardness)\n\nusing PyPlot\nfigure()\nplot(x, f1x)\nplot(x, f2x)\nplot(x, y)\nlegend([\"f1(x)\",\"f2(x)\",\"sigmoid\"])\nsavefig(\"sigmoid.svg\"); nothing # hide","category":"page"},{"location":"#","page":"Home","title":"Home","text":"(Image: )","category":"page"},{"location":"#","page":"Home","title":"Home","text":"sigmoid_blend","category":"page"},{"location":"#FLOWMath.sigmoid_blend","page":"Home","title":"FLOWMath.sigmoid_blend","text":"sigmoid_blend(f1x, f2x, x, xt, hardness=50)\n\nSmoothly transitions the results of functions f1 and f2 using the sigmoid function, with the transition between the functions located at xt. hardness controls the sharpness of the transition between the two functions.\n\n\n\n\n\n","category":"function"},{"location":"#Blending-functions-using-cubic-or-quintic-polynomials-1","page":"Home","title":"Blending functions using cubic or quintic polynomials","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Cubic or quintic polynomials can also be used to construct a piecewise function that smoothly blends two functions.  The advantage of this approach compared to sigmoid_blend is that the blending can be restricted to a small interval defined by the half-width delta_x.  The disadvantage of this approach is that the resulting function is only C1 continuous when cubic_blend is used, and C2 continuous when quintic_blend is used.  The method implemented in this package uses a user-specified transition location (xt).  The smoothness of the transition between the two functions can be adjusted by modifying delta_x, which is the half-width of the transition interval.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"using FLOWMath: cubic_blend, quintic_blend\n\nx = 0.05\nf1x = x\nf2x = x^2\nxt = 0.0\ndelta_x = 0.1\ny1 = cubic_blend(f1x, f2x, x, xt, delta_x)\ny2 = quintic_blend(f1x, f2x, x, xt, delta_x)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"cubic_blend and quintic_blend can also be used with vector inputs using broadcasting.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"x = -0.25:0.001:0.25\nf1x = x\nf2x = x.^2\nxt = 0.0\ndelta_x = 0.1\ny1 = cubic_blend.(f1x, f2x, x, xt, delta_x)\ny2 = quintic_blend.(f1x, f2x, x, xt, delta_x)\n\nusing PyPlot\nfigure()\nplot(x, f1x)\nplot(x, f2x)\nplot(x, y1)\nplot(x, y2)\nlegend([\"f1(x)\",\"f2(x)\",\"cubic\", \"quintic\"])\nsavefig(\"cubic.svg\"); nothing # hide","category":"page"},{"location":"#","page":"Home","title":"Home","text":"(Image: )","category":"page"},{"location":"#","page":"Home","title":"Home","text":"cubic_blend\nquintic_blend","category":"page"},{"location":"#FLOWMath.cubic_blend","page":"Home","title":"FLOWMath.cubic_blend","text":"cubic_blend(f1x, f2x, x, xt, delta_x)\n\nSmoothly transitions the results of functions f1 and f2 using a cubic polynomial, with the transition between the functions located at xt. delta_x is the half width of the smoothing interval.  The resulting function is C1 continuous.\n\n\n\n\n\n","category":"function"},{"location":"#FLOWMath.quintic_blend","page":"Home","title":"FLOWMath.quintic_blend","text":"quintic_blend(f1x, f2x, x, xt, delta_x)\n\nSmoothly transitions the results of functions f1 and f2 using a quintic polynomial, with the transition between the functions located at xt. delta_x is the half width of the smoothing interval.  The resulting function is C2 continuous.\n\n\n\n\n\n","category":"function"},{"location":"#Complex-step-safe-functions-1","page":"Home","title":"Complex-step safe functions","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"The complex-step derivative approximation can be used to easily and accurately approximate first derivatives. However, the function f one wishes to differentiate must be composed of functions that are compatible with the method. Most elementary functions are, but a few common ones are not:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"abs\nabs2\nnorm\ndot\nthe two argument form of atan (often called atan2 or arctan2 in other languages)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"FLOWMath provides complex-step safe versions of these functions. These functions use Julia's multiple dispatch to fall back on the standard implementations when given real arguments, and so shouldn't impose any performance penalty when not used with the complex step method.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"abs_cs_safe\nabs2_cs_safe\nnorm_cs_safe\ndot_cs_safe\natan_cs_safe","category":"page"},{"location":"#FLOWMath.abs_cs_safe","page":"Home","title":"FLOWMath.abs_cs_safe","text":"abs_cs_safe(x)\n\nCalculate the absolute value of x in a manner compatible with the complex-step derivative approximation.\n\nSee also: abs.\n\n\n\n\n\n","category":"function"},{"location":"#FLOWMath.abs2_cs_safe","page":"Home","title":"FLOWMath.abs2_cs_safe","text":"abs2_cs_safe(x)\n\nCalculate the squared absolute value of x in a manner compatible with the complex-step derivative approximation.\n\nSee also: abs2.\n\n\n\n\n\n","category":"function"},{"location":"#FLOWMath.norm_cs_safe","page":"Home","title":"FLOWMath.norm_cs_safe","text":"norm_cs_safe(x, p)\n\nCalculate the p-norm value of iterable x in a manner compatible with the complex-step derivative approximation.\n\nSee also: norm.\n\n\n\n\n\n","category":"function"},{"location":"#FLOWMath.dot_cs_safe","page":"Home","title":"FLOWMath.dot_cs_safe","text":"dot_cs_safe(a, b)\n\nCalculate the dot product of vectors a and b in a manner compatible with the complex-step derivative approximation.\n\nSee also: norm.\n\n\n\n\n\n","category":"function"},{"location":"#FLOWMath.atan_cs_safe","page":"Home","title":"FLOWMath.atan_cs_safe","text":"atan_cs_safe(y, x)\n\nCalculate the two-argument arctangent function in a manner compatible with the complex-step derivative approximation.\n\nSee also: atan.\n\n\n\n\n\n","category":"function"}]
}