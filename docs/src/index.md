# FLOWMath.jl

Examples of the available methods are shown below.  More examples are available in the test suite (/test/runtests.jl)

## Quadrature

### Trapezoidal Integration

This is just simple trapezoidal integration using vectors.  [Gaussian quadrature](https://github.com/JuliaMath/QuadGK.jl) is much better, but for times when we need to define a mesh for other purposes and cannot use an adaptive method a simple trapezoidal integration fits the bill.

```@example trapz
using FLOWMath: trapz

x = range(0.0, stop=pi+1e-15, step=pi/100)
y = sin.(x)
z = trapz(x, y)
```

```@docs
trapz
```

## Root Finding

### Brent's Method (1D functions)

Brent's method is an effective 1D root finding method as it combines bracketing methods (bisection) with fast quadratic interpolation.  Thus, you can get near quadratic convergence but with safeguarding.

```@example
using FLOWMath: brent

f(x) = x^2 - 1.0
xstar = brent(f, -2.0, 0)
```

The above example shows basic usage.  Additional inputs and outputs are available as described below.

```@docs
brent
```

## Interpolation

### Akima Spline

An Akima spline is a 1D spline that avoids overshooting issues common with many other polynomial splines resulting in a more natural curve.  It also only uses local support allowing for more efficient computation.

Interpolation is perhaps clearest through plotting so we'll load a plotting package for this examples.

```@example akima
using PyPlot
nothing # hide
```


```@example akima
using FLOWMath: akima, Akima

x = 0:pi/4:2*pi
y = sin.(x)
xpt = 0:pi/16:2*pi

ypt = akima(x, y, xpt)

figure()
plot(x, y, "o")
plot(xpt, ypt)
savefig("interp.svg"); nothing # hide
```

![](interp.svg)

or if you plan to evaluate the spline repeatedly

```@example akima

spline = Akima(x, y)
ypt = similar(xpt)
ypt .= spline.(xpt) # ypt change in place
ypt = spline(xpt)
nothing # hide
```


```@docs
Akima
akima
```

You can also compute the gradient of the spline.

```@example akima

dydx = gradient(spline, xpt)
nothing # hide
```

```@docs
gradient
```

### 2D/3D/4D Interpolation using Recursive 1D Interpolation

The functions `interp2d`, `interp3d`, and `interp4d` are generic and will accept any method that performs 1D interpolation as the first argument.  In the below examples, akima is used.  These examples are based off of examples from Matlab's interpn documentation.

2D:
```@example akima
using FLOWMath: interp2d

x = -5.0:5.0
y = -5.0:5.0
z = zeros(11, 11)
for i = 1:11
    for j = 1:11
        v = sqrt(x[i]^2 + y[j]^2) + 1e-15
        z[i, j] = sin(v) / v
    end
end

xpt = range(-5.0, 5.0, length=100)
ypt = range(-5.0, 5.0, length=100)

zpt = interp2d(akima, x, y, z, xpt, ypt)

figure()
contour(xpt, ypt, zpt)
savefig("contour.svg"); nothing # hide
```

![](contour.svg)

4D:
```@example akima
using FLOWMath: interp4d

x = -1:0.2:1
y = -1:0.2:1
z = -1:0.2:1
t = 0:2:10.0

nx = length(x)
ny = length(y)
nz = length(z)
nt = length(t)

f = Array{typeof(x[1])}(undef, nx, ny, nz, nt)

for i = 1:nx
    for j = 1:ny
        for k = 1:nz
            for l = 1:nt
                f[i, j, k, l] = t[l]*exp(-x[i]^2 - y[j]^2 - z[k]^2)
            end
        end
    end
end

xpt = -1:0.05:1
ypt = -1:0.08:1
zpt = -1:0.05:1
tpt = 0:0.5:10.0

fpt = interp4d(akima, x, y, z, t, f, xpt, ypt, zpt, tpt)
nothing # hide
```


```@docs
interp2d
interp3d
interp4d
```

## Smoothing

### Absolute value

The absolute value function is not differentiable at x = 0.  The below function smoothly adds a small quadratic function in place of the cusp with a half-width given by `delta_x`.  This small rounding at the bottom can prevent numerical issues with gradient-based optimization.

```@example
using FLOWMath: abs_smooth

x = range(-2.0, 2.0, length=100)
delta_x = 0.1

y = abs_smooth.(x, delta_x)

using PyPlot
figure()
plot(x, y)
savefig("abs.svg"); nothing # hide
```

![](abs.svg)


```@docs
abs_smooth
```

### Kreisselmeier-Steinhauser Constraint Aggregation Function

The Kreisselmeier-Steinhauser (KS) function is often used with constrained gradient-based optimization problems to smoothly aggregate an arbitrary number of constraints into a single constraint.  It may also be used as a smooth approximation of the maximum function (or minimum function).  A salient feature of this function is that it is guaranteed to overestimate the maximum function (or underestimate the minimum function).  This feature of the function can be used to ensure that the resulting constraint is conservative.  

We provide two implementations of this function: `ksmax` and `ksmin`.  `ksmax` and `ksmin` may be used to smoothly approximate the maximum and minimum functions, respectively.  Both functions take the optional parameter `hardness` that controls the smoothness of the resulting function.  As `hardness` increases the function more and more closely approximates the maximum (or minimum) function.

```@example ks
using FLOWMath: ksmax, ksmin

x = [1.2, 0.0, 0.5]
hardness = 100
max_x = ksmax(x, hardness)
```

```@example ks
min_x = ksmin(x, hardness)
```

```@docs
ksmax
ksmin
```

### Blending functions using the sigmoid function

The sigmoid function may be used to smoothly blend the results of two continuous one-dimensional functions.  The method implemented in this package uses a user-specified transition location (`xt`) and scales the input of the sigmoid function using the input `hardness` in order to adjust the smoothness of the transition between the two functions.

```@example sb
using FLOWMath: sigmoid_blend

x = 0.1
f1x = x
f2x = x^2
xt = 0.0
hardness = 25
y = sigmoid_blend(f1x, f2x, x, xt, hardness)
```

`sigmoid_blend` can also be used with vector inputs using broadcasting.

```@example sb
x = -0.25:0.001:0.25
f1x = x
f2x = x.^2
xt = 0.0
hardness = 25
y = sigmoid_blend.(f1x, f2x, x, xt, hardness)

using PyPlot
figure()
plot(x, f1x)
plot(x, f2x)
plot(x, y)
legend(["f1(x)","f2(x)","sigmoid"])
savefig("sigmoid.svg"); nothing # hide
```

![](sigmoid.svg)

```@docs
sigmoid_blend
```

### Blending functions using cubic or quintic polynomials

Cubic or quintic polynomials can also be used to construct a piecewise function that smoothly blends two functions.  The advantage of this approach compared to `sigmoid_blend` is that the blending can be restricted to a small interval defined by the half-width `delta_x`.  The disadvantage of this approach is that the resulting function is only C1 continuous when `cubic_blend` is used, and C2 continuous when `quintic_blend` is used.  The method implemented in this package uses a user-specified transition location (`xt`).  The smoothness of the transition between the two functions can be adjusted by modifying `delta_x`, which is the half-width of the transition interval.

```@example poly
using FLOWMath: cubic_blend, quintic_blend

x = 0.05
f1x = x
f2x = x^2
xt = 0.0
delta_x = 0.1
y1 = cubic_blend(f1x, f2x, x, xt, delta_x)
y2 = quintic_blend(f1x, f2x, x, xt, delta_x)
```

`cubic_blend` and `quintic_blend` can also be used with vector inputs using broadcasting.

```@example poly
x = -0.25:0.001:0.25
f1x = x
f2x = x.^2
xt = 0.0
delta_x = 0.1
y1 = cubic_blend.(f1x, f2x, x, xt, delta_x)
y2 = quintic_blend.(f1x, f2x, x, xt, delta_x)

using PyPlot
figure()
plot(x, f1x)
plot(x, f2x)
plot(x, y1)
plot(x, y2)
legend(["f1(x)","f2(x)","cubic", "quintic"])
savefig("cubic.svg"); nothing # hide
```

![](cubic.svg)

```@docs
cubic_blend
quintic_blend
```
