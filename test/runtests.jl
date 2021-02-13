using FLOWMath
using Test
import ForwardDiff
using LinearAlgebra: diag

@testset "FLOWMath.jl" begin


# ------ trapz --------
# tests from matlab trapz docs
x = 1:5
y = [1 4 9 16 25]
z = trapz(x, y)
@test z == 42.0

x = range(0.0, stop=pi+1e-15, step=pi/100)
y = sin.(x)
z = trapz(x, y)
@test isapprox(z, 1.999835503887444, atol=1e-15)

# -------------------------

# ------ Brent's method ------

f(x) = x^2 - 1.0

xstar, _ = brent(f, -2.0, 0)
@test xstar == -1.0
xstar, _ = brent(f, 0.0, 2)
@test xstar == 1.0

f(x) = x^3 - 1
xstar, _ = brent(f, 0, 3)
@test xstar == 1.0

f(x) = sin(x)
atol = 2e-12
xstar, _ = brent(f, 1, 4, atol=atol)
@test isapprox(xstar, pi, atol=atol)

atol = 1e-15
xstar, _ = brent(f, 1, 4, atol=atol)
@test isapprox(xstar, pi, atol=atol)

# -------------------------

# ------ abs_smooth ---------
x = -3.0
delta_x = 0.0
y = abs_smooth(x, delta_x)
@test y == 3.0

x = 3.0
delta_x = 0.1
y = abs_smooth(x, delta_x)
@test y == 3.0

x = -3.0
delta_x = 0.1
y = abs_smooth(x, delta_x)
@test y == 3.0

x = 0
delta_x = 0.1
y = abs_smooth(x, delta_x)
@test y == 0.05

x = 0.1
delta_x = 0.1
y = abs_smooth(x, delta_x)
@test y == 0.1

x = -0.1
delta_x = 0.1
y = abs_smooth(x, delta_x)
@test y == 0.1

x = 0.05
delta_x = 0.1
y = abs_smooth(x, delta_x)
@test y == 0.0625

x = -0.05
delta_x = 0.1
y = abs_smooth(x, delta_x)
@test y == 0.0625
# -------------------------

# ------ ksmax ---------
x = [0.0, 0.0]
x_max_smooth = ksmax(x)
@test isapprox(x_max_smooth, 0.013862943611198907)

# overflow
x = [1e6, 1e6]
x_max_smooth = ksmax(x)
@test isapprox(x_max_smooth, 1.0000000138629436e6)

# underflow
x = [-1e6, -1e6]
x_max_smooth = ksmax(x)
@test isapprox(x_max_smooth, -999999.9861370564)

# hardness
x = [0.0, 0.0]
hardness = 100.0
x_max_smooth = ksmax(x, hardness)
@test isapprox(x_max_smooth, 0.006931471805599453)

# ------ ksmin ---------
x = [0.0, 0.0]
x_max_smooth = ksmin(x)
@test isapprox(x_max_smooth, -0.013862943611198907)

# overflow
x = [1e6, 1e6]
x_max_smooth = ksmin(x)
@test isapprox(x_max_smooth, 999999.9861370564)

# underflow
x = [-1e6, -1e6]
x_max_smooth = ksmin(x)
@test isapprox(x_max_smooth, -1.0000000138629436e6)

# hardness
x = [0.0, 0.0]
hardness = 100.0
x_max_smooth = ksmin(x, hardness)
@test isapprox(x_max_smooth, -0.006931471805599453)

# -------------------------

# ------ ksmax_adaptive ---------
x = [0.0, 0.0]
x_max_smooth = ksmax_adaptive(x)
@test isapprox(x_max_smooth, 0.0008325546111576975)

# overflow
x = [1e6, 1e6]
x_max_smooth = ksmax_adaptive(x)
@test isapprox(x_max_smooth, 1.0000000008325547e6)

# underflow
x = [-1e6, -1e6]
x_max_smooth = ksmax_adaptive(x)
@test isapprox(x_max_smooth, -999999.9991674453)

# hardness
x = [0.0, 0.0]
hardness = 100.0
x_max_smooth = ksmax_adaptive(x, hardness)
@test isapprox(x_max_smooth, 0.0008325546111576975)

x = [-0.1, 0.0]
hardness = 100.0
x_max_smooth = ksmax_adaptive(x, hardness)
@test isapprox(x_max_smooth, 4.5398899216870535e-7)

# tolerance
x = [0.0, 0.0]
tol = 1e-3
x_max_smooth = ksmax_adaptive(x, tol=tol)
@test isapprox(x_max_smooth, 0.013862943611198907)

# smoothing_fraction
x = [-0.165, 0.0]
smoothing_fraction = 0.1
x_max_smooth = ksmax_adaptive(x, smoothing_fraction=smoothing_fraction)
@test isapprox(x_max_smooth, 5.261253727654178e-6)

x = [-0.165, 0.0]
smoothing_fraction = 0.2
x_max_smooth = ksmax_adaptive(x, smoothing_fraction=smoothing_fraction)
@test isapprox(x_max_smooth, 5.2856933329025475e-6)

# ------ ksmin_adaptive ---------
x = [0.0, 0.0]
x_max_smooth = ksmin_adaptive(x)
@test isapprox(x_max_smooth, -0.0008325546111576975)

# overflow
x = [1e6, 1e6]
x_max_smooth = ksmin_adaptive(x)
@test isapprox(x_max_smooth, 999999.9991674453)

# underflow
x = [-1e6, -1e6]
x_max_smooth = ksmin_adaptive(x)
@test isapprox(x_max_smooth, -1.0000000008325547e6)

# hardness
x = [0.0, 0.0]
hardness = 100.0
x_max_smooth = ksmin_adaptive(x, hardness)
@test isapprox(x_max_smooth, -0.0008325546111576975)

x = [0.1, 0.0]
hardness = 100.0
x_max_smooth = ksmin_adaptive(x, hardness)
@test isapprox(x_max_smooth, -4.5398899216870535e-7)

# tolerance
x = [0.0, 0.0]
tol = 1e-3
x_max_smooth = ksmin_adaptive(x, tol=tol)
@test isapprox(x_max_smooth, -0.013862943611198907)

# smoothing_fraction
x = [0.165, 0.0]
smoothing_fraction = 0.1
x_max_smooth = ksmin_adaptive(x, smoothing_fraction=smoothing_fraction)
@test isapprox(x_max_smooth, -5.261253727654178e-6)

x = [0.165, 0.0]
smoothing_fraction = 0.2
x_max_smooth = ksmin_adaptive(x, smoothing_fraction=smoothing_fraction)
@test isapprox(x_max_smooth, -5.2856933329025475e-6)

# -------------------------

# ------ sigmoid ---------
x = -0.5
y = sigmoid(x)
@test isapprox(y, 0.37754066879814546)

x = 0.0
y = sigmoid(x)
@test isapprox(y, 0.5)

x = 0.5
y = sigmoid(x)
@test isapprox(y, 0.6224593312018546)

# overflow
x = 1e6
y = sigmoid(x)
@test isapprox(y, 1.0)

# underflow
x = -1e6
y = sigmoid(x)
@test isapprox(y, 0.0)

# -------------------------

# ------ sigmoid_blend ---------
x = -0.1
f1x = x
f2x = x^2
xt = 0.0
y = sigmoid_blend(f1x, f2x, x, xt)
@test isapprox(y, -0.09926378639832867)

x = 0.0
f1x = x
f2x = x^2
xt = 0.0
y = sigmoid_blend(f1x, f2x, x, xt)
@test isapprox(y, 0.0)

x = 0.1
f1x = x
f2x = x^2
xt = 0.0
y = sigmoid_blend(f1x, f2x, x, xt)
@test isapprox(y, 0.010602356583185632)

# overflow
x = 1e6
f1x = x
f2x = x^2
xt = 0.0
y = sigmoid_blend(f1x, f2x, x, xt)
@test isapprox(y, f2x)

# underflow
x = -1e6
f1x = x
f2x = x^2
xt = 0.0
y = sigmoid_blend(f1x, f2x, x, xt)
@test isapprox(y, f1x)

# hardness
x = -0.1
f1x = x
f2x = x^2
xt = 0.0
hardness = 100
y = sigmoid_blend(f1x, f2x, x, xt, hardness)
@test isapprox(y, -0.09999500623444274)

x = 0.0
f1x = x
f2x = x^2
xt = 0.0
hardness = 100
y = sigmoid_blend(f1x, f2x, x, xt, hardness)
@test isapprox(y, 0.0)

x = 0.1
f1x = x
f2x = x^2
xt = 0.0
hardness = 100
y = sigmoid_blend(f1x, f2x, x, xt, hardness)
@test isapprox(y, 0.010004085808183225)

# vectorized
x = -0.25:0.05:0.25
f1x = x
f2x = x.^2
xt = 0.0
hardness = 100
y = sigmoid_blend.(f1x, f2x, x, xt, hardness)
ytest = [-0.24999999999566003,
         -0.19999999950532316,
         -0.14999994723186585,
         -0.09999500623444274,
         -0.04964862532647505,
          0.0,
          0.0028179104189035298,
          0.010004085808183225,
          0.022500039002533945,
          0.040000000329784596,
          0.062500000002604]
@test isapprox(y, ytest)

# -------------------------

# ------ cubic_blend ---------
x = -3.0
f1x = x
f2x = x^2
xt = 0.0
delta_x = 0.1
y = cubic_blend(f1x, f2x, x, xt, delta_x)
@test y == -3.0

x = 3.0
f1x = x
f2x = x^2
xt = 0.0
delta_x = 0.1
y = cubic_blend(f1x, f2x, x, xt, delta_x)
@test y == 9.0

x = 0.0
f1x = x
f2x = x^2
xt = 0.0
delta_x = 0.1
y = cubic_blend(f1x, f2x, x, xt, delta_x)
@test y == 0.0

x = -0.1
f1x = x
f2x = x^2
xt = 0.0
delta_x = 0.1
y = cubic_blend(f1x, f2x, x, xt, delta_x)
@test y == f1x

x = 0.1
f1x = x
f2x = x^2
xt = 0.0
delta_x = 0.1
y = cubic_blend(f1x, f2x, x, xt, delta_x)
@test y == f2x

x = -0.05
f1x = x
f2x = x^2
xt = 0.0
delta_x = 0.1
y = cubic_blend(f1x, f2x, x, xt, delta_x)
@test isapprox(y, -0.041796875000000004)

x = 0.05
f1x = x
f2x = x^2
xt = 0.0
delta_x = 0.1
y = cubic_blend(f1x, f2x, x, xt, delta_x)
@test isapprox(y, 0.009921875000000004)

# vectorized
x = -0.25:0.05:0.25
f1x = x
f2x = x.^2
xt = 0.0
delta_x = 0.1
y = cubic_blend.(f1x, f2x, x, xt, delta_x)
ytest = [-0.25,
         -0.2,
         -0.15,
         -0.1,
         -0.041796875000000004,
          0.0,
          0.009921875000000004,
          0.010000000000000002,
          0.0225,
          0.04000000000000001,
          0.0625]
@test isapprox(y, ytest)

# -------------------------
# ------ quintic_blend ---------
x = -3.0
f1x = x
f2x = x^2
xt = 0.0
delta_x = 0.1
y = quintic_blend(f1x, f2x, x, xt, delta_x)
@test y == -3.0

x = 3.0
f1x = x
f2x = x^2
xt = 0.0
delta_x = 0.1
y = quintic_blend(f1x, f2x, x, xt, delta_x)
@test y == 9.0

x = 0.0
f1x = x
f2x = x^2
xt = 0.0
delta_x = 0.1
y = quintic_blend(f1x, f2x, x, xt, delta_x)
@test y == 0.0

x = -0.1
f1x = x
f2x = x^2
xt = 0.0
delta_x = 0.1
y = quintic_blend(f1x, f2x, x, xt, delta_x)
@test y == f1x

x = 0.1
f1x = x
f2x = x^2
xt = 0.0
delta_x = 0.1
y = quintic_blend(f1x, f2x, x, xt, delta_x)
@test y == f2x

x = -0.05
f1x = x
f2x = x^2
xt = 0.0
delta_x = 0.1
y = quintic_blend(f1x, f2x, x, xt, delta_x)
@test isapprox(y, -0.044565429687500005)

x = 0.05
f1x = x
f2x = x^2
xt = 0.0
delta_x = 0.1
y = quintic_blend(f1x, f2x, x, xt, delta_x)
@test isapprox(y, 0.007416992187499999)

# vectorized
x = -0.25:0.05:0.25
f1x = x
f2x = x.^2
xt = 0.0
delta_x = 0.1
y = quintic_blend.(f1x, f2x, x, xt, delta_x)
ytest = [ -0.25,
          -0.2,
          -0.15,
          -0.1,
          -0.044565429687500005,
           0.0,
           0.007416992187499999,
           0.010000000000000002,
           0.0225,
           0.04000000000000001,
           0.0625]
@test isapprox(y, ytest)
# -------------------------

# ---------- akima interpolate -------------

x = 0:pi/4:2*pi
y = sin.(x) 
xpt = 0:pi/16:2*pi

ypt = akima(x, y, xpt)

# FIXME: this is not an independent test.  right now it is just testing against refactoring.
ytest = [0.0, 0.21394356492851746, 0.40088834764831843, 0.5641656518405971, 0.7071067811865475, 0.8281808021479407, 0.920495128834866, 0.9793385864009951, 1.0, 0.9873810649285174, 0.9419417382415921, 0.8523082377305078, 0.7071067811865476, 0.5303300858899105, 0.35355339059327384, 0.17677669529663714, 1.2246467991473532e-16, -0.17677669529663698, -0.35355339059327373, -0.5303300858899104, -0.7071067811865475, -0.8523082377305078, -0.941941738241592, -0.9873810649285175, -1.0, -0.9793385864009951, -0.9204951288348657, -0.8281808021479408, -0.7071067811865477, -0.5641656518405971, -0.400888347648319, -0.21394356492851788, 0.0]

@test isapprox(ypt, ytest, atol=1e-8)

spl = Akima(x, y)

dydx = derivative(spl, pi/16)

@test isapprox(dydx, 1.0180260961104746, atol=1e-12)

dydx = gradient(spl, xpt)

dydx_test = [1.1640128599466306, 1.0180260961104746, 0.889005522603991, 0.7769511394271804, 0.6818629465800423, 0.5473879346340718, 0.3889191062220746, 0.20645646134405082, 1.8419755006770065e-16, -0.13818978335121798, -0.33430576382780836, -0.5883479414297708, -0.900316316157106, -0.9003163161571062, -0.9003163161571062, -0.9003163161571061, -0.9003163161571061, -0.9003163161571064, -0.9003163161571064, -0.9003163161571065, -0.9003163161571061, -0.5883479414297709, -0.33430576382780913, -0.13818978335121862, -1.8419755006770062e-16, 0.20645646134405057, 0.3889191062220747, 0.5473879346340714, 0.6818629465800421, 0.7769511394271802, 0.8890055226039906, 1.0180260961104746, 1.1640128599466313]

@test isapprox(dydx, dydx_test, atol=1e-12)

# differentiate with dual numbers

dydx = ForwardDiff.derivative(spl, pi/16)
@test isapprox(dydx, 1.0180260961104746, atol=1e-12)

J = ForwardDiff.jacobian(spl, xpt)
dydx = diag(J)
@test isapprox(dydx, dydx_test, atol=1e-12)

# differentiate coordinates of spline
# these don't test against numerical values (assumes forwarddiff works)
# it is just a test that dual numbers are properly supported and that this runs without error.

wrapper(y) = akima(x, y, xpt)
J = ForwardDiff.jacobian(wrapper, y)

wrapper2(x) = akima(x, y, xpt)
J = ForwardDiff.jacobian(wrapper2, x)

# ---------------------------

# ----- 1D linear interpolation ------
xvec = [1.0, 2.0, 4.0, 5.0]
yvec = [2.0, 3.0, 5.0, 8.0]

y = linear(xvec, yvec, 1.0)
@test y == 2.0
y = linear(xvec, yvec, 1.5)
@test y == 2.5
y = linear(xvec, yvec, 3.0)
@test y == 4.0
y = linear(xvec, yvec, 4.5)
@test y == 6.5
y = linear(xvec, yvec, 5.0)
@test y == 8.0

y = linear(xvec, yvec, [1.0, 1.5, 3.0, 4.5, 5.0])
@test y == [2.0, 2.5, 4.0, 6.5, 8.0]

dydx = derivative(xvec, yvec, 1.0)
@test dydx == 1.0
dydx = derivative(xvec, yvec, 1.5)
@test dydx == 1.0
dydx = derivative(xvec, yvec, 3.0)
@test dydx == 1.0
dydx = derivative(xvec, yvec, 4.5)
@test dydx == 3.0
dydx = derivative(xvec, yvec, 5.0)
@test dydx == 3.0

dydx = gradient(xvec, yvec, [1.0, 1.5, 3.0, 4.5, 5.0])
@test dydx == [1.0, 1.0, 1.0, 3.0, 3.0]

# ---------------------------

# ----- 2D linear interpolation -----
xdata = [1.0, 2.0, 3.0, 4.0]
ydata = [2.0, 4.0, 6.0, 8.0]
fdata = [1.0 2.0 3.0 4.0;
        5.0 6.0 7.0 8.0;
        9.0 10.0 11.0 12.0;
        13.0 14.0 15.0 16.0]
interp1d = linear

f = interp2d(interp1d, xdata, ydata, fdata, 1.0, 2.0)
@test f[1,1] == 1.0
f = interp2d(interp1d, xdata, ydata, fdata, 1.5, 2.0)
@test f[1,1] == 3.0
f = interp2d(interp1d, xdata, ydata, fdata, 3.5, 5.0)
@test f[1,1] == 12.5

# with ForwardDiff
wrapper(x) = interp2d(interp1d, xdata, ydata, fdata, x, 5.0)
J = ForwardDiff.derivative(wrapper, 2.5)
J = ForwardDiff.jacobian(wrapper, [2.5, 3.0, 4.0])

wrapper2(y) = interp2d(interp1d, xdata, ydata, fdata, 2.5, y)
J = ForwardDiff.derivative(wrapper2, 3.0)
J = ForwardDiff.jacobian(wrapper2, [3.0, 5.0, 7.0])

# ---------------------------

# ----- 3D linear interpolation -----
xdata = [1.0, 2.0, 3.0, 4.0]
ydata = [2.0, 4.0, 6.0, 8.0]
zdata = [3.0, 5.0, 7.0, 9.0]

fdata = zeros(4, 4, 4)
fdata[:,:,1] = [1.0 2.0 3.0 4.0;
                5.0 6.0 7.0 8.0;
                9.0 10.0 11.0 12.0;
                13.0 14.0 15.0 16.0]
fdata[:,:,2] = fdata[:,:,1] .+ 1
fdata[:,:,3] = fdata[:,:,1] .+ 2
fdata[:,:,4] = fdata[:,:,1] .+ 3
interp1d = linear

f = interp3d(interp1d, xdata, ydata, zdata, fdata, 1.0, 2.0, 3.0)
@test f[1,1,1] == 1.0
f = interp3d(interp1d, xdata, ydata, zdata, fdata, 1.5, 2.0, 3.0)
@test f[1,1,1] == 3.0
f = interp3d(interp1d, xdata, ydata, zdata, fdata, 1.0, 3.0, 3.0)
@test f[1,1,1] == 1.5
f = interp3d(interp1d, xdata, ydata, zdata, fdata, 1.0, 2.0, 6.0)
@test f[1,1,1] == 2.5
f = interp3d(interp1d, xdata, ydata, zdata, fdata, 3.5, 5.0, 4.0)
@test f[1,1,1] == 13.0

# with ForwardDiff
wrapper(x) = interp3d(interp1d, xdata, ydata, zdata, fdata, x, 2.0, 3.0)
J = ForwardDiff.derivative(wrapper, 1.5)
J = ForwardDiff.jacobian(wrapper, [1.5; 2.5; 3.5])

wrapper2(y) = interp3d(interp1d, xdata, ydata, zdata, fdata, 1.0, y, 3.0)
J = ForwardDiff.derivative(wrapper2, 5.0)
J = ForwardDiff.jacobian(wrapper2, [3.0; 5.0; 7.0])

wrapper3(z) = interp3d(interp1d, xdata, ydata, zdata, fdata, 1.0, 2.0, z)
J = ForwardDiff.derivative(wrapper3, 8.0)
J = ForwardDiff.jacobian(wrapper3, [4.0; 6.0; 8.0])
# ---------------------------

end
