using FLOWMath
using Test

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

xstar = brent(f, -2.0, 0)
@test xstar == -1.0
xstar = brent(f, 0.0, 2)
@test xstar == 1.0

f(x) = x^3 - 1
xstar = brent(f, 0, 3)
@test xstar == 1.0

f(x) = sin(x)
atol = 2e-12
xstar = brent(f, 1, 4, atol=atol)
@test isapprox(xstar, pi, atol=atol)

atol = 1e-15
xstar = brent(f, 1, 4, atol=atol)
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

# ------- forward/backwards/central/complex diff ----------

f(x) = @. exp(x) / sqrt(sin(x)^3 + cos(x)^3)

function dfdx(x)
    B = @. sqrt(sin(x)^3 + cos(x)^3)
    T = @. exp(x)
    dT = @. exp(x)
    dB = @. 0.5/sqrt(sin(x)^3 + cos(x)^3)*3*(sin(x)^2*cos(x) - cos(x)^2*sin(x))
    return @. (B*dT - T*dB)/B^2
end

x = [1.5, 0.5]
dff = forwarddiff(f, x)
dfb = backwarddiff(f, x)
dfc = centraldiff(f, x)
dfcplx = complexstep(f, x)
dfexact = dfdx(x)
@test isapprox(dff[1, 1], dfexact[1], atol=1e-4)
@test isapprox(dfb[1, 1], dfexact[1], atol=1e-4)
@test isapprox(dfc[1, 1], dfexact[1], atol=1e-6)
@test isapprox(dfcplx[1, 1], dfexact[1], atol=1e-15)
@test isapprox(dff[2, 2], dfexact[2], atol=1e-4)
@test isapprox(dfb[2, 2], dfexact[2], atol=1e-4)
@test isapprox(dfc[2, 2], dfexact[2], atol=1e-6)
@test isapprox(dfcplx[2, 2], dfexact[2], atol=1e-15)



# -------------------------------------------------------


# ---------- interpolate -------------

x = 0:pi/4:2*pi
y = sin.(x)
xpt = 0:pi/16:2*pi

ypt = akima(x, y, xpt)

# ---------------------------


end
