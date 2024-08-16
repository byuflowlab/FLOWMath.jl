module FLOWMath

include("cs_safe.jl")
export abs_cs_safe, abs2_cs_safe
export norm_cs_safe
export dot_cs_safe
export atan_cs_safe

include("quadrature.jl")
export trapz, cumtrapz, cumtrapz!

include("roots.jl")
export brent

include("smooth.jl")
export abs_smooth
export ksmax, ksmin
export ksmax_adaptive, ksmin_adaptive
export sigmoid
export sigmoid_blend
export cubic_blend
export quintic_blend

include("interpolate.jl")
export Akima
export derivative
export gradient
export akima
export linear
export interp2d
export interp3d
export interp4d

end # module
