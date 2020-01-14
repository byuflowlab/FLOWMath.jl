module FLOWMath

include("quadrature.jl")
export trapz

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
export akima_setup
export akima_interp
export akima
export interp2d
export interp3d
export interp4d

end # module
