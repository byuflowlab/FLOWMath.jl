module FLOWMath

include("quadrature.jl")
export trapz

include("roots.jl")
export brent

include("smooth.jl")
export abs_smooth
export ks

include("interpolate.jl")
export akima_setup
export akima_interp
export akima
export interp2d
export interp3d
export interp4d

include("derivatives.jl")
export forwarddiff
export backwarddiff
export centraldiff
export complexstep


end # module
