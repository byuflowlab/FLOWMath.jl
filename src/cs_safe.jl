using LinearAlgebra: norm, dot


"""
    abs_cs_safe(x)

Calculate the absolute value of `x` in a manner compatible with the complex-step derivative approximation.

See also: [`abs`](@ref).
"""
abs_cs_safe

@inline function abs_cs_safe(x::T) where {T<:Complex}
    return x*sign(real(x))
end

@inline function abs_cs_safe(x)
    return abs(x)
end

"""
    abs2_cs_safe(x)

Calculate the squared absolute value of `x` in a manner compatible with the complex-step derivative approximation.

See also: [`abs2`](@ref).
"""
abs2_cs_safe

@inline function abs2_cs_safe(x::T) where {T<:Complex}
    return abs_cs_safe(x)^2
end

@inline function abs2_cs_safe(x)
    return abs2(x)
end

"""
    norm_cs_safe(x, p)

Calculate the `p`-norm value of iterable `x` in a manner compatible with the complex-step derivative approximation.

See also: [`norm`](@ref).
"""
norm_cs_safe

@inline function norm_cs_safe(x::AbstractArray{T}, p::Real=2) where {T<:Complex}
    return sum(x.^p)^(1/p)
end

@inline function norm_cs_safe(x, p::Real=2)
    return norm(x, p)
end

"""
    dot_cs_safe(a, b)

Calculate the dot product of vectors `a` and `b` in a manner compatible with the complex-step derivative approximation.

See also: [`norm`](@ref).
"""
dot_cs_safe

@inline function dot_cs_safe(a::AbstractVector{T}, b) where {T<:Complex}
    # `dot` conjugates its first argument, so we only need to worry about the case where the first argument is complex.
    # return sum(a.*b)
    return dot(conj.(a), b)
end

@inline function dot_cs_safe(a, b)
    return dot(a, b)
end
