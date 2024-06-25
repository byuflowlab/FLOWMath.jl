using LinearAlgebra: norm, dot

"""
    abs_cs_safe(x)

Calculate the absolute value of `x` in a manner compatible with the complex-step derivative approximation.
"""
abs_cs_safe

@inline function abs_cs_safe(x::T) where {T<:Complex}
    return x * sign(real(x))
end

@inline function abs_cs_safe(x)
    return abs(x)
end

"""
    abs2_cs_safe(x)

Calculate the squared absolute value of `x` in a manner compatible with the complex-step derivative approximation.
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
"""
norm_cs_safe

@inline function norm_cs_safe(x::AbstractArray{T}, p::Real=2) where {T<:Complex}
    return sum(x .^ p)^(1 / p)
end

@inline function norm_cs_safe(x, p::Real=2)
    return norm(x, p)
end

"""
    dot_cs_safe(a, b)

Calculate the dot product of vectors `a` and `b` in a manner compatible with the complex-step derivative approximation.
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

"""
    atan_cs_safe(y, x)

Calculate the two-argument arctangent function in a manner compatible with the complex-step derivative approximation.
"""
atan_cs_safe

@inline function atan_cs_safe(y, x)
    return atan_cs_safe(promote(y, x)...)
end

@inline function atan_cs_safe(y::T, x::T) where {T<:Complex}
    # Stolen from openmdao/utils/cs_safe.py
    # a = np.real(y)
    # b = np.imag(y)
    # c = np.real(x)
    # d = np.imag(x)
    # return np.arctan2(a, c) + 1j * (c * b - a * d) / (a**2 + c**2)
    a = real(y)
    b = imag(y)
    c = real(x)
    d = imag(x)
    return complex(atan(a, c), (c * b - a * d) / (a^2 + c^2))
end

@inline function atan_cs_safe(y::T, x::T) where {T}
    return atan(y, x)
end

