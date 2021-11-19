function abs_cs_safe(x::T) where {T<:Complex}
    return x*sign(real(x))
end

function abs_cs_safe(x)
    return abs(x)
end
