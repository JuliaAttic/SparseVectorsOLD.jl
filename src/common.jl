# not exported, used mainly for testing

_copy_convert{T}(::Type{T}, x::Vector{T}) = copy(x)
_copy_convert{R,T}(::Type{R}, x::AbstractVector{T}) = convert(Vector{R}, x)

import Base: Func, AddFun, MulFun, MaxFun, MinFun

if VERSION < v"0.4-dev"
    call(f::Function, x) = f(x)
    call(f::Function, x, y) = f(x, y)
    call(f::Func{1}, x) = Base.evaluate(f, x)
    call(f::Func{2}, x, y) = Base.evaluate(f, x, y)
else
    import Base: call
end

immutable RealFun <: Func{1} end
call(::RealFun, x) = real(x)

immutable ImagFun <: Func{1} end
call(::ImagFun, x) = imag(x)

immutable SubFun <: Func{2} end
call(::SubFun, x, y) = x - y

immutable ComplexFun <: Func{2} end
call(::ComplexFun, x::Real, y::Real) = complex(x, y)

immutable DotFun <: Func{2} end
_dot(x::Number, y::Number) = conj(x) * y
_dot(x::Real, y::Real) = x * y
call(::DotFun, x::Number, y::Number) = _dot(x, y)

typealias UnaryOp Union(Function, Func{1})
typealias BinaryOp Union(Function, Func{2})
