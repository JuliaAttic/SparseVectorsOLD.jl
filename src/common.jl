# not exported, used mainly for testing

abstract UnaryOp

immutable RealOp <: UnaryOp end
_eval(::RealOp, x::Number) = real(x)

immutable ImagOp <: UnaryOp end
_eval(::ImagOp, x::Number) = imag(x)


abstract BinaryOp

immutable AddOp <: BinaryOp end
_eval(::AddOp, x::Number, y::Number) = x + y

immutable SubOp <: BinaryOp end
_eval(::SubOp, x::Number, y::Number) = x - y

immutable MulOp <: BinaryOp end
_eval(::MulOp, x::Number, y::Number) = x * y

immutable ComplexOp <: BinaryOp end
_eval(::ComplexOp, x::Real, y::Real) = complex(x, y)
