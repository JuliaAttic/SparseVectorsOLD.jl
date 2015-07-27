# not exported, used mainly for testing

abstract BinaryOp

immutable AddOp <: BinaryOp end
_eval(::AddOp, x::Number, y::Number) = x + y

immutable SubOp <: BinaryOp end
_eval(::SubOp, x::Number, y::Number) = x - y

immutable ComplexOp <: BinaryOp end
_eval(::ComplexOp, x::Real, y::Real) = complex(x, y)
