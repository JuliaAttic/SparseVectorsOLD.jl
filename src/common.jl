# not exported, used mainly for testing

abstract ZeroAwareBinaryOp
immutable _Zero end
const _zero = _Zero()

immutable ZeroAwareAdd <: ZeroAwareBinaryOp end
_eval(::ZeroAwareAdd, x::Number, y::Number) = x + y
_eval(::ZeroAwareAdd, x::Number, ::_Zero) = x
_eval(::ZeroAwareAdd, ::_Zero, y) = y

immutable ZeroAwareSub <: ZeroAwareBinaryOp end
_eval(::ZeroAwareSub, x::Number, y::Number) = x - y
_eval(::ZeroAwareSub, x::Number, ::_Zero) = x
_eval(::ZeroAwareSub, ::_Zero, y::Number) = -y
