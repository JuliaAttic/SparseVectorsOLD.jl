
### Unary Map

# zero-preserving functions (z->z, nz->nz)
for op in [:-, :abs, :abs2, :conj]
    @eval begin
        $(op)(x::AbstractSparseVector) =
            SparseVector(length(x), copy(nonzeroinds(x)), $(op)(nonzeros(x)))
    end
end

# functions f, such that
#   f(x) can be zero or non-zero when x != 0
#   f(x) = 0 when x == 0
#
macro unarymap_nz2z_z2z(op, TF)
    esc(quote
        function $(op){Tv<:$(TF),Ti<:Integer}(x::AbstractSparseVector{Tv,Ti})
            R = typeof($(op)(zero(Tv)))
            xnzind = nonzeroinds(x)
            xnzval = nonzeros(x)
            m = length(xnzind)

            ynzind = Array(Ti, m)
            ynzval = Array(R, m)
            ir = 0
            @inbounds for j = 1:m
                i = xnzind[j]
                v = $(op)(xnzval[j])
                if v != zero(v)
                    ir += 1
                    ynzind[ir] = i
                    ynzval[ir] = v
                end
            end
            resize!(ynzind, ir)
            resize!(ynzval, ir)
            SparseVector(length(x), ynzind, ynzval)
        end
    end)
end

real{T<:Real}(x::AbstractSparseVector{T}) = x
@unarymap_nz2z_z2z real Complex

imag{Tv<:Real,Ti<:Integer}(x::AbstractSparseVector{Tv,Ti}) = SparseVector(length(x), Ti[], Tv[])
@unarymap_nz2z_z2z imag Complex

for op in [:floor, :ceil, :trunc, :round]
    @eval @unarymap_nz2z_z2z $(op) Real
end

for op in [:log1p, :expm1,
           :sin, :tan, :sinpi, :sind, :tand,
           :asin, :atan, :asind, :atand,
           :sinh, :tanh, :asinh, :atanh]
    @eval @unarymap_nz2z_z2z $(op) Number
end

# function that does not preserve zeros

macro unarymap_z2nz(op, TF)
    quote
        function $(op){Tv<:$(TF),Ti<:Integer}(x::AbstractSparseVector{Tv,Ti})
            v0 = $(op)(zero())
            R = typeof(v0)
            xnzind = nonzeroinds(x)
            xnzval = nonzeros(x)
            n = length(x)
            m = length(xnzind)
            y = fill(v0, n)
            @inbounds for j = 1:m
                y[xnzind[j]] = $(op)(xnzval[j])
            end
            y
        end
    end
end

for op in [:exp, :exp2, :exp10, :log, :log2, :log10,
           :cos, :csc, :cot, :sec, :cospi,
           :cosd, :cscd, :cotd, :secd,
           :acos, :acot, :acosd, :acotd,
           :cosh, :csch, :coth, :sech,
           :acsch, :asech]
    @eval @unarymap_z2nz $(op) Number
end


### Binary Map

# mode:
# 0: f(nz, nz) -> nz, f(z, nz) -> z, f(nz, z) ->  z
# 1: f(nz, nz) -> z/nz, f(z, nz) -> nz, f(nz, z) -> nz

function _binarymap{Tx,Ty}(f::BinaryOp,
                           x::AbstractSparseVector{Tx},
                           y::AbstractSparseVector{Ty},
                           mode::Int)

    0 <= mode <= 1 || throw(ArgumentError("Incorrect mode $mode."))
    R = typeof(call(f, zero(Tx), zero(Ty)))
    n = length(x)
    length(y) == n || throw(DimensionMismatch())

    xnzind = nonzeroinds(x)
    xnzval = nonzeros(x)
    ynzind = nonzeroinds(y)
    ynzval = nonzeros(y)
    mx = length(xnzind)
    my = length(ynzind)
    (mx == 0 || my == 0) && return SparseVector(R, 0)
    cap = (mode == 0 ? min(mx, my) : mx + my)::Int

    rind = Array(Int, cap)
    rval = Array(R, cap)
    ir = 0
    ix = 1
    iy = 1

    if mode == 0
        @inbounds while ix <= mx && iy <= my
            jx = xnzind[ix]
            jy = ynzind[iy]
            if jx == jy
                v = call(f, xnzval[ix], ynzval[iy])
                ir += 1
                rind[ir] = jx
                rval[ir] = v
                ix += 1
                iy += 1
            elseif jx < jy
                ix += 1
            else
                iy += 1
            end

        end

    else # mode == 1
        @inbounds while ix <= mx && iy <= my
            jx = xnzind[ix]
            jy = ynzind[iy]
            if jx == jy
                v = call(f, xnzval[ix], ynzval[iy])
                if v != zero(v)
                    ir += 1
                    rind[ir] = jx
                    rval[ir] = v
                end
                ix += 1
                iy += 1
            elseif jx < jy
                v = call(f, xnzval[ix], zero(Ty))
                ir += 1
                rind[ir] = jx
                rval[ir] = v
                ix += 1
            else
                v = call(f, zero(Tx), ynzval[iy])
                ir += 1
                rind[ir] = jy
                rval[ir] = v
                iy += 1
            end
        end
        @inbounds while ix <= mx
            v = call(f, xnzval[ix], zero(Ty))
            ir += 1
            rind[ir] = xnzind[ix]
            rval[ir] = v
            ix += 1
        end
        @inbounds while iy <= my
            v = call(f, zero(Tx), ynzval[iy])
            ir += 1
            rind[ir] = ynzind[iy]
            rval[ir] = v
            iy += 1
        end
    end

    resize!(rind, ir)
    resize!(rval, ir)
    return SparseVector(n, rind, rval)
end

function _binarymap{Tx,Ty}(f::BinaryOp,
                           x::AbstractVector{Tx},
                           y::AbstractSparseVector{Ty},
                           mode::Int)

    0 <= mode <= 1 || throw(ArgumentError("Incorrect mode $mode."))
    R = typeof(call(f, zero(Tx), zero(Ty)))
    n = length(x)
    length(y) == n || throw(DimensionMismatch())

    ynzind = nonzeroinds(y)
    ynzval = nonzeros(y)
    m = length(ynzind)

    dst = Array(R, n)
    if mode == 0
        ii = 1
        @inbounds for i = 1:m
            j = ynzind[i]
            while ii < j
                dst[ii] = zero(R)
                ii += 1
            end
            dst[j] = call(f, x[j], ynzval[i])
            ii += 1
        end
        @inbounds while ii <= n
            dst[ii] = zero(R)
            ii += 1
        end
    else # mode == 1
        ii = 1
        @inbounds for i = 1:m
            j = ynzind[i]
            while ii < j
                dst[ii] = call(f, x[ii], zero(Ty))
                ii += 1
            end
            dst[j] = call(f, x[j], ynzval[i])
            ii += 1
        end
        @inbounds while ii <= n
            dst[ii] = call(f, x[ii], zero(Ty))
            ii += 1
        end
    end
    return dst
end

function _binarymap{Tx,Ty}(f::BinaryOp,
                           x::AbstractSparseVector{Tx},
                           y::AbstractVector{Ty},
                           mode::Int)

    0 <= mode <= 1 || throw(ArgumentError("Incorrect mode $mode."))
    R = typeof(call(f, zero(Tx), zero(Ty)))
    n = length(x)
    length(y) == n || throw(DimensionMismatch())

    xnzind = nonzeroinds(x)
    xnzval = nonzeros(x)
    m = length(xnzind)

    dst = Array(R, n)
    if mode == 0
        ii = 1
        @inbounds for i = 1:m
            j = xnzind[i]
            while ii < j
                dst[ii] = zero(R)
                ii += 1
            end
            dst[j] = call(f, xnzval[i], y[j])
            ii += 1
        end
        @inbounds while ii <= n
            dst[ii] = zero(R)
            ii += 1
        end
    else # mode == 1
        ii = 1
        @inbounds for i = 1:m
            j = xnzind[i]
            while ii < j
                dst[ii] = call(f, zero(Tx), y[ii])
                ii += 1
            end
            dst[j] = call(f, xnzval[i], y[j])
            ii += 1
        end
        @inbounds while ii <= n
            dst[ii] = call(f, zero(Tx), y[ii])
            ii += 1
        end
    end
    return dst
end


### Binary arithmetics: +, -, *

_vadd(x::AbstractSparseVector, y::AbstractSparseVector) = _binarymap(AddFun(), x, y, 1)
_vsub(x::AbstractSparseVector, y::AbstractSparseVector) = _binarymap(SubFun(), x, y, 1)
_vmul(x::AbstractSparseVector, y::AbstractSparseVector) = _binarymap(MulFun(), x, y, 0)

_vadd(x::StridedVector, y::AbstractSparseVector) = _binarymap(AddFun(), x, y, 1)
_vsub(x::StridedVector, y::AbstractSparseVector) = _binarymap(SubFun(), x, y, 1)
_vmul(x::StridedVector, y::AbstractSparseVector) = _binarymap(MulFun(), x, y, 0)

_vadd(x::AbstractSparseVector, y::StridedVector) = _binarymap(AddFun(), x, y, 1)
_vsub(x::AbstractSparseVector, y::StridedVector) = _binarymap(SubFun(), x, y, 1)
_vmul(x::AbstractSparseVector, y::StridedVector) = _binarymap(MulFun(), x, y, 0)

# to workaround the ambiguities with vectorized dates/arithmetic.jl functions
if VERSION > v"0.4-dev"
    +{T<:Dates.TimeType,P<:Dates.GeneralPeriod}(x::StridedVector{P}, y::AbstractSparseVector{T}) = _vadd(x, y)
    -{T<:Dates.TimeType,P<:Dates.GeneralPeriod}(x::StridedVector{P}, y::AbstractSparseVector{T}) = _vsub(x, y)
    +{T<:Dates.TimeType,P<:Dates.GeneralPeriod}(x::AbstractSparseVector{T}, y::StridedVector{P}) = _vadd(x, y)
    -{T<:Dates.TimeType,P<:Dates.GeneralPeriod}(x::AbstractSparseVector{T}, y::StridedVector{P}) = _vsub(x, y)
end

# to workaround the ambiguities with BitVector
.*(x::BitVector, y::AbstractSparseVector{Bool}) = _vmul(x, y)
.*(x::AbstractSparseVector{Bool}, y::BitVector) = _vmul(x, y)

# definition of operators

+(x::AbstractSparseVector, y::AbstractSparseVector) = _vadd(x, y)
-(x::AbstractSparseVector, y::AbstractSparseVector) = _vsub(x, y)
+(x::StridedVector, y::AbstractSparseVector) = _vadd(x, y)
-(x::StridedVector, y::AbstractSparseVector) = _vsub(x, y)
+(x::AbstractSparseVector, y::StridedVector) = _vadd(x, y)
-(x::AbstractSparseVector, y::StridedVector) = _vsub(x, y)

.+(x::AbstractSparseVector, y::AbstractSparseVector) = _vadd(x, y)
.-(x::AbstractSparseVector, y::AbstractSparseVector) = _vsub(x, y)
.*(x::AbstractSparseVector, y::AbstractSparseVector) = _vmul(x, y)

.+(x::StridedVector, y::AbstractSparseVector) = _vadd(x, y)
.-(x::StridedVector, y::AbstractSparseVector) = _vsub(x, y)
.*(x::StridedVector, y::AbstractSparseVector) = _vmul(x, y)

.+(x::AbstractSparseVector, y::StridedVector) = _vadd(x, y)
.-(x::AbstractSparseVector, y::StridedVector) = _vsub(x, y)
.*(x::AbstractSparseVector, y::StridedVector) = _vmul(x, y)

# complex

complex{Tx<:Real,Ty<:Real}(x::AbstractSparseVector{Tx}, y::AbstractSparseVector{Ty}) =
    _binarymap(ComplexFun(), x, y, 1)
complex{Tx<:Real,Ty<:Real}(x::StridedVector{Tx}, y::AbstractSparseVector{Ty}) =
    _binarymap(ComplexFun(), x, y, 1)
complex{Tx<:Real,Ty<:Real}(x::AbstractSparseVector{Tx}, y::StridedVector{Ty}) =
    _binarymap(ComplexFun(), x, y, 1)


### Reduction

sum(x::AbstractSparseVector) = sum(nonzeros(x))
sumabs(x::AbstractSparseVector) = sumabs(nonzeros(x))
sumabs2(x::AbstractSparseVector) = sumabs2(nonzeros(x))

vecnorm(x::AbstractSparseVector, p::Real=2) = vecnorm(nonzeros(x), p)
