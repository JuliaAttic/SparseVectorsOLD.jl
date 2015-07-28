
### Unary Map

function _unarymap_selectnz{Tv,Ti<:Integer}(f::UnaryOp, x::AbstractSparseVector{Tv,Ti})
    R = typeof(_eval(f, zero(Tv)))
    xnzind = nonzeroinds(x)
    xnzval = nonzeros(x)
    m = length(xnzind)

    ynzind = Array(Ti, m)
    ynzval = Array(R, m)
    ir = 0
    @inbounds for j = 1:m
        i = xnzind[j]
        v = _eval(f, xnzval[j])
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

- (x::AbstractSparseVector) =
    SparseVector(length(x), copy(nonzeroinds(x)), -nonzeros(x))

abs(x::AbstractSparseVector) =
    SparseVector(length(x), copy(nonzeroinds(x)), abs(nonzeros(x)))
abs2(x::AbstractSparseVector) =
    SparseVector(length(x), copy(nonzeroinds(x)), abs2(nonzeros(x)))

real{T<:Real}(x::AbstractSparseVector{T}) = x
real{T<:Complex}(x::AbstractSparseVector{T}) = _unarymap_selectnz(RealOp(), x)

imag{Tv<:Real,Ti<:Integer}(x::AbstractSparseVector{Tv,Ti}) = SparseVector(Tv, Ti, length(x))
imag{T<:Complex}(x::AbstractSparseVector{T}) = _unarymap_selectnz(ImagOp(), x)


### Binary Map

# mode:
# 0: f(nz, nz) -> nz, f(z, nz) -> z, f(nz, z) ->  z
# 1: f(nz, nz) -> z/nz, f(z, nz) -> nz, f(nz, z) -> nz


function _binarymap{Tx,Ty}(f::BinaryOp,
                           x::AbstractSparseVector{Tx},
                           y::AbstractSparseVector{Ty},
                           mode::Int)

    0 <= mode <= 1 || throw(ArgumentError("Incorrect mode $mode."))
    R = typeof(_eval(f, zero(Tx), zero(Ty)))
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
                v = _eval(f, xnzval[ix], ynzval[iy])
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
                v = _eval(f, xnzval[ix], ynzval[iy])
                if v != zero(v)
                    ir += 1
                    rind[ir] = jx
                    rval[ir] = v
                end
                ix += 1
                iy += 1
            elseif jx < jy
                v = _eval(f, xnzval[ix], zero(Ty))
                ir += 1
                rind[ir] = jx
                rval[ir] = v
                ix += 1
            else
                v = _eval(f, zero(Tx), ynzval[iy])
                ir += 1
                rind[ir] = jy
                rval[ir] = v
                iy += 1
            end
        end
        @inbounds while ix <= mx
            v = _eval(f, xnzval[ix], zero(Ty))
            ir += 1
            rind[ir] = xnzind[ix]
            rval[ir] = v
            ix += 1
        end
        @inbounds while iy <= my
            v = _eval(f, zero(Tx), ynzval[iy])
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
    R = typeof(_eval(f, zero(Tx), zero(Ty)))
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
            dst[j] = _eval(f, x[j], ynzval[i])
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
                dst[ii] = _eval(f, x[ii], zero(Ty))
                ii += 1
            end
            dst[j] = _eval(f, x[j], ynzval[i])
            ii += 1
        end
        @inbounds while ii <= n
            dst[ii] = _eval(f, x[ii], zero(Ty))
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
    R = typeof(_eval(f, zero(Tx), zero(Ty)))
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
            dst[j] = _eval(f, xnzval[i], y[j])
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
                dst[ii] = _eval(f, zero(Tx), y[ii])
                ii += 1
            end
            dst[j] = _eval(f, xnzval[i], y[j])
            ii += 1
        end
        @inbounds while ii <= n
            dst[ii] = _eval(f, zero(Tx), y[ii])
            ii += 1
        end
    end
    return dst
end


### Arithmetics: +, -, *

_vadd(x::AbstractSparseVector, y::AbstractSparseVector) = _binarymap(AddOp(), x, y, 1)
_vsub(x::AbstractSparseVector, y::AbstractSparseVector) = _binarymap(SubOp(), x, y, 1)
_vmul(x::AbstractSparseVector, y::AbstractSparseVector) = _binarymap(MulOp(), x, y, 0)

_vadd(x::StridedVector, y::AbstractSparseVector) = _binarymap(AddOp(), x, y, 1)
_vsub(x::StridedVector, y::AbstractSparseVector) = _binarymap(SubOp(), x, y, 1)
_vmul(x::StridedVector, y::AbstractSparseVector) = _binarymap(MulOp(), x, y, 0)

_vadd(x::AbstractSparseVector, y::StridedVector) = _binarymap(AddOp(), x, y, 1)
_vsub(x::AbstractSparseVector, y::StridedVector) = _binarymap(SubOp(), x, y, 1)
_vmul(x::AbstractSparseVector, y::StridedVector) = _binarymap(MulOp(), x, y, 0)

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
    _binarymap(ComplexOp(), x, y, 1)
complex{Tx<:Real,Ty<:Real}(x::StridedVector{Tx}, y::AbstractSparseVector{Ty}) =
    _binarymap(ComplexOp(), x, y, 1)
complex{Tx<:Real,Ty<:Real}(x::AbstractSparseVector{Tx}, y::StridedVector{Ty}) =
    _binarymap(ComplexOp(), x, y, 1)

###
