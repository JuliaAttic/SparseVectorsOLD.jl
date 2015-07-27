
### Unary Map

abs(x::AbstractSparseVector) =
    SparseVector(length(x), copy(nonzeroinds(x)), abs(nonzeros(x)))
abs2(x::AbstractSparseVector) =
    SparseVector(length(x), copy(nonzeroinds(x)), abs2(nonzeros(x)))


### Binary Map

function _binarymap{Tx,Ty}(f, x::AbstractSparseVector{Tx}, y::AbstractSparseVector{Ty})
    R = typeof(_eval(f, zero(Tx), zero(Ty)))
    n = length(x)
    length(y) == n || throw(DimensionMismatch())

    xnzind = nonzeroinds(x)
    xnzval = nonzeros(x)
    ynzind = nonzeroinds(y)
    ynzval = nonzeros(y)
    mx = length(xnzind)
    my = length(ynzind)

    ix = 1
    iy = 1
    rind = Array(Int, 0)
    rval = Array(R, 0)
    sizehint!(rind, mx + my)
    sizehint!(rval, mx + my)

    @inbounds while ix <= mx && iy <= my
        jx = xnzind[ix]
        jy = ynzind[iy]

        if jx == jy
            v = _eval(f, xnzval[ix], ynzval[iy])
            if v != zero(v)
                push!(rind, jx)
                push!(rval, v)
            end
            ix += 1
            iy += 1
        elseif jx < jy
            push!(rind, jx)
            push!(rval, _eval(f, xnzval[ix], _zero))
            ix += 1
        else
            push!(rind, jy)
            push!(rval, _eval(f, _zero, ynzval[iy]))
            iy += 1
        end
    end

    @inbounds while ix <= mx
        push!(rind, xnzind[ix])
        push!(rval, _eval(f, xnzval[ix], _zero))
        ix += 1
    end

    @inbounds while iy <= my
        push!(rind, ynzind[iy])
        push!(rval, _eval(f, _zero, ynzval[iy]))
        iy += 1
    end

    return SparseVector(n, rind, rval)
end

function _binarymap{Tx,Ty}(f, x::AbstractVector{Tx}, y::AbstractSparseVector{Ty})
    R = typeof(_eval(f, zero(Tx), zero(Ty)))
    n = length(x)
    length(y) == n || throw(DimensionMismatch())

    ynzind = nonzeroinds(y)
    ynzval = nonzeros(y)
    m = length(ynzind)

    dst = Array(R, n)
    ii = 1
    @inbounds for i = 1:m
        j = ynzind[i]
        while ii < j
            dst[ii] = _eval(f, x[ii], _zero)
            ii += 1
        end
        # at this point: ii == j
        dst[j] = _eval(f, x[j], ynzval[i])
        ii += 1
    end

    @inbounds while ii <= n
        dst[ii] = _eval(f, x[ii], _zero)
        ii += 1
    end
    return dst
end

function _binarymap{Tx,Ty}(f, x::AbstractSparseVector{Tx}, y::AbstractVector{Ty})
    R = typeof(_eval(f, zero(Tx), zero(Ty)))
    n = length(x)
    length(y) == n || throw(DimensionMismatch())

    xnzind = nonzeroinds(x)
    xnzval = nonzeros(x)
    m = length(xnzind)

    dst = Array(R, n)
    ii = 1
    @inbounds for i = 1:m
        j = xnzind[i]
        while ii < j
            dst[ii] = _eval(f, _zero, y[ii])
            ii += 1
        end
        # at this point: ii == j
        dst[j] = _eval(f, xnzval[i], y[j])
        ii += 1
    end

    @inbounds while ii <= n
        dst[ii] = _eval(f, _zero, y[ii])
        ii += 1
    end
    return dst
end

_vadd(x::AbstractSparseVector, y::AbstractSparseVector) = _binarymap(ZeroAwareAdd(), x, y)
_vsub(x::AbstractSparseVector, y::AbstractSparseVector) = _binarymap(ZeroAwareSub(), x, y)
_vadd(x::StridedVector, y::AbstractSparseVector) = _binarymap(ZeroAwareAdd(), x, y)
_vsub(x::StridedVector, y::AbstractSparseVector) = _binarymap(ZeroAwareSub(), x, y)
_vadd(x::AbstractSparseVector, y::StridedVector) = _binarymap(ZeroAwareAdd(), x, y)
_vsub(x::AbstractSparseVector, y::StridedVector) = _binarymap(ZeroAwareSub(), x, y)

# to workaround the ambiguities with vectorized dates/arithmetic.jl functions
if VERSION > v"0.4-dev"
    +{T<:Dates.TimeType,P<:Dates.GeneralPeriod}(x::StridedVector{P}, y::AbstractSparseVector{T}) = _vadd(x, y)
    -{T<:Dates.TimeType,P<:Dates.GeneralPeriod}(x::StridedVector{P}, y::AbstractSparseVector{T}) = _vsub(x, y)
    +{T<:Dates.TimeType,P<:Dates.GeneralPeriod}(x::AbstractSparseVector{T}, y::StridedVector{P}) = _vadd(x, y)
    -{T<:Dates.TimeType,P<:Dates.GeneralPeriod}(x::AbstractSparseVector{T}, y::StridedVector{P}) = _vsub(x, y)
end

###

+(x::AbstractSparseVector, y::AbstractSparseVector) = _vadd(x, y)
-(x::AbstractSparseVector, y::AbstractSparseVector) = _vsub(x, y)
+(x::StridedVector, y::AbstractSparseVector) = _vadd(x, y)
-(x::StridedVector, y::AbstractSparseVector) = _vsub(x, y)
+(x::AbstractSparseVector, y::StridedVector) = _vadd(x, y)
-(x::AbstractSparseVector, y::StridedVector) = _vsub(x, y)

.+(x::AbstractSparseVector, y::AbstractSparseVector) = _vadd(x, y)
.-(x::AbstractSparseVector, y::AbstractSparseVector) = _vsub(x, y)
.+(x::StridedVector, y::AbstractSparseVector) = _vadd(x, y)
.-(x::StridedVector, y::AbstractSparseVector) = _vsub(x, y)
.+(x::AbstractSparseVector, y::StridedVector) = _vadd(x, y)
.-(x::AbstractSparseVector, y::StridedVector) = _vsub(x, y)
