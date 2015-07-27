typealias GenericSparseVector{Tv,Ti} Union(SparseVector{Tv,Ti}, SparseVectorView{Tv,Ti})

abs(x::GenericSparseVector) = SparseVector(x.n, copy(x.nzind), abs(x.nzval))
abs2(x::GenericSparseVector) = SparseVector(x.n, copy(x.nzind), abs2(x.nzval))

abstract _BinOp

type _AddOp <: _BinOp end

_eval(::_AddOp, x, y) = x + y
_eval1(::_AddOp, x) = x
_eval2(::_AddOp, y) = y

type _SubOp <: _BinOp end

_eval(::_SubOp, x, y) = x - y
_eval1(::_SubOp, x) = x
_eval2(::_SubOp, y) = -y


function _mapbinop{Tx,Ty}(op::_BinOp, x::GenericSparseVector{Tx}, y::GenericSparseVector{Ty})
    R = typeof(_eval(op, zero(Tx), zero(Ty)))
    n = length(x)
    length(y) == n || throw(DimensionMismatch())

    xnzind = x.nzind
    xnzval = x.nzval
    ynzind = y.nzind
    ynzval = y.nzval
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
            v = _eval(op, xnzval[ix], ynzval[iy])
            if v != zero(v)
                push!(rind, jx)
                push!(rval, v)
            end
            ix += 1
            iy += 1
        elseif jx < jy
            push!(rind, jx)
            push!(rval, _eval1(op, xnzval[ix]))
            ix += 1
        else
            push!(rind, jy)
            push!(rval, _eval2(op, ynzval[iy]))
            iy += 1
        end
    end

    @inbounds while ix <= mx
        push!(rind, xnzind[ix])
        push!(rval, _eval1(op, xnzval[ix]))
        ix += 1
    end

    @inbounds while iy <= my
        push!(rind, ynzind[iy])
        push!(rval, _eval2(op, ynzval[iy]))
        iy += 1
    end

    return SparseVector(n, rind, rval)
end

function _mapbinop{Tx,Ty}(op::_BinOp, x::StridedVector{Tx}, y::GenericSparseVector{Ty})
    R = typeof(_eval(op, zero(Tx), zero(Ty)))
    n = length(x)
    length(y) == n || throw(DimensionMismatch())

    ynzind = y.nzind
    ynzval = y.nzval
    m = length(ynzind)

    dst = Array(R, n)
    ii = 1
    @inbounds for i = 1:m
        j = ynzind[i]
        while ii < j
            dst[ii] = _eval1(op, x[ii])
            ii += 1
        end
        # at this point: ii == j
        dst[j] = _eval(op, x[j], ynzval[i])
        ii += 1
    end

    @inbounds while ii <= n
        dst[ii] = _eval1(op, x[ii])
        ii += 1
    end
    return dst
end

function _mapbinop{Tx,Ty}(op::_BinOp, x::GenericSparseVector{Tx}, y::StridedVector{Ty})
    R = typeof(_eval(op, zero(Tx), zero(Ty)))
    n = length(x)
    length(y) == n || throw(DimensionMismatch())

    xnzind = x.nzind
    xnzval = x.nzval
    m = length(xnzind)

    dst = Array(R, n)
    ii = 1
    @inbounds for i = 1:m
        j = xnzind[i]
        while ii < j
            dst[ii] = _eval2(op, y[ii])
            ii += 1
        end
        # at this point: ii == j
        dst[j] = _eval(op, xnzval[i], y[j])
        ii += 1
    end

    @inbounds while ii <= n
        dst[ii] = _eval2(op, y[ii])
        ii += 1
    end
    return dst
end


+(x::GenericSparseVector, y::GenericSparseVector) = _mapbinop(_AddOp(), x, y)
-(x::GenericSparseVector, y::GenericSparseVector) = _mapbinop(_SubOp(), x, y)
+(x::StridedVector, y::GenericSparseVector) = _mapbinop(_AddOp(), x, y)
-(x::StridedVector, y::GenericSparseVector) = _mapbinop(_SubOp(), x, y)
+(x::GenericSparseVector, y::StridedVector) = _mapbinop(_AddOp(), x, y)
-(x::GenericSparseVector, y::StridedVector) = _mapbinop(_SubOp(), x, y)

.+(x::GenericSparseVector, y::GenericSparseVector) = (x + y)
.-(x::GenericSparseVector, y::GenericSparseVector) = (x - y)

.+(x::StridedVector, y::GenericSparseVector) = (x + y)
.-(x::StridedVector, y::GenericSparseVector) = (x - y)

.+(x::GenericSparseVector, y::StridedVector) = (x + y)
.-(x::GenericSparseVector, y::StridedVector) = (x - y)
