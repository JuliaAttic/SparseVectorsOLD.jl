
### BLAS Level-1

# axpy

function Base.LinAlg.axpy!(a::Number, x::GenericSparseVector, y::StridedVector)
    length(x) == length(y) || throw(DimensionMismatch())

    nzind = x.nzind
    nzval = x.nzval
    m = length(nzind)

    if a == one(a)
        for i = 1:m
            @inbounds ii = nzind[i]
            @inbounds v = nzval[i]
            y[ii] += v
        end
    elseif a == -one(a)
        for i = 1:m
            @inbounds ii = nzind[i]
            @inbounds v = nzval[i]
            y[ii] -= v
        end
    else
        for i = 1:m
            @inbounds ii = nzind[i]
            @inbounds v = nzval[i]
            y[ii] += a * v
        end
    end
    return y
end


# scale

Base.scale!(x::GenericSparseVector, a::Number) = (scale!(x.nzval, a); x)

Base.scale!(a::Number, x::GenericSparseVector) = scale!(x, a)

Base.scale{T<:Number,S<:Number}(x::GenericSparseVector{T}, a::S) =
    SparseVector(x.n, copy(x.nzind), scale(x.nzval, a))

Base.scale(a::Number, x::GenericSparseVector) = scale(x, a)

* (x::GenericSparseVector, a::Number) = scale(x, a)
* (a::Number, x::GenericSparseVector) = scale(x, a)
.* (x::GenericSparseVector, a::Number) = scale(x, a)
.* (a::Number, x::GenericSparseVector) = scale(x, a)


# dot

function Base.dot{Tx<:Real,Ty<:Real}(x::StridedVector{Tx}, y::GenericSparseVector{Ty})
    n = length(x)
    length(y) == n || throw(DimensionMismatch())
    nzind = y.nzind
    nzval = y.nzval
    s = zero(Tx) * zero(Ty)
    for i = 1:length(nzind)
        s += x[nzind[i]] * nzval[i]
    end
    return s
end

Base.dot{Tx<:Real,Ty<:Real}(x::GenericSparseVector{Tx}, y::StridedVector{Ty}) = dot(y, x)

function Base.dot{Tx<:Real,Ty<:Real}(x::GenericSparseVector{Tx}, y::GenericSparseVector{Ty})
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
    s = zero(Tx) * zero(Ty)
    while ix <= mx && iy <= my
        jx = xnzind[ix]
        jy = ynzind[iy]
        if jx == jy
            s += xnzval[ix] * ynzval[iy]
            ix += 1
            iy += 1
        elseif jx < jy
            ix += 1
        else
            iy += 1
        end
    end
    return s
end
