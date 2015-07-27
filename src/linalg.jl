
### BLAS Level-1

# axpy

function LinAlg.axpy!(a::Number, x::GenericSparseVector, y::StridedVector)
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

scale!(x::GenericSparseVector, a::Number) = (scale!(x.nzval, a); x)

scale!(a::Number, x::GenericSparseVector) = scale!(x, a)

scale{T<:Number,S<:Number}(x::GenericSparseVector{T}, a::S) =
    SparseVector(x.n, copy(x.nzind), scale(x.nzval, a))

scale(a::Number, x::GenericSparseVector) = scale(x, a)

*(x::GenericSparseVector, a::Number) = scale(x, a)
*(a::Number, x::GenericSparseVector) = scale(x, a)
.*(x::GenericSparseVector, a::Number) = scale(x, a)
.*(a::Number, x::GenericSparseVector) = scale(x, a)


# dot

function dot{Tx<:Real,Ty<:Real}(x::StridedVector{Tx}, y::GenericSparseVector{Ty})
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

dot{Tx<:Real,Ty<:Real}(x::GenericSparseVector{Tx}, y::StridedVector{Ty}) = dot(y, x)

function dot{Tx<:Real,Ty<:Real}(x::GenericSparseVector{Tx}, y::GenericSparseVector{Ty})
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


### BLAS Level-2

function *{Ta,Tx}(A::StridedMatrix{Ta}, x::GenericSparseVector{Tx})
    m, n = size(A)
    length(x) == n || throw(DimensionMismatch())
    Ty = promote_type(Ta, Tx)
    y = Array(Ty, m)
    A_mul_B!(y, A, x)
end

A_mul_B!{Tx,Ty}(y::StridedVector{Ty}, A::StridedMatrix, x::GenericSparseVector{Tx}) =
    A_mul_B!(one(Tx), A, x, zero(Ty), y)

function A_mul_B!(α::Number, A::StridedMatrix, x::GenericSparseVector, β::Number, y::StridedVector)
    m, n = size(A)
    length(x) == n && length(y) == m || throw(DimensionMismatch())
    m == 0 && return y

    nzind = x.nzind
    nzval = x.nzval

    if β != one(β)
        β == zero(β) ? fill!(y, zero(eltype(y))) : scale!(y, β)
    end

    for i = 1:length(nzind)
        j = nzind[i]
        v = nzval[i] * α
        for r = 1:m
            y[r] += A[r,j] * v
        end
    end
    return y
end

function At_mul_B{Ta,Tx}(A::StridedMatrix{Ta}, x::GenericSparseVector{Tx})
    m, n = size(A)
    length(x) == m || throw(DimensionMismatch())
    Ty = promote_type(Ta, Tx)
    y = Array(Ty, n)
    At_mul_B!(y, A, x)
end

At_mul_B!{Tx,Ty}(y::StridedVector{Ty}, A::StridedMatrix, x::GenericSparseVector{Tx}) =
    At_mul_B!(one(Tx), A, x, zero(Ty), y)

function At_mul_B!(α::Number, A::StridedMatrix, x::GenericSparseVector, β::Number, y::StridedVector)
    m, n = size(A)
    length(x) == m && length(y) == n || throw(DimensionMismatch())
    n == 0 && return y

    nzind = x.nzind
    nzval = x.nzval

    if β != one(β)
        β == zero(β) ? fill!(y, zero(eltype(y))) : scale!(y, β)
    end
    _nnz = length(nzind)
    _nnz == 0 && return y

    s0 = zero(eltype(A)) * zero(eltype(x))
    for j = 1:n
        s = zero(s0)
        for i = 1:_nnz
            s += A[nzind[i], j] * nzval[i]
        end
        y[j] += s * α
    end
    return y
end
