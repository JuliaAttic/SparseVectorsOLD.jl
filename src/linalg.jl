
### BLAS Level-1

# axpy

function LinAlg.axpy!(a::Number, x::AbstractSparseVector, y::StridedVector)
    length(x) == length(y) || throw(DimensionMismatch())
    nzind = nonzeroinds(x)
    nzval = nonzeros(x)
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

scale!(x::AbstractSparseVector, a::Real) = (scale!(nonzeros(x), a); x)
scale!(x::AbstractSparseVector, a::Complex) = (scale!(nonzeros(x), a); x)
scale!(a::Real, x::AbstractSparseVector) = scale!(nonzeros(x), a)
scale!(a::Complex, x::AbstractSparseVector) = scale!(nonzeros(x), a)

scale(x::AbstractSparseVector, a::Real) =
    SparseVector(length(x), copy(nonzeroinds(x)), scale(nonzeros(x), a))
scale(x::AbstractSparseVector, a::Complex) =
    SparseVector(length(x), copy(nonzeroinds(x)), scale(nonzeros(x), a))

scale(a::Real, x::AbstractSparseVector) = scale(x, a)
scale(a::Complex, x::AbstractSparseVector) = scale(x, a)

*(x::AbstractSparseVector, a::Number) = scale(x, a)
*(a::Number, x::AbstractSparseVector) = scale(x, a)
.*(x::AbstractSparseVector, a::Number) = scale(x, a)
.*(a::Number, x::AbstractSparseVector) = scale(x, a)


# dot

function dot{Tx<:Real,Ty<:Real}(x::StridedVector{Tx}, y::AbstractSparseVector{Ty})
    n = length(x)
    length(y) == n || throw(DimensionMismatch())
    nzind = nonzeroinds(y)
    nzval = nonzeros(y)
    s = zero(Tx) * zero(Ty)
    for i = 1:length(nzind)
        s += x[nzind[i]] * nzval[i]
    end
    return s
end

dot{Tx<:Real,Ty<:Real}(x::AbstractSparseVector{Tx}, y::AbstractVector{Ty}) = dot(y, x)

function dot{Tx<:Real,Ty<:Real}(x::AbstractSparseVector{Tx}, y::AbstractSparseVector{Ty})
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

function *{Ta,Tx}(A::StridedMatrix{Ta}, x::AbstractSparseVector{Tx})
    m, n = size(A)
    length(x) == n || throw(DimensionMismatch())
    Ty = promote_type(Ta, Tx)
    y = Array(Ty, m)
    A_mul_B!(y, A, x)
end

A_mul_B!{Tx,Ty}(y::StridedVector{Ty}, A::StridedMatrix, x::AbstractSparseVector{Tx}) =
    A_mul_B!(one(Tx), A, x, zero(Ty), y)

function A_mul_B!(α::Number, A::StridedMatrix, x::AbstractSparseVector, β::Number, y::StridedVector)
    m, n = size(A)
    length(x) == n && length(y) == m || throw(DimensionMismatch())
    m == 0 && return y

    nzind = nonzeroinds(x)
    nzval = nonzeros(x)

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

function At_mul_B{Ta,Tx}(A::StridedMatrix{Ta}, x::AbstractSparseVector{Tx})
    m, n = size(A)
    length(x) == m || throw(DimensionMismatch())
    Ty = promote_type(Ta, Tx)
    y = Array(Ty, n)
    At_mul_B!(y, A, x)
end

At_mul_B!{Tx,Ty}(y::StridedVector{Ty}, A::StridedMatrix, x::AbstractSparseVector{Tx}) =
    At_mul_B!(one(Tx), A, x, zero(Ty), y)

function At_mul_B!(α::Number, A::StridedMatrix, x::AbstractSparseVector, β::Number, y::StridedVector)
    m, n = size(A)
    length(x) == m && length(y) == n || throw(DimensionMismatch())
    n == 0 && return y

    nzind = nonzeroinds(x)
    nzval = nonzeros(x)

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
