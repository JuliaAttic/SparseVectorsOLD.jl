
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

function _spdot(xj::Int, xj_last::Int, xnzind, xnzval,
                yj::Int, yj_last::Int, ynzind, ynzval)
    # dot product between ranges of non-zeros,
    s = zero(eltype(xnzval)) * zero(eltype(ynzval))
    @inbounds while xj <= xj_last && yj <= yj_last
        ix = xnzind[xj]
        iy = ynzind[yj]
        if ix == iy
            s += xnzval[xj] * ynzval[yj]
            xj += 1
            yj += 1
        elseif ix < iy
            xj += 1
        else
            yj += 1
        end
    end
    s
end

function dot{Tx<:Real,Ty<:Real}(x::AbstractSparseVector{Tx}, y::AbstractSparseVector{Ty})
    n = length(x)
    length(y) == n || throw(DimensionMismatch())

    xnzind = nonzeroinds(x)
    ynzind = nonzeroinds(y)
    xnzval = nonzeros(x)
    ynzval = nonzeros(y)

    _spdot(1, length(xnzind), xnzind, xnzval,
           1, length(ynzind), ynzind, ynzval)
end


### BLAS-2 / dense A * sparse x -> dense y

# A_mul_B

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
    if β != one(β)
        β == zero(β) ? fill!(y, zero(eltype(y))) : scale!(y, β)
    end
    α == zero(α) && return y

    xnzind = nonzeroinds(x)
    xnzval = nonzeros(x)
    @inbounds for i = 1:length(xnzind)
        v = xnzval[i]
        if v != zero(v)
            j = xnzind[i]
            αv = v * α
            for r = 1:m
                y[r] += A[r,j] * αv
            end
        end
    end
    return y
end

# At_mul_B

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
    if β != one(β)
        β == zero(β) ? fill!(y, zero(eltype(y))) : scale!(y, β)
    end
    α == zero(α) && return y

    xnzind = nonzeroinds(x)
    xnzval = nonzeros(x)
    _nnz = length(xnzind)
    _nnz == 0 && return y

    s0 = zero(eltype(A)) * zero(eltype(x))
    @inbounds for j = 1:n
        s = zero(s0)
        for i = 1:_nnz
            s += A[xnzind[i], j] * xnzval[i]
        end
        y[j] += s * α
    end
    return y
end


### BLAS-2 / sparse A * sparse x -> dense y

function sparsemv_to_dense(A::SparseMatrixCSC, x::AbstractSparseVector; trans::Bool=false)
    m, n = size(A)
    xlen = ifelse(trans, m, n)::Int
    xlen == length(x) || throw(DimensionMismatch())
    T = promote_type(eltype(A), eltype(x))
    y = Array(T, ifelse(trans, n, m)::Int)
    if trans
        At_mul_B!(y, A, x)
    else
        A_mul_B!(y, A, x)
    end
    y
end

# A_mul_B

A_mul_B!{Tx,Ty}(y::StridedVector{Ty}, A::SparseMatrixCSC, x::AbstractSparseVector{Tx}) =
    A_mul_B!(one(Tx), A, x, zero(Ty), y)

function A_mul_B!(α::Number, A::SparseMatrixCSC, x::AbstractSparseVector, β::Number, y::StridedVector)
    m, n = size(A)
    length(x) == n && length(y) == m || throw(DimensionMismatch())
    m == 0 && return y
    if β != one(β)
        β == zero(β) ? fill!(y, zero(eltype(y))) : scale!(y, β)
    end
    α == zero(α) && return y

    xnzind = nonzeroinds(x)
    xnzval = nonzeros(x)
    Acolptr = A.colptr
    Arowval = A.rowval
    Anzval = A.nzval

    @inbounds for i = 1:length(xnzind)
        v = xnzval[i]
        if v != zero(v)
            αv = v * α
            j = xnzind[i]
            for r = A.colptr[j]:(Acolptr[j+1]-1)
                y[Arowval[r]] += Anzval[r] * αv
            end
        end
    end
    return y
end

# At_mul_B

At_mul_B!{Tx,Ty}(y::StridedVector{Ty}, A::SparseMatrixCSC, x::AbstractSparseVector{Tx}) =
    At_mul_B!(one(Tx), A, x, zero(Ty), y)

function At_mul_B!{Tx,Ty}(α::Number, A::SparseMatrixCSC, x::AbstractSparseVector{Tx}, β::Number, y::StridedVector{Ty})
    m, n = size(A)
    length(x) == m && length(y) == n || throw(DimensionMismatch())
    n == 0 && return y
    if β != one(β)
        β == zero(β) ? fill!(y, zero(eltype(y))) : scale!(y, β)
    end
    α == zero(α) && return y

    xnzind = nonzeroinds(x)
    xnzval = nonzeros(x)
    Acolptr = A.colptr
    Arowval = A.rowval
    Anzval = A.nzval
    mx = length(xnzind)

    for j = 1:n
        # s <- dot(A[:,j], x)
        s = _spdot(Acolptr[j], Acolptr[j+1]-1, Arowval, Anzval,
                   1, mx, xnzind, xnzval)
        @inbounds y[j] += s * α
    end
    return y
end


### BLAS-2 / sparse A * sparse x -> dense y

function At_mul_B{TvA,TiA,TvX,TiX}(A::SparseMatrixCSC{TvA,TiA}, x::AbstractSparseVector{TvX,TiX})
    m, n = size(A)
    length(x) == m || throw(DimensionMismatch())
    Tv = promote_type(TvA, TvX)
    Ti = promote_type(TiA, TiX)

    xnzind = nonzeroinds(x)
    xnzval = nonzeros(x)
    Acolptr = A.colptr
    Arowval = A.rowval
    Anzval = A.nzval
    mx = length(xnzind)

    ynzind = Array(Ti, n)
    ynzval = Array(Tv, n)

    jr = 0
    for j = 1:n
        s = _spdot(Acolptr[j], Acolptr[j+1]-1, Arowval, Anzval,
                   1, mx, xnzind, xnzval)
        if s != zero(s)
            jr += 1
            ynzind[jr] = j
            ynzval[jr] = s
        end
    end
    if jr < n
        resize!(ynzind, jr)
        resize!(ynzval, jr)
    end
    SparseVector(n, ynzind, ynzval)
end
