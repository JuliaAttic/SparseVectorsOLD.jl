
### Types

typealias CVecView{T} ContiguousView{T,1,Vector{T}}

immutable SparseVector{Tv,Ti<:Integer} <: AbstractSparseVector{Tv,Ti}
    n::Int              # the number of elements
    nzind::Vector{Ti}   # the indices of nonzeros
    nzval::Vector{Tv}   # the values of nonzeros

    function SparseVector(n::Integer, nzind::Vector{Ti}, nzval::Vector{Tv})
        n >= 0 || throw(ArgumentError("The number of elements must be non-negative."))
        length(nzind) == length(nzval) ||
            throw(DimensionMismatch("The lengths of nzind and nzval are inconsistent."))
        new(convert(Int, n), nzind, nzval)
    end
end

SparseVector{Tv,Ti}(n::Integer, nzind::Vector{Ti}, nzval::Vector{Tv}) =
    SparseVector{Tv,Ti}(n, nzind, nzval)

immutable SparseVectorView{Tv,Ti<:Integer} <: AbstractSparseVector{Tv,Ti}
    n::Int                  # the number of elements
    nzind::CVecView{Ti}     # the indices of nonzeros
    nzval::CVecView{Tv}     # the values of nonzeros

    function SparseVectorView(n::Integer, nzind::CVecView{Ti}, nzval::CVecView{Tv})
        n >= 0 || throw(ArgumentError("The number of elements must be non-negative."))
        length(nzind) == length(nzval) ||
            throw(DimensionMismatch("The lengths of nzind and nzval are inconsistent."))
        new(convert(Int, n), nzind, nzval)
    end
end

SparseVectorView{Tv,Ti}(n::Integer, nzind::CVecView{Ti}, nzval::CVecView{Tv}) =
    SparseVectorView{Tv,Ti}(n, nzind, nzval)

typealias GenericSparseVector{Tv,Ti} Union(SparseVector{Tv,Ti}, SparseVectorView{Tv,Ti})


### Basic properties

Base.length(x::GenericSparseVector) = x.n
Base.size(x::GenericSparseVector) = (x.n,)

Base.nnz(x::GenericSparseVector) = length(x.nzval)
Base.countnz(x::GenericSparseVector) = countnz(x.nzval)
Base.nonzeros(x::GenericSparseVector) = x.nzval

### Element access

function Base.getindex{Tv}(x::GenericSparseVector{Tv}, i::Int)
    m = length(x.nzind)
    ii = searchsortedfirst(x.nzind, i)
    (ii <= m && x.nzind[ii] == i) ? x.nzval[ii] : zero(Tv)
end

Base.getindex(x::GenericSparseVector, i::Integer) = x[convert(Int, i)]


### Conversion

# convert SparseMatrixCSC to SparseVector
function Base.convert{Tv,Ti<:Integer}(::Type{SparseVector{Tv,Ti}}, s::SparseMatrixCSC{Tv,Ti})
    size(s, 2) == 1 || throw(ArgumentError("The input argument must have a single-column."))
    SparseVector(s.m, s.rowval, s.nzval)
end

Base.convert{Tv,Ti}(::Type{SparseVector{Tv}}, s::SparseMatrixCSC{Tv,Ti}) =
    convert(SparseVector{Tv,Ti}, s)

Base.convert{Tv,Ti}(::Type{SparseVector}, s::SparseMatrixCSC{Tv,Ti}) =
    convert(SparseVector{Tv,Ti}, s)


# convert Vector to SparseVector
function Base.convert{Tv}(::Type{SparseVector{Tv,Int}}, s::Vector{Tv})
    n = length(s)
    nzind = Array(Int, 0)
    nzval = Array(Tv, 0)
    for i = 1:n
        @inbounds v = s[i]
        if v != zero(v)
            push!(nzind, i)
            push!(nzval, v)
        end
    end
    return SparseVector(n, nzind, nzval)
end

Base.convert{Tv}(::Type{SparseVector{Tv}}, s::Vector{Tv}) =
    convert(SparseVector{Tv,Int}, s)

Base.convert{Tv}(::Type{SparseVector}, s::Vector{Tv}) =
    convert(SparseVector{Tv,Int}, s)


# make SparseVector from an iterable collection of (ind, value), e.g. dictionary
function _make_sparsevec{Tv,Ti<:Integer}(::Type{Tv}, ::Type{Ti}, n::Integer, iter, checkrep::Bool)
    m = length(iter)
    nzind = Array(Ti, 0)
    nzval = Array(Tv, 0)
    sizehint!(nzind, m)
    sizehint!(nzval, m)

    for (k, v) in iter
        if v != zero(v)
            push!(nzind, k)
            push!(nzval, v)
        end
    end

    p = sortperm(nzind)
    permute!(nzind, p)
    if checkrep
        for i = 1:length(nzind)-1
            nzind[i] != nzind[i+1] || error("Repeated indices.")
        end
    end
    permute!(nzval, p)

    return SparseVector{Tv,Ti}(convert(Int, n), nzind, nzval)
end

SparseVector{Tv,Ti<:Integer}(n::Integer, s::Associative{Ti,Tv}) =
    _make_sparsevec(Tv, Ti, n, s, false)


SparseVector{Tv,Ti}(n::Integer, s::AbstractVector{@compat(Tuple{Ti,Tv})}) =
    _make_sparsevec(Tv, Ti, n, s, true)


### View

view(x::SparseVector) = SparseVectorView(length(x), view(x.nzind), view(x.nzval))

### Array manipulation

function Base.full{Tv}(x::GenericSparseVector{Tv})
    n = x.n
    nzind = x.nzind
    nzval = x.nzval
    r = zeros(Tv, n)
    for i = 1:length(nzind)
        r[nzind[i]] = nzval[i]
    end
    return r
end

Base.vec(x::GenericSparseVector) = x

Base.copy(x::GenericSparseVector) = SparseVector(x.n, copy(x.nzind), copy(x.nzval))


### Computation

Base.scale!(x::GenericSparseVector, a::Number) = (scale!(x.nzval, a); x)

Base.scale!(a::Number, x::GenericSparseVector) = scale!(x, a)

Base.scale{T<:Number,S<:Number}(x::GenericSparseVector{T}, a::S) =
    SparseVector(x.n, copy(x.nzind), scale(x.nzval, a))

Base.scale(a::Number, x::GenericSparseVector) = scale(x, a)

* (x::GenericSparseVector, a::Number) = scale(x, a)
* (a::Number, x::GenericSparseVector) = scale(x, a)
.* (x::GenericSparseVector, a::Number) = scale(x, a)
.* (a::Number, x::GenericSparseVector) = scale(x, a)


function + {Tx,Ty}(x::GenericSparseVector{Tx}, y::GenericSparseVector{Ty})
    R = typeof(zero(Tx) + zero(Ty))
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

    while ix <= mx && iy <= my
        jx = xnzind[ix]
        jy = ynzind[iy]

        if jx == jy
            v = xnzval[ix] + ynzval[iy]
            if v != zero(v)
                push!(rind, jx)
                push!(rval, v)
            end
            ix += 1
            iy += 1
        elseif jx < jy
            push!(rind, jx)
            push!(rval, xnzval[ix])
            ix += 1
        else
            push!(rind, jy)
            push!(rval, ynzval[iy])
            iy += 1
        end
    end

    while ix <= mx
        push!(rind, xnzind[ix])
        push!(rval, xnzval[ix])
        ix += 1
    end

    while iy <= my
        push!(rind, ynzind[iy])
        push!(rval, ynzval[iy])
        iy += 1
    end

    return SparseVector(n, rind, rval)
end

.+ (x::GenericSparseVector, y::GenericSparseVector) = (x + y)


function Base.LinAlg.axpy!(a::Number, x::GenericSparseVector, y::StridedVector)
    length(x) == length(y) || throw(DimensionMismatch())

    nzind = x.nzind
    nzval = x.nzval
    m = length(nzind)

    if a == one(a)
        for i = 1:m
            y[nzind[i]] += nzval[i]
        end
    elseif a == -one(a)
        for i = 1:m
            y[nzind[i]] -= nzval[i]
        end
    else
        for i = 1:m
            y[nzind[i]] += a * nzval[i]
        end
    end
    return y
end


Base.sum(x::GenericSparseVector) = sum(x.nzval)
Base.sumabs(x::GenericSparseVector) = sumabs(x.nzval)
Base.sumabs2(x::GenericSparseVector) = sumabs2(x.nzval)

Base.vecnorm(x::GenericSparseVector) = vecnorm(x.nzval)
Base.vecnorm(x::GenericSparseVector, p::Real) = vecnorm(x.nzval, p)


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
