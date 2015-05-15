
### Types

typealias CVecView{T} ContiguousView{T,1,Vector{T}}

immutable SparseVector{Tv,Ti<:Integer} <: AbstractSparseVector{Tv,Ti}
    n::Int              # the number of elements
    nzind::Vector{Ti}   # the indices of nonzeros
    nzval::Vector{Tv}   # the values of nonzeros

    function SparseVector(n::Int, nzind::Vector{Ti}, nzval::Vector{Tv})
        n >= 0 || throw(ArgumentError("The number of elements must be non-negative."))
        length(nzind) == length(nzval) ||
            throw(DimensionMismatch("The lengths of nzind and nzval are inconsistent."))
        new(n, nzind, nzval)
    end
end

SparseVector{Tv,Ti}(n::Integer, nzind::Vector{Ti}, nzval::Vector{Tv}) =
    SparseVector{Tv,Ti}(convert(Int, n), nzind, nzval)


immutable SparseVectorView{Tv,Ti<:Integer} <: AbstractSparseVector{Tv,Ti}
    n::Int              # the number of elements
    nzind::Vector{Ti}   # the indices of nonzeros
    nzval::Vector{Tv}   # the values of nonzeros

    function SparseVectorView(n::Int, nzind::CVecView{Ti}, nzval::CVecView{Tv})
        n >= 0 || throw(ArgumentError("The number of elements must be non-negative."))
        length(nzind) == length(nzval) ||
            throw(DimensionMismatch("The lengths of nzind and nzval are inconsistent."))
        new(n, nzind, nzval)
    end
end

SparseVectorView{Tv,Ti}(n::Integer, nzind::CVecView{Ti}, nzval::CVecView{Tv}) =
    SparseVectorView{Tv,Ti}(convert(Int, n), nzind, nzval)


typealias GenericSparseVector{Tv,Ti} Union(SparseVector{Tv,Ti}, SparseVectorView{Tv,Ti})


### Basic properties

Base.length(x::GenericSparseVector) = x.n
Base.size(x::GenericSparseVector) = (x.n,)

Base.nnz(x::GenericSparseVector) = length(x.nzval)
Base.countnz(x::GenericSparseVector) = countnz(x.nzval)
Base.nonzeros(x::GenericSparseVector) = x.nzval


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

Base.sum(x::GenericSparseVector) = sum(x.nzval)
Base.sumabs(x::GenericSparseVector) = sumabs(x.nzval)
Base.sumabs2(x::GenericSparseVector) = sumabs2(x.nzval)

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
