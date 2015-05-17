
### Types

immutable SparseMatrixCSCView{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
    m::Int                  # the number of rows
    n::Int                  # the number of columns
    colptr::Vector{Ti}      # column i in colptr[i]:(colptr[i+1]-1)
    rowval::CVecView{Ti}    # row indices of nonzeros
    nzval::CVecView{Tv}     # nonzero values

    function SparseMatrixCSCView(m::Integer, n::Integer, colptr::Vector{Ti}, rowval::CVecView{Ti}, nzval::CVecView{Tv})
        m >= 0 || throw(ArgumentError("The number of rows must be non-negative."))
        n >= 0 || throw(ArgumentError("The number of columns must be non-negative."))
        length(colptr) == n+1 || throw(DimensionMismatch())
        length(rowval) == length(nzval) || throw(DimensionMismatch())
        new(convert(Int, m), convert(Int, n), colptr, rowval, nzval)
    end
end

SparseMatrixCSCView{Tv,Ti<:Integer}(m::Integer, n::Integer, colptr::Vector{Ti}, rowval::CVecView{Ti}, nzval::CVecView{Tv}) =
    SparseMatrixCSCView{Tv,Ti}(m, n, colptr, rowval, nzval)


typealias GenericSparseMatrixCSC{Tv,Ti} Union(SparseMatrixCSC{Tv,Ti},SparseMatrixCSCView{Tv,Ti})

### Basic properties

Base.length(x::SparseMatrixCSCView) = x.m * x.n
Base.size(x::SparseMatrixCSCView) = (x.m, x.n)

Base.nnz(x::SparseMatrixCSCView) = length(x.nzval)
Base.countnz(x::SparseMatrixCSCView) = countnz(x.nzval)
Base.nonzeros(x::SparseMatrixCSCView) = x.nzval


### Element access

function Base.getindex{Tv}(x::SparseMatrixCSCView{Tv}, i::Int, j::Int)
    (1 <= i <= x.m && 1 <= j <= x.n) || throw(BoundsError())
    r1 = convert(Int, x.colptr[j])
    r2 = convert(Int, x.colptr[j+1]) - 1
    (r1 > r2) && return zero(Tv)
    r1 = searchsortedfirst(x.rowval, i, r1, r2, Base.Forward)
    (r1 <= r2 && x.rowval[r1] == i) ? x.nzval[r1] : zero(Tv)
end

Base.getindex(x::SparseMatrixCSCView, i::Integer, j::Integer) = x[convert(Int, i), convert(Int, j)]


### Views

view(x::SparseMatrixCSC) = SparseMatrixCSCView(x.m, x.n, x.colptr, view(x.rowval), view(x.nzval))

function view(x::GenericSparseMatrixCSC, ::Colon, j::Integer)
    1 <= j <= x.n || throw(BoundsError())
    r1 = convert(Int, x.colptr[j])
    r2 = convert(Int, x.colptr[j+1]) - 1
    rgn = r1:r2
    SparseVectorView(x.m, view(x.rowval, rgn), view(x.nzval, rgn))
end

function view(x::GenericSparseMatrixCSC, ::Colon, J::UnitRange)
    jfirst = first(J)
    jlast = last(J)
    (1 <= jfirst <= x.n && jlast <= x.n) || throw(BoundsError())
    r1 = x.colptr[jfirst]
    r2 = x.colptr[jlast+1] - one(r1)
    newcolptr = view(x.colptr, jfirst:jlast+1) - (r1 - one(r1))
    rgn = r1:r2
    SparseMatrixCSCView(x.m, length(J), newcolptr,
        view(x.rowval, rgn), view(x.nzval, rgn))
end

# internal function, wrap SparseMatrixCSCView as SparseMatrixCSC (without copying)
# the results should only be used within a local scope!!
as_sparsemat{Tv,Ti<:Integer}(x::SparseMatrixCSCView{Tv,Ti}) =
    SparseMatrixCSC{Tv,Ti}(x.m, x.n, x.colptr, as_jvec(x.rowval), as_jvec(x.nzval))

as_jvec{T}(x::ContiguousView{T,1,Vector{T}}) = pointer_to_array(pointer(x), length(x), false)


### Array manipulation

function Base.full{Tv}(x::SparseMatrixCSCView{Tv})
    m, n = size(x)
    r = zeros(Tv, m, n)
    rowind = x.rowval
    nzval = x.nzval
    for j = 1:n
        r1 = convert(Int, x.colptr[j])
        r2 = convert(Int, x.colptr[j+1]) - 1
        for i = r1:r2
            r[rowind[i], j] = nzval[i]
        end
    end
    return r
end

Base.copy{Tv,Ti<:Integer}(x::SparseMatrixCSCView{Tv,Ti}) =
    SparseMatrixCSC{Tv,Ti}(x.m, x.n, x.colptr, copy(x.rowval), copy(x.nzval))


### sum

Base.sum(x::GenericSparseMatrixCSC) = sum(x.nzval)
Base.sumabs(x::GenericSparseMatrixCSC) = sumabs(x.nzval)
Base.sumabs2(x::GenericSparseMatrixCSC) = sumabs2(x.nzval)

Base.vecnorm(x::SparseMatrixCSCView, p::Real=2) = vecnorm(x.nzval, p)


### sum along dimensions

function _sum{Tv}(x::GenericSparseMatrixCSC{Tv}, dim::Integer)
    m, n = size(x)
    Td = typeof(zero(Tv) + zero(Tv))
    if 1 <= dim <= 2
        dsiz = dim == 1 ? (1, n) : (m, 1)
        dst = zeros(Td, dsiz)
        if dim == 1
            nzval = x.nzval
            for j = 1:n
                r1 = convert(Int, x.colptr[j])
                r2 = convert(Int, x.colptr[j+1]) - 1
                s = zero(Td)
                @inbounds for i = r1:r2
                    s += nzval[i]
                end
                dst[j] = s
            end
        else  # dim == 2
            rowind = x.rowval
            nzval = x.nzval
            for j = 1:n
                r1 = convert(Int, x.colptr[j])
                r2 = convert(Int, x.colptr[j+1]) - 1
                for i = r1:r2
                    @inbounds v = nzval[i]
                    @inbounds ii = rowind[i]
                    dst[ii] += v
                end
            end
        end
        dst
    else
        convert(Matrix{Td}, full(x))
    end::Matrix{Td}
end

Base.sum(x::SparseMatrixCSC, dim::Integer) = _sum(x, dim)
Base.sum(x::SparseMatrixCSCView, dim::Integer) = _sum(x, dim)
