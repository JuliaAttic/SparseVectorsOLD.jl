
### Types

immutable SparseMatrixCSCView{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
    m::Int                  # the number of rows
    n::Int                  # the number of columns
    colptr::Vector{Ti}      # column i in colptr[i]:(colptr[i+1]-1)
    rowval::CVecView{Ti}    # row indices of nonzeros
    nzval::CVecView{Tv}     # nonzero values

    function SparseMatrixCSCView(m::Integer, n::Integer, colptr::Vector{Ti}, rowval::Vector{Ti}, nzval::Vector{Tv})
        m >= 0 || throw(ArgumentError("The number of rows must be non-negative."))
        n >= 0 || throw(ArgumentError("The number of columns must be non-negative."))
        length(colptr) == n+1 || throw(DimensionMismatch())
        length(rowval) == length(nzval) || throw(DimensionMismatch())
        new(convert(Int, m), convert(Int, n), colptr, rowval, nzval)
    end
end

SparseMatrixCSCView{Tv,Ti<:Integer}(m::Integer, n::Integer, colptr::Vector{Ti}, rowval::Vector{Ti}, nzval::Vector{Tv}) =
    SparseMatrixCSCView{Tv,Ti}(m, n, colptr, rowval, nzval)


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
    r1 = searchsorted(x.rowval, i, r1, r2, Forward)
    (r1 <= r2 && x.rowval[r1] == i) ? x.nzval[r1] : zero(Tv)
end

Base.getindex(x::SparseMatrixCSCView, i::Integer, j::Integer) = x[convert(Int, i), convert(Int, j)]


### Views

view(x::SparseMatrixCSC) = SparseMatrixCSCView(x.m, x.n, x.colptr, view(x.rowval), view(x.nzval))

function view(x::Union(SparseMatrixCSC,SparseMatrixCSCView), ::Colon, j::Integer)
    1 <= j <= x.n || throw(BoundsError())
    r1 = convert(Int, x.colptr[j])
    r2 = convert(Int, x.colptr[j+1]) - 1
    rgn = r1:r2
    SparseVectorView(x.m, view(x.rowval, rgn), view(x.nzval, rgn))
end

function view(x::Union(SparseMatrixCSC,SparseMatrixCSCView), ::Colon, J::UnitRange)
    jfirst = first(J)
    jlast = last(J)
    (1 <= jfirst <= x.n && jlast <= x.n) || throw(BoundsError())
    r1 = x.colptr[jfirst]
    r2 = x.colptr[jlast+1] - one(r1)
    newcolptr = view(x.colptr, J) - (r1 - one(r1))
    rgn = convert(Int, r1) : convert(Int, r2)
    SparseMatrixCSCView(x.m, length(J), newcolptr,
        view(x.rowval, rgn), view(x.nzval, rgn))
end
