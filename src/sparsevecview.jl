using ArrayViews
import ArrayViews: view

typealias CVecView{T} ContiguousView{T,1,Vector{T}}

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

### Construction

SparseVectorView{Tv,Ti}(n::Integer, nzind::CVecView{Ti}, nzval::CVecView{Tv}) =
    SparseVectorView{Tv,Ti}(n, nzind, nzval)

view(x::AbstractSparseVector) =
    SparseVectorView(length(x), view(nonzeroinds(x)), view(nonzeros(x)))

### Basic properties

length(x::SparseVectorView) = x.n
size(x::SparseVectorView) = (x.n,)
nnz(x::SparseVectorView) = length(x.nzval)
countnz(x::SparseVectorView) = countnz(x.nzval)
nonzeros(x::SparseVectorView) = x.nzval
nonzeroinds(x::SparseVectorView) = x.nzind
