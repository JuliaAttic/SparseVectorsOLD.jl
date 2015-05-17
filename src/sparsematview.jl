
### Views

jvec_rgn(x::Vector, first::Int, n::Int) = pointer_to_array(pointer(x, first), n, false)

as_jvec{T}(x::ContiguousView{T,1,Vector{T}}) = pointer_to_array(pointer(x), length(x), false)

function view(x::SparseMatrixCSC, ::Colon, j::Integer)
    1 <= j <= x.n || throw(BoundsError())
    r1 = convert(Int, x.colptr[j])
    r2 = convert(Int, x.colptr[j+1]) - 1
    rgn = r1:r2
    SparseVectorView(x.m, view(x.rowval, rgn), view(x.nzval, rgn))
end

function unsafe_colrange{Tv,Ti}(x::SparseMatrixCSC{Tv,Ti}, J::UnitRange)
    jfirst = first(J)
    jlast = last(J)
    (1 <= jfirst <= x.n && jlast <= x.n) || throw(BoundsError())
    r1 = x.colptr[jfirst]
    r2 = x.colptr[jlast+1] - one(r1)
    newcolptr = view(x.colptr, jfirst:jlast+1) - (r1 - one(r1))

    fi = convert(Int, r1)
    nc = convert(Int, r2 - r1) + 1
    SparseMatrixCSC{Tv, Ti}(x.m, length(J), newcolptr,
        jvec_rgn(x.rowval, fi, nc), jvec_rgn(x.nzval, fi, nc))
end
