module SparseVectors

using ArrayViews
using Compat

import Base:
    +, .+, -, .-, *, .*,
    length, size, nnz, countnz, nonzeros,
    getindex, setindex!, convert, full, vec, copy,
    showarray, show, writemime,
    abs, abs2, sum, sumabs, sumabs2, vecnorm,
    sprand, sprandn

import ArrayViews: view
import Base.LinAlg: axpy!

export

    # reexport
    view, axpy!,

    # sparsevec
    SparseVector, SparseVectorView,

    # sparsematview
    unsafe_colrange

# sources
include("sparsevec.jl")
include("sparsematview.jl")
include("linalg.jl")

end # module
