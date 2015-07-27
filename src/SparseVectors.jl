module SparseVectors

using Compat

import Base:
    +, .+, -, .-, *, .*,
    length, size, nnz, countnz, nonzeros,
    getindex, setindex!, convert, full, vec, copy,
    showarray, show, writemime,
    abs, abs2, sum, sumabs, sumabs2, vecnorm,
    sprand, sprandn, scale, scale!, dot,
    A_mul_B!, At_mul_B, At_mul_B!

import Base.LinAlg

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
include("common.jl")

include("sparsevec.jl")
include("sparsevecview.jl")
include("sparsematview.jl")

include("generics.jl")
include("arithmetic.jl")
include("linalg.jl")

end # module
