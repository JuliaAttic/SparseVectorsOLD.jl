module SparseVectors

using Compat

import Base:
    +, .+, -, .-, *, .*,
    length, size, nnz, countnz, nonzeros, getindex, setindex!,
    showarray, show, writemime,
    convert, full, vec, copy, reinterpret,
    float, complex, real, imag,
    abs, abs2, scale, scale!, dot,
    sum, sumabs, sumabs2, vecnorm,
    sparse, sprand, sprandn, 
    A_mul_B!, At_mul_B, At_mul_B!

import Base.LinAlg

import ArrayViews: view
import Base.LinAlg: axpy!

export
    # reexport
    view, axpy!,

    # types
    SparseVector,
    SparseVectorView,

    # functions
    nonzeroinds,
    sparsevector,
    unsafe_colrange,
    sparsemv_to_dense

# sources
include("common.jl")

include("sparsevec.jl")
include("sparsevecview.jl")
include("sparsematview.jl")

include("generics.jl")
include("arithmetic.jl")
include("linalg.jl")

end # module
