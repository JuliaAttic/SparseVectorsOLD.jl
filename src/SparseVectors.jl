module SparseVectors

using ArrayViews
using Compat

import Base: +, .+, -, .-, *, .*
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
