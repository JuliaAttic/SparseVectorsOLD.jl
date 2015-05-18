module SparseVectors

using ArrayViews
using Compat

import Base: +, .+, -, .-, *, .*
import ArrayViews: view

export

    # reexport view from ArrayViews
    view,

    # sparsevec
    SparseVector, SparseVectorView,

    # sparsematview
    unsafe_colrange


# sources
include("sparsevec.jl")
include("sparsematview.jl")
include("linalg.jl")

end # module
