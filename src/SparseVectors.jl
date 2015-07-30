module SparseVectors

using Compat

import Base:
    length, size, nnz, countnz, nonzeros, getindex, setindex!,
    showarray, show, writemime,
    convert, full, vec, copy, reinterpret, float, complex,
    scale, scale!, dot, vecnorm, hcat, vcat,
    sum, sumabs, sumabs2, maximum, minimum, maxabs, minabs,
    sparse, sprand, sprandn,
    A_mul_B!, At_mul_B, At_mul_B!, Ac_mul_B, Ac_mul_B!

import Base:
    +, .+, -, .-, *, .*, max, min,
    abs, abs2, conj, real, imag,
    floor, ceil, trunc, round,
    exp, exp2, exp10, log, log2, log10,
    log1p, expm1,
    sin, cos, tan, csc, cot, sec, sinpi, cospi,
    sind, cosd, tand, cscd, cotd, secd,
    asin, acos, atan, acot,
    asind, acosd, atand, acotd,
    sinh, cosh, tanh, csch, coth, sech,
    asinh, atanh, acsch, asech

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
    getcol,
    unsafe_colrange,
    densemv

# sources
include("common.jl")

include("sparsevec.jl")
include("sparsevecview.jl")
include("sparsematview.jl")

include("generics.jl")
include("math.jl")
include("linalg.jl")

end # module
