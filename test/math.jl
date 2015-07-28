using SparseVectors
using Base.Test


### Data

rnd_x0 = sprand(50, 0.6)
rnd_x0f = full(rnd_x0)

rnd_x1 = sprand(50, 0.7) * 4.0
rnd_x1f = full(rnd_x1)

spv_x1 = SparseVector(8, [2, 5, 6], [1.25, -0.75, 3.5])
_x2 = SparseVector(8, [1, 2, 6, 7], [3.25, 4.0, -5.5, -6.0])
spv_x2 = view(_x2)


### Arithmetic operations

let x = spv_x1, x2 = spv_x2
    # negate
    @test exact_equal(-x, SparseVector(8, [2, 5, 6], [-1.25, 0.75, -3.5]))

    # abs and abs2
    @test exact_equal(abs(x), SparseVector(8, [2, 5, 6], abs([1.25, -0.75, 3.5])))
    @test exact_equal(abs2(x), SparseVector(8, [2, 5, 6], abs2([1.25, -0.75, 3.5])))

    # plus and minus
    xa = SparseVector(8, [1,2,5,6,7], [3.25,5.25,-0.75,-2.0,-6.0])

    @test exact_equal(x + x, x * 2)
    @test exact_equal(x + x2, xa)
    @test exact_equal(x2 + x, xa)

    xb = SparseVector(8, [1,2,5,6,7], [-3.25,-2.75,-0.75,9.0,6.0])
    @test exact_equal(x - x, SparseVector(8, Int[], Float64[]))
    @test exact_equal(x - x2, xb)
    @test exact_equal(x2 - x, -xb)

    @test full(x) + x2 == full(xa)
    @test full(x) - x2 == full(xb)
    @test x + full(x2) == full(xa)
    @test x - full(x2) == full(xb)

    # multiplies
    xm = SparseVector(8, [2, 6], [5.0, -19.25])
    @test exact_equal(x .* x, abs2(x))
    @test exact_equal(x .* x2, xm)
    @test exact_equal(x2 .* x, xm)

    @test full(x) .* x2 == full(xm)
    @test x .* full(x2) == full(xm)
end

### Complex

let x = spv_x1, x2 = spv_x2
    # complex
    @test exact_equal(complex(x, x),
        SparseVector(8, [2,5,6], [1.25+1.25im, -0.75-0.75im, 3.5+3.5im]))
    @test exact_equal(complex(x, x2),
        SparseVector(8, [1,2,5,6,7], [3.25im, 1.25+4.0im, -0.75+0.im, 3.5-5.5im, -6.0im]))
    @test exact_equal(complex(x2, x),
        SparseVector(8, [1,2,5,6,7], [3.25+0.im, 4.0+1.25im, -0.75im, -5.5+3.5im, -6.0+0.im]))

    # real & imag

    @test is(real(x), x)
    @test exact_equal(imag(x), sparsevector(Float64, length(x)))

    xcp = complex(x, x2)
    @test exact_equal(real(xcp), x)
    @test exact_equal(imag(xcp), x2)
end

### Zero-preserving math functions

function check_sp2sp{T}(f::Function, x::SparseVector{T}, xf::Vector{T})
    R = typeof(f(zero(T)))
    r = f(x)
    isa(r, AbstractSparseVector) || error("$f(x) is not a sparse vector.")
    eltype(r) == R || error("$f(x) results in eltype = $(eltype(r)), expect $R")
    all(r.nzval .!= 0) || error("$f(x) contains zeros in nzval.")
    full(r) == f(xf) || error("Incorrect results found in $f(x).")
end

for f in [floor, ceil, trunc, round]
    check_sp2sp(f, rnd_x1, rnd_x1f)
end

for f in [log1p, expm1,
          sin, tan, sinpi, sind, tand,
          asin, atan, asind, atand,
          sinh, tanh, asinh, atanh]
    check_sp2sp(f, rnd_x0, rnd_x0f)
end
