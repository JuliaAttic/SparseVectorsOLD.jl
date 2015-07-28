using Base.Test
using SparseVectors

import SparseVectors: exact_equal

### BLAS Level-1

let x = sprand(16, 0.5), x2 = sprand(16, 0.4)
    xf = full(x)
    xf2 = full(x2)

    # axpy!
    for c in [1.0, -1.0, 2.0, -2.0]
        y = full(x)
        @test is(axpy!(c, x2, y), y)
        @test y == full(x2 * c + x)
    end

    # scale
    let sx = SparseVector(x.n, x.nzind, x.nzval * 2.5)
        @test exact_equal(scale(x, 2.5), sx)
        @test exact_equal(scale(2.5, x), sx)
        @test exact_equal(x * 2.5, sx)
        @test exact_equal(2.5 * x, sx)
        @test exact_equal(x .* 2.5, sx)
        @test exact_equal(2.5 .* x, sx)

        xc = copy(x)
        @test is(scale!(xc, 2.5), xc)
        @test exact_equal(xc, sx)
    end

    # dot
    let dv = dot(xf, xf2)
        @test_approx_eq dot(x, x) sumabs2(x)
        @test_approx_eq dot(x, x2) dv
        @test_approx_eq dot(x2, x) dv
        @test_approx_eq dot(full(x), x2) dv
        @test_approx_eq dot(x, full(x2)) dv
    end
end

### BLAS Level-2:

## dense A * sparse x -> dense y

let A = randn(9, 16), x = sprand(16, 0.7)
    xf = full(x)
    for α in [0.0, 1.0, 2.0], β in [0.0, 0.5, 1.0]
        y = rand(9)
        rr = α * A * xf + β * y
        @test is(A_mul_B!(α, A, x, β, y), y)
        @test_approx_eq y rr
    end
    y = A * x
    @test isa(y, Vector{Float64})
    @test_approx_eq A * x A * xf
end

let A = randn(16, 9), x = sprand(16, 0.7)
    xf = full(x)
    for α in [0.0, 1.0, 2.0], β in [0.0, 0.5, 1.0]
        y = rand(9)
        rr = α * A'xf + β * y
        @test is(At_mul_B!(α, A, x, β, y), y)
        @test_approx_eq y rr
    end
    y = At_mul_B(A, x)
    @test isa(y, Vector{Float64})
    @test_approx_eq y At_mul_B(A, xf)
end

## sparse A * sparse x -> dense y

let A = sprandn(9, 16, 0.5), x = sprand(16, 0.7)
    Af = full(A)
    xf = full(x)
    for α in [0.0, 1.0, 2.0], β in [0.0, 0.5, 1.0]
        y = rand(9)
        rr = α * Af * xf + β * y
        @test is(A_mul_B!(α, A, x, β, y), y)
        @test_approx_eq y rr
    end
    y = sparsemv_to_dense(A, x)
    @test isa(y, Vector{Float64})
    @test_approx_eq y Af * xf
end

let A = sprandn(16, 9, 0.5), x = sprand(16, 0.7)
    Af = full(A)
    xf = full(x)
    for α in [0.0, 1.0, 2.0], β in [0.0, 0.5, 1.0]
        y = rand(9)
        rr = α * Af'xf + β * y
        @test is(At_mul_B!(α, A, x, β, y), y)
        @test_approx_eq y rr
    end
    y = sparsemv_to_dense(A, x; trans=true)
    @test isa(y, Vector{Float64})
    @test_approx_eq y At_mul_B(Af, xf)
end

## sparse A * sparse x -> sparse y

let A = sprandn(9, 16, 0.5), x = sprand(16, 0.7)
    Af = full(A)
    xf = full(x)
    y = A * x
    @test isa(y, SparseVector{Float64,Int})
    @test_approx_eq full(y) Af * xf
end

let A = sprandn(16, 9, 0.5), x = sprand(16, 0.7)
    Af = full(A)
    xf = full(x)
    y = At_mul_B(A, x)
    @test isa(y, SparseVector{Float64,Int})
    @test all(nonzeros(y) .!= 0.0)
    @test_approx_eq full(y) At_mul_B(Af, xf)
end
