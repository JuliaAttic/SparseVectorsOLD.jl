using Base.Test
using SparseVectors

import SparseVectors: exact_equal

### BLAS Level-1

x = sprand(16, 0.5)
x2 = sprand(16, 0.4)

xf = full(x)
xf2 = full(x2)

# axpy

for c in [1.0, -1.0, 2.0, -2.0]
    y = full(x)
    @test is(axpy!(c, x2, y), y)
    @test y == full(x2 * c + x)
end

# scale

sx = SparseVector(x.n, x.nzind, x.nzval * 2.5)

@test exact_equal(scale(x, 2.5), sx)
@test exact_equal(scale(2.5, x), sx)
@test exact_equal(x * 2.5, sx)
@test exact_equal(2.5 * x, sx)
@test exact_equal(x .* 2.5, sx)
@test exact_equal(2.5 .* x, sx)

xc = copy(x)
@test is(scale!(xc, 2.5), xc)
@test exact_equal(xc, sx)

# dot

dv = dot(xf, xf2)

@test_approx_eq dot(x, x) sumabs2(x)
@test_approx_eq dot(x, x2) dv
@test_approx_eq dot(x2, x) dv
@test_approx_eq dot(full(x), x2) dv
@test_approx_eq dot(x, full(x2)) dv


### BLAS Level-2

# A * x

A = randn(9, 16)

for α in [0.0, 1.0, 2.0], β in [0.0, 0.5, 1.0]
    y = rand(9)
    rr = α * A * xf + β * y
    @test is(A_mul_B!(α, A, x, β, y), y)
    @test_approx_eq y rr
end

@test_approx_eq A * x A * xf

# At_mul_B

A = randn(16, 9)

for α in [0.0, 1.0, 2.0], β in [0.0, 0.5, 1.0]
    y = rand(9)
    rr = α * A'xf + β * y
    @test is(At_mul_B!(α, A, x, β, y), y)
    @test_approx_eq y rr
end

@test_approx_eq At_mul_B(A, x) At_mul_B(A, xf)
