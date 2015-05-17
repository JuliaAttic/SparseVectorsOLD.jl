using Base.Test
using SparseExtensions

import SparseExtensions: GenericSparseVector
import Base.LinAlg: axpy!

# auxiliary tools

function exact_equal(x::GenericSparseVector, y::GenericSparseVector)
    x.n == y.n && x.nzind == y.nzind && x.nzval == y.nzval
end


# construction

x = SparseVector(8, [2, 5, 6], [1.25, -0.75, 3.5])

@test eltype(x) == Float64
@test ndims(x) == 1
@test length(x) == 8
@test size(x) == (8,)
@test size(x,1) == 8
@test size(x,2) == 1
@test !isempty(x)

@test countnz(x) == 3
@test nnz(x) == 3
@test nonzeros(x) == [1.25, -0.75, 3.5]


# view

_x2 = SparseVector(8, [1, 2, 6, 7], [3.25, 4.0, -5.5, -6.0])
x2 = view(_x2)

@test isa(x2, SparseVectorView{Float64,Int})

@test eltype(x2) == Float64
@test ndims(x2) == 1
@test length(x2) == 8
@test size(x2) == (8,)
@test size(x2,1) == 8
@test size(x2,2) == 1
@test !isempty(x2)

@test countnz(x2) == 4
@test nnz(x2) == 4
@test nonzeros(x2) == [3.25, 4.0, -5.5, -6.0]

# full

xf = zeros(8)
xf[2] = 1.25
xf[5] = -0.75
xf[6] = 3.5
@test isa(full(x), Vector{Float64})
@test full(x) == xf

xf2 = zeros(8)
xf2[1] = 3.25
xf2[2] = 4.0
xf2[6] = -5.5
xf2[7] = -6.0
@test isa(full(x2), Vector{Float64})
@test full(x2) == xf2


# conversion

s = sparse([2, 5, 6], [1, 1, 1], [1.25, -0.75, 3.5], 8, 1)
xs = convert(SparseVector, s)

@test isa(xs, SparseVector{Float64,Int})
@test xs == x

xc = convert(SparseVector, xf)
@test isa(xc, SparseVector{Float64,Int})
@test exact_equal(x, xc)

dct = Dict{Int,Float64}()
dct[2] = 1.25
dct[5] = -0.75
dct[6] = 3.5
xc = SparseVector(8, dct)
@test isa(xc, SparseVector{Float64,Int})
@test exact_equal(x, xc)

ps = [(5, -0.75), (2, 1.25), (6, 3.5)]
xc = SparseVector(8, ps)
@test isa(xc, SparseVector{Float64,Int})
@test exact_equal(x, xc)

# copy

xc = copy(x)
@test !is(x.nzind, xc.nzval)
@test !is(x.nzval, xc.nzval)
@test exact_equal(x, xc)

# getindex

for i = 1:length(x)
    @test x[i] == xf[i]
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

# add

xa = SparseVector(8, [1,2,5,6,7], [3.25,5.25,-0.75,-2.0,-6.0])

@test exact_equal(x + x, x * 2)
@test exact_equal(x + x2, xa)

# axpy

y = full(x)
@test is(axpy!(2.0, x2, y), y)
@test y == full(x2 * 2.0 + x)


# reduction

@test sum(x) == 4.0
@test sumabs(x) == 5.5
@test sumabs2(x) == 14.375

@test vecnorm(x) == sqrt(14.375)
@test vecnorm(x, 1) == 5.5
@test vecnorm(x, 2) == sqrt(14.375)
@test vecnorm(x, Inf) == 3.5

# dot

dv = dot(xf, xf2)

@test dot(x, x) == sumabs2(x)
@test dot(x, x2) == dv
@test dot(x2, x) == dv
@test dot(full(x), x2) == dv
@test dot(x, full(x2)) == dv
