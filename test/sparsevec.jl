using Base.Test
using Compat
using SparseVectors

import SparseVectors: exact_equal

### Data

x = SparseVector(8, [2, 5, 6], [1.25, -0.75, 3.5])
_x2 = SparseVector(8, [1, 2, 6, 7], [3.25, 4.0, -5.5, -6.0])
x2 = view(_x2)

@test isa(x, SparseVector{Float64,Int})
@test isa(x2, SparseVectorView{Float64,Int})


### Basic Properties

@test eltype(x) == Float64
@test ndims(x) == 1
@test length(x) == 8
@test size(x) == (8,)
@test size(x,1) == 8
@test size(x,2) == 1
@test !isempty(x)

@test countnz(x) == 3
@test nnz(x) == 3
@test nonzeroinds(x) == [2, 5, 6]
@test nonzeros(x) == [1.25, -0.75, 3.5]

@test eltype(x2) == Float64
@test ndims(x2) == 1
@test length(x2) == 8
@test size(x2) == (8,)
@test size(x2,1) == 8
@test size(x2,2) == 1
@test !isempty(x2)

@test countnz(x2) == 4
@test nnz(x2) == 4
@test nonzeroinds(x2) == [1, 2, 6, 7]
@test nonzeros(x2) == [3.25, 4.0, -5.5, -6.0]

# full

x_full = zeros(length(x))
x_full[nonzeroinds(x)] = nonzeros(x)
@test isa(full(x), Vector{Float64})
@test full(x) == x_full

x2_full = zeros(length(x2))
x2_full[nonzeroinds(x2)] = nonzeros(x2)
@test isa(full(x2), Vector{Float64})
@test full(x2) == x2_full


### Show

showstr = "Sparse vector, length = 8, with 3 Float64 entries:\n" *
"  [2]  =  1.25\n" *
"  [5]  =  -0.75\n" *
"  [6]  =  3.5\n"
@test string(x) == showstr


### Other Constructors

x0 = SparseVector(8)
@test isa(x0, SparseVector{Float64,Int})
@test length(x0) == 8
@test nnz(x0) == 0

x0 = SparseVector(Float32, 8)
@test isa(x0, SparseVector{Float32,Int})
@test length(x0) == 8
@test nnz(x0) == 0

x0 = SparseVector(Float32, Int32, 8)
@test isa(x0, SparseVector{Float32, Int32})
@test length(x0) == 8
@test nnz(x0) == 0

xr = sprand(1000, 0.3)
@test isa(xr, SparseVector{Float64,Int})
@test length(xr) == 1000
@test all(nonzeros(xr) .>= 0.0)

xr = sprand(1000, 0.3, Float32)
@test isa(xr, SparseVector{Float32,Int})
@test length(xr) == 1000
@test all(nonzeros(xr) .>= 0.0)

xr = sprandn(1000, 0.3)
@test isa(xr, SparseVector{Float64,Int})
@test length(xr) == 1000
@test any(nonzeros(xr) .> 0.0) && any(nonzeros(xr) .< 0.0)

dct = Dict{Int,Float64}()
for i in nonzeroinds(x)
    dct[i] = x_full[i]
end
xc = sparsevector(8, dct)
@test isa(xc, SparseVector{Float64,Int})
@test exact_equal(x, xc)


### Element access

# getindex

for i = 1:length(x)
    @test x[i] == x_full[i]
end

# setindex

xc = SparseVector(8)
xc[3] = 2.0
@test exact_equal(xc, SparseVector(8, [3], [2.0]))

xc = copy(x)
xc[5] = 2.0
@test exact_equal(xc, SparseVector(8, [2, 5, 6], [1.25, 2.0, 3.5]))

xc = copy(x)
xc[3] = 4.0
@test exact_equal(xc, SparseVector(8, [2, 3, 5, 6], [1.25, 4.0, -0.75, 3.5]))

xc[1] = 6.0
@test exact_equal(xc, SparseVector(8, [1, 2, 3, 5, 6], [6.0, 1.25, 4.0, -0.75, 3.5]))

xc[8] = -1.5
@test exact_equal(xc, SparseVector(8, [1, 2, 3, 5, 6, 8], [6.0, 1.25, 4.0, -0.75, 3.5, -1.5]))

xc = copy(x)
xc[5] = 0.0
@test exact_equal(xc, SparseVector(8, [2, 6], [1.25, 3.5]))

xc[6] = 0.0
@test exact_equal(xc, SparseVector(8, [2], [1.25]))

xc[2] = 0.0
@test exact_equal(xc, SparseVector(8, Int[], Float64[]))


### Array manipulation

# copy

xc = copy(x)
@test isa(xc, SparseVector{Float64,Int})
@test !is(x.nzind, xc.nzval)
@test !is(x.nzval, xc.nzval)
@test exact_equal(x, xc)

# reinterpret

a = SparseVector(8, [2, 5, 6], Int32[12, 35, 72])
au = reinterpret(UInt32, a)

@test isa(au, SparseVector{UInt32,Int})
@test exact_equal(au, SparseVector(8, [2, 5, 6], UInt32[12, 35, 72]))

# float

af = float(a)
@test isa(af, SparseVector{Float64,Int})
@test exact_equal(af, SparseVector(8, [2, 5, 6], [12., 35., 72.]))

# complex

acp = complex(af)
@test isa(acp, SparseVector{Complex128,Int})
@test exact_equal(acp, SparseVector(8, [2, 5, 6], complex([12., 35., 72.])))


### Type conversion

x_from_mat = convert(SparseVector,
    sparse([2, 5, 6], [1, 1, 1], [1.25, -0.75, 3.5], 8, 1))

@test isa(x_from_mat, SparseVector{Float64,Int})
@test exact_equal(x_from_mat, x)

xc = convert(SparseVector, x_full)
@test isa(xc, SparseVector{Float64,Int})
@test exact_equal(x, xc)

xc = convert(SparseVector{Float32,Int}, x)
xf32 = SparseVector(8, [2, 5, 6], [1.25f0, -0.75f0, 3.5f0])
@test isa(xc, SparseVector{Float32,Int})
@test exact_equal(xc, xf32)

xc = convert(SparseVector{Float32}, x)
@test isa(xc, SparseVector{Float32,Int})
@test exact_equal(xc, xf32)

xm = convert(SparseMatrixCSC, x)
@test isa(xm, SparseMatrixCSC{Float64,Int})
@test full(xm) == reshape(x_full, 8, 1)

xm = convert(SparseMatrixCSC{Float32}, x)
@test isa(xm, SparseMatrixCSC{Float32,Int})
@test full(xm) == reshape(convert(Vector{Float32}, x_full), 8, 1)


### Arithmetics

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

# complex

@test exact_equal(complex(x, x),
    SparseVector(8, [2,5,6], [1.25+1.25im, -0.75-0.75im, 3.5+3.5im]))

@test exact_equal(complex(x, x2),
    SparseVector(8, [1,2,5,6,7], [3.25im, 1.25+4.0im, -0.75+0.im, 3.5-5.5im, -6.0im]))

@test exact_equal(complex(x2, x),
    SparseVector(8, [1,2,5,6,7], [3.25+0.im, 4.0+1.25im, -0.75im, -5.5+3.5im, -6.0+0.im]))

# real & imag

@test is(real(x), x)
@test exact_equal(imag(x), SparseVector(Float64, length(x)))

xcp = complex(x, x2)
@test exact_equal(real(xcp), x)
@test exact_equal(imag(xcp), x2)

### Reduction

@test sum(x) == 4.0
@test sumabs(x) == 5.5
@test sumabs2(x) == 14.375

@test vecnorm(x) == sqrt(14.375)
@test vecnorm(x, 1) == 5.5
@test vecnorm(x, 2) == sqrt(14.375)
@test vecnorm(x, Inf) == 3.5
