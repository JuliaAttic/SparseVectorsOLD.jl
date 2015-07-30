using Base.Test
using Compat
using SparseVectors

import SparseVectors: exact_equal

### Data

spv_x1 = SparseVector(8, [2, 5, 6], [1.25, -0.75, 3.5])
_x2 = SparseVector(8, [1, 2, 6, 7], [3.25, 4.0, -5.5, -6.0])
spv_x2 = view(_x2)

@test isa(spv_x1, SparseVector{Float64,Int})
@test isa(spv_x2, SparseVectorView{Float64,Int})

x1_full = zeros(length(spv_x1))
x1_full[nonzeroinds(spv_x1)] = nonzeros(spv_x1)

x2_full = zeros(length(spv_x2))
x2_full[nonzeroinds(spv_x2)] = nonzeros(spv_x2)


### Basic Properties

let x = spv_x1
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
end

let x = spv_x2
    @test eltype(x) == Float64
    @test ndims(x) == 1
    @test length(x) == 8
    @test size(x) == (8,)
    @test size(x,1) == 8
    @test size(x,2) == 1
    @test !isempty(x)

    @test countnz(x) == 4
    @test nnz(x) == 4
    @test nonzeroinds(x) == [1, 2, 6, 7]
    @test nonzeros(x) == [3.25, 4.0, -5.5, -6.0]
end

# full

for (x, xf) in [(spv_x1, x1_full), (spv_x2, x2_full)]
    @test isa(full(x), Vector{Float64})
    @test full(x) == xf
end

### Show

@test string(spv_x1) == "Sparse vector, length = 8, with 3 Float64 entries:\n" *
"  [2]  =  1.25\n" *
"  [5]  =  -0.75\n" *
"  [6]  =  3.5\n"


### Other Constructors

# construct empty sparse vector

@test exact_equal(sparsevector(Float64, 8), SparseVector(8, Int[], Float64[]))

# from list of indices and values

@test exact_equal(
    sparsevector(Int[], Float64[], 8),
    SparseVector(8, Int[], Float64[]))

@test exact_equal(
    sparsevector(Int[], Float64[]),
    SparseVector(0, Int[], Float64[]))

@test exact_equal(
    sparsevector([3, 3], [5.0, -5.0], 8),
    sparsevector(Float64, 8))

@test exact_equal(
    sparsevector([2, 3, 6], [12.0, 18.0, 25.0]),
    SparseVector(6, [2, 3, 6], [12.0, 18.0, 25.0]))

let x0 = SparseVector(8, [2, 3, 6], [12.0, 18.0, 25.0])
    @test exact_equal(
        sparsevector([2, 3, 6], [12.0, 18.0, 25.0], 8), x0)

    @test exact_equal(
        sparsevector([3, 6, 2], [18.0, 25.0, 12.0], 8), x0)

    @test exact_equal(
        sparsevector([2, 3, 4, 4, 6], [12.0, 18.0, 5.0, -5.0, 25.0], 8),
        x0)

    @test exact_equal(
        sparsevector([1, 1, 1, 2, 3, 3, 6], [2.0, 3.0, -5.0, 12.0, 10.0, 8.0, 25.0], 8),
        x0)

    @test exact_equal(
        sparsevector([2, 3, 6, 7, 7], [12.0, 18.0, 25.0, 5.0, -5.0], 8), x0)
end

# from dictionary

function my_intmap(x)
    a = Dict{Int,eltype(x)}()
    for i in nonzeroinds(x)
        a[i] = x[i]
    end
    return a
end

let x = spv_x1
    a = my_intmap(x)
    xc = sparsevector(a, 8)
    @test exact_equal(x, xc)

    xc = sparsevector(a)
    @test exact_equal(xc, SparseVector(6, [2, 5, 6], [1.25, -0.75, 3.5]))
end

# sprand & sprandn

let xr = sprand(1000, 0.3)
    @test isa(xr, SparseVector{Float64,Int})
    @test length(xr) == 1000
    @test all(nonzeros(xr) .>= 0.0)
end

let xr = sprand(1000, 0.3, Float32)
    @test isa(xr, SparseVector{Float32,Int})
    @test length(xr) == 1000
    @test all(nonzeros(xr) .>= 0.0)
end

let xr = sprandn(1000, 0.3)
    @test isa(xr, SparseVector{Float64,Int})
    @test length(xr) == 1000
    @test any(nonzeros(xr) .> 0.0) && any(nonzeros(xr) .< 0.0)
end


### Element access

# getindex

# single integer index
for (x, xf) in [(spv_x1, x1_full), (spv_x2, x2_full)]
    for i = 1:length(x)
        @test x[i] == xf[i]
    end
end

# range index
let x = spv_x2
    # range that contains no non-zeros
    @test exact_equal(x[3:2], sparsevector(Float64, 0))
    @test exact_equal(x[3:3], sparsevector(Float64, 1))
    @test exact_equal(x[3:5], sparsevector(Float64, 3))

    # range with non-zeros
    @test exact_equal(x[1:length(x)], x)
    @test exact_equal(x[1:5], SparseVector(5, [1,2], [3.25, 4.0]))
    @test exact_equal(x[2:6], SparseVector(5, [1,5], [4.0, -5.5]))
    @test exact_equal(x[2:8], SparseVector(7, [1,5,6], [4.0, -5.5, -6.0]))
end

# generic array index
let x = sprand(100, 0.5)
    I = rand(1:length(x), 20)
    @which x[I]
    r = x[I]
    @test isa(r, SparseVector{Float64,Int})
    @test all(nonzeros(r) .!= 0.0)
    @test full(r) == full(x)[I]
end

# setindex

let xc = sparsevector(Float64, 8)
    xc[3] = 2.0
    @test exact_equal(xc, SparseVector(8, [3], [2.0]))
end

let xc = copy(spv_x1)
    xc[5] = 2.0
    @test exact_equal(xc, SparseVector(8, [2, 5, 6], [1.25, 2.0, 3.5]))
end

let xc = copy(spv_x1)
    xc[3] = 4.0
    @test exact_equal(xc, SparseVector(8, [2, 3, 5, 6], [1.25, 4.0, -0.75, 3.5]))

    xc[1] = 6.0
    @test exact_equal(xc, SparseVector(8, [1, 2, 3, 5, 6], [6.0, 1.25, 4.0, -0.75, 3.5]))

    xc[8] = -1.5
    @test exact_equal(xc, SparseVector(8, [1, 2, 3, 5, 6, 8], [6.0, 1.25, 4.0, -0.75, 3.5, -1.5]))
end

let xc = copy(spv_x1)
    xc[5] = 0.0
    @test exact_equal(xc, SparseVector(8, [2, 6], [1.25, 3.5]))

    xc[6] = 0.0
    @test exact_equal(xc, SparseVector(8, [2], [1.25]))

    xc[2] = 0.0
    @test exact_equal(xc, SparseVector(8, Int[], Float64[]))
end


### Array manipulation

# copy

let x = spv_x1
    xc = copy(x)
    @test isa(xc, SparseVector{Float64,Int})
    @test !is(x.nzind, xc.nzval)
    @test !is(x.nzval, xc.nzval)
    @test exact_equal(x, xc)
end

let a = SparseVector(8, [2, 5, 6], Int32[12, 35, 72])
    # reinterpret
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
end

### Type conversion

let x = convert(SparseVector, sparse([2, 5, 6], [1, 1, 1], [1.25, -0.75, 3.5], 8, 1))
    @test isa(x, SparseVector{Float64,Int})
    @test exact_equal(x, spv_x1)
end

let x = spv_x1, xf = x1_full
    xc = convert(SparseVector, xf)
    @test isa(xc, SparseVector{Float64,Int})
    @test exact_equal(xc, x)

    xc = convert(SparseVector{Float32,Int}, x)
    xf32 = SparseVector(8, [2, 5, 6], [1.25f0, -0.75f0, 3.5f0])
    @test isa(xc, SparseVector{Float32,Int})
    @test exact_equal(xc, xf32)

    xc = convert(SparseVector{Float32}, x)
    @test isa(xc, SparseVector{Float32,Int})
    @test exact_equal(xc, xf32)

    xm = convert(SparseMatrixCSC, x)
    @test isa(xm, SparseMatrixCSC{Float64,Int})
    @test full(xm) == reshape(xf, 8, 1)

    xm = convert(SparseMatrixCSC{Float32}, x)
    @test isa(xm, SparseMatrixCSC{Float32,Int})
    @test full(xm) == reshape(convert(Vector{Float32}, xf), 8, 1)
end


### Concatenation

let m = 80, n = 100
    A = Array(SparseVector{Float64,Int}, n)
    tnnz = 0
    for i = 1:length(A)
        A[i] = sprand(m, 0.3)
        tnnz += nnz(A[i])
    end

    H = hcat(A...)
    @test isa(H, SparseMatrixCSC{Float64,Int})
    @test size(H) == (m, n)
    @test nnz(H) == tnnz
    Hr = zeros(m, n)
    for j = 1:n
        Hr[:,j] = full(A[j])
    end
    @test full(H) == Hr
end
