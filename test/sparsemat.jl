using Base.Test
using SparseVectors

S = sprand(4, 8, 0.5)
Sf = full(S)
@assert isa(Sf, Matrix{Float64})

# column views

for j = 1:size(S,2)
    col = view(S, :, j)
    @test isa(col, SparseVectorView{Float64,Int})
    @test length(col) == size(S,1)
    @test full(col) == Sf[:,j]
end

# column-range views

# non-empty range
V = unsafe_colrange(S, 2:6)
@test isa(V, SparseMatrixCSC{Float64,Int})
@test size(V) == (4, 5)
@test full(V) == Sf[:, 2:6]
@test !isempty(V)

# empty range
V0 = unsafe_colrange(S, 2:1)
@test isa(V0, SparseMatrixCSC{Float64,Int})
@test size(V0) == (4, 0)
@test isempty(V0)
