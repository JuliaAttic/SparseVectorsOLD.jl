using Base.Test
using SparseExtensions


S = sprand(4, 8, 0.5)
Sf = full(S)
Sv = view(S)

@test isa(Sv, SparseMatrixCSCView{Float64,Int})

@test eltype(Sv) == Float64
@test ndims(Sv) == 2
@test length(Sv) == 32
@test size(Sv) == (4, 8)
@test size(Sv, 1) == 4
@test size(Sv, 2) == 8
@test !isempty(Sv)

@test countnz(Sv) == countnz(S)
@test nnz(Sv) == nnz(S)
@test nonzeros(Sv) == nonzeros(S)

# getindex

for j = 1:size(Sv,2), i = 1:size(Sv,1)
    @test Sv[i,j] == Sf[i,j]
end


# full

@test full(Sv) == Sf

# copy

Sc = copy(Sv)
@test isa(Sc, SparseMatrixCSC{Float64,Int})
@test size(Sc) == size(S)
# @test !is(Sc.colptr, S.colptr)
@test !is(Sc.rowval, S.rowval)
@test !is(Sc.nzval, S.nzval)
@test Sc.colptr == S.colptr
@test Sc.rowval == S.rowval
@test Sc.nzval == S.nzval


# column views

for X in Any[S, Sv]
    for j = 1:size(X,2)
        col = view(X, :, j)
        @test isa(col, SparseVectorView{Float64,Int})
        @test length(col) == size(X,1)
        @test full(col) == Sf[:,j]
    end
end

# column-range views

for X in Any[S, Sv]
    # non-empty range
    V = view(X, :, 2:6)
    @test isa(V, SparseMatrixCSCView{Float64,Int})
    @test size(V) == (4, 5)
    @test full(V) == Sf[:, 2:6]
    @test !isempty(V)

    # empty range
    V0 = view(X, :, 2:1)
    @test isa(V0, SparseMatrixCSCView{Float64,Int})
    @test size(V0) == (4, 0)
    @test isempty(V0)
end
