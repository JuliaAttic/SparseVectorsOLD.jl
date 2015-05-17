using Base.Test
using SparseExtensions


S = sprand(4, 8, 0.5)
Sf = full(S)
@assert isa(Sf, Matrix{Float64})
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


# as_sparsemat

S_a = SparseExtensions.as_sparsemat(Sv)
@test isa(S_a, SparseMatrixCSC{Float64,Int})
@test size(S_a) == size(S)
@test full(S_a) == full(S)


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


# sum & vecnorm

for X in Any[S, Sv]
    @test_approx_eq sum(X) sum(Sf)
    @test_approx_eq sumabs(X) sumabs(Sf)
    @test_approx_eq sumabs2(X) sumabs2(Sf)

    @test_approx_eq vecnorm(X) vecnorm(Sf)
    @test_approx_eq vecnorm(X, 1) vecnorm(Sf, 1)
    @test_approx_eq vecnorm(X, 2) vecnorm(Sf, 2)
    @test_approx_eq vecnorm(X, 3) vecnorm(Sf, 3)
    @test_approx_eq vecnorm(X, Inf) vecnorm(Sf, Inf)
end

# sum along dimensions

for X in Any[S, Sv]
    s1 = sum(X, 1)
    @test isa(s1, Matrix{Float64})
    @test size(s1) == (1, size(X,2))
    @test_approx_eq s1 sum(Sf, 1)

    s2 = sum(X, 2)
    @test isa(s2, Matrix{Float64})
    @test size(s2) == (size(X,1), 1)
    @test_approx_eq s2 sum(Sf, 2)

    s3 = sum(X, 3)
    @test s3 == Sf
end
