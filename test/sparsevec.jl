using Base.Test
using SparseData

# construction

x = SparseVector(8, [2, 5, 6], [1.25, -0.75, 3.5])

@test eltype(x) == Float64
@test ndims(x) == 1
@test length(x) == 8
@test size(x) == (8,)
@test size(x,1) == 8
@test size(x,2) == 1

@test countnz(x) == 3
@test nnz(x) == 3
@test nonzeros(x) == [1.25, -0.75, 3.5]


# full

xf = zeros(8)
xf[2] = 1.25
xf[5] = -0.75
xf[6] = 3.5
@test full(x) == xf

# copy

xc = copy(x)
@test !is(x.nzind, xc.nzval)
@test !is(x.nzval, xc.nzval)

@test x.n == xc.n
@test x.nzind == xc.nzind
@test x.nzval == xc.nzval

# getindex

for i = 1:length(x)
    @test x[i] == xf[i]
end

# sum

@test sum(x) == 4.0
@test sumabs(x) == 5.5
@test sumabs2(x) == 14.375

# dot

x2 = SparseVector(8, [1, 2, 6, 7], [3.25, 4.0, -5.5, -6.0])
dv = dot(full(x), full(x2))

@test dot(x, x) == sumabs2(x)
@test dot(x, x2) == dv
@test dot(x2, x) == dv
@test dot(full(x), x2) == dv
@test dot(x, full(x2)) == dv
