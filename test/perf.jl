# performance benchmark

using SparseVectors

function bench_vadd(x::AbstractSparseVector, y::AbstractSparseVector, nr::Int)
    println("Benchmark on sparse vector addition ...")
    xmat = convert(SparseMatrixCSC, x)
    ymat = convert(SparseMatrixCSC, y)
    x + y
    xmat + ymat
    et1 = @elapsed for i = 1:nr
        x + y
    end
    et2 = @elapsed for i = 1:nr
        xmat + ymat
    end
    @printf("  sparse vector: %9.4fms (gain = %.4fx)\n", et1 * 1e3, et2 / et1)
    @printf("  sparse matrix: %9.4fms\n", et2 * 1e3)
end

function bench_vmul(x::AbstractSparseVector, y::AbstractSparseVector, nr::Int)
    println("Benchmark on sparse vector multiplication (elem-wise) ...")
    xmat = convert(SparseMatrixCSC, x)
    ymat = convert(SparseMatrixCSC, y)
    x .* y
    xmat .* ymat
    et1 = @elapsed for i = 1:nr
        x .* y
    end
    et2 = @elapsed for i = 1:nr
        xmat .* ymat
    end
    @printf("  sparse vector: %9.4fms (gain = %.4fx)\n", et1 * 1e3, et2 / et1)
    @printf("  sparse matrix: %9.4fms\n", et2 * 1e3)
end


# data

n = 1_000_000
x = sprand(n, 0.25)
y = sprand(n, 0.25)

gc();
gc_enable(false);
bench_vadd(x, y, 20)
bench_vmul(x, y, 20)
gc_enable(true);
