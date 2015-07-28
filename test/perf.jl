# performance benchmark

using SparseVectors

macro benchfun2(bfun, tfun, descr, xs, ys, nrtimes)
    esc(quote
        function $(bfun)(x::AbstractSparseVector, y::AbstractSparseVector, nr::Int)
            println(string("Benchmark on ", $(string(descr)), " ..."))
            xmat = convert(SparseMatrixCSC, x)
            ymat = convert(SparseMatrixCSC, y)

            gc();
            gc_enable(false)

            # warming
            $(tfun)(x, y)
            $(tfun)(xmat, ymat)

            # main profile
            et1 = @elapsed for i = 1:nr
                $(tfun)(x, y)
            end
            et2 = @elapsed for i = 1:nr
                $(tfun)(xmat, ymat)
            end

            gc_enable(true)

            # show result
            @printf("  sparse vector: %9.4fms (gain = %.4fx)\n", et1 * 1e3, et2 / et1)
            @printf("  sparse matrix: %9.4fms\n", et2 * 1e3)
        end
        $(bfun)($xs, $ys, $nrtimes)
    end)
end

# data

n = 1_000_000
x = sprand(n, 0.25)
y = sprand(n, 0.25)

@benchfun2(bench_vadd, +, "sparse vector addition", x, y, 20)
@benchfun2(bench_vmul, .*, "sparse vector multiplication", x, y, 20)
