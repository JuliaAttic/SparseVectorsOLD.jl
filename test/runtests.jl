tests = ["sparsevec"]

for t in tests
    fp = "$t.jl"
    println("Running $fp ...")
    include(fp)
end
