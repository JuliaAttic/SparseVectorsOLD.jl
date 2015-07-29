# Generic functions operating on AbstractSparseVector

### getindex

function _spgetindex{Tv,Ti}(m::Int, nzind::AbstractVector{Ti}, nzval::AbstractVector{Tv}, i::Integer)
    ii = searchsortedfirst(nzind, convert(Ti, i))
    (ii <= m && nzind[ii] == i) ? nzval[ii] : zero(Tv)
end

getindex{Tv}(x::AbstractSparseVector{Tv}, i::Integer) =
    _spgetindex(nnz(x), nonzeroinds(x), nonzeros(x), i)

function getindex{Tv,Ti}(x::AbstractSparseVector{Tv,Ti}, I::UnitRange)
    xlen = length(x)
    i0 = first(I)
    i1 = last(I)
    (i0 >= 1 && i1 <= xlen) || throw(BoundsError())

    xnzind = nonzeroinds(x)
    xnzval = nonzeros(x)
    m = length(xnzind)

    # locate the first j0, s.t. xnzind[j0] >= i0
    j0 = 1
    while j0 <= m && xnzind[j0] < i0
        j0 += 1
    end
    # locate the last j1, s.t. xnzind[j1] <= i1
    j1 = j0 - 1
    while j1 < m && xnzind[j1+1] <= i1
        j1 += 1
    end

    # compute the number of non-zeros
    jrgn = j0:j1
    mr = length(jrgn)
    rind = Array(Ti, mr)
    rval = Array(Tv, mr)
    if mr > 0
        c = 0
        for j in jrgn
            c += 1
            rind[c] = convert(Ti, xnzind[j] - i0 + 1)
            rval[c] = xnzval[j]
        end
    end
    SparseVector(length(I), rind, rval)
end

function getindex{Tv,Ti}(x::AbstractSparseVector{Tv,Ti}, I::AbstractArray)
    xnzind = nonzeroinds(x)
    xnzval = nonzeros(x)
    m = length(xnzind)
    n = length(I)
    nzind = Array(Ti, n)
    nzval = Array(Tv, n)
    c = 0
    for j = 1:n
        v = _spgetindex(m, xnzind, xnzval, I[j])
        if v != zero(v)
            c += 1
            nzind[c] = convert(Ti, j)
            nzval[c] = v
        end
    end
    if c < n
        resize!(nzind, c)
        resize!(nzval, c)
    end
    SparseVector(n, nzind, nzval)
end

### show and friends

function showarray(io::IO, x::AbstractSparseVector;
                   header::Bool=true, limit::Bool=Base._limit_output,
                   rows = Base.tty_size()[1], repr=false)

    n = length(x)
    nzind = nonzeroinds(x)
    nzval = nonzeros(x)
    xnnz = length(nzind)

    if header
        print(io, "Sparse vector, length = ", n,
            ", with ", xnnz, " ", eltype(nzval), " entries:", "\n")
    end
    half_screen_rows = limit ? div(rows - 8, 2) : typemax(Int)
    pad = ndigits(n)
    k = 0
    sep = "\n\t"
    for k = 1:length(nzind)
        if k < half_screen_rows || k > xnnz - half_screen_rows
            print(io, "  ", '[', rpad(nzind[k], pad), "]  =  ")
            showcompact(io, nzval[k])
        elseif k == half_screen_rows
            print(io, sep, '\u22ee')
        end
        print(io, "\n")
        k += 1
    end
end

show(io::IO, x::AbstractSparseVector) = showarray(io, x)
writemime(io::IO, ::MIME"text/plain", x::AbstractSparseVector) = show(io, x)


### Comparison

function exact_equal(x::AbstractSparseVector, y::AbstractSparseVector)
    eltype(x) == eltype(y) &&
    eltype(nonzeroinds(x)) == eltype(nonzeroinds(y)) &&
    length(x) == length(y) &&
    nonzeroinds(x) == nonzeroinds(y) &&
    nonzeros(x) == nonzeros(y)
end

### Conversion to matrix

function convert{TvD,TiD,Tv,Ti}(::Type{SparseMatrixCSC{TvD,TiD}}, x::AbstractSparseVector{Tv,Ti})
    n = length(x)
    xnzind = nonzeroinds(x)
    xnzval = nonzeros(x)
    m = length(xnzind)
    colptr = TiD[1, m+1]
    rowval = _copy_convert(TiD, xnzind)
    nzval = _copy_convert(TvD, xnzval)
    SparseMatrixCSC(n, 1, colptr, rowval, nzval)
end

convert{TvD,Tv,Ti}(::Type{SparseMatrixCSC{TvD}}, x::AbstractSparseVector{Tv,Ti}) =
    convert(SparseMatrixCSC{TvD,Ti}, x)

convert{Tv,Ti}(::Type{SparseMatrixCSC}, x::AbstractSparseVector{Tv,Ti}) =
    convert(SparseMatrixCSC{Tv,Ti}, x)


### Array manipulation

function full{Tv}(x::AbstractSparseVector{Tv})
    n = length(x)
    nzind = nonzeroinds(x)
    nzval = nonzeros(x)
    r = zeros(Tv, n)
    for i = 1:length(nzind)
        r[nzind[i]] = nzval[i]
    end
    return r
end

vec(x::AbstractSparseVector) = x
copy(x::AbstractSparseVector) =
    SparseVector(length(x), copy(nonzeroinds(x)), copy(nonzeros(x)))

function reinterpret{T,Tv}(::Type{T}, x::AbstractSparseVector{Tv})
    sizeof(T) == sizeof(Tv) ||
        throw(ArgumentError("reinterpret of sparse vectors only supports element types of the same size."))
    SparseVector(length(x), copy(nonzeroinds(x)), reinterpret(T, nonzeros(x)))
end

float{Tv<:FloatingPoint}(x::AbstractSparseVector{Tv}) = x
float(x::AbstractSparseVector) =
    SparseVector(length(x), copy(nonzeroinds(x)), float(nonzeros(x)))

complex{Tv<:Complex}(x::AbstractSparseVector{Tv}) = x
complex(x::AbstractSparseVector) =
    SparseVector(length(x), copy(nonzeroinds(x)), complex(nonzeros(x)))
