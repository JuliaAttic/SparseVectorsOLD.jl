# Generic functions operating on AbstractSparseVector

### getindex

function getindex{Tv}(x::AbstractSparseVector{Tv}, i::Integer)
    nzind = nonzeroinds(x)
    nzval = nonzeros(x)
    m = length(nzind)
    ii = searchsortedfirst(nzind, i)
    (ii <= m && nzind[ii] == i) ? nzval[ii] : zero(Tv)
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
            print(io, "\t", '[', rpad(nzind[k], pad), "]  =  ")
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

import Base.SparseMatrix.indtype

function exact_equal(x::AbstractSparseVector, y::AbstractSparseVector)
    eltype(x) == eltype(y) &&
    indtype(x) == indtype(y) &&
    length(x) == length(y) &&
    nonzeroinds(x) == nonzeroinds(y) &&
    nonzeros(x) == nonzeros(y)
end


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


### Reduction

sum(x::AbstractSparseVector) = sum(nonzeros(x))
sumabs(x::AbstractSparseVector) = sumabs(nonzeros(x))
sumabs2(x::AbstractSparseVector) = sumabs2(nonzeros(x))

vecnorm(x::AbstractSparseVector, p::Real=2) = vecnorm(nonzeros(x), p)
