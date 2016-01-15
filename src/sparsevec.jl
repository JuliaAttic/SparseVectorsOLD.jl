
### Types

immutable SparseVector{Tv,Ti<:Integer} <: AbstractSparseVector{Tv,Ti}
    n::Int              # the number of elements
    nzind::Vector{Ti}   # the indices of nonzeros
    nzval::Vector{Tv}   # the values of nonzeros

    function SparseVector(n::Integer, nzind::Vector{Ti}, nzval::Vector{Tv})
        n >= 0 || throw(ArgumentError("The number of elements must be non-negative."))
        length(nzind) == length(nzval) ||
            throw(DimensionMismatch("The lengths of nzind and nzval are inconsistent."))
        new(convert(Int, n), nzind, nzval)
    end
end

SparseVector{Tv,Ti}(n::Integer, nzind::Vector{Ti}, nzval::Vector{Tv}) =
    SparseVector{Tv,Ti}(n, nzind, nzval)

### Basic properties

length(x::SparseVector) = x.n
size(x::SparseVector) = (x.n,)
nnz(x::SparseVector) = length(x.nzval)
countnz(x::SparseVector) = countnz(x.nzval)
nonzeros(x::SparseVector) = x.nzval
nonzeroinds(x::SparseVector) = x.nzind

### Construct empty sparse vector

sparsevector{T}(::Type{T}, len::Integer) = SparseVector(len, Int[], T[])

### Construction from lists of indices and values

function _sparsevector!{Ti<:Integer}(I::Vector{Ti}, V::Vector, len::Integer)
    # pre-condition: no duplicate indexes in I
    if !isempty(I)
        p = sortperm(I)
        permute!(I, p)
        permute!(V, p)
    end
    SparseVector(len, I, V)
end

function _sparsevector!{Tv,Ti<:Integer}(I::Vector{Ti}, V::Vector{Tv}, len::Integer, combine::BinaryOp)
    if !isempty(I)
        p = sortperm(I)
        permute!(I, p)
        permute!(V, p)
        m = length(I)

        # advances to the first non-zero element
        r = 1     # index of last processed entry
        while r <= m
            if V[r] == zero(Tv)
                r += 1
            else
                break
            end
        end
        r > m && SparseVector(len, Ti[], Tv[])

        # move r-th to l-th
        l = 1       # length of processed part
        i = I[r]    # row-index of current element
        if r > 1
            I[l] = i; V[l] = V[r]
        end

        # main loop
        while r < m
            r += 1
            i2 = I[r]
            if i2 == i  # accumulate r-th to the l-th entry
                V[l] = call(combine, V[l], V[r])
            else  # advance l, and move r-th to l-th
                pv = V[l]
                if pv != zero(Tv)
                    l += 1
                end
                i = i2
                if l < r
                    I[l] = i; V[l] = V[r]
                end
            end
        end
        if V[l] == zero(Tv)
            l -= 1
        end
        if l < m
            resize!(I, l)
            resize!(V, l)
        end
    end
    SparseVector(len, I, V)
end

function sparsevector{Tv,Ti<:Integer}(I::AbstractVector{Ti}, V::AbstractVector{Tv}, combine::BinaryOp)
    length(I) == length(V) ||
        throw(DimensionMismatch("The lengths of I and V are inconsistent."))
    len = 0
    for i in I
        i >= 1 || error("Index must be positive.")
        if i > len
            len = i
        end
    end
    _sparsevector!(_copy_convert(Ti, I), _copy_convert(Tv, V), len, combine)
end

function sparsevector{Tv,Ti<:Integer}(I::AbstractVector{Ti}, V::AbstractVector{Tv}, len::Integer, combine::BinaryOp)
    length(I) == length(V) ||
        throw(DimensionMismatch("The lengths of I and V are inconsistent."))
    maxi = convert(Ti, len)
    for i in I
        1 <= i <= maxi || error("An index is out of bound.")
    end
    _sparsevector!(_copy_convert(Ti, I), _copy_convert(Tv, V), len, combine)
end

sparsevector{Ti<:Integer}(I::AbstractVector{Ti}, V::AbstractVector) =
    sparsevector(I, V, AddFun())

sparsevector{Ti<:Integer}(I::AbstractVector{Ti}, V::AbstractVector, len::Integer) =
    sparsevector(I, V, len, AddFun())

sparsevector{Ti<:Integer}(I::AbstractVector{Ti}, v::Number, combine::BinaryOp) =
    sparsevector(I, fill(v, length(I)), combine)

sparsevector{Ti<:Integer}(I::AbstractVector{Ti}, v::Number, len::Integer, combine::BinaryOp) =
    sparsevector(I, fill(v, length(I)), len, combine)

sparsevector{Ti<:Integer}(I::AbstractVector{Ti}, v::Number) =
    sparsevector(I, v, AddFun())

sparsevector{Ti<:Integer}(I::AbstractVector{Ti}, v::Number, len::Integer) =
    sparsevector(I, v, len, AddFun())


### Construction from dictionary

function sparsevector{Tv,Ti<:Integer}(dict::Associative{Ti,Tv})
    m = length(dict)
    nzind = Array(Ti, m)
    nzval = Array(Tv, m)

    cnt = 0
    len = zero(Ti)
    for (k, v) in dict
        k >= 1 || error("Index must be positive.")
        if k > len
            len = k
        end
        if v != zero(v)
            cnt += 1
            @inbounds nzind[cnt] = k
            @inbounds nzval[cnt] = v
        end
    end
    resize!(nzind, cnt)
    resize!(nzval, cnt)
    _sparsevector!(nzind, nzval, len)
end

function sparsevector{Tv,Ti<:Integer}(dict::Associative{Ti,Tv}, len::Integer)
    m = length(dict)
    nzind = Array(Ti, m)
    nzval = Array(Tv, m)

    cnt = 0
    maxk = convert(Ti, len)
    for (k, v) in dict
        1 <= k <= maxk || error("An index (key) is out of bound.")
        if v != zero(v)
            cnt += 1
            @inbounds nzind[cnt] = k
            @inbounds nzval[cnt] = v
        end
    end
    resize!(nzind, cnt)
    resize!(nzval, cnt)
    _sparsevector!(nzind, nzval, len)
end


### Element access

function setindex!{Tv,Ti<:Integer}(x::SparseVector{Tv,Ti}, v::Tv, i::Ti)
    nzind = nonzeroinds(x)
    nzval = nonzeros(x)

    m = length(nzind)
    k = searchsortedfirst(nzind, i)
    if 1 <= k <= m && nzind[k] == i  # i found
        if v == zero(v)
            deleteat!(nzind, k)
            deleteat!(nzval, k)
        else
            nzval[k] = v
        end
    else  # i not found
        if v != zero(v)
            insert!(nzind, k, i)
            insert!(nzval, k, v)
        end
    end
    x
end

setindex!{Tv, Ti<:Integer}(x::SparseVector{Tv,Ti}, v, i::Integer) =
    setindex!(x, convert(Tv, v), convert(Ti, i))


### Conversion

# convert SparseMatrixCSC to SparseVector
function convert{Tv,Ti<:Integer}(::Type{SparseVector{Tv,Ti}}, s::SparseMatrixCSC{Tv,Ti})
    size(s, 2) == 1 || throw(ArgumentError("The input argument must have a single-column."))
    SparseVector(s.m, s.rowval, s.nzval)
end

convert{Tv,Ti}(::Type{SparseVector{Tv}}, s::SparseMatrixCSC{Tv,Ti}) =
    convert(SparseVector{Tv,Ti}, s)

convert{Tv,Ti}(::Type{SparseVector}, s::SparseMatrixCSC{Tv,Ti}) =
    convert(SparseVector{Tv,Ti}, s)

# convert Vector to SparseVector

function _dense2sparsevec{Tv}(s::Vector{Tv}, initcap::Int)
    # pre-condition: initcap > 0
    n = length(s)
    cap = initcap
    nzind = Array(Int, cap)
    nzval = Array(Tv, cap)
    c = 0
    @inbounds for i = 1:n
        v = s[i]
        if v != zero(v)
            if c >= cap
                cap *= 2
                resize!(nzind, cap)
                resize!(nzval, cap)
            end
            c += 1
            nzind[c] = i
            nzval[c] = v
        end
    end
    if c < cap
        resize!(nzind, c)
        resize!(nzval, c)
    end
    SparseVector(n, nzind, nzval)
end

convert{Tv}(::Type{SparseVector{Tv,Int}}, s::Vector{Tv}) =
    _dense2sparsevec(s, max(8, div(length(s), 8)))

convert{Tv}(::Type{SparseVector{Tv}}, s::Vector{Tv}) =
    convert(SparseVector{Tv,Int}, s)

convert{Tv}(::Type{SparseVector}, s::Vector{Tv}) =
    convert(SparseVector{Tv,Int}, s)


# convert between different types of SparseVector
convert{Tv,Ti,TvS,TiS}(::Type{SparseVector{Tv,Ti}}, s::SparseVector{TvS,TiS}) =
    SparseVector{Tv,Ti}(s.n, convert(Vector{Ti}, s.nzind), convert(Vector{Tv}, s.nzval))

convert{Tv,TvS,TiS}(::Type{SparseVector{Tv}}, s::SparseVector{TvS,TiS}) =
    SparseVector{Tv,TiS}(s.n, s.nzind, convert(Vector{Tv}, s.nzval))


### Rand Construction

@compat function sprand{T}(n::Integer, p::AbstractFloat, rfn::Function, ::Type{T})
    I = randsubseq(1:convert(Int, n), p)
    V = rfn(T, length(I))
    SparseVector(n, I, V)
end

@compat function sprand(n::Integer, p::AbstractFloat, rfn::Function)
    I = randsubseq(1:convert(Int, n), p)
    V = rfn(length(I))
    SparseVector(n, I, V)
end

@compat sprand{T}(n::Integer, p::AbstractFloat, ::Type{T}) = sprand(n, p, rand, T)
@compat sprand(n::Integer, p::AbstractFloat) = sprand(n, p, rand)
@compat sprandn(n::Integer, p::AbstractFloat) = sprand(n, p, randn)
