
### Types

typealias CVecView{T} ContiguousView{T,1,Vector{T}}

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

immutable SparseVectorView{Tv,Ti<:Integer} <: AbstractSparseVector{Tv,Ti}
    n::Int                  # the number of elements
    nzind::CVecView{Ti}     # the indices of nonzeros
    nzval::CVecView{Tv}     # the values of nonzeros

    function SparseVectorView(n::Integer, nzind::CVecView{Ti}, nzval::CVecView{Tv})
        n >= 0 || throw(ArgumentError("The number of elements must be non-negative."))
        length(nzind) == length(nzval) ||
            throw(DimensionMismatch("The lengths of nzind and nzval are inconsistent."))
        new(convert(Int, n), nzind, nzval)
    end
end

SparseVectorView{Tv,Ti}(n::Integer, nzind::CVecView{Ti}, nzval::CVecView{Tv}) =
    SparseVectorView{Tv,Ti}(n, nzind, nzval)

typealias GenericSparseVector{Tv,Ti} Union(SparseVector{Tv,Ti}, SparseVectorView{Tv,Ti})


### Basic properties

Base.length(x::GenericSparseVector) = x.n
Base.size(x::GenericSparseVector) = (x.n,)

Base.nnz(x::GenericSparseVector) = length(x.nzval)
Base.countnz(x::GenericSparseVector) = countnz(x.nzval)
Base.nonzeros(x::GenericSparseVector) = x.nzval

# not exported, used mainly for testing
function exact_equal(x::GenericSparseVector, y::GenericSparseVector)
    x.n == y.n && x.nzind == y.nzind && x.nzval == y.nzval
end


### Element access

function Base.getindex{Tv}(x::GenericSparseVector{Tv}, i::Int)
    m = length(x.nzind)
    ii = searchsortedfirst(x.nzind, i)
    (ii <= m && x.nzind[ii] == i) ? x.nzval[ii] : zero(Tv)
end

Base.getindex(x::GenericSparseVector, i::Integer) = x[convert(Int, i)]


function Base.setindex!{Tv,Ti<:Integer}(x::SparseVector{Tv,Ti}, v::Tv, i::Ti)
    nzind = x.nzind
    nzval = x.nzval

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

Base.setindex!{Tv, Ti<:Integer}(x::SparseVector{Tv,Ti}, v, i::Integer) =
    setindex!(x, convert(Tv, v), convert(Ti, i))


### Conversion

# convert SparseMatrixCSC to SparseVector
function Base.convert{Tv,Ti<:Integer}(::Type{SparseVector{Tv,Ti}}, s::SparseMatrixCSC{Tv,Ti})
    size(s, 2) == 1 || throw(ArgumentError("The input argument must have a single-column."))
    SparseVector(s.m, s.rowval, s.nzval)
end

Base.convert{Tv,Ti}(::Type{SparseVector{Tv}}, s::SparseMatrixCSC{Tv,Ti}) =
    convert(SparseVector{Tv,Ti}, s)

Base.convert{Tv,Ti}(::Type{SparseVector}, s::SparseMatrixCSC{Tv,Ti}) =
    convert(SparseVector{Tv,Ti}, s)


# convert Vector to SparseVector
function Base.convert{Tv}(::Type{SparseVector{Tv,Int}}, s::Vector{Tv})
    n = length(s)
    nzind = Array(Int, 0)
    nzval = Array(Tv, 0)
    for i = 1:n
        @inbounds v = s[i]
        if v != zero(v)
            push!(nzind, i)
            push!(nzval, v)
        end
    end
    return SparseVector(n, nzind, nzval)
end

Base.convert{Tv}(::Type{SparseVector{Tv}}, s::Vector{Tv}) =
    convert(SparseVector{Tv,Int}, s)

Base.convert{Tv}(::Type{SparseVector}, s::Vector{Tv}) =
    convert(SparseVector{Tv,Int}, s)


# convert between different types of SparseVector
Base.convert{Tv,Ti,TvS,TiS}(::Type{SparseVector{Tv,Ti}}, s::SparseVector{TvS,TiS}) =
    SparseVector{Tv,Ti}(s.n, convert(Vector{Ti}, s.nzind), convert(Vector{Tv}, s.nzval))

Base.convert{Tv,TvS,TiS}(::Type{SparseVector{Tv}}, s::SparseVector{TvS,TiS}) =
    SparseVector{Tv,TiS}(s.n, s.nzind, convert(Vector{Tv}, s.nzval))


### Construction

SparseVector(n::Integer) = SparseVector(n, Int[], Float64[])
SparseVector{Tv}(::Type{Tv}, n::Integer) = SparseVector(n, Int[], Tv[])
SparseVector{Tv,Ti<:Integer}(::Type{Tv}, ::Type{Ti}, n::Integer) = SparseVector(n, Ti[], Tv[])

# make SparseVector from an iterable collection of (ind, value), e.g. dictionary
function _make_sparsevec{Tv,Ti<:Integer}(::Type{Tv}, ::Type{Ti}, n::Integer, iter, checkrep::Bool)
    m = length(iter)
    nzind = Array(Ti, 0)
    nzval = Array(Tv, 0)
    sizehint!(nzind, m)
    sizehint!(nzval, m)

    for (k, v) in iter
        if v != zero(v)
            push!(nzind, k)
            push!(nzval, v)
        end
    end

    p = sortperm(nzind)
    permute!(nzind, p)
    if checkrep
        for i = 1:length(nzind)-1
            nzind[i] != nzind[i+1] || error("Repeated indices.")
        end
    end
    permute!(nzval, p)

    return SparseVector{Tv,Ti}(convert(Int, n), nzind, nzval)
end

SparseVector{Tv,Ti<:Integer}(n::Integer, s::Associative{Ti,Tv}) =
    _make_sparsevec(Tv, Ti, n, s, false)


SparseVector{Tv,Ti}(n::Integer, s::AbstractVector{@compat(Tuple{Ti,Tv})}) =
    _make_sparsevec(Tv, Ti, n, s, true)


# sprand

function Base.sprand{T}(n::Integer, p::FloatingPoint, rfn::Function, ::Type{T})
    I = randsubseq(1:convert(Int, n), p)
    V = rfn(T, length(I))
    SparseVector(n, I, V)
end

function Base.sprand(n::Integer, p::FloatingPoint, rfn::Function)
    I = randsubseq(1:convert(Int, n), p)
    V = rfn(length(I))
    SparseVector(n, I, V)
end

Base.sprand{T}(n::Integer, p::FloatingPoint, ::Type{T}) = sprand(n, p, rand, T)
Base.sprand(n::Integer, p::FloatingPoint) = sprand(n, p, rand)

Base.sprandn(n::Integer, p::FloatingPoint) = sprand(n, p, randn)


### View

view(x::SparseVector) = SparseVectorView(length(x), view(x.nzind), view(x.nzval))


### Show

function Base.showarray(io::IO, x::GenericSparseVector;
                        header::Bool=true, limit::Bool=Base._limit_output,
                        rows = Base.tty_size()[1], repr=false)
    if header
        print(io, "Sparse vector, length = ", x.n,
            ", with ", nnz(x), " ", eltype(x), " entries:", "\n")
    end

    if limit
        half_screen_rows = div(rows - 8, 2)
    else
        half_screen_rows = typemax(Int)
    end
    pad = ndigits(x.n)
    k = 0

    for k = 1:length(x.nzind)
        if k < half_screen_rows || k > nnz(x)-half_screen_rows
            print(io, "\t", '[', rpad(x.nzind[k], pad), "]  =  ")
            showcompact(io, x.nzval[k])
            print(io, "\n")
        elseif k == half_screen_rows
            print(io, sep, '\u22ee')
        end
        k += 1
    end
end

Base.show(io::IO, x::GenericSparseVector) = Base.showarray(io, x)
Base.writemime(io::IO, ::MIME"text/plain", x::GenericSparseVector) = show(io, x)


### Array manipulation

function Base.full{Tv}(x::GenericSparseVector{Tv})
    n = x.n
    nzind = x.nzind
    nzval = x.nzval
    r = zeros(Tv, n)
    for i = 1:length(nzind)
        r[nzind[i]] = nzval[i]
    end
    return r
end

Base.vec(x::GenericSparseVector) = x

Base.copy(x::GenericSparseVector) = SparseVector(x.n, copy(x.nzind), copy(x.nzval))


### Computation

Base.abs(x::GenericSparseVector) = SparseVector(x.n, copy(x.nzind), abs(x.nzval))
Base.abs2(x::GenericSparseVector) = SparseVector(x.n, copy(x.nzind), abs2(x.nzval))

abstract _BinOp

type _AddOp <: _BinOp end

_eval(::_AddOp, x, y) = x + y
_eval1(::_AddOp, x) = x
_eval2(::_AddOp, y) = y

type _SubOp <: _BinOp end

_eval(::_SubOp, x, y) = x - y
_eval1(::_SubOp, x) = x
_eval2(::_SubOp, y) = -y


function _mapbinop{Tx,Ty}(op::_BinOp, x::GenericSparseVector{Tx}, y::GenericSparseVector{Ty})
    R = typeof(_eval(op, zero(Tx), zero(Ty)))
    n = length(x)
    length(y) == n || throw(DimensionMismatch())

    xnzind = x.nzind
    xnzval = x.nzval
    ynzind = y.nzind
    ynzval = y.nzval
    mx = length(xnzind)
    my = length(ynzind)

    ix = 1
    iy = 1
    rind = Array(Int, 0)
    rval = Array(R, 0)
    sizehint!(rind, mx + my)
    sizehint!(rval, mx + my)

    @inbounds while ix <= mx && iy <= my
        jx = xnzind[ix]
        jy = ynzind[iy]

        if jx == jy
            v = _eval(op, xnzval[ix], ynzval[iy])
            if v != zero(v)
                push!(rind, jx)
                push!(rval, v)
            end
            ix += 1
            iy += 1
        elseif jx < jy
            push!(rind, jx)
            push!(rval, _eval1(op, xnzval[ix]))
            ix += 1
        else
            push!(rind, jy)
            push!(rval, _eval2(op, ynzval[iy]))
            iy += 1
        end
    end

    @inbounds while ix <= mx
        push!(rind, xnzind[ix])
        push!(rval, _eval1(op, xnzval[ix]))
        ix += 1
    end

    @inbounds while iy <= my
        push!(rind, ynzind[iy])
        push!(rval, _eval2(op, ynzval[iy]))
        iy += 1
    end

    return SparseVector(n, rind, rval)
end

function _mapbinop{Tx,Ty}(op::_BinOp, x::StridedVector{Tx}, y::GenericSparseVector{Ty})
    R = typeof(_eval(op, zero(Tx), zero(Ty)))
    n = length(x)
    length(y) == n || throw(DimensionMismatch())

    ynzind = y.nzind
    ynzval = y.nzval
    m = length(ynzind)

    dst = Array(R, n)
    ii = 1
    @inbounds for i = 1:m
        j = ynzind[i]
        while ii < j
            dst[ii] = _eval1(op, x[ii])
            ii += 1
        end
        # at this point: ii == j
        dst[j] = _eval(op, x[j], ynzval[i])
        ii += 1
    end

    @inbounds while ii <= n
        dst[ii] = _eval1(op, x[ii])
        ii += 1
    end
    return dst
end

function _mapbinop{Tx,Ty}(op::_BinOp, x::GenericSparseVector{Tx}, y::StridedVector{Ty})
    R = typeof(_eval(op, zero(Tx), zero(Ty)))
    n = length(x)
    length(y) == n || throw(DimensionMismatch())

    xnzind = x.nzind
    xnzval = x.nzval
    m = length(xnzind)

    dst = Array(R, n)
    ii = 1
    @inbounds for i = 1:m
        j = xnzind[i]
        while ii < j
            dst[ii] = _eval2(op, y[ii])
            ii += 1
        end
        # at this point: ii == j
        dst[j] = _eval(op, xnzval[i], y[j])
        ii += 1
    end

    @inbounds while ii <= n
        dst[ii] = _eval2(op, y[ii])
        ii += 1
    end
    return dst
end


+ (x::GenericSparseVector, y::GenericSparseVector) = _mapbinop(_AddOp(), x, y)
- (x::GenericSparseVector, y::GenericSparseVector) = _mapbinop(_SubOp(), x, y)
+ (x::StridedVector, y::GenericSparseVector) = _mapbinop(_AddOp(), x, y)
- (x::StridedVector, y::GenericSparseVector) = _mapbinop(_SubOp(), x, y)
+ (x::GenericSparseVector, y::StridedVector) = _mapbinop(_AddOp(), x, y)
- (x::GenericSparseVector, y::StridedVector) = _mapbinop(_SubOp(), x, y)

.+ (x::GenericSparseVector, y::GenericSparseVector) = (x + y)
.- (x::GenericSparseVector, y::GenericSparseVector) = (x - y)

.+ (x::StridedVector, y::GenericSparseVector) = (x + y)
.- (x::StridedVector, y::GenericSparseVector) = (x - y)

.+ (x::GenericSparseVector, y::StridedVector) = (x + y)
.- (x::GenericSparseVector, y::StridedVector) = (x - y)


Base.sum(x::GenericSparseVector) = sum(x.nzval)
Base.sumabs(x::GenericSparseVector) = sumabs(x.nzval)
Base.sumabs2(x::GenericSparseVector) = sumabs2(x.nzval)

Base.vecnorm(x::GenericSparseVector, p::Real=2) = vecnorm(x.nzval, p)
