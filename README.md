# SparseVectors

A Julia package to support sparse vectors.

[![Build Status](https://travis-ci.org/JuliaSparse/SparseVectors.jl.svg?branch=master)](https://travis-ci.org/JuliaSparse/SparseVectors.jl)

## Overview

Sparse data has become increasingly common in machine learning and related areas. For example, in document analysis, each document is often represented as a *sparse vector*, which each entry represents the number of occurrences of a certain word. However, the support of sparse vectors remains quite limited in Julia base.

This package provides two types ``SparseVector`` and ``SparseVectorView`` and a series of methods to work with *sparse vectors*. Specifically, this package provides the following functionalities:

- Construction of sparse vectors, either with given non-zero entries or randomly.
- Get a view of a column in a sparse matrix (of CSC format), or a view of a range of columns.
- Specialized arithmetic functions on sparse vectors, *e.g.* ``+``, ``-``, ``*``, etc.
- Specialized math functions on sparse vectors, *e.g.* ``abs``, ``abs2``, ``exp``, ``sin``, etc.
- Specialized reduction functions on sparse vectors, *e.g.* ``sum``, ``vecnorm``, etc.
- Specialized linear algebraic functions, *e.g.* ``axpy!``, ``dot``, ``A * x``, ``At_mul_B``, etc.

**Note:** Many of the functionalities implemented in this package may be migrated to Julia Base in `v0.5` development cycle.


## Types

This package defines two types.

- ``SparseVector``: a sparse vector that owns its memory
- ``SparseVectorView``: a view of external data as a sparse vector.

The formal definition of these types are listed below:

```julia
immutable SparseVector{Tv,Ti<:Integer} <: AbstractSparseVector{Tv,Ti}
    n::Int              # the number of elements
    nzind::Vector{Ti}   # the indices of nonzeros
    nzval::Vector{Tv}   # the values of nonzeros
end

typealias CVecView{T} ContiguousView{T,1,Vector{T}}

immutable SparseVectorView{Tv,Ti<:Integer} <: AbstractSparseVector{Tv,Ti}
    n::Int                  # the number of elements
    nzind::CVecView{Ti}     # the indices of nonzeros
    nzval::CVecView{Tv}     # the values of nonzeros
end
```

## Constructors

An instance of ``SparseVector`` can be constructed as follows:
```julia
SparseVector(n, nzind, nzval)  # constructs an instance by providing all fields
```
Here, all inputs will be used as fields as they are. The constructor will ensure that ``length(nzind) == length(nzval)``. However, it will *NOT* examine the elements of ``nzind`` (*e.g.* the indexes are sorted, without duplication, and within the range ``1:n``).

The package also provides a `sparsevector` function to construct a sparse vector in a variety of ways:

#### zero sparse vector

```julia
# construct a zerp sparse vector of length `len` and element type `T`
sparsevector(T, len)  
```

#### with given lists of non-zero entries

The following methods construct a sparse vector of length `len`, with:

- non-zero indices, given by an integer vector `I`
- non-zero values, given by `V` (either a vector or a number)

**Note:**

- When `len` is omitted, it is determined by `maximum(I)`.
- Multiple values are allowed to be corresponding to the same index. These values are combined with a binary function/functor `combine`. When it is omitted, `AddFun` is used by default, meaning that the values are summed.

```julia
sparsevector(I, V, len, combine)
sparsevector(I, V, combine)
sparsevector(I, V, len)
sparsevector(I, V)
```

#### from an associative collection (e.g. *Dict*)

```julia
sparsevector(a, len)  # a is an instance of `Associative{Ti<:Integer, Tv}`
sparsevector(a)  # length inferred as the maximum index
```

#### random sparse vector

```julia
sprand(n, p)       # construct a random sparse vector with length n and density p
                   # the non-zero values are generated using rand(nnz)

sprand(n, p, T)    # construct a random sparse vector of element type T
                   # the non-zero values are generated using rand(T, nnz)

sprandn(n, p)      # construct a random sparse vector,
                   # where values follow standard Normal distribution

sprand(n, p, rfn)  # construct a random sparse vector,
                   # where the non-zero values are generated using rfn
```


## Basic methods

Like other array types, `SparseVector` and `SparseVectorView` support all the basic methods for arrays:

```julia
eltype(x)   # get the element type
ndims(x)    # get the number of dimensions (1)
length(x)   # get the length
size(x)     # get the size, i.e. (length(x),)

x[i]        # get the i-th element of x
x[i] = v    # set the i-th element of x to v
```

They also provide methods for extracting internal data structures:

```julia
nnz(x)          # the number of stored entries
countnz(x)      # count the actual number of nonzero entries
nonzeroinds(x)  # get the vector of indices of non-zero values  
nonzeros(x)     # get the vector of non-zero values
```

## Conversion

The package supports conversion between sparse vectors and other types of arrays.

```julia
convert(SparseVector, s)  # convert s to an instance of SparseVector
                          # s can be an instance of Vector or SparseMatrixCSC

convert(SparseVector{Tv}, s)  # convert the element-type of s to Tv
                              # where s is an instance of SparseVector

convert(SparseVector{Tv,Ti}, s)  # convert the element-type of s to Tv,
                                 # and the index-type of s to Ti,
                                 # where s is an instance of SparseVector

convert(SparseMatrixCSC, v)  # convert a sparse vector v to a sparse matrix
                             # with a single column

convert(SparseMatrixCSC{Tv}, v)
convert(SparseMatrixCSC{Tv,Ti}, v)
```

## Views

The package provides methods to obtain views of sparse vectors

```julia
view(x)   # construct a SparseVectorView instance as a view of x
          # where x is an instance of SparseVector

view(A, :, i)   # construct a view of the i-th column of X
                # where X is an instance of SparseMatrixCSC
                # returns a instance of SparseVectorView

unsafe_colrange(A, i1:i2)  # construct an unsafe view of a range of columns
                           # i.e. from the i1-th to i2-th column.
                           # returns an instance of SparseMatrixCSC
```

**Note:** `unsafe_colrange` uses ``pointer_to_array`` to obtain the internal vectors, and therefore the returned array should only be used within the local scope.


## Math Functions

The package implement a number of specialized math functions for sparse vectors.

#### Arithmetics

```julia
scale!(x, c)   # x <- x * c, where c is a scalar
scale!(c, x)   # i.e. scale!(x, c)
scale(x, c)    # returns x * c
scale(c, x)    # i.e. scale(x, c)

x * c, x .* c  # multiple x and a scalar c
c * x, c .* x  # i.e. x * c

- x            # negate x
x + y, x .+ y  # add x and y, x and y can be either dense or sparse
x - y, x .- y  # subtract y from x, x and y can be either dense or sparse
x .* y         # multiply x and y (element-wise),
               # x and y can be either dense or sparse

axpy!(a, x, y)  # y <- y + a * x
                # a: a scalar number
                # x: a sparse vector
                # y: a dense vector
```

#### Element-wise math functions

```julia
# Input: (sparse, sparse) --> Output: sparse
# Input: (sparse, dense)  --> Output: dense
# Input: (dense, sparse)  --> Output: sparse

max(x, y), min(x, y)
complex(x, y)

# zero-preserving functions
#   Input: sparse --> Output: sparse

abs(x), abs2(x)
real(x), imag(x), conj(x)
floor(x), ceil(x), trunc(x), round(x)
log1p(x), expm1(x),
sin(x), tan(x), sinpi(x), sind(x), tand(x),
asin(x), atan(x), asind(x), atand(x),
sinh(x), tanh(x), asinh(x), atanh(x)

# Non-zero-preserving functions
#   Input: sparse --> Output: dense

exp(x), exp2(x), exp10(x), log(x), log2(x), log10(x),
cos(x), csc(x), cot(x), sec(x), cospi(x),
cosd(x), cscd(x), cotd(x), secd(x),
acos(x), acot(x), acosd(x), acotd(x),
cosh(x), csch(x), coth(x), sech(x),
acsch(x), asech(x)
```


## Reduction

```julia
sum(x)      # Compute the sum of elements
sumabs(x)   # Compute the sum of absolute values
sumabs2(x)  # Compute the sum of squared absolute values
maximum(x)  # Compute the maximum of elements (including zeros)
minimum(x)  # Compute the minimum of elements (including zeros)
maxabs(x)   # Compute the maximum of absolute values
minabs(x)   # Compute the minimum of absolute values

vecnorm(x, p=2)  # Compute the p-th order vector-norm

dot(x, y)   # Compute the dot product between x and y
            # x and y can be either dense or sparse vectors
```

## Linear Algebra: Matrix-vector product

```julia
# Note: the product is dense iff A is dense
A * x                     # A * x (matrix-vector product)
At_mul_B(A, x)            # A' * x, without explicitly transposing A

# If you want to get a dense result even when both A and x are sparse
# then you can write:
densemv(A, x)               # A * x --> dense vector
densemv(A, x; trans='N')    # A * x, as above
densemv(A, x; trans='T')    # transpose(A) * x -> dense vector
densemv(A, x; trans='C')    # ctranspose(A) * x -> dense vector

# Note: the following functions are only for cases where y is a strided vector
A_mul_B!(y, A, x)         # y <- A * x
A_mul_B!(a, A, x, b, y)   # y <- a * A * x + b * y
At_mul_B(y, A, x)         # y <- A' * x
At_mul_B!(a, A, x, b, y)  # y <- a * A' * x + b * y
```
