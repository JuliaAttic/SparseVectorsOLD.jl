# SparseVectors

A Julia package to support sparse vectors.

[![Build Status](https://travis-ci.org/lindahua/SparseVectors.jl.svg?branch=master)](https://travis-ci.org/lindahua/SparseVectors.jl)
[![Coverage Status](https://img.shields.io/coveralls/lindahua/SparseVectors.jl.svg)](https://coveralls.io/r/lindahua/SparseVectors.jl)

## Overview

Sparse data has become increasingly common in machine learning and related areas. For example, in document analysis, each document is often represented as a *sparse vector*, which each entry represents the number of occurrences of a certain word. However, the support of sparse vectors remains quite limited in Julia base.

This package provides two types ``SparseVector`` and ``SparseVectorView`` and a series of methods to work with *sparse vectors*. Specifically, this package provides the following functionalities:

- Construction of sparse vectors
- Get a view of a column in a sparse matrix (of CSC format), or a view of a range of columns.
- Specialized arithmetic functions on sparse vectors, *e.g.* ``+``, ``-``, ``*``, etc.
- Specialized reduction functions on sparse vectors, *e.g.* ``sum``, ``vecnorm``, etc.


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

typealias GenericSparseVector{Tv,Ti} Union(SparseVector{Tv,Ti}, SparseVectorView{Tv,Ti})
```

## Methods

In addition to the methods for all subtypes of ``AbstractVector`` (*e.g.* ``length``, ``size``, ``ndims``, ``eltype``, ``isempty``, etc), this package provides specialized implementation of the following methods for both ``SparseVector`` and ``SparseVectorView``:

```julia
## Construction
SparseVector(n, nzind, nzval)  # constructs an instance by providing all fields

SparseVector(n)          # construct a zero sparse vector of length n, Tv == Float64
SparseVector(Tv, n)      # construct a zero sparse vector of length n, with element type Tv
SparseVector(Tv, Ti, n)  # construct a zero sparse vector of length n,
                         # with element type Tv, and index type Ti

SparseVector(n, src)   # construct a sparse vector of length n, with entries from src
                       # src can be either of the following:
                       # - an index to value map
                       # - a sequence of (index, value) tuple

## Random Construction
sprand(n, p)       # construct a random sparse vector with length n and density p
                   # the non-zero values are generated using rand(nnz)

sprand(n, p, T)    # construct a random sparse vector of element type T
                   # the non-zero values are generated using rand(T, nnz)

sprandn(n, p)      # construct a random sparse vector,
                   # where values follow standard Normal distribution

sprand(n, p, rfn)  # construct a random sparse vector,
                   # where the non-zero values are generated using rfn


## Basics
nnz(x)       # the number of stored entries
countnz(x)   # count the actual number of nonzero entries
nonzeros(x)  # a vector of stored entries (i.e. x.nzval)

full(x)      # construct a full vector, of type Vector{Tv}

copy(x)      # return a copy of x, of type SparseVector{Tv,Ti}

vec(x)       # return x itself (because `x` itself is a vector)

x[i]         # get the i-th element of x

x[i] = v     # set the i-th element of x to v


## Conversion
convert(SparseVector, s)  # convert s to an instance of SparseVector
                          # s can be an instance of Vector or SparseMatrixCSC

convert(SparseVector{Tv}, s)  # convert the element-type of s to Tv
                              # where s is an instance of SparseVector

convert(SparseVector{Tv, Ti}, s)  # convert the element-type of s to Tv,
                                  # and the index-type of s to Ti,
                                  # where s is an instance of SparseVector

## Element-wise Computation
scale!(x, c)   # x <- x * c, where c is a scalar
scale!(c, x)   # i.e. scale!(x, c)
scale(x, c)    # returns x * c
scale(c, x)    # i.e. scale(x, c)

x * c, x .* c  # multiple x and a scalar c
c * x, c .* x  # i.e. x * c

x + y, x .+ y  # add x and y, x and y can be either dense or sparse
x - y, x .- y  # subtract y from x, x and y can be either dense or sparse

axpy!(a, x, y)  # y <- y + a * x
                # a: a scalar number
                # x: a sparse vector
                # y: a dense vector
                # This operation is very common in machine learning context

## Reduction
sum(x)      # Compute the sum of elements
sumabs(x)   # Compute the sum of absolute values
sumabs2(x)  # Compute the sum of squared absolute values

vecnorm(x, p=2)  # Compute the p-th order vector-norm

dot(x, y)   # Compute the dot product between x and y
            # x and y can be either dense or sparse vectors


## Matrix-vector products between
# a strided dense matrix A and a sparse vector x

A * x                     # matrix-vector product
A_mul_B!(y, A, x)         # y <- A * x
A_mul_B!(a, A, x, b, y)   # y <- a * A * x + b * y

At_mul_B(A, x)            # A' * x, without explicitly transposing A
At_mul_B(y, A, x)         # y <- A' * x
At_mul_B!(a, A, x, b, y)  # y <- a * A' * x + b * y            
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
