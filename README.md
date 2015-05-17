# SparseVectors

A Julia package to support sparse vectors.

[![Build Status](https://travis-ci.org/lindahua/SparseVectors.jl.svg?branch=master)](https://travis-ci.org/lindahua/SparseVectors.jl)

## Overview

Sparse data has become increasingly common in machine learning and related areas. For example, in document analysis, each document is often represented as a *sparse vector*, which each entry represents the number of occurrences of a certain word. However, the support of sparse vectors remains quite limited in Julia base.

This package provides two types ``SparseVector`` and ``SparseVectorView`` and a series of methods to work with *sparse vectors*. Specifically, this package provides the following functionalities:

- Construction of sparse vectors
- Get a view of a column in a sparse matrix (of CSC format), or a view of a range of columns.
- Specialized arithmetic functions on sparse vectors, *e.g.* ``+``, ``-``, ``*``, etc.
- Specialized reduction functions on sparse vectors, *e.g.* ``sum``, ``vecnorm``, etc.
