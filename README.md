# SparseData

A Julia package to support working with sparse data sets (e.g. text corpus).

[![Build Status](https://travis-ci.org/lindahua/SparseData.jl.svg?branch=master)](https://travis-ci.org/lindahua/SparseData.jl)

## Overview

Sparse data has become increasingly common in machine learning and related areas. However, the support of sparse data in Julia base remains quite limited (*e.g.* it does not provide 1D sparse vectors and views of sparse matrices). The primary purpose of this package is to mitigate this situation by providing *sparse vectors*, *sparse views*, and functions to construct and manipulate sparse data structures.
