---
title: Compressed Sparse Columns
# draft: true
alias: CSC
tags:
  - linear algebra
  - computer science
type: note
foam_template:
  description: Standard blank notes with YAML frontmatter
---

* Stores [[sparse matrices]] as three arrays:
  * `val`: the non-zero values, column by column.
  * `row_ind`: the row indices corresponding to each value in `val`.
  * `col_ptr`: an array such that `col_ptr[j]` is the index in `val` where column `j` starts, and `col_ptr[j+1]` is where it ends.
* This lets you efficiently iterate over all non-zeros in a given column — ideal for operations like \( A^T y \) or Jacobian transposes that are common in [[sequential quadratic programming|SQP]] methods.
