---
title: SNOPT
# draft: true
tags:
  - software libraries
  - optimization
type: note
foam_template:
  description: Standard blank notes with YAML frontmatter
---

* SNOPT (Sparse Nonlinear OPTimizer)
* Stores [[sparse-matrices|sparse matrices]] in [[compressed-sparse-columns|CSC]] format
  * The [[jacobian-matrix|Jacobian matrix]] in constrained optimization often has many more rows than columns (i.e. many constraints and relatively few variables), so it's efficient to store and access **columns**.
  * SNOPT solves [[sqp|SQP]] subproblems using **quasi-Newton approximations**, so efficient access to columns of the Jacobian and [[hessian|Hessian]] is important during factorization and linear solves.
