using BenchmarkTools
# stdlib
using LinearAlgebra, Logging, Printf, SparseArrays

# JSO packages
using Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverBenchmark, SolverCore, SolverTools, ADNLPModels, SolverTest, JSOSolvers, CUTEst

function run_cutest_problem(nlp::CUTEstModel)
  lbfgs(nlp)
  finalize(nlp)
end

problem_names = CUTEst.select(;only_free_var=true, max_con=0, min_var=5, max_var=500)
cutest_problems = ((p, CUTEstModel(p)) for p in problem_names)

const SUITE = BenchmarkGroup()
SUITE[:cutest_lbfgs] = BenchmarkGroup()

for (p, nlp) ∈ cutest_problems
  SUITE[:cutest_lbfgs][p] = @benchmarkable run_cutest_problem($nlp)
  finalize(nlp)
end

