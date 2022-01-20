using BenchmarkTools
# stdlib
using LinearAlgebra, Logging, Printf, SparseArrays

# JSO packages
using Krylov, LinearOperators, NLPModels, NLPModelsModifiers, NLPModelsJuMP, SolverBenchmark, SolverCore, SolverTools, ADNLPModels, SolverTest, JSOSolvers, OptimizationProblems, OptimizationProblems.PureJuMP

problems = (MathOptNLPModel(eval(p)(),name=string(p)) for p ∈ filter(x -> x != :PureJuMP, names(OptimizationProblems.PureJuMP)))
problems = Iterators.filter(p -> unconstrained(p) &&  5 ≤ get_nvar(p) ≤ 1000 && get_minimize(p), problems)

const SUITE = BenchmarkGroup()
SUITE[:lbfgs] = BenchmarkGroup()

for nlp ∈ problems
  SUITE[:lbfgs][get_name(nlp)] = @benchmarkable lbfgs($nlp)
end

