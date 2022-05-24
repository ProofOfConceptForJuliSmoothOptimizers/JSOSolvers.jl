using Pkg
using Distributed
using SolverTuning

# 1. Launch workers
init_workers(;exec_flags="--project=$(@__DIR__)")

# 2. make modules available to all workers:
@everywhere begin
  using JSOSolvers,
  SolverTuning,
  OptimizationProblems,
  OptimizationProblems.ADNLPProblems,
  NLPModels,
  ADNLPModels,
  Statistics,
  BBModels
end


# 3. Setup problems
T = Float64
problems = (eval(p)(type=Val(T)) for p ∈ filter(x -> x != :ADNLPProblems && x != :scosine, names(OptimizationProblems.ADNLPProblems)))
problems = Iterators.filter(p -> unconstrained(p) &&  1 ≤ get_nvar(p) ≤ 1000 && get_minimize(p), problems)

# 4. get parameters from solver:
solver = TrunkSolver(first(problems))
x = solver.p

# 5. Define a BBModel problem:

# 5.1 define a function that executes your solver. It must take an nlp followed by a vector of real values:
@everywhere function solver_func(nlp::AbstractNLPModel, p::NamedTuple)
  return trunk(nlp, p; verbose=0, max_time=60.0)
end

# 5.2 Define a function that takes a ProblemMetric object. This function must return one real number.

@everywhere function aux_func(p_metric::ProblemMetrics)
  median_time = median(get_times(p_metric))
  memory = get_memory(p_metric)
  solved = get_solved(p_metric)
  counters = get_counters(p_metric)

  return median_time + memory + counters.neval_obj + (Float64(!solved) * 5.0 * median_time)
end

# 5.4 Define the BBModel:
problems = collect(problems)
# problems = [p for (i,p) in zip(1:3, problems)]
lvar = Real[false, 20, 5, T(0.0), T(0.0), zero(T), zero(T), one(T)]
uvar = Real[true, 30, 15, T(1.0), T(1.0), T(0.9499), one(T), T(2.0)]
lcon = [eps(T)]
ucon = [T(Inf)]
@everywhere function c(x)
  return [x[8] - x[7];]
end
bbmodel = BBModel(x, solver_func, aux_func, c, lcon, ucon, problems;lvar=lvar, uvar=uvar)

solve_with_nomad(bbmodel;
display_all_eval = true,
# max_time = 300,
# max_bb_eval =3,
display_stats = ["BBE", "EVAL", "SOL", "OBJ"],
)

