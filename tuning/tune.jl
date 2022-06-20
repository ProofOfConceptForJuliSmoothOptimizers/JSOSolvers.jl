using Pkg
using Distributed
using SolverTuning
using JSON

# 1. Launch workers
init_workers(; exec_flags="--project=$(@__DIR__)")

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
problems = Iterators.filter(p -> unconstrained(p) &&  100 ≤ get_nvar(p) ≤ 1000 && get_minimize(p), problems)

# 4. get parameters from solver:
solver = LBFGSSolver(first(problems))
x = solver.p

# 5. Define a BBModel problem:

# 5.1 define a function that executes your solver. It must take an nlp followed by a vector of real values:
@everywhere function solver_func(nlp::AbstractNLPModel, p::NamedTuple)
  return lbfgs(nlp, p; verbose=0, max_time=60.0)
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
# problems = [p for (i,p) in zip(1:40, problems)]
bbmodel = BBModel(x, solver_func, aux_func, problems;lvar=Real[1, false, T(0.0), 10], uvar=Real[100, true, T(0.9999), 30])

best_params, param_opt_problem = solve_bb_model(bbmodel;lb_choice=:C,
display_all_eval = true,
# max_time = 300,
# max_bb_eval =100,
display_stats = ["BBE", "SOL", "CONS_H", "TIME", "OBJ"],
)

# worker_data = Dict(w_id => [sum(sum(get_times(p);init=0.0) for p in iteration;init=0.0) for iteration in data] for (w_id, data) in param_opt_problem.worker_data)
# file_path = joinpath(@__DIR__, "plots", "combine", "lbfgs_worker_times_combine_algorithm.json")
# open(file_path, "w") do f
#   JSON.print(f, worker_data)
# end