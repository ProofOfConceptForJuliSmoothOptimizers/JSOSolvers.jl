using Distributed
using SolverTuning
using JSON

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
  BBModels,
  SolverCore,
  SolverParameters
end

# 5. Define a BBModel problem:

# 5.1 define a function that executes your solver. It must take an nlp followed by an AbstractParameterSet
@everywhere function solver_func(nlp::AbstractNLPModel, params::LBFGSParameterSet)
  solver = LBFGSSolver(nlp; mem=value(params.mem), scaling=value(params.scaling))
  stats = GenericExecutionStats(nlp)
  x = nlp.meta.x0
  return JSOSolvers.SolverCore.solve!(solver, params, nlp, stats; x = x, verbose=0, max_time = 5.0)
end

# 5.2 Define a function that takes a ProblemMetric object. This function must return one real number.

@everywhere function aux_func(p_metric::ProblemMetrics)
  median_time = median(get_times(p_metric))
  memory = get_memory(p_metric)
  solved = get_solved(p_metric)
  counters = get_counters(p_metric)
  # convert time to seconds:
  median_time /= 1.0e9
  # convert memory to Mb:
  memory /= (2^20)

  return median_time * memory + counters.neval_obj + (Float64(!solved) * 5.0 * median_time)
end


function create_json(param_opt_problem::ParameterOptimizationProblem)
  worker_data = Dict(w_id => [sum(sum(get_times(p);init=0.0) for p in iteration;init=0.0) for iteration in data] for (w_id, data) in param_opt_problem.worker_data)
  file_path = joinpath(@__DIR__, "plots", "combine", "lbfgs_worker_times_combine_algorithm.json")
  open(file_path, "w") do f
    JSON.print(f, worker_data)
  end
end

function main()
  # 3. Setup problems
  @info "Defining problem set:"
  T = Float64
  R = T
  I = Int64
  N = 30
  problems = (eval(p)(type=Val(T)) for p ∈ filter(x -> x != :ADNLPProblems, names(OptimizationProblems.ADNLPProblems)))
  problems = Iterators.filter(p -> unconstrained(p) &&  5 ≤ get_nvar(p) ≤ 1000 && get_minimize(p), problems)

  # 4. get parameters from solver:
  @info "Defining bb model:"

  params = LBFGSParameterSet{R, I}()
  # 5.4 Define the BBModel:
  # problems = [p for (_, p) in zip(1:N, problems)]
  
  bbmodel = BBModel(params, solver_func, aux_func, collect(problems);)
  
  @info "Starting NOMAD:"
  best_params, param_opt_problem = solve_bb_model(bbmodel;lb_choice=:C,
  display_all_eval = true,
  # max_time = 300,
  max_bb_eval = 200,
  display_stats = ["BBE", "SOL", "CONS_H", "TIME", "OBJ"],
  )

  # create_json(param_opt_problem)
end

main()

