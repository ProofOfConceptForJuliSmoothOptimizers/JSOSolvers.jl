using Distributed
using SolverTuning
using JSON
using Dates

# 1. Launch workers

init_workers(;nb_nodes=15, exec_flags="--project=$(@__DIR__)")

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
  mem = value(params.mem)
  scaling = value(params.scaling)
  τ₀ = value(params.τ₀)
  τ₁ = value(params.τ₁)
  return lbfgs(nlp; mem=mem, scaling=scaling, τ₀=τ₀, τ₁=τ₁, verbose=0, max_time=30.0)
end

# 5.2 Define a function that takes a ProblemMetric object. This function must return one real number.

@everywhere function f(p_metrics::Vector{ProblemMetrics})
  return time_only(p_metrics;penalty=50.0)
end

function create_json(param_opt_problem::ParameterOptimizationProblem, filename::String)
  worker_data = Dict(w_id => [sum(sum(get_times(p);init=0.0) for p in iteration;init=0.0) for iteration in data] for (w_id, data) in param_opt_problem.worker_data)
  file_path = joinpath(@__DIR__, "plots", "combine", "$(filename)_$(now()).json")
  open(file_path, "w") do f
    JSON.print(f, worker_data)
  end
end

function main()
  # 3. Setup problems
  @info "Defining problem set:"
  T = Float64
  I = Int64
  N = 3
  problems = (eval(p)(type=Val(T)) for p ∈ filter(x -> x ∉ [:ADNLPProblems, :spmsrtls, :scosine], names(OptimizationProblems.ADNLPProblems)))
  problems = Iterators.filter(p -> unconstrained(p) &&  1 ≤ get_nvar(p) ≤ 500 && get_minimize(p), problems)

  # 4. get parameters from solver:
  @info "Defining bb model:"

  params = LBFGSParameterSet{T, I}()
  # 5.4 Define the BBModel:
  # problems = [p for (_, p) in zip(1:N, problems)]
  problems = collect(problems)
  bbmodel = BBModel(params, problems, solver_func, f;)
  
  @info "Starting NOMAD:"
  best_params, param_opt_problem = solve_bb_model(bbmodel;lb_choice=:C,
  display_all_eval = true,
  # max_time = 60,
  # max_bb_eval = 3,
  display_stats = ["BBE", "SOL", "TIME", "OBJ"],
  )

  # create_json(param_opt_problem, "lbfgs_float64")
  rmprocs(workers())
end

main()
