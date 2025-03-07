export TronSolverNLS

const tronls_allowed_subsolvers = [:cgls, :crls, :lsqr, :lsmr]

tron(nls::AbstractNLSModel; variant = :GaussNewton, kwargs...) = tron(Val(variant), nls; kwargs...)

"""
    slope, qs = compute_As_slope_qs!(As, A, s, Fx)

Compute `slope = dot(As, Fx)` and `qs = dot(As, As) / 2 + slope`. Use `As` to store `A * s`.
"""
function compute_As_slope_qs!(
  As::AbstractVector{T},
  A::Union{AbstractMatrix, AbstractLinearOperator},
  s::AbstractVector{T},
  Fx::AbstractVector{T},
) where {T <: Real}
  As .= A * s
  slope = dot(As, Fx)
  qs = dot(As, As) / 2 + slope
  return slope, qs
end

"""
    tron(nls; kwargs...)

A pure Julia implementation of a trust-region solver for bound-constrained
nonlinear least-squares problems:

    min ½‖F(x)‖²    s.t.    ℓ ≦ x ≦ u

For advanced usage, first define a `TronSolverNLS` to preallocate the memory used in the algorithm, and then call `solve!`:
    solver = TronSolverNLS(nls; kwargs...)
    solve!(solver, nls; kwargs...)

# Arguments
- `nls::AbstractNLSModel{T, V}` represents the model to solve, see `NLPModels.jl`.
The keyword arguments may include
- `subsolver_logger::AbstractLogger = NullLogger()`: subproblem's logger.
- `x::V = nlp.meta.x0`: the initial guess.
- `subsolver::Symbol = :lsmr`: `Krylov.jl` method used as subproblem solver, see `JSOSolvers.tronls_allowed_subsolvers` for a list.
- `μ₀::T = T(1e-2)`: algorithm parameter in (0, 0.5).
- `μ₁::T = one(T)`: algorithm parameter in (0, +∞).
- `σ::T = T(10)`: algorithm parameter in (1, +∞).
- `max_eval::Int = -1`: maximum number of objective function evaluations.
- `max_time::Float64 = 30.0`: maximum time limit in seconds.
- `max_iter::Int = typemax(Int)`: maximum number of iterations.
- `max_cgiter::Int = 50`: subproblem iteration limit.
- `cgtol::T = T(0.1)`: subproblem tolerance.
- `atol::T = √eps(T)`: absolute tolerance.
- `rtol::T = √eps(T)`: relative tolerance, the algorithm stops when ‖∇f(xᵏ)‖ ≤ atol + rtol * ‖∇f(x⁰)‖.
- `Fatol::T = √eps(T)`: absolute tolerance on the residual.
- `Frtol::T = eps(T)`: relative tolerance on the residual, the algorithm stops when ‖F(xᵏ)‖ ≤ Fatol + Frtol * ‖F(x⁰)‖.
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration.

The keyword arguments of `TronSolverNLS` are passed to the [`TRONTrustRegion`](https://github.com/JuliaSmoothOptimizers/SolverTools.jl/blob/main/src/trust-region/tron-trust-region.jl) constructor.

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
The callback is called at each iteration.
The expected signature of the callback is `callback(nlp, solver, stats)`, and its output is ignored.
Changing any of the input arguments will affect the subsequent iterations.
In particular, setting `stats.status = :user` will stop the algorithm.
All relevant information should be available in `nlp` and `solver`.
Notably, you can access, and modify, the following:
- `solver.x`: current iterate;
- `solver.gx`: current gradient;
- `stats`: structure holding the output of the algorithm (`GenericExecutionStats`), which contains, among other things:
  - `stats.dual_feas`: norm of current gradient;
  - `stats.iter`: current iteration counter;
  - `stats.objective`: current objective function value;
  - `stats.status`: current status of the algorithm. Should be `:unknown` unless the algorithm attained a stopping criterion. Changing this to anything will stop the algorithm, but you should use `:user` to properly indicate the intention.
  - `stats.elapsed_time`: elapsed time in seconds.

# References
This is an adaptation for bound-constrained nonlinear least-squares problems of the TRON method described in

    Chih-Jen Lin and Jorge J. Moré, *Newton's Method for Large Bound-Constrained
    Optimization Problems*, SIAM J. Optim., 9(4), 1100–1127, 1999.
    DOI: 10.1137/S1052623498345075

# Examples
```jldoctest; output = false
using JSOSolvers, ADNLPModels
F(x) = [x[1] - 1.0; 10 * (x[2] - x[1]^2)]
x0 = [-1.2; 1.0]
nls = ADNLSModel(F, x0, 2, zeros(2), 0.5 * ones(2))
stats = tron(nls)
# output
"Execution stats: first-order stationary"
```

```jldoctest; output = false
using JSOSolvers, ADNLPModels
F(x) = [x[1] - 1.0; 10 * (x[2] - x[1]^2)]
x0 = [-1.2; 1.0]
nls = ADNLSModel(F, x0, 2, zeros(2), 0.5 * ones(2))
solver = TronSolverNLS(nls)
stats = solve!(solver, nls)
# output
"Execution stats: first-order stationary"
```
"""
mutable struct TronSolverNLS{T, V <: AbstractVector{T}} <: AbstractOptimizationSolver
  x::V
  xc::V
  gx::V
  gt::V
  gpx::V
  tr::TRONTrustRegion{T, V}
  Fc::V
  Av::V
  Atv::V
end

function TronSolverNLS(
  nlp::AbstractNLSModel{T, V};
  max_radius::T = min(one(T) / sqrt(2 * eps(T)), T(100)),
  kwargs...,
) where {T, V <: AbstractVector{T}}
  nvar = nlp.meta.nvar
  nequ = nlp.nls_meta.nequ
  x = V(undef, nvar)
  xc = V(undef, nvar)
  gx = V(undef, nvar)
  gt = V(undef, nvar)
  gpx = V(undef, nvar)
  tr = TRONTrustRegion(gt, min(one(T), max_radius - eps(T)); max_radius = max_radius, kwargs...)

  Fc = V(undef, nequ)
  Av = V(undef, nequ)
  Atv = V(undef, nvar)

  return TronSolverNLS{T, V}(x, xc, gx, gt, gpx, tr, Fc, Av, Atv)
end

function SolverCore.reset!(solver::TronSolverNLS)
  solver.tr.good_grad = false
  solver
end
SolverCore.reset!(solver::TronSolverNLS, ::AbstractNLPModel) = reset!(solver)

@doc (@doc TronSolverNLS) function tron(
  ::Val{:GaussNewton},
  nlp::AbstractNLSModel{T, V};
  x::V = nlp.meta.x0,
  kwargs...,
) where {T, V}
  dict = Dict(kwargs)
  subsolver_keys = intersect(keys(dict), tron_keys)
  subsolver_kwargs = Dict(k => dict[k] for k in subsolver_keys)
  solver = TronSolverNLS(nlp; subsolver_kwargs...)
  for k in subsolver_keys
    pop!(dict, k)
  end
  return solve!(solver, nlp; x = x, dict...)
end

function SolverCore.solve!(
  solver::TronSolverNLS{T, V},
  nlp::AbstractNLSModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  subsolver_logger::AbstractLogger = NullLogger(),
  x::V = nlp.meta.x0,
  subsolver::Symbol = :lsmr,
  μ₀::Real = T(1e-2),
  μ₁::Real = one(T),
  σ::Real = T(10),
  max_eval::Int = -1,
  max_iter::Int = typemax(Int),
  max_time::Real = 30.0,
  max_cgiter::Int = 50,
  cgtol::T = T(0.1),
  atol::T = √eps(T),
  rtol::T = √eps(T),
  Fatol::T = √eps(T),
  Frtol::T = eps(T),
  verbose::Int = 0,
) where {T, V <: AbstractVector{T}}
  if !(nlp.meta.minimize)
    error("tron only works for minimization problem")
  end
  if !(unconstrained(nlp) || bound_constrained(nlp))
    error("tron should only be called for unconstrained or bound-constrained problems")
  end

  reset!(stats)
  ℓ = T.(nlp.meta.lvar)
  u = T.(nlp.meta.uvar)
  n = nlp.meta.nvar
  m = nlp.nls_meta.nequ

  start_time = time()
  set_time!(stats, 0.0)

  solver.x .= x
  x = solver.x
  gx = solver.gx

  # Preallocation
  Av = solver.Av
  Atv = solver.Atv
  gpx = solver.gpx
  xc = solver.xc

  F(x) = residual(nlp, x)
  A(x) = jac_op_residual!(nlp, x, Av, Atv)

  x .= max.(ℓ, min.(x, u))
  Fx = F(x)
  fx = dot(Fx, Fx) / 2
  Ax = A(x)
  mul!(gx, Ax', Fx)
  gt = solver.gt

  Fc = solver.Fc
  num_success_iters = 0

  # Optimality measure
  project_step!(gpx, x, gx, ℓ, u, -one(T))
  πx = nrm2(n, gpx)
  ϵ = atol + rtol * πx
  fmin = min(-one(T), fx) / eps(eltype(x))
  optimal = πx <= ϵ
  unbounded = fx < fmin
  ϵF = Fatol + Frtol * 2 * √fx
  project_step!(gpx, x, x, ℓ, u, zero(T)) # Proj(x) - x
  small_residual = (norm(gpx) <= ϵ) && (2 * √fx <= ϵF)

  set_iter!(stats, 0)
  set_objective!(stats, fx)
  set_dual_residual!(stats, πx)

  αC = one(T)
  tr = solver.tr
  tr.radius = tr.initial_radius = min(max(one(T), πx / 10), tr.max_radius)
  verbose > 0 && @info log_header(
    [:iter, :f, :dual, :radius, :ratio, :cgstatus],
    [Int, T, T, T, T, String],
    hdr_override = Dict(:f => "f(x)", :dual => "π", :radius => "Δ"),
  )

  set_status!(
    stats,
    get_status(
      nlp,
      elapsed_time = stats.elapsed_time,
      optimal = optimal,
      small_residual = small_residual,
      unbounded = unbounded,
      max_eval = max_eval,
      iter = stats.iter,
      max_iter = max_iter,
      max_time = max_time,
    ),
  )

  callback(nlp, solver, stats)

  done = stats.status != :unknown

  while !done
    # Current iteration
    xc .= x
    fc = fx
    Fc .= Fx
    Δ = tr.radius

    αC, s, cauchy_status = cauchy_ls(x, Ax, Fx, gx, Δ, αC, ℓ, u, μ₀ = μ₀, μ₁ = μ₁, σ = σ)

    if cauchy_status != :success
      @error "Cauchy step returned: $cauchy_status"
      stats.status = cauchy_status
      done = true
      continue
    end
    s, As, cgits, cginfo = with_logger(subsolver_logger) do
      projected_gauss_newton!(
        x,
        Ax,
        Fx,
        Δ,
        cgtol,
        s,
        ℓ,
        u,
        subsolver = subsolver,
        max_cgiter = max_cgiter,
      )
    end
    slope = dot(m, Fx, As)
    qs = dot(As, As) / 2 + slope
    Fx = F(x)
    fx = dot(Fx, Fx) / 2

    ared, pred = aredpred!(tr, nlp, fc, fx, qs, x, s, slope)
    if pred ≥ 0
      stats.status = :neg_pred
      done = true
      continue
    end
    tr.ratio = ared / pred
    verbose > 0 &&
      mod(stats.iter, verbose) == 0 &&
      @info log_row([stats.iter, fx, πx, Δ, tr.ratio, cginfo])

    s_norm = nrm2(n, s)
    if num_success_iters == 0
      tr.radius = min(Δ, s_norm)
    end

    # Update the trust region
    update!(tr, s_norm)

    if acceptable(tr)
      num_success_iters += 1
      Ax = A(x)
      if tr.good_grad
        gx .= tr.gt
        tr.good_grad = false
      else
        gx .= Ax' * Fx
      end
      project_step!(gpx, x, gx, ℓ, u, -one(T))
      πx = nrm2(n, gpx)
    end

    # No post-iteration

    if !acceptable(tr)
      Fx .= Fc
      fx = fc
      x .= xc
    end

    set_iter!(stats, stats.iter + 1)
    set_objective!(stats, fx)
    set_time!(stats, time() - start_time)
    set_dual_residual!(stats, πx)

    optimal = πx <= ϵ
    project_step!(gpx, x, x, ℓ, u, zero(T)) # Proj(x) - x
    small_residual = (norm(gpx) <= ϵ) && (2 * √fx <= ϵF)
    unbounded = fx < fmin

    set_status!(
      stats,
      get_status(
        nlp,
        elapsed_time = stats.elapsed_time,
        optimal = optimal,
        small_residual = small_residual,
        unbounded = unbounded,
        max_eval = max_eval,
        iter = stats.iter,
        max_iter = max_iter,
        max_time = max_time,
      ),
    )

    callback(nlp, solver, stats)

    done = stats.status != :unknown
  end
  verbose > 0 && @info log_row(Any[stats.iter, fx, πx, tr.radius])

  set_solution!(stats, x)
  stats
end

"""`s = projected_line_search_ls!(x, A, g, d, ℓ, u; μ₀ = 1e-2)`

Performs a projected line search, searching for a step size `t` such that

    ½‖As + Fx‖² ≤ ½‖Fx‖² + μ₀FxᵀAs

where `s = P(x + t * d) - x`, while remaining on the same face as `x + d`.
Backtracking is performed from t = 1.0. `x` is updated in place.
"""
function projected_line_search_ls!(
  x::AbstractVector{T},
  A::Union{AbstractMatrix, AbstractLinearOperator},
  Fx::AbstractVector{T},
  d::AbstractVector{T},
  ℓ::AbstractVector{T},
  u::AbstractVector{T};
  μ₀::Real = T(1e-2),
) where {T <: Real}
  α = one(T)
  _, brkmin, _ = breakpoints(x, d, ℓ, u)
  nsteps = 0
  n = length(x)
  m = length(Fx)

  s = zeros(T, n)
  As = zeros(T, m)

  search = true
  while search && α > brkmin
    nsteps += 1
    project_step!(s, x, d, ℓ, u, α)
    slope, qs = compute_As_slope_qs!(As, A, s, Fx)
    if qs <= μ₀ * slope
      search = false
    else
      α /= 2
    end
  end
  if α < 1 && α < brkmin
    α = brkmin
    project_step!(s, x, d, ℓ, u, α)
    slope, qs = compute_As_slope_qs!(As, A, s, Fx)
  end

  project_step!(s, x, d, ℓ, u, α)
  x .= x .+ s

  return s
end

"""`α, s = cauchy_ls(x, A, Fx, g, Δ, ℓ, u; μ₀ = 1e-2, μ₁ = 1.0, σ=10.0)`

Computes a Cauchy step `s = P(x - α g) - x` for

    min  q(s) = ½‖As + Fx‖² - ½‖Fx‖²     s.t.    ‖s‖ ≦ μ₁Δ,  ℓ ≦ x + s ≦ u,

with the sufficient decrease condition

    q(s) ≦ μ₀gᵀs,

where g = AᵀFx.
"""
function cauchy_ls(
  x::AbstractVector{T},
  A::Union{AbstractMatrix, AbstractLinearOperator},
  Fx::AbstractVector{T},
  g::AbstractVector{T},
  Δ::Real,
  α::Real,
  ℓ::AbstractVector{T},
  u::AbstractVector{T};
  μ₀::Real = T(1e-2),
  μ₁::Real = one(T),
  σ::Real = T(10),
) where {T <: Real}
  # TODO: Use brkmin to care for g direction
  _, _, brkmax = breakpoints(x, -g, ℓ, u)
  n = length(x)
  m = length(Fx)
  s = zeros(T, n)
  As = zeros(T, m)

  status = :success

  project_step!(s, x, g, ℓ, u, -α)

  # Interpolate or extrapolate
  s_norm = nrm2(n, s)
  if s_norm > μ₁ * Δ
    interp = true
  else
    slope, qs = compute_As_slope_qs!(As, A, s, Fx)
    interp = qs >= μ₀ * slope
  end

  if interp
    search = true
    while search
      α /= σ
      project_step!(s, x, g, ℓ, u, -α)
      s_norm = nrm2(n, s)
      if s_norm <= μ₁ * Δ
        slope, qs = compute_As_slope_qs!(As, A, s, Fx)
        search = qs >= μ₀ * slope
      end
      # TODO: Correctly assess why this fails
      if α < sqrt(nextfloat(zero(α)))
        search = false
        status = :small_step
      end
    end
  else
    search = true
    αs = α
    while search && α <= brkmax
      α *= σ
      project_step!(s, x, g, ℓ, u, -α)
      s_norm = nrm2(n, s)
      if s_norm <= μ₁ * Δ
        slope, qs = compute_As_slope_qs!(As, A, s, Fx)
        if qs <= μ₀ * slope
          αs = α
        end
      else
        search = false
      end
    end
    # Recover the last successful step
    α = αs
    s = project_step!(s, x, g, ℓ, u, -α)
  end
  return α, s, status
end

"""`projected_gauss_newton!(x, A, Fx, Δ, gctol, s, max_cgiter, ℓ, u)`

Compute an approximate solution `d` for

    min q(d) = ½‖Ad + Fx‖² - ½‖Fx‖²     s.t.    ℓ ≦ x + d ≦ u,  ‖d‖ ≦ Δ

starting from `s`.  The steps are computed using the conjugate gradient method
projected on the active bounds.
"""
function projected_gauss_newton!(
  x::AbstractVector{T},
  A::Union{AbstractMatrix, AbstractLinearOperator},
  Fx::AbstractVector{T},
  Δ::Real,
  cgtol::Real,
  s::AbstractVector{T},
  ℓ::AbstractVector{T},
  u::AbstractVector{T};
  subsolver::Symbol = :lsmr,
  max_cgiter::Int = 50,
) where {T <: Real}
  n = length(x)
  status = ""
  subsolver in tronls_allowed_subsolvers ||
    error("subproblem solver must be one of $tronls_allowed_subsolvers")
  lssolver = eval(subsolver)

  As = A * s

  # Projected Newton Step
  exit_optimal = false
  exit_pcg = false
  exit_itmax = false
  iters = 0
  xt = x .+ s
  project!(xt, xt, ℓ, u)
  while !(exit_optimal || exit_pcg || exit_itmax)
    ifree = setdiff(1:n, active(xt, ℓ, u))
    if length(ifree) == 0
      exit_optimal = true
      continue
    end
    Z = opExtension(ifree, n)
    @views wa = Fx
    @views Ffree = As + wa
    wanorm = norm(wa)

    AZ = A * Z
    st, stats = lssolver(AZ, -Ffree, radius = Δ, rtol = cgtol, atol = zero(T))
    iters += 1
    status = stats.status

    # Projected line search
    xfree = @view xt[ifree]
    @views w = projected_line_search_ls!(xfree, AZ, Ffree, st, ℓ[ifree], u[ifree])
    @views s[ifree] .+= w

    As .= A * s

    @views Ffree .= As .+ Fx
    if norm(Z' * A' * Ffree) <= cgtol * wanorm
      exit_optimal = true
    elseif status == "on trust-region boundary"
      exit_pcg = true
    elseif iters >= max_cgiter
      exit_itmax = true
    end
  end
  status = if exit_optimal
    "stationary point found"
  elseif exit_itmax
    "maximum number of iterations"
  else
    status # on trust-region
  end

  x .= xt

  return s, As, iters, status
end
