export LBFGSParameterSet

struct LBFGSParameterSet{R <: AbstractFloat,I  <: Integer} <: AbstractParameterSet
  mem::Parameter{I, IntegerRange{I}}
  scaling::Parameter{Bool, BinaryRange{Bool}}
  τ₀::Parameter{R, RealInterval{R}}
  τ₁::Parameter{R, RealInterval{R}}
  bk_max::Parameter{I, IntegerRange{I}}
  bW_max::Parameter{I, IntegerRange{I}}
  
  function LBFGSParameterSet{R, I}(;
    mem::I=I(5),
    scaling::Bool=true,
    τ₀::R=max(R(1.0e-4), sqrt(eps(R))),
    τ₁::R=R(0.999),
    bk_max::I=I(10),
    bW_max::I=I(5)
    ) where {R <: AbstractFloat, I <: Integer}
    (τ₀ < τ₁) || throw(DomainError("Inavlid τ₀ and/or τ₁: slope factor should satisfy τ₁ > τ₀."))
    p_set = new{R, I}(
      Parameter(mem, IntegerRange(I(5), I(30))),
      Parameter(scaling, BinaryRange()),
      Parameter(τ₀, RealInterval(eps(R), R(1/2);)),
      Parameter(τ₁, RealInterval(R(2/3), R(1 - eps(R));)),
      Parameter(bk_max, IntegerRange(I(5), I(15))),
      Parameter(bW_max, IntegerRange(I(1), I(5))),
    )
    set_names!(p_set)
    return p_set
  end
end


