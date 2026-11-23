import Lean

/-! Finite-product and polynomial checks accompanying the SLDI manuscript.
    The finite sums use exact rational arithmetic. -/

namespace SLDI

def weighted {α : Type} (weights : List (α × Rat)) (f : α → Rat) : Rat :=
  match weights with
  | [] => 0
  | (x, weight) :: rest => weight * f x + weighted rest f

theorem weighted_add {α : Type} (weights : List (α × Rat)) (f g : α → Rat) :
    weighted weights (fun x => f x + g x) = weighted weights f + weighted weights g := by
  induction weights with
  | nil => simp [weighted]; grind
  | cons pair rest ih =>
    obtain ⟨x, weight⟩ := pair
    simp only [weighted, ih]
    grind

theorem weighted_scale {α : Type} (weights : List (α × Rat)) (f : α → Rat) (c : Rat) :
    weighted weights (fun x => c * f x) = c * weighted weights f := by
  induction weights with
  | nil => simp [weighted]
  | cons pair rest ih =>
    obtain ⟨x, weight⟩ := pair
    simp only [weighted, ih]
    grind

theorem weighted_zero {α : Type} (weights : List (α × Rat)) :
    weighted weights (fun _ => 0) = 0 := by
  induction weights with
  | nil => rfl
  | cons pair rest ih =>
    obtain ⟨x, weight⟩ := pair
    simp [weighted, ih]; grind

theorem weighted_sub {α : Type} (weights : List (α × Rat)) (f g : α → Rat) :
    weighted weights (fun x => f x - g x) = weighted weights f - weighted weights g := by
  have h := weighted_add weights f (fun x => (-1 : Rat) * g x)
  rw [weighted_scale] at h
  have negOne (u : Rat) : (-1 : Rat) * u = -u := by grind
  simpa [Rat.sub_eq_add_neg, negOne] using h

/-- Product distribution of a finite history and a fresh innovation. -/
def productMean {H X : Type} (past : List (H × Rat)) (fresh : List (X × Rat))
    (f : H → X → Rat) : Rat := weighted past (fun h => weighted fresh (f h))

/-- The coefficient can use the history arbitrarily, with an independent centered innovation. -/
theorem predictable_centered {H X : Type}
    (past : List (H × Rat)) (fresh : List (X × Rat))
    (coefficient : H → Rat) (basis : X → Rat)
    (centered : weighted fresh basis = 0) :
    productMean past fresh (fun h x => coefficient h * basis x) = 0 := by
  simp [productMean, weighted_scale, centered, weighted_zero]

/-- A one-step correction preserves the finite-product expectation of any gradient. -/
theorem predictable_unbiased {H X : Type}
    (past : List (H × Rat)) (fresh : List (X × Rat))
    (gradient : H → X → Rat) (coefficient : H → Rat) (basis : X → Rat)
    (centered : weighted fresh basis = 0) :
    productMean past fresh (fun h x => gradient h x - coefficient h * basis x) =
      productMean past fresh gradient := by
  simp [productMean, weighted_sub, weighted_scale, centered, weighted_zero]; grind

/-- An earlier scalar term measurable in the history is orthogonal to the correction. -/
theorem orthogonal_to_past {H X : Type}
    (past : List (H × Rat)) (fresh : List (X × Rat))
    (earlier coefficient : H → Rat) (basis : X → Rat)
    (centered : weighted fresh basis = 0) :
    productMean past fresh (fun h x => earlier h * (coefficient h * basis x)) = 0 := by
  simpa only [Rat.mul_assoc] using
    predictable_centered past fresh (fun h => earlier h * coefficient h) basis centered

theorem weighted_square_split {α : Type} (weights : List (α × Rat))
    (f g : α → Rat) (orthogonal : weighted weights (fun x => f x * g x) = 0) :
    weighted weights (fun x => (f x + g x)^2) =
      weighted weights (fun x => (f x)^2) + weighted weights (fun x => (g x)^2) := by
  have expansion : (fun x => (f x + g x)^2) =
      (fun x => (f x)^2 + (g x)^2 + 2 * (f x * g x)) := by
    funext x
    grind
  rw [expansion, weighted_add, weighted_add, weighted_scale, orthogonal]
  grind

/-- One Hermite step for the OU example, using the unnormalized quadratic
    innovation. Its factor is algebraically the same as the product of
    the normalized feature and its coefficient in the manuscript. -/
theorem ou_one_step (d sigma accumulated innovation : Rat) :
    (d * (accumulated + innovation) - sigma * (accumulated + innovation)^2) -
      ((d - 2 * sigma * accumulated) * innovation - sigma * (innovation^2 - 1)) =
    d * accumulated - sigma * accumulated^2 - sigma := by
  grind

def ouCorrection (d sigma accumulated : Rat) (steps : List (Rat × Rat)) : Rat :=
  match steps with
  | [] => 0
  | (weight, noise) :: rest =>
    (d - 2 * sigma * accumulated) * (weight * noise) -
      sigma * (weight^2 * (noise^2 - 1)) +
      ouCorrection d sigma (accumulated + weight * noise) rest

def realized (steps : List (Rat × Rat)) : Rat :=
  match steps with
  | [] => 0
  | (weight, noise) :: rest => weight * noise + realized rest

def squaredWeights (steps : List (Rat × Rat)) : Rat :=
  match steps with
  | [] => 0
  | (weight, _) :: rest => weight^2 + squaredWeights rest

/-- The full OU correction telescopes on every path for an arbitrary grid. -/
theorem ou_telescope (steps : List (Rat × Rat)) (d sigma accumulated : Rat) :
    (d * (accumulated + realized steps) - sigma * (accumulated + realized steps)^2) -
      ouCorrection d sigma accumulated steps =
    d * accumulated - sigma * accumulated^2 - sigma * squaredWeights steps := by
  induction steps generalizing accumulated with
  | nil => simp [ouCorrection, realized, squaredWeights]; grind
  | cons pair rest ih =>
    obtain ⟨weight, noise⟩ := pair
    have h := ih (accumulated + weight * noise)
    simp only [ouCorrection, realized, squaredWeights]
    grind

theorem ou_zero_variance (steps : List (Rat × Rat)) (d sigma : Rat) :
    d * realized steps - sigma * (realized steps)^2 - ouCorrection d sigma 0 steps =
      -(sigma * squaredWeights steps) := by
  have h := ou_telescope steps d sigma 0
  grind

#print axioms predictable_unbiased
#print axioms orthogonal_to_past
#print axioms weighted_square_split
#print axioms ou_zero_variance

end SLDI
