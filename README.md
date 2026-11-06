# NSmatter

**Variational Monte Carlo for nuclear and neutron-star matter with local chiral interactions.**

NSmatter computes the equation of state and the spin-isospin structure of infinite
nucleonic matter: pure neutron matter, symmetric nuclear matter, isospin-asymmetric
matter and spin-polarized matter, with the four populations
n↑, n↓, p↑, p↓ treated independently. The interaction is the family of local
chiral potentials with explicit Δ(1232) degrees of freedom of Piarulli *et al.*
(the Norfolk potentials NV2-Ia, Ib, IIa, IIb, and their three-nucleon partners NV3).
The many-body method is variational Monte Carlo (VMC) with a correlated trial
wavefunction optimized by stochastic reconfiguration.

The project is built to be compared, Hamiltonian for Hamiltonian, with
Brueckner–Hartree–Fock (BHF) calculations of the same interactions. Using the same
potential in two very different many-body methods separates the uncertainty that
comes from the many-body approximation from the spread between interaction models.

---

## Contents

1. [Physics goals](#physics-goals)
2. [The Hamiltonian](#the-hamiltonian)
3. [Variational Monte Carlo](#variational-monte-carlo)
4. [Spin and isospin in the configuration basis](#spin-and-isospin-in-the-configuration-basis)
5. [The local energy, operator by operator](#the-local-energy-operator-by-operator)
6. [The trial wavefunction](#the-trial-wavefunction)
7. [Optimization](#optimization)
8. [Infinite matter in a periodic box](#infinite-matter-in-a-periodic-box)
9. [Statistics and error analysis](#statistics-and-error-analysis)
10. [Observables and structure](#observables-and-structure)
11. [From the equation of state to neutron stars](#from-the-equation-of-state-to-neutron-stars)
12. [Validation strategy](#validation-strategy)
13. [Implementation and infrastructure](#implementation-and-infrastructure)
14. [Installation and usage](#installation-and-usage)
15. [Outputs and the report](#outputs-and-the-report)
16. [Repository layout](#repository-layout)
17. [Units and conventions](#units-and-conventions)
18. [References](#references)

---

## Physics goals

Neutron stars probe matter from a fraction of nuclear saturation density
ρ₀ ≈ 0.16 fm⁻³ in the crust to several times ρ₀ in the core. Their structure
(mass–radius relation, tidal deformability, composition, cooling) is set by the
equation of state of nucleonic matter and by how that matter responds to
isospin asymmetry and to spin polarization. NSmatter computes, from a realistic
nuclear Hamiltonian:

- **Energy per particle** E/A(ρ) of pure neutron matter (PNM) and symmetric
  nuclear matter (SNM), from about 0.04 to 0.32 fm⁻³.
- **Asymmetric matter** E/A(ρ, x) at proton fractions x between 0 and 1/2, and
  from it the **symmetry energy** S(ρ) and its slope L.
- **β-equilibrium composition**: the proton fraction x(ρ) of neutron-star matter
  from charge neutrality and the chemical-equilibrium condition
  μ_n = μ_p + μ_e, and whether the direct Urca threshold (x ≈ 0.11–0.14) is reached.
- **Spin-polarized matter** E/A(ρ, Δ) for neutron and proton polarizations
  Δ_n, Δ_p, and from it the **magnetic susceptibility** χ/χ_F. This decides whether
  nuclear matter can develop a ferromagnetic instability, a question of direct
  relevance for magnetars.
- **Operator decomposition** of the potential energy, ⟨v_p⟩ for each operator of
  the interaction, and **pair distribution functions** g(r) per spin-isospin
  channel, which show which parts of the force drive each effect.
- **Model dependence**: the four NV2 models (classes I and II, regulators a and b)
  give an estimate of the theoretical uncertainty of the chiral expansion.

The four populations are independent inputs, so any combination of isospin
asymmetry and spin polarization can be computed, as in a magnetized,
β-stable stellar environment.

---

## The Hamiltonian

Nucleons are non-relativistic point particles interacting through two- and
three-body forces:

```
H = - ħ²/2m Σ_i ∇_i²  +  Σ_{i<j} v_ij  +  Σ_{i<j<k} V_ijk
```

### Two-nucleon interaction: NV2

The Norfolk two-nucleon potentials (Piarulli *et al.* 2015, 2016) are local in
coordinate space and are derived in chiral effective field theory with Δ isobars.
They contain one-pion exchange, Δ-full two-pion exchange up to next-to-next-to-leading
order, and contact terms up to N3LO. They are written in an AV18-like operator form,

```
v_ij = Σ_p  v^p(r_ij) O^p_ij
```

with the operators

| Group | Operators O^p |
|---|---|
| central and spin-isospin | 1, σ_i·σ_j, τ_i·τ_j, (σ_i·σ_j)(τ_i·τ_j) |
| tensor | S_ij, S_ij τ_i·τ_j, with S_ij = 3(σ_i·r̂)(σ_j·r̂) − σ_i·σ_j |
| spin-orbit | L·S, L·S τ_i·τ_j |
| quadratic in momentum | L², L² σ_i·σ_j, (L·S)² |
| charge dependent | T_ij, σ_i·σ_j T_ij, S_ij T_ij, L·S T_ij, τ_i^z + τ_j^z, with T_ij = 3τ_i^z τ_j^z − τ_i·τ_j (electromagnetic terms are left out: negligible in neutron matter) |

The long-range one-pion-exchange part is regularized with a cutoff R_L and the
short-range contact part with a Gaussian of range R_S:

| Model | Fitted to NN data up to | (R_S, R_L) |
|---|---|---|
| NV2-Ia | 125 MeV lab energy | (0.8, 1.2) fm |
| NV2-Ib | 125 MeV | (0.7, 1.0) fm |
| NV2-IIa | 200 MeV | (0.8, 1.2) fm |
| NV2-IIb | 200 MeV | (0.7, 1.0) fm |

The radial functions v^p(r) are implemented in NSmatter from the published
formulas and low-energy constants, independently of any existing code. They are
evaluated once on a fine radial grid, written to tables, and interpolated with
cubic splines during the simulation. The potential is never differentiated (only
the wavefunction is), so the interpolation costs nothing in accuracy as long as
the grid is fine. The tables are validated point by point against an independent
reference implementation of the same potentials.

### Three-nucleon interaction: NV3

Pure neutron matter is computed with the two-body force alone as well as with the
three-body force, which allows a clean comparison with two-body BHF. Matter with
protons is not realistic without three-nucleon forces: two-body potentials alone
saturate at the wrong density and energy, in VMC as in BHF. NSmatter includes the
consistent NV3 forces (Piarulli *et al.* 2018; Baroni *et al.* 2018): two-pion
exchange with the c₁ and c₃ couplings and the Δ(1232) (Fujita–Miyazawa) term, the
one-pion-exchange–contact term c_D and the three-nucleon contact c_E, with the same
regulators as the two-body model:

```
V = Σ_{i<j<k} Σ_cyclic [ a_X {X_ij, X_jk} + a_1 Z(r_ij)Z(r_jk)(σ_i·r̂_ij)(σ_k·r̂_jk)
                         + a_D X_ik [C_RS(r_ij) + C_RS(r_jk)] + a_E C_RS(r_ij) C_RS(r_jk) ]
X_ij = Y(r_ij) σ_i·σ_j + T(r_ij) S_ij
```

(written for neutron matter, where τ_i·τ_j = 1 and the commutator terms vanish).
Two families of contact couplings exist: the original fits to the trinucleon
energies and the nd doublet scattering length (NV2+3-Ia … IIb), and the "starred"
fits that also reproduce the tritium Gamow–Teller matrix element (Ia* … IIb*). The
original c_E values make pure neutron matter collapse (Lovato *et al.* 2022), so
neutron matter and neutron stars use the starred models. The couplings are written
with F_π = 92.4 MeV, the convention in which c_D and c_E are defined.
### What the method does not include

Relativistic corrections, hyperons and quark degrees of freedom are outside the
scope of the project. The chiral models are fitted to data up to 125 or 200 MeV;
above about 2ρ₀ the relative momenta of pairs exceed that range, and results there
should be read as extrapolations of the interaction rather than as predictions.

---

## Variational Monte Carlo

For any trial state Ψ the energy is an upper bound to the ground-state energy:

```
E[Ψ] = ⟨Ψ|H|Ψ⟩ / ⟨Ψ|Ψ⟩  =  ∫ dX |Ψ(X)|² E_L(X) / ∫ dX |Ψ(X)|²,
E_L(X) = (HΨ)(X) / Ψ(X)
```

X is a configuration: positions, spin coordinates and isospin labels of all
particles. Configurations are sampled from |Ψ|² with the Metropolis algorithm, and
the energy is the average of the **local energy** E_L over the samples.

Two properties make the method attractive here:

- |Ψ|² is a positive probability density, so VMC has **no fermion sign problem**,
  at any density and composition.
- The variance of E_L vanishes for an exact eigenstate. It is a direct,
  sample-by-sample measure of the quality of the trial state, and it is what the
  optimizer reduces.

The price is the variational bias: the result is only as good as the trial
wavefunction. The design of the wavefunction, and checks of its convergence, are
therefore central to the project.

Because the local energy is evaluated exactly for the given Ψ, VMC can treat every
operator of the NV2 interaction, including L², (L·S)² and the full three-body
force, without the approximations that projector methods such as auxiliary-field
diffusion Monte Carlo (AFDMC) need for those terms.

---

## Spin and isospin in the configuration basis

The wavefunction is represented in the configuration basis,
Ψ(X) = ⟨r_1 ξ_1 t_1, …, r_N ξ_N t_N | Ψ⟩, rather than as a vector over all
2^N spin and isospin states (as in Green's function Monte Carlo). The cost then
grows polynomially with N, and systems of tens to about a hundred nucleons are
accessible.

**Isospin** is a discrete label t_i = +1 (proton) or −1 (neutron). The Hamiltonian
conserves the total isospin projection, so the numbers of protons and neutrons are
exact. Because |Ψ|² is symmetric under permutations of particles, the isospin labels
can be kept fixed for the whole simulation without loss of generality.

**Spin** is a continuous spinor ξ_i = (ξ↑, ξ↓) on the unit sphere of ℂ², sampled
together with the positions. The wavefunction is linear in each spinor,

```
Ψ(…, ξ_j, …) = ξ_j↑ Ψ(…, ↑, …) + ξ_j↓ Ψ(…, ↓, …),
```

and the spinor states resolve the identity, so sampling them with weight |Ψ|²
gives exact expectation values. Discrete labels (↑ = (1, 0), ↓ = (0, 1)) are a special
case. Continuous spinors are needed because the tensor and spin-orbit forces do not
conserve the total spin projection: a correlated wavefunction has components in
every S_z sector, and a discrete-spin walker would sit in configurations where the
reference determinant vanishes identically. Spinors overlap all sectors at once and
keep every determinant non-singular. Spinors are moved by random SU(2) rotations,
whose proposal is symmetric with respect to the invariant measure.

---

## The local energy, operator by operator

### Kinetic energy

```
E_L^kin = − ħ²/2m Σ_i [ ∇_i² ln Ψ + (∇_i ln Ψ)² ]
```

Only ln Ψ is differentiated, never Ψ itself, which overflows for large N. ln Ψ is
complex; the (∇ ln Ψ)² term is a plain square, not a modulus, and for plane waves
it is exactly where the kinetic energy comes from. The derivatives are computed by
automatic differentiation in forward mode, which handles complex outputs of real
inputs directly.

### Spin-isospin exchange operators

σ_i·σ_j and τ_i·τ_j are exchange operators:

```
σ_i·σ_j = 2 P^σ_ij − 1,   τ_i·τ_j = 2 P^τ_ij − 1
```

so their action on Ψ is Ψ evaluated with the spin (or isospin) coordinates of i and
j swapped. For the four central operators the local energy of a pair is

```
a_0 + a_σ ρ^σ_ij + a_τ ρ^τ_ij + a_στ ρ^στ_ij,   ρ^σ_ij = Ψ(spins of i, j swapped)/Ψ, …
a_0 = v_c − v_σ − v_τ + v_στ,  a_σ = 2(v_σ − v_στ),  a_τ = 2(v_τ − v_στ),  a_στ = 4 v_στ
```

### General two-body operators, tensor force

Operators that are not exchanges, starting with the tensor S_ij (which can flip both
spins of a pair), are written as explicit 4 × 4 matrices on the two-body spin states
and applied as an ordinary matrix–vector product:

```
E_L^pot = Σ_{i<j} Σ_targets ⟨current| v_ij |target⟩ Ψ(target) / Ψ(current)
```

The current pair spin state is the product spinor ξ_i ⊗ ξ_j; the targets are the 4
basis spin states of the pair times 2 isospin states (unchanged, or swapped when
t_i ≠ t_j). The tensor radial functions vanish as r² at the origin, so the code
works with v_T(r) r̂_a r̂_b = w(r) r_a r_b and never divides by r.

### Momentum-dependent operators

The spin-orbit operator L·S, with L = r_ij × p_ij and S = (σ_i + σ_j)/2, needs the
first derivatives of Ψ with respect to the relative coordinate of the pair, at each
target spin configuration; L² and (L·S)² need second derivatives. All of them are
obtained by automatic differentiation of the trial wavefunction, so they are exact
for the given Ψ.

These operators dominate the cost: for every pair and every spin target, the first
and second derivatives of Ψ with respect to the pair's relative coordinate. A move of
one pair changes only two columns of the Slater matrix, so the ratio Ψ(X′)/Ψ(X) is
computed by a rank-2 update — the new determinant from a 2×2 determinant, all pair
ratios of the correlation factor from D′ = E − (GU)R⁻¹E[p,q] — in O(N²) instead of
O(N³), and only the 2N − 3 correlation amplitudes of pairs that contain the moved
particles are recomputed. The static operators use the same pair moves at u = 0.

### Three-body force

In neutron matter σ_j enters {X_ij, X_jk} only through {σ_j^a, σ_j^b} = 2δ^ab, so

```
{X_ij, X_jk} = 2 σ_i · [A(r_ij) A(r_jk)] · σ_k,   A(r) = Y(r) 1 + T(r)(3 r̂r̂ − 1)
```

acts on the spins of the outer pair only; every NV3 term has this form. The
three-body force is therefore a sum of pair operators σ_a^α M^αβ(a,b) σ_b^β whose
3×3 matrices collect the contributions of all middle particles m, and its local
energy is computed with the same pair machinery as the two-body potential.
With all periodic images of m and b summed independently, every term factorizes into
image sums of two-body quantities, Σ_images A(r_am)A(r_mb) = Ā(a,m)·Ā(m,b), so the
whole force costs O(N³) small matrix products per configuration. In matter with
protons the isospin structure is genuinely three-body (commutator terms) and the
operators act on the three-particle spin-isospin targets of each triplet.

---

## The trial wavefunction

```
Ψ(X) = exp[J(X)] × [ 1 + Σ_{i<j} Σ_p u_p(r_ij) O^p_ij ] × Φ(X)
```

**Φ** is a Slater determinant of spin-isospin plane-wave orbitals
e^{ik·r} χ_spin χ_isospin, with momenta k = 2π n / L filled in closed shells for each
of the four populations. It is the exact ground state of the free gas in the box and
fixes the nodal and phase structure.

**J** is a central Jastrow factor, J = Σ_{i<j} u_c(r_ij), with separate correlation
functions for like (nn, pp) and unlike (np) pairs. It suppresses close pairs where the
short-range repulsion acts.

**Operator correlations** u_p(r) O^p_ij, with O^p running over σ_i·σ_j, τ_i·τ_j,
(σ_i·σ_j)(τ_i·τ_j), S_ij, S_ij τ_i·τ_j, build in the spin-isospin
structure of the interaction. Without them the tensor and spin-orbit forces average
to almost exactly zero: in a closed-shell Slater determinant the tensor force gives
exactly zero, by cubic symmetry, and it contributes to the binding only through
tensor correlations, which are also what produce the D-state of the deuteron.
Applying a pair operator to Φ changes the spin-isospin coordinates of two particles,
that is, two columns of the Slater matrix; the resulting determinants are obtained
from the inverse of the reference matrix by low-rank updates instead of new
factorizations.

**Spin-orbit correlations** u_LS(r) L_ij·S_ij and u_LSτ(r) L_ij·S_ij τ_i·τ_j (option
`--spin-orbit-corr`) let the wavefunction gain the spin-orbit attraction of triplet
P waves, which averages to almost zero otherwise. L acting on Φ needs the derivatives
of the two Slater columns of the pair, analytic for plane waves, so these terms use
the same low-rank updates as the others; they make a run about 2.5 times slower.

**Radial functions.** All correlation functions are expanded on the basis

```
u(r) = (1 − x²)³ Σ_m a_m x^{2m},   x = r / r_c,   u = 0 for r ≥ r_c ≤ L/2
```

The envelope makes u, u′ and u″ vanish at the cutoff (a kink in ln Ψ would hide a
delta-function contribution in the kinetic energy), the even powers give u′(0) = 0
(no cusp, appropriate for regular potentials, and a finite derivative of r² in
automatic differentiation), and the functions are linear in the variational
parameters. Tensor correlations carry an extra factor x², so that u_T S_ij is smooth
at the origin.

The variational parameters, a few dozen in total, are the coefficients a_m of all
correlation functions.

---

## Optimization

The parameters are optimized by **stochastic reconfiguration** (SR), the natural
gradient of the energy with respect to the geometry of the wavefunction. From the
log-derivatives O_k = ∂ ln Ψ / ∂ p_k, sampled with the energy:

```
g_k  = 2 Re ⟨(O_k − ⟨O_k⟩)* (E_L − ⟨E_L⟩)⟩           energy gradient
S_kl =   Re ⟨(O_k − ⟨O_k⟩)* (O_l − ⟨O_l⟩)⟩            metric
p   →  p − η (S + ε 1)⁻¹ g
```

SR measures steps by how much the wavefunction changes, not the parameters, so it
is invariant under rescaling of the parameters; it is equivalent to imaginary-time
evolution projected on the variational manifold, with η playing the role of the
time step. The diagonal shift ε regularizes directions the samples cannot resolve.

A run has two phases. In the optimization phase the parameters move at every step,
and the energy and the variance of E_L are monitored; a drop in the variance is the
clearest sign of convergence. In the evaluation phase the parameters are frozen and
a long run gives the variational energy with a proper error bar. Only evaluation
energies are reported.

---

## Infinite matter in a periodic box

N nucleons are placed in a cubic box of side L = (N/ρ)^{1/3} with periodic boundary
conditions. Separations are minimum-image vectors, with components in [−L/2, L/2].

- **Potential.** At ρ₀ the box is only a few fm across, comparable to the range of
  the interaction, so every pair interacts with all periodic images of its partner.
  The image sum is taken over a cube of images large enough that the neglected tail
  is below 10⁻¹⁰ MeV. Tensor terms are summed as matrices, since each image has its
  own direction.
- **Shells.** Each population must fill complete momentum shells
  (1, 7, 19, 27, 33, 57, 81, … particles), so the free ground state is unique and keeps
  the cubic, parity and translation symmetries of the box. Standard sizes are
  N = 66 for PNM and N = 28 or 76 for SNM.
- **Finite-size corrections.** The difference between the free-gas energy in the box
  and in the thermodynamic limit is added to the kinetic energy. Twist-averaged
  boundary conditions, with phases e^{iθ} at the box faces averaged over θ, remove most
  of the remaining shell effects and allow arbitrary particle numbers per population,
  which gives a fine grid in proton fraction and polarization.

---

## Statistics and error analysis

Many independent Markov chains (walkers) run in parallel. The step sizes of the
position and spinor moves are tuned during thermalization only, towards about 50%
acceptance, and are then kept fixed, to preserve detailed balance.

Successive measurements are correlated. Errors are computed by the
Flyvbjerg–Petersen **blocking** analysis: the series is repeatedly averaged in pairs,
and the error estimate rises and then plateaus at the true statistical error. The
reported error is the conservative plateau value, and the integrated autocorrelation
time τ_int is computed alongside it. Independent runs with different random seeds
are used as a check of the error bars.

---

## Observables and structure

Besides the energy, each run measures (all per particle, with errors):

- **the energy decomposition**: kinetic energy and the expectation value of every
  operator channel of NV2 (c, τ, σ, στ, tensor, spin-orbit, L², L²σ, (L·S)², the
  charge-dependent terms) and of NV3 (c₁, c₃, Δ, c_D, c_E), also summed by operator
  group. In neutron matter τ·τ = 1, so only group sums such as c + t are
  basis-independent;
- **pair distributions** g_nn(r), g_np(r), g_pp(r), with the normalization of
  Fore *et al.* (2025), and **spin and isospin correlations** ⟨σ_i·σ_j⟩(r),
  ⟨τ_i·τ_j⟩(r). The operator correlations are computed with local estimators,
  O_L = Σ_targets ⟨X|O|X′⟩ Ψ(X′)/Ψ(X), so they include the spin-isospin exchange
  structure of the wavefunction;
- **total spin and isospin** ⟨S²⟩ and ⟨T²⟩, through Σ_{i<j} σ_i·σ_j = 2S² − 3N/2;
- **configurations**: positions, spinors and spin directions of all walkers at the
  end of the run, and trajectories of a few walkers for animations.

The trial wavefunction is built on plane waves and describes uniform matter: the
snapshots show the correlation hole and the spin pairing of neutrons, not the
clusters that appear at very low density with wavefunctions able to localize nucleons.

---

## From the equation of state to neutron stars

The energies per particle E/N(ρ) of pure neutron matter are fitted to
E/N = a u^α + b u^β (u = ρ/ρ₀), symmetric matter to the expansion
E₀ + K x²/2 + Q x³/6 (x = (ρ − ρ₀)/3ρ₀). From the fits:

- pressure P = ρ² dE/dρ, energy density ε = ρ(m + E/N), speed of sound c_s² = dP/dε;
- β-equilibrated n p e μ matter with the quadratic isospin expansion
  E(ρ, x) = E_SNM + S(ρ)(1 − 2x)², μ_n − μ_p = 4S(1 − 2x) + (m_n − m_p) = μ_e = μ_μ;
- where a fit would become acausal (c_s² > 1), the stiffest causal continuation
  c_s² = 1 is used;
- the Tolman–Oppenheimer–Volkoff equations and the l = 2 tidal perturbation give
  M(R), the maximum mass, R₁.₄ and the tidal deformability Λ(M).

The statistical errors of the VMC energies are propagated to the star by a
parametric bootstrap (refits of E/N + δ with δ drawn from the error bars). The
central density of each star is reported next to the largest computed density, so
the part of a result that rests on the extrapolation of the fit is visible. The EOS
table extends to 2 fm⁻³; a heaviest star at its end is flagged (its M_max is a
lower bound).

**Crust.** Below ρ = 0.08 fm⁻³ the neutron-matter core is replaced by the SLy crust
(Douchin & Haensel 2001), in the analytic representation of Haensel & Potekhin
(2004); core and crust are joined at the core pressure P(0.08 fm⁻³). Through the
same TOV code the full SLy EOS gives M_max = 2.05 M☉ and R₁.₄ = 11.7 km, the
published values (a test). The crust adds about 0.3–0.5 km to R₁.₄ and barely
changes M_max and Λ₁.₄. `--no-crust` (report.py, neutron_star.py) switches it off.

**Uncertainties.** For every NV2+3 model the report separates
- *statistical*: the bootstrap spread above;
- *EOS fit form*: a second fit E/N = a u^{2/3} + b u + c u² (the form of Lovato
  et al.) against the default one, which measures how much the extrapolation beyond
  the computed densities depends on the chosen function;
- *crust*: with minus without the crust;
- *many-body method*: the published AFDMC fit for the same interaction minus our
  VMC fit, both through the same TOV code. VMC is a variational upper bound, so this
  shift is one-sided (AFDMC lower).

Over the NV2+3 models the combined result is the mean, with the *interaction*
uncertainty taken as half the spread between the models (the cutoff and fit
dependence of the Norfolk family), and the other systematics as the largest shift
over the models (for the many-body shift: with its sign when every model shifts the
same way, as for E/N, otherwise as a ± band, so that opposite shifts cannot cancel). They are listed separately and added
in quadrature. Quantities: E/N and P of neutron matter at ρ₀, M_max, R₁.₄, Λ₁.₄.

---

## Validation strategy

In VMC a bug rarely crashes the program: it produces a smooth, plausible energy that
is wrong, and the variational principle does not flag it. Every component of
NSmatter is therefore tested against an independent exact result:

| Component | Exact reference |
|---|---|
| geometry, shells | closed-shell counts, parity Σk = 0, minimum-image definition |
| Slater determinant | antisymmetry across all species, Pauli nodes, periodicity, translation invariance |
| kinetic energy | the free gas is an exact eigenstate (E_L constant to 10⁻⁹); analytic Gaussian and plane-wave cases; finite differences |
| sampler | flat and von Mises distributions with exact moments |
| error analysis | white noise and AR(1) series with exact errors and τ_int |
| central potential | brute-force image sums; Hartree–Fock energy of the Slater determinant, computed analytically and by sampling |
| spin-isospin operators | explicit Pauli matrices on all 16 pair states; exact Hartree–Fock expectation values with spin-isospin exchange |
| tensor force | S_ij algebra (singlet annihilation, S_ij² = 6 + 2σ·σ − 2S_ij); matrix–vector product over all 2^N spin states with random spinors; exact zero in a closed-shell determinant |
| optimizer | linear models with known solution; scale invariance; energy and variance reduction |
| NV2 | deuteron binding energy and D-state probability, ¹S₀ nn scattering length; line-by-line port of the original Fortran implementation, compared channel by channel |
| NV2 tables | spline tables against the analytic radial functions; brute-force 16×16 spin-isospin matrices for an np pair |
| charge-dependent operators | explicit isospin matrices; L·S T_ij = −3 L·S − L·S τ·τ for np pairs |
| spin-orbit correlations | the correlation factor built from the definition, L·S applied to Φ at every spin-isospin target with derivatives by automatic differentiation of the determinant |
| pair moves (rank-2 updates) | full evaluation of Ψ at moved positions, new spinors and swapped isospins (10⁻¹⁵); potential terms with and without the fast path |
| NV3 | full 2⁸-dimensional spin space of 8 neutrons, three-body operators as explicit matrices (no reduction), every channel |
| energy decomposition | the channels add up to the total local energy |
| pair observables | Σ_{i<j} σ_i·σ_j = 2S(S+1) − 3N/2 and the same for τ, exact at every configuration for S², T² eigenstates |
| EOS and TOV | Schwarzschild interior solution; Newtonian n = 1 polytrope (radius and Love number); thermodynamic consistency, also along β-equilibrium |
| crust | the full SLy EOS reproduces its published M_max and R₁.₄; crust and core joined with ε(P) increasing |

At the level of physics results, three cross-checks are used: the two-body PNM
energies are compared with BHF for the same NV2 model; light-nucleus energies
(²H, ³H, ⁴He) in free space test the isospin machinery and the three-body force; and
the published QMC results for the NV2+3 models in neutron matter fix the size of the
variational bias.

---

## Implementation and infrastructure

### Language and libraries

The whole simulation is written in **Python with JAX**:

- `jax.jit` compiles every function in the inner loop to optimized CPU or GPU code;
- `jax.vmap` runs the same code over hundreds of walkers, and over all
  spin-isospin target configurations of the local energy, at once;
- automatic differentiation provides the kinetic energy, the derivatives needed by
  the momentum-dependent operators, and the log-derivatives for the optimizer, all
  exactly;
- the same code runs on a laptop CPU and on cluster GPUs without changes.

All calculations are done in **double precision** (`jax_enable_x64`): determinants
and the SR metric lose accuracy quickly in single precision. NumPy and SciPy are used
for setup (shells, tables) and analysis (blocking, fits).

### Design principles

- **Pure functions.** Every wavefunction is a pair `(params, log_psi)` with
  `log_psi(params, r, s, t) → ln Ψ` for one configuration; batching is done outside
  with `vmap`. Every potential term has the signature
  `term(log_psi, params, r, s, t) → complex`, and receives `log_psi` itself because
  spin-isospin operators need Ψ at modified configurations.
- **Setup versus loop.** Quantities computed once (shells, pair lists, image vectors,
  potential tables) use NumPy; everything evaluated in the Monte Carlo loop uses
  `jax.numpy` with static array shapes, so it can be compiled.
- **Explicit randomness.** JAX random keys are passed and split explicitly, so every
  run is exactly reproducible from its seed.
- **Tests first.** Each module was written against a test file that fixed its
  interface and its exact reference results before the code existed.

### Development environment

The project is managed with [uv](https://docs.astral.sh/uv/): a project-local Python
(3.12) and virtual environment, exact dependency versions in `uv.lock`, and no root
privileges required. GPU support comes from the CUDA builds of JAX installed as pip
packages; only an NVIDIA driver is needed on the machine.

### Hardware

Jastrow–Slater calculations of a few tens of nucleons run on a single workstation GPU
or on a CPU. Operator-correlated wavefunctions at N = 28–66 and density scans for four
models benefit from data-centre GPUs with fast double-precision arithmetic; the code
scales to several GPUs by sharding the walkers.

---

## Installation and usage

```bash
# install uv (once), then:
git clone https://github.com/Alessio-Paradisi/NSmatter.git
cd NSmatter
uv sync                      # creates the environment from uv.lock
uv run pytest -q             # runs the full test suite
```

For CPU-only machines, replace the CUDA build of JAX with the plain one
(`uv add jax`). Check the device with:

```bash
uv run python -c "import jax; print(jax.devices())"
```

Example runs (on a multi-core CPU the walkers are spread over all cores; the number
of walkers must be a multiple of the number of cores):

```bash
# one density: pure neutron matter, 14 neutrons, NV2-Ib (+ NV3-Ib*)
uv run python scripts/run_nv2.py --model Ib --density 0.16 --walkers 132 --iterations 60 --measurements 40
uv run python scripts/run_nv2.py --model Ib --three-body "Ib*" --density 0.16 --walkers 132

# a density scan (each point in its own process; finished points are skipped on restart)
uv run python scripts/eos_scan.py --models Ib --densities 0.04 0.08 0.12 0.16 0.20 0.24 \
    --out results/vmc_Ib.jsonl -- --walkers 132 --iterations 60 --measurements 40

# the complete report (EOS, energy decomposition, structure, neutron stars, quality)
uv run python scripts/report.py --results results/vmc_Ib.jsonl
uv run python scripts/report.py --results results/vmc_Ib.jsonl --quality-only

# single figures: neutron stars (also beta equilibrium, comparison data), plots, animations
uv run python scripts/neutron_star.py --results results/vmc_Ib.jsonl --models Ib
uv run python scripts/plot_results.py pairs --results results/vmc_Ib.jsonl --model Ib --densities 0.08 0.16
uv run python scripts/plot_results.py config --results results/vmc_Ib.jsonl --model Ib --density 0.16 --animate
```

A point of pure neutron matter (14 neutrons, 132 walkers, 60 SR iterations,
40 measurements) takes about 40 minutes on a 12-core workstation CPU.

---

## Outputs and the report

```
results/
├── vmc_<model>.jsonl     one JSON record per run: E/N, energy decomposition, pair
│                         observables, <S²>, <T²>, free-gas energies, settings,
│                         optimized parameters, code version, timings
├── configs/              walker configurations and trajectories of each run (.npz)
├── logs/                 terminal output of each run
├── figures/              single figures (PDF) and animations (GIF)
├── reports/              report.pdf (complete) and quality.pdf
└── summary/              numerical summaries (neutron_star.json, report_errors.json:
                          every model, every uncertainty, the combined result)
```

`report.pdf` collects, for every model variant present in the results: an overview
table; the equation of state with fit, residuals, finite-size-corrected points,
pressure, speed of sound and published benchmarks; the energy decomposition by
channel and group; pair distributions and spin correlations (also scaled with k_F);
the neutron-star sequence with bootstrap bands, M(ρ_c) and Λ(M); the uncertainty
pages (EOS and M(R) bands of the models against the AFDMC fits, the error budget
table and the shift of every model); configuration snapshots; and a quality page for
every run. The energy decomposition has one page per NV2 model, followed by a page with every
channel at the lowest, the saturation and the highest computed density; a variant computed at
a single density is drawn with open markers and no line. Each run is checked automatically for
an unconverged optimization (drift), a mismatch between optimization and
measurement, correlated measurements (τ_int, with the corrected error), badly tuned
step sizes and the total spin.

---

## Repository layout

```
NSmatter/
├── pyproject.toml, uv.lock     project definition and exact dependency versions
├── src/nsmatter/               the package
│   ├── constants.py            ħc, nucleon masses, ħ²/2m
│   ├── system.py               the four populations, density, box, labels
│   ├── geometry.py             periodic box: wrapping, minimum image, momentum shells
│   ├── spin.py                 spin coordinates: labels and spinors
│   ├── wavefunctions/
│   │   ├── slater.py           spin-isospin plane-wave Slater determinant
│   │   ├── jastrow.py          central Jastrow factor
│   │   └── correlations.py     operator correlations, rank-2 pair moves
│   ├── hamiltonian/
│   │   ├── kinetic.py          kinetic local energy by automatic differentiation
│   │   ├── radial.py           radial functions: Gaussian sums, spline tables, image sums
│   │   ├── central.py          central test potentials
│   │   ├── exchange.py         σ·σ, τ·τ by exchange of coordinates
│   │   ├── pair_operators.py   static pair operators: v6, charge-dependent terms, tensor
│   │   ├── momentum.py         L·S, L², (L·S)² (with τ·τ and T_ij forms)
│   │   └── three_body.py       three-nucleon force as pair operators (neutron matter)
│   ├── potentials/
│   │   ├── nv2.py              NV2 radial functions (long range, contacts), models, LECs
│   │   ├── two_body.py         deuteron and ¹S₀ scattering length (validation)
│   │   ├── nv2_vmc.py          NV2 spline tables and potential term, operator groups
│   │   └── nv3.py              NV3 couplings and radial functions
│   ├── local_energy.py         E_L and its decomposition by channel
│   ├── observables.py          pair distributions, spin-isospin correlations
│   ├── sampler.py              Metropolis sampling of positions and spinors
│   ├── optimize.py             stochastic reconfiguration
│   ├── stats.py                blocking analysis, autocorrelation time
│   ├── eos.py                  EOS fits, thermodynamics, β-equilibrium, causality
│   ├── tov.py                  TOV and tidal deformability
│   ├── results.py              loading and selecting run records
│   ├── benchmarks.py           published reference results
│   └── plotting.py             drawing of configurations in the periodic box
├── scripts/
│   ├── run_nv2.py              one VMC run (optimization + measurement)
│   ├── eos_scan.py             density scans, restartable
│   ├── report.py               the complete PDF report
│   ├── neutron_star.py         EOS fit and neutron stars (β-equilibrium, comparisons)
│   └── plot_results.py         single figures and animations
└── tests/                      one test file per module, against exact references
```

---

## Units and conventions

- Energies and masses in MeV, lengths in fm, densities in fm⁻³; ħc = 197.3269804 MeV fm
  and ħ²/2m = 20.7355 MeV fm² with the average nucleon mass.
- Spin label +1 = up, −1 = down; isospin +1 = proton, −1 = neutron.
- Particles are stored in species order n↑, n↓, p↑, p↓.
- Two-body basis states |a b⟩ have index 2·idx(a) + idx(b), with idx(↑) = 0, idx(↓) = 1.
- Energies are printed per particle; internally the local energy is the total energy
  of the box, so the SR step η has units of MeV⁻¹ and must shrink as N grows.

---

## References

**Interaction**

- M. Piarulli *et al.*, *Minimally nonlocal nucleon-nucleon potentials with chiral
  two-pion exchange including Δ resonances*, Phys. Rev. C **91**, 024003 (2015).
- M. Piarulli *et al.*, *Local chiral potentials with Δ-intermediate states and the
  structure of light nuclei*, Phys. Rev. C **94**, 054007 (2016).
- M. Piarulli *et al.*, *Light-nuclei spectra from chiral dynamics*,
  Phys. Rev. Lett. **120**, 052503 (2018).
- A. Baroni *et al.*, *Local chiral interactions, the tritium
  Gamow-Teller matrix element, and the three-nucleon contact term*, Phys. Rev. C **98**,
  044003 (2018).
- R. B. Wiringa, V. G. J. Stoks, R. Schiavilla, *Accurate nucleon-nucleon potential
  with charge-independence breaking*, Phys. Rev. C **51**, 38 (1995).

**Quantum Monte Carlo for nuclear systems**

- J. Carlson *et al.*, *Quantum Monte Carlo methods for nuclear physics*,
  Rev. Mod. Phys. **87**, 1067 (2015).
- J. E. Lynn, I. Tews, S. Gandolfi, A. Lovato, *Quantum Monte Carlo methods in nuclear
  physics: recent advances*, Annu. Rev. Nucl. Part. Sci. **69**, 279 (2019).
- F. Becca, S. Sorella, *Quantum Monte Carlo Approaches for Correlated Systems*
  (Cambridge University Press, 2017).
- W. M. C. Foulkes, L. Mitas, R. J. Needs, G. Rajagopal, *Quantum Monte Carlo
  simulations of solids*, Rev. Mod. Phys. **73**, 33 (2001).
- C. Lin, F. H. Zong, D. M. Ceperley, *Twist-averaged boundary conditions in
  continuum quantum Monte Carlo algorithms*, Phys. Rev. E **64**, 016702 (2001).
- S. Sorella, *Green function Monte Carlo with stochastic reconfiguration*,
  Phys. Rev. Lett. **80**, 4558 (1998).
- H. Flyvbjerg, H. G. Petersen, *Error estimates on averages of correlated data*,
  J. Chem. Phys. **91**, 461 (1989).

**Neutron matter and neutron stars**

- S. Fantoni, A. Sarsa, K. E. Schmidt, *Spin susceptibility of neutron matter at zero
  temperature*, Phys. Rev. Lett. **87**, 181101 (2001).
- I. Vidaña, I. Bombaci, *Equation of state and magnetic susceptibility of
  spin-polarized isospin-asymmetric nuclear matter*, Phys. Rev. C **66**, 045801 (2002).
- M. Piarulli *et al.*, *Benchmark calculations of pure neutron matter with realistic
  nucleon-nucleon interactions*, Phys. Rev. C **101**, 045801 (2020).
- A. Lovato, I. Bombaci, D. Logoteta, M. Piarulli, R. B. Wiringa, *Benchmark
  calculations of infinite neutron matter with realistic two- and three-nucleon
  potentials*, Phys. Rev. C **105**, 055808 (2022).
- T. Hinderer, *Tidal Love numbers of neutron stars*, Astrophys. J. **677**, 1216 (2008);
  S. Postnikov, M. Prakash, J. M. Lattimer, Phys. Rev. D **82**, 024016 (2010).
- F. Douchin, P. Haensel, *A unified equation of state of dense matter and neutron
  star structure*, Astron. Astrophys. **380**, 151 (2001) (SLy).
- P. Haensel, A. Y. Potekhin, *Analytical representations of unified equations of
  state of neutron-star matter*, Astron. Astrophys. **428**, 191 (2004).
- B. Fore, J. Kim, M. Hjorth-Jensen, A. Lovato, *Investigating the crust of neutron
  stars with neural-network quantum states*, Commun. Phys. **8**, 108 (2025).
