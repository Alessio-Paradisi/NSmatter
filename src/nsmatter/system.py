"""Composition of the simulated system: four populations in a periodic box.

Particles are stored in a fixed species order:
    neutron up, neutron down, proton up, proton down.
Each particle keeps its spin and isospin label for the whole run
("fixed labels"). This is exact for permutation-symmetric quantities,
because |Psi|^2 is unchanged when (r, s, t) triples are permuted.

Conventions: spin s = +1 (up) / -1 (down); isospin t = +1 proton / -1 neutron.
"""

from dataclasses import dataclass

import numpy as np

# Species names in storage order, and the labels each one carries.
SPECIES = ("n_up", "n_down", "p_up", "p_down")
SPIN = {"n_up": +1, "n_down": -1, "p_up": +1, "p_down": -1}
ISOSPIN = {"n_up": -1, "n_down": -1, "p_up": +1, "p_down": +1}


# frozen=True: the object cannot be modified after creation. That makes it
# hashable, which JAX needs to use it as a static (compile-time) argument.
@dataclass(frozen=True)
class System:
    """Four populations at baryon density `density` (fm^-3)."""

    n_up: int  # neutrons with spin up
    n_down: int  # neutrons with spin down
    p_up: int  # protons with spin up
    p_down: int  # protons with spin down
    density: float  # baryon density rho (fm^-3)

    # --- basic counts ---------------------------------------------------------
    @property
    def counts(self) -> tuple:
        """Number of particles per species, in storage order."""
        return (self.n_up, self.n_down, self.p_up, self.p_down)

    @property
    def n_particles(self) -> int:
        """Total number of nucleons N."""
        return sum(self.counts)

    @property
    def box_length(self) -> float:
        """Side L of the cubic box (fm), from N / L^3 = rho."""
        return (self.n_particles / self.density) ** (1.0 / 3.0)

    # --- where each species sits in the particle arrays ----------------------
    @property
    def species_slices(self) -> tuple:
        """Index range of each species, as Python slices (empty ones included).

        Example: counts (2, 1, 0, 3) -> slice(0,2), slice(2,3), slice(3,3), slice(3,6).
        Later, r[slice] selects the positions of one species.
        """
        out, start = [], 0
        for n in self.counts:
            out.append(slice(start, start + n))
            start += n
        return tuple(out)

    # --- per-particle labels (setup only: NumPy) -----------------------------
    @property
    def spins(self) -> np.ndarray:
        """Spin label of every particle (+1 / -1), shape (N,)."""
        return np.concatenate([np.full(n, SPIN[k]) for k, n in zip(SPECIES, self.counts)]).astype(np.int32)

    @property
    def isospins(self) -> np.ndarray:
        """Isospin label of every particle (+1 proton / -1 neutron), shape (N,)."""
        return np.concatenate([np.full(n, ISOSPIN[k]) for k, n in zip(SPECIES, self.counts)]).astype(np.int32)

    # --- derived physical quantities -----------------------------------------
    @property
    def proton_fraction(self) -> float:
        """x = N_p / N."""
        return (self.p_up + self.p_down) / self.n_particles

    @property
    def polarization(self) -> float:
        """Total spin polarization Delta = (N_up - N_down) / N of the reference state."""
        return (self.n_up + self.p_up - self.n_down - self.p_down) / self.n_particles

    def describe(self) -> str:
        """One-line summary for printing at the start of a run."""
        return (
            f"N = {self.n_particles} (n_up {self.n_up}, n_down {self.n_down}, "
            f"p_up {self.p_up}, p_down {self.p_down}), rho = {self.density} fm^-3, "
            f"L = {self.box_length:.4f} fm, x_p = {self.proton_fraction:.4f}, "
            f"Delta = {self.polarization:.4f}"
        )


# -----------------------------------------------------------------------------
# Shortcuts for the two most common cases
# -----------------------------------------------------------------------------
def pure_neutron_matter(n_per_spin: int, density: float) -> System:
    """Unpolarized pure neutron matter, N = 2 * n_per_spin."""
    return System(n_per_spin, n_per_spin, 0, 0, density)


def symmetric_matter(n_per_species: int, density: float) -> System:
    """Unpolarized symmetric nuclear matter, N = 4 * n_per_species."""
    return System(n_per_species, n_per_species, n_per_species, n_per_species, density)
