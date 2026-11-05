"""Composition of the simulated system: four populations in a periodic box.

Particles are stored in a fixed order of species:
    neutron up, neutron down, proton up, proton down.
Each particle keeps its spin and isospin label for the whole run
("fixed labels"); this is exact for permutation-symmetric observables
because |Psi|^2 is invariant under permutations of (r, s, t) triples.

Conventions: spin s = +1 (up) / -1 (down); isospin t = +1 proton, -1 neutron.
"""

from dataclasses import dataclass

import numpy as np

SPECIES = ("n_up", "n_down", "p_up", "p_down")
SPIN = {"n_up": +1, "n_down": -1, "p_up": +1, "p_down": -1}
ISOSPIN = {"n_up": -1, "n_down": -1, "p_up": +1, "p_down": +1}


@dataclass(frozen=True)
class System:
    """Four populations at baryon density `density` (fm^-3)."""

    n_up: int
    n_down: int
    p_up: int
    p_down: int
    density: float

    @property
    def counts(self) -> tuple:
        return (self.n_up, self.n_down, self.p_up, self.p_down)

    @property
    def n_particles(self) -> int:
        return sum(self.counts)

    @property
    def box_length(self) -> float:
        """Side L of the cubic box, from N / L^3 = density."""
        return (self.n_particles / self.density) ** (1.0 / 3.0)

    @property
    def species_slices(self) -> tuple:
        """Index range of each species in the particle array (empty ones included)."""
        out, start = [], 0
        for n in self.counts:
            out.append(slice(start, start + n))
            start += n
        return tuple(out)

    @property
    def spins(self) -> np.ndarray:
        return np.concatenate([np.full(n, SPIN[k]) for k, n in zip(SPECIES, self.counts)]).astype(np.int32)

    @property
    def isospins(self) -> np.ndarray:
        return np.concatenate([np.full(n, ISOSPIN[k]) for k, n in zip(SPECIES, self.counts)]).astype(np.int32)

    @property
    def proton_fraction(self) -> float:
        return (self.p_up + self.p_down) / self.n_particles

    @property
    def polarization(self) -> float:
        """Total spin polarization (N_up - N_down) / N of the reference state."""
        return (self.n_up + self.p_up - self.n_down - self.p_down) / self.n_particles

    def describe(self) -> str:
        return (
            f"N = {self.n_particles} (n_up {self.n_up}, n_down {self.n_down}, "
            f"p_up {self.p_up}, p_down {self.p_down}), rho = {self.density} fm^-3, "
            f"L = {self.box_length:.4f} fm, x_p = {self.proton_fraction:.4f}, "
            f"Delta = {self.polarization:.4f}"
        )


def pure_neutron_matter(n_per_spin: int, density: float) -> System:
    """Unpolarized pure neutron matter, N = 2 * n_per_spin."""
    return System(n_per_spin, n_per_spin, 0, 0, density)


def symmetric_matter(n_per_species: int, density: float) -> System:
    """Unpolarized symmetric nuclear matter, N = 4 * n_per_species."""
    return System(n_per_species, n_per_species, n_per_species, n_per_species, density)
