"""Total local energy E_L = (H Psi)(X) / Psi(X) for one walker.

Potential terms are functions
    term(log_psi, params, r, s, t) -> complex
so that spin-isospin operators can evaluate Psi at swapped configurations.
make_local_energy_components gives the same energy split into kinetic energy
and one part per potential channel.
"""

from .hamiltonian.kinetic import kinetic_local_energy


def make_local_energy(log_psi, potential_terms=()):
    def local_energy(params, r, s, t):
        e = kinetic_local_energy(log_psi, params, r, s, t)
        for term in potential_terms:
            e = e + term(log_psi, params, r, s, t)
        return e

    return local_energy


def make_local_energy_components(log_psi, potential_terms=()):
    """E_L split into parts: dict with "kinetic" and one entry per potential channel.

    Terms with a by_channel attribute (pair_operators, momentum, nv2_vmc) give
    one entry per channel ("v_<channel>"); other terms one entry each ("term<k>").
    The sum of all entries is the local energy of make_local_energy.
    """

    def components(params, r, s, t):
        out = {"kinetic": kinetic_local_energy(log_psi, params, r, s, t)}
        for k, term in enumerate(potential_terms):
            if hasattr(term, "by_channel"):
                out.update({f"v_{ch}": v for ch, v in term.by_channel(log_psi, params, r, s, t).items()})
            else:
                out[f"term{k}"] = term(log_psi, params, r, s, t)
        return out

    return components
