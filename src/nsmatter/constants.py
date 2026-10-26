"""Physical constants. Units: energies and masses in MeV, lengths in fm."""

# Conversion constant hbar * c (MeV fm). It turns momenta in fm^-1 into MeV.
HBARC = 197.3269804

# Nucleon rest masses (MeV), CODATA values.
M_PROTON = 938.27208816
M_NEUTRON = 939.56542052

# Average nucleon mass (MeV). For now protons and neutrons get the same mass,
# as is common in QMC calculations; the NV2 mass convention comes later.
M_NUCLEON = 0.5 * (M_PROTON + M_NEUTRON)

# hbar^2 / 2m (MeV fm^2), the coefficient of the kinetic energy:
# T = -hbar^2/2m * laplacian. Written as (hbar c)^2 / (2 m c^2) so that
# masses in MeV give the result in MeV fm^2. Value: about 20.7355.
HBAR2_2M = HBARC**2 / (2.0 * M_NUCLEON)
