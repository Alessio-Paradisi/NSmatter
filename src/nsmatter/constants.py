"""Physical constants in MeV and fm."""

HBARC = 197.3269804  # MeV fm
M_PROTON = 938.27208816  # MeV
M_NEUTRON = 939.56542052  # MeV
M_NUCLEON = 0.5 * (M_PROTON + M_NEUTRON)  # equal-mass approximation

# hbar^2 / 2m for the average nucleon mass, in MeV fm^2 (about 20.736)
HBAR2_2M = HBARC**2 / (2.0 * M_NUCLEON)
