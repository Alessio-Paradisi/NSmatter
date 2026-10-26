"""Error analysis for correlated Monte Carlo time series (plain NumPy)."""

import numpy as np


def blocking(series):
    """Flyvbjerg-Petersen blocking analysis.

    Repeatedly averages neighbouring pairs of the series and records the
    naive standard error at each level. For correlated data the error grows
    with the level and then plateaus: the plateau is the true error.

    Returns (mean, errors per level, number of blocks per level).
    """
    x = np.asarray(series, dtype=float)
    mean = x.mean()
    errors, nblocks = [], []
    while x.size >= 2:
        # Naive error of the mean at this level.
        errors.append(x.std(ddof=1) / np.sqrt(x.size))
        nblocks.append(x.size)
        # Average pairs (x0+x1)/2, (x2+x3)/2, ...; an odd last element is dropped.
        n = x.size // 2
        x = 0.5 * (x[: 2 * n : 2] + x[1 : 2 * n : 2])
    return mean, np.array(errors), np.array(nblocks)


def blocking_error(series, min_blocks=32):
    """Mean and a conservative error estimate.

    Takes the largest blocking error among levels that still have at least
    `min_blocks` blocks: deeper levels have too few blocks to be reliable.
    """
    mean, errors, nblocks = blocking(series)
    ok = nblocks >= min_blocks
    err = errors[ok].max() if ok.any() else errors[0]
    return mean, err


def integrated_autocorrelation_time(series, c=5.0):
    """Integrated autocorrelation time tau_int, in units of the series spacing.

    tau_int = 1/2 + sum_{w>=1} rho(w), with rho the normalized autocorrelation.
    The sum is cut at the first window W >= c * tau (Sokal's automatic
    window): summing the noisy tail would only add noise.
    The autocorrelation is computed with an FFT (zero-padded to 2n so the
    circular correlation does not wrap around).
    """
    x = np.asarray(series, dtype=float) - np.mean(series)
    n = x.size
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    if acf[0] == 0.0:  # constant series: no fluctuations at all
        return 0.5
    acf = acf / acf[0]
    tau = 0.5
    for w in range(1, n):
        tau += acf[w]
        if w >= c * tau:
            break
    return tau
