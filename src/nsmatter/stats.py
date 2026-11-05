"""Error analysis for correlated Monte Carlo time series."""

import numpy as np


def blocking(series):
    """Flyvbjerg-Petersen blocking analysis.

    Repeatedly averages neighbouring pairs and records the naive standard
    error at each level. For correlated data the error grows with the level
    and then plateaus; the plateau is the true statistical error.

    Returns (mean, errors per level, number of blocks per level).
    """
    x = np.asarray(series, dtype=float)
    mean = x.mean()
    errors, nblocks = [], []
    while x.size >= 2:
        errors.append(x.std(ddof=1) / np.sqrt(x.size))
        nblocks.append(x.size)
        n = x.size // 2
        x = 0.5 * (x[: 2 * n : 2] + x[1 : 2 * n : 2])
    return mean, np.array(errors), np.array(nblocks)


def blocking_error(series, min_blocks=32):
    """Mean and a conservative error: the largest blocking error with at least min_blocks blocks."""
    mean, errors, nblocks = blocking(series)
    ok = nblocks >= min_blocks
    err = errors[ok].max() if ok.any() else errors[0]
    return mean, err


def integrated_autocorrelation_time(series, c=5.0):
    """tau_int with Sokal's automatic window (window W is the first with W >= c * tau)."""
    x = np.asarray(series, dtype=float) - np.mean(series)
    n = x.size
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    if acf[0] == 0.0:
        return 0.5
    acf = acf / acf[0]
    tau = 0.5
    for w in range(1, n):
        tau += acf[w]
        if w >= c * tau:
            break
    return tau
