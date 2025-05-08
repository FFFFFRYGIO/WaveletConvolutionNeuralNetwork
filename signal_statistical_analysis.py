import numpy as np
import scipy.stats as stats

try:
    from scipy.stats import gaussian_kde
except ImportError:
    gaussian_kde = None  # KDE will be unavailable if scipy isn't installed


def data_distribution(
        signal: np.ndarray,
        bins: int = 50,
        density: bool = True,
        compute_kde: bool = False,
        kde_points: int = 200
) -> dict:
    """
    Compute the distribution and summary stats of a 1D ECG signal.

    Parameters
    ----------
    signal : np.ndarray
        1D array of ECG amplitude values.
    bins : int
        Number of bins to use for the histogram.
    density : bool
        If True, the histogram is normalized to form a probability density.
    compute_kde : bool
        If True and scipy is available, compute a Gaussian KDE estimate.
    kde_points : int
        Number of points at which to evaluate the KDE.

    Returns
    -------
    result : dict
        {
            'histogram': np.ndarray of shape (bins,),
            'bin_edges': np.ndarray of shape (bins+1,),
            'mean': float,
            'median': float,
            'std': float,
            'min': float,
            'max': float,
            'percentiles': {
                p: float for p in [5, 25, 75, 95]
            },
            'kde': {
                'xs': np.ndarray (kde_points,),
                'density': np.ndarray (kde_points,)
            }           # only if compute_kde=True and scipy installed
        }
    """
    # Compute histogram (no display)
    hist, bin_edges = np.histogram(signal, bins=bins, density=density)

    # Summary statistics
    mean = float(np.mean(signal))
    median = float(np.median(signal))
    std = float(np.std(signal))
    mn, mx = float(np.min(signal)), float(np.max(signal))
    percentiles = {p: float(np.percentile(signal, p)) for p in (5, 25, 75, 95)}

    result: dict = {
        'histogram': hist,
        'bin_edges': bin_edges,
        'mean': mean,
        'median': median,
        'std': std,
        'min': mn,
        'max': mx,
        'percentiles': percentiles
    }

    # Optional KDE computation
    if compute_kde:
        if gaussian_kde is None:
            raise ImportError("scipy is required for KDE but is not installed.")
        kde = gaussian_kde(signal)
        xs = np.linspace(mn, mx, kde_points)
        density_vals = kde(xs)
        result['kde'] = {
            'xs': xs,
            'density': density_vals
        }

    return result


def detect_distribution_type(
        signal: np.ndarray,
        bins: int = 100,
        distributions: list[stats.rv_continuous] = None,
        use_ks: bool = False
) -> dict:
    """
    Try to detect which theoretical distribution best fits `signal`.

    Parameters
    ----------
    signal : np.ndarray
        1D array of samples.
    bins : int
        Number of bins for histogram (only used for SSE criterion).
    distributions : list of scipy.stats continuous distributions, optional
        If None, defaults to [norm, expon, gamma, beta, lognorm, uniform].
    use_ks : bool
        If True, also perform a KS test and return its p-value for the winner.

    Returns
    -------
    result : dict
        {
           'name': <best dist name>,
           'params': <fitted params tuple>,
           'sse': <sum squared error on PDF>,
           'ks_pvalue': <if use_ks, p-value of KS test>
        }
    """
    if distributions is None:
        distributions = [
            stats.norm,
            stats.expon,
            stats.gamma,
            stats.beta,
            stats.lognorm,
            stats.uniform
        ]

    # 1) histogram
    hist, bin_edges = np.histogram(signal, bins=bins, density=True)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    best = {'name': None, 'params': None, 'sse': np.inf, 'ks_pvalue': None}

    for dist in distributions:
        name = dist.name
        try:
            # 2) fit distribution to data
            params = dist.fit(signal)

            # 3) compute SSE between hist and PDF at bin centers
            pdf_vals = dist.pdf(centers, *params)
            sse = np.sum((hist - pdf_vals) ** 2)

            # 4) optionally do KS test on raw data
            pvalue = None
            if use_ks:
                _, pvalue = stats.kstest(signal, name, args=params)

            # 5) check if this is the new best
            if sse < best['sse']:
                best.update({
                    'name': name,
                    'params': params,
                    'sse': sse,
                    'ks_pvalue': pvalue
                })

        except Exception:
            # some distributions may fail to fit; just skip them
            continue

    return best
