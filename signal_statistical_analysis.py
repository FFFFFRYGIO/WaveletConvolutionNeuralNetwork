import numpy as np
from typing import Dict, Any, Optional

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
) -> Dict[str, Any]:
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
    # Basic histogram
    hist, bin_edges = np.histogram(signal, bins=bins, density=density)

    # Summary stats
    mean = np.mean(signal)
    median = np.median(signal)
    std = np.std(signal)
    mn, mx = np.min(signal), np.max(signal)
    percentiles = {p: float(np.percentile(signal, p)) for p in (5, 25, 75, 95)}

    result: Dict[str, Any] = {
        'histogram': hist,
        'bin_edges': bin_edges,
        'mean': float(mean),
        'median': float(median),
        'std': float(std),
        'min': float(mn),
        'max': float(mx),
        'percentiles': percentiles
    }

    # Optional KDE
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
