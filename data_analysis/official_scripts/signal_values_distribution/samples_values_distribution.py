"""Get example ECG signal and detect its samples values distribution type."""
import os
from typing import cast

import matplotlib
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.io import loadmat

matplotlib.use("TkAgg")


def detect_distribution_type(signal: NDArray) -> dict:
    """Use Python tools to detect distribution best fit for signal."""

    available_distributions = [stats.expon, stats.gamma, stats.beta, stats.lognorm, stats.uniform]

    hist, bin_edges = np.histogram(signal, bins=100, density=True)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    best = {'name': None, 'params': None, 'sse': np.inf, 'dist': None}

    for distribution in available_distributions:
        params = distribution.fit(signal)
        pdf_vals = distribution.pdf(centers, *params)
        sse = np.sum((hist - pdf_vals) ** 2)

        if sse < best['sse']:
            best.update({
                'name': distribution.name,
                'params': params,
                'sse': sse,
                'dist': distribution
            })

    xs = np.linspace(centers.min(), centers.max(), 200)
    *shape_args, loc, scale = best['params']
    plot_pdf_vals = best['dist'].pdf(xs, *shape_args, loc=loc, scale=scale)
    best['plot_values'] = {'xs': xs, 'pdf_vals': plot_pdf_vals}

    distribution_type_full_text = (
        f"distribution type name: {best['name']}\n"
        f"\n"
        f"distribution parameters:\n"
        f"{'\n'.join(str(val) for val in best['params'])}\n"
        f"\n"
        f"sum of squared errors:\n"
        f"{best['sse']}")

    best['text'] = distribution_type_full_text

    return best


def get_example_ecg_signal() -> tuple[NDArray, str]:
    """Get example ECG signal from ECGData from Physionet"""

    data_source = os.getenv('DATA_SOURCE')
    mat_data = loadmat(data_source)
    raw_data = mat_data['ECGData']
    source_signals, labels = raw_data[0, 0]

    # Get healthy signal based on documentation
    nsr_signal = source_signals[96 + 30]

    return nsr_signal, 'NSR'


def main():
    """Get signal, detect its samples values distribution, finally display the results."""

    ecg_signal, tag = get_example_ecg_signal()

    fig, axs = plt.subplots(
        ncols=3,
        figsize=(12, 4),
    )

    signal_ax = cast(Axes, axs[0])
    signal_ax.plot(ecg_signal)
    signal_ax.set_title(f"ECG signal: {tag} (len={len(ecg_signal)})")
    signal_ax.set_xlabel("Probe number")
    signal_ax.set_ylabel("Amplitude [mV]")

    distribution_ax = cast(Axes, axs[1])
    distribution_ax.hist(ecg_signal, bins=100, density=True)
    distribution_ax.set_title(f"Signal distribution: {tag} (len={len(ecg_signal)})")
    distribution_ax.set_xlabel("Value")
    distribution_ax.set_ylabel("Occurrence")

    distribution_type = detect_distribution_type(ecg_signal)

    distribution_type_ax = cast(Axes, axs[2])
    distribution_type_ax.set_title("Best fit")
    distribution_type_ax.text(
        0.5, 0.5, distribution_type['text'],
        ha='center', va='center',
        fontsize=10,
        family='monospace'
    )

    distribution_ax.plot(
        distribution_type['plot_values']['xs'],
        distribution_type['plot_values']['pdf_vals'],
        '--',
        label=f"{distribution_type['name']} fit"
    )

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
