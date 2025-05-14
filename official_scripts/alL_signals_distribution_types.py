"""Get example ECG signal and detect its samples values distribution type."""
import os

import matplotlib
import numpy as np
import pandas as pd
import scipy.stats as stats
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
            })

    return best


def get_ecg_signals() -> tuple[tuple[NDArray, str]]:
    """Get example ECG signal from ECGData from Physionet"""

    data_source = os.getenv('DATA_SOURCE')
    mat_data = loadmat(data_source)
    raw_data = mat_data['ECGData']
    source_signals, labels = raw_data[0, 0]

    return zip(source_signals, labels)


def generate_signals_distribution_analysis(file_name: str):
    """Generate distribution type detection for all signals from ECGData"""

    signals_distribution_analysis = []

    for ecg_signal, tag in get_ecg_signals():
        dist_type = detect_distribution_type(ecg_signal)
        print(dist_type)
        signals_distribution_analysis.append({
            'tag': tag,
            'distribution': dist_type['name'],
            'params': dist_type['params'],
            'sse': dist_type['sse']
        })

    pd.DataFrame(signals_distribution_analysis).to_csv(file_name)


def main():
    """Get signal, detect its samples values distribution, finally display the results."""

    file_name = 'signals_distribution_analysis.csv'

    # generate_signals_distribution_analysis(file_name)

    generated_distribution_analysis = pd.read_csv(file_name)

    pct = (generated_distribution_analysis['distribution']
           .value_counts(normalize=True).mul(100).round(2))

    pct_df = pct.rename_axis('Distribution').reset_index(name='Percentage')

    print(pct_df)


if __name__ == '__main__':
    main()
