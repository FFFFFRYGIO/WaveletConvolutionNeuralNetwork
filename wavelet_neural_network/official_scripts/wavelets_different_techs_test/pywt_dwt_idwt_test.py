"""This script is creating default and custom wavelets with PyWavelets and tests if the approach is correct."""
from copy import deepcopy

import pandas as pd
import pywt


def main():
    """Create default and custom wavelets and compare them."""
    wavelet_name = 'db4'

    original_wavelet = pywt.Wavelet(wavelet_name)
    original_filter_bank = original_wavelet.filter_bank
    original_rec_lo_filter = deepcopy(original_wavelet.rec_lo)

    copied_rec_lo_filter = deepcopy(original_wavelet.rec_lo)
    custom_rec_lo_filter = [p * (1 + i / 1000) for i, p in enumerate(copied_rec_lo_filter)]

    assert original_rec_lo_filter != custom_rec_lo_filter, 'Original and custom filters do not differ!'

    wavelet_recreated_filter_bank = pywt.Wavelet(f'rec_fb_{wavelet_name}', filter_bank=original_filter_bank)
    wavelet_recreated_filter_bank.orthogonal = True
    wavelet_recreated_filter_bank.biorthogonal = True

    wavelet_recreated_rec_lo_filter_bank = pywt.orthogonal_filter_bank(original_rec_lo_filter)
    wavelet_recreated_rec_lo = pywt.Wavelet(f'rec_lo_{wavelet_name}', filter_bank=wavelet_recreated_rec_lo_filter_bank)
    wavelet_recreated_rec_lo.orthogonal = True
    wavelet_recreated_rec_lo.biorthogonal = True

    wavelet_custom_rec_lo_filter_bank = pywt.orthogonal_filter_bank(custom_rec_lo_filter)
    wavelet_custom_rec_lo = pywt.Wavelet(f'custom_{wavelet_name}', filter_bank=wavelet_custom_rec_lo_filter_bank)
    wavelet_custom_rec_lo.orthogonal = True
    wavelet_custom_rec_lo.biorthogonal = True

    comparison_rows = []
    for wave_rec_fb, wave_rec_lo, wave_cust in zip(
            wavelet_recreated_filter_bank.filter_bank,
            wavelet_recreated_rec_lo.filter_bank,
            wavelet_custom_rec_lo.filter_bank
    ):
        for wave1_ft, wave2_ft, wave3_ft in zip(wave_rec_fb, wave_rec_lo, wave_cust):
            comparison_rows.append({
                'wave_rec_fb': wave1_ft,
                'wave_rec_lo': wave2_ft,
                'wave_cust': wave3_ft,
                'Diff1-2': wave1_ft - wave2_ft,
                'Diff1-3': wave1_ft - wave3_ft,
                'Diff2-3': wave2_ft - wave3_ft,
            })

    df = pd.DataFrame(comparison_rows)

    assert (df['Diff1-2'].abs() < 1e-10).all(), 'Reconstructed wavelets differ too much'

    reconstructed_custom_diff = df['Diff1-3'] - df['Diff2-3']
    assert (reconstructed_custom_diff.abs() < 1e-10).all(), 'Custom wavelet differ with reconstructed too much'

    print('Confirmed that pywt correctly creates wavelets with custom filters!')


if __name__ == '__main__':
    main()
