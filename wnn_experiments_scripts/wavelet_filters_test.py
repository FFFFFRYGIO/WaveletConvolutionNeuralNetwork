import numpy as np
import pywt

# pick your original wavelet
w = pywt.Wavelet('db4')

# extract the four filters
dec_lo, dec_hi, rec_lo, rec_hi = w.filter_bank

# rebuild a perfectly equivalent filter bank:
fb = pywt.orthogonal_filter_bank(rec_lo)  # note: DEC-LO here, not DEC-HI
cust1 = pywt.Wavelet('cust1', filter_bank=fb)

assert np.allclose(cust1.dec_lo, w.dec_lo)
assert np.allclose(cust1.dec_hi, w.dec_hi)
assert np.allclose(cust1.rec_lo, w.rec_lo)
assert np.allclose(cust1.rec_hi, w.rec_hi)
