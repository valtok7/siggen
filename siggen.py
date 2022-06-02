#%% Import module
#%matplotlib inline   # Uncomment when .ipynb
import yui
import math


#%% Test
freq = 128
fs = 1024
length = 8000
"""
cw = yui.gen_cw(1, freq, fs, length, log=False)
#pulse = gen_pulse(1, 10, 5, length, log=False)
#pulsed_cw = modulation(cw, pulse, log=False)
#noise = gen_noise(1, fs, length, log=True)
#output_csv("cw.csv", cw)
#output_bin("cw.bin", cw)
yui.anal_power(cw, length, log=True)
yui.anal_spect(cw, 128, fs=fs, window="flattop", log=True)
fs2 = 2000
tmp, start = yui.rate_conv(cw, 2, 1, log=True)
cw2 = tmp[start:len(tmp)]
yui.anal_power(cw2, len(cw2), log=True)
yui.anal_spect(cw2, 256, fs=fs2, window="flattop", log=True)

fs3 = 900
cw3 = yui.resample(cw, fs, fs3, len(cw), 51, log=True)
yui.anal_power(cw3, len(cw3), log=True)
yui.anal_spect(cw3, 256, 100, fs=fs3, window="flattop", log=True)
"""
sig_real = yui.gen_cos(1, freq, fs, length, "real", log=False)
spect, spect_abs, spect_abs_db, spect_phase, spect_freq = yui.anal_spect(sig_real, fs, fs=fs, window="rect", log=False)
print(spect[freq])
print(math.atan2(spect[freq].imag, spect[freq].real))
sig_real = yui.gen_cos(1, freq, fs, length, "real", 1, log=False)
spect, spect_abs, spect_abs_db, spect_phase, spect_freq = yui.anal_spect(sig_real, fs, fs=fs, window="rect", log=False)
print(spect[freq])
print(math.atan2(spect[freq].imag, spect[freq].real))
sig_real = yui.gen_cos(1, freq, fs, length, "real", 2, log=False)
spect, spect_abs, spect_abs_db, spect_phase, spect_freq = yui.anal_spect(sig_real, fs, fs=fs, window="rect", log=False)
print(spect[freq])
print(math.atan2(spect[freq].imag, spect[freq].real))
sig_real = yui.gen_cos(1, freq, fs, length, "real", 3, log=False)
spect, spect_abs, spect_abs_db, spect_phase, spect_freq = yui.anal_spect(sig_real, fs, fs=fs, window="rect", log=False)
print(spect[freq])
print(math.atan2(spect[freq].imag, spect[freq].real))
sig_real = yui.gen_cos(1, freq, fs, length, "real", 4, log=False)
spect, spect_abs, spect_abs_db, spect_phase, spect_freq = yui.anal_spect(sig_real, fs, fs=fs, window="rect", log=False)
print(spect[freq])
print(math.atan2(spect[freq].imag, spect[freq].real))

print(1)
# %%
