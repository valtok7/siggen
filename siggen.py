#%% Import module
#%matplotlib inline   # Uncomment when .ipynb
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as npsig
import struct

#%% Generate CW
def gen_cw(ampl, freq, fs, length, init_phase=0, log=False):
    n = np.arange(length)
    iq = ampl * np.cos((2 * np.pi * freq * n / fs) + init_phase) \
        + 1j * (ampl * np.sin((2 * np.pi * freq * n / fs) + init_phase))
    if log:
        plt.plot(n, iq.real, label="i")
        plt.plot(n, iq.imag, label="q")
        plt.title(f"gen_cw ampl={ampl},freq={freq},fs={fs}")
        plt.xlabel("sample")
        plt.ylabel("amplitude")
        plt.legend()
        plt.show()
    return iq

#%% Modulation
def modulation(iq, mod_sig, log=False):
    iq_out = np.multiply(iq, mod_sig)
    if log:
        n  = np.arange(len(iq_out))
        plt.plot(n, iq_out.real, label="i")
        plt.plot(n, iq_out.imag, label="q")
        plt.plot(n, mod_sig.real, label="mod_sig_i")
        plt.plot(n, mod_sig.imag, label="mod_sig_q")
        plt.title(f"modulation")
        plt.xlabel("sample")
        plt.ylabel("amplitude")
        plt.legend()
        plt.show()
    return iq_out

#%% Generate Pulse
def gen_pulse(amplitude, on_start, on_length, total_length, log=False):
    pulse = np.zeros(total_length)
    on_range = range(on_start, on_start + on_length)
    for index in on_range:
        pulse[index] = amplitude
    if log:
        n = range(len(pulse))
        plt.plot(n, pulse, label="pulse")
        plt.title(f"gen_pulse")
        plt.xlabel("sample")
        plt.ylabel("amplitude")
        plt.legend()
        plt.show()        
    return pulse

#%% Output CSV
def output_csv(filename, iq):
    np.savetxt(filename, np.stack((iq.real, iq.imag)).T, delimiter=",")
    return

#%% Output BIN
def output_bin(filename, iq):
    with open(filename, "wb") as fp:
        rng = range(iq.shape[0])
        data = np.insert(iq.imag, rng, iq.real)
        fmt = "f" * len(data)  # output as float
        bin = struct.pack(fmt, *data)   # * means unpack
        fp.write(bin)
    return

#%% Spectrum Analysis
def anal_spect(iq, length, start=0, fs=1.0, window="rect", log=False):
    # Windowing
    signal = iq[start:start+length]
    if window != "rect":
        signal *= npsig.windows.get_window(window, length, False)

    spect = np.fft.fft(signal, n=length)
    spect_shifted = np.fft.fftshift(spect)
    spect_abs = np.abs(spect_shifted) / length
    spect_phase = np.angle(spect_shifted)
    spect_abs_db = 20 * np.log10(spect_abs)
    spect_freq = np.fft.fftshift(np.fft.fftfreq(length, 1/fs))
    
    if log:
        n  = np.arange(len(spect_abs))
        plt.plot(spect_freq, spect_abs, label="spect_abs")
        plt.title(f"anal_spect")
        plt.xlabel("freq")
        plt.ylabel("amplitude(linear)")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(spect_freq, spect_abs_db, label="spect_abs_db")
        plt.title(f"anal_spect")
        plt.xlabel("freq")
        plt.ylabel("amplitude(dB)")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(spect_freq, spect_phase, label="spect_phase")
        plt.title(f"anal_spect")
        plt.xlabel("freq")
        plt.ylabel("phase(rad)")
        plt.legend()
        plt.show()
    return spect, spect_abs, spect_abs_db, spect_phase, spect_freq


#%% Test
freq = 100.0
fs = 1000.0
length = 2000
cw = gen_cw(1, freq, fs, length, log=False)
pulse = gen_pulse(1, 10, 5, length, log=False)
pulsed_cw = modulation(cw, pulse, log=False)
output_csv("cw.csv", cw)
output_bin("cw.bin", cw)
anal_spect(cw, 128, fs=fs, window="blackmanharris", log=True)



# %%
