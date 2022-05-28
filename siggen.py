#%% Import module
#%matplotlib inline   # Uncomment when .ipynb
import matplotlib.pyplot as plt
import numpy as np
import struct

#%% Generate CW
def gen_cw(ampl, freq, fs, length, log=False):
    n = np.arange(length)
    iq = [ampl * np.cos(2 * np.pi * freq * n / fs), ampl * np.sin(2 * np.pi * freq * n / fs)]
    if log:
        plt.plot(n, iq[0], label="i")
        plt.plot(n, iq[1], label="q")
        plt.title(f"gen_cw ampl={ampl},freq={freq},fs={fs}")
        plt.xlabel("sample")
        plt.ylabel("amplitude")
        plt.legend()
        plt.show()
    return iq

#%% Amplitude Modulation
def mod_ampl(iq, mod_sig, log=False):
    iq_out = [np.multiply(iq[0], mod_sig), np.multiply(iq[1], mod_sig)]
    if log:
        n  = np.arange(len(iq_out[0]))
        plt.plot(n, iq_out[0], label="i")
        plt.plot(n, iq_out[1], label="q")
        plt.plot(n, mod_sig, label="mod_sig")
        plt.title(f"mod_ampl")
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
    np.savetxt(filename, np.stack((iq[0], iq[1])).T, delimiter=",")
    return

#%% Output BIN
def output_bin(filename, iq):
    with open(filename, "wb") as fp:
        rng = range(iq[0].shape[0])
        data = np.insert(iq[1], rng, iq[0])
        fmt = "f" * len(data)  # output as float
        bin = struct.pack(fmt, *data)   # * means unpack
        fp.write(bin)
    return

#%% Test
cw = gen_cw(1, 100, 1000, 20, True)
pulse = gen_pulse(1, 10, 5, 20, True)
pulsed_cw = mod_ampl(cw, pulse, True)
#output_csv("cw.csv", cw)
#output_bin("cw.bin", cw)

# %%
