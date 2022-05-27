#%% Import module
#%matplotlib inline   # Uncomment when .ipynb
import matplotlib.pyplot as plt
import numpy as np
import struct

#%% Generate CW
def gen_cw(ampl, freq, fs, length, log=False):
    n = np.arange(1)
    t = np.linspace(0, length, fs)
    i = ampl * np.cos(2 * np.pi * freq * t)
    q = ampl * np.sin(2 * np.pi * freq * t)
    if log:
        plt.plot(t, i, label="i")
        plt.plot(t, q, label="q")
        plt.title(f"gen_cw ampl={ampl},freq={freq},fs={fs}")
        plt.xlabel("time")
        plt.ylabel("amplitude")
        plt.legend()
        plt.show()
    return t, i, q

#%% Output CSV
def output_csv(filename, i, q):
    np.savetxt(filename, np.stack((i, q)).T, delimiter=",")
    return

#%% Output BIN
def output_bin(filename, i, q):
    with open(filename, "wb") as fp:
        rng = range(i.shape[0])
        data = np.insert(q, rng, i)
        fmt = "f" * len(data)  # output as float
        bin = struct.pack(fmt, *data)   # * means unpack
        fp.write(bin)
    return

#%% Test
t, i, q = gen_cw(1, 100, 1000, 20)
t, i, q = gen_cw(2, 200, 1000, 20)
t, i, q = gen_cw(3, 0, 1000, 20, True)
output_csv("cw.csv", i, q)
output_bin("cw.bin", i, q)

# %%
