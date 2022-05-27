#%% Import module
#%matplotlib inline   # Uncomment when .ipynb
import matplotlib.pyplot as plt
import numpy as np

#%% Generate CW
def gen_cw(ampl, freq, fs, length):
    n = np.arange(1)
    t = np.linspace(0, length, fs)
    i = ampl * np.cos(2 * np.pi * freq * t)
    q = ampl * np.sin(2 * np.pi * freq * t)
    plt.plot(t, i, label="i")
    plt.plot(t, q, label="q")
    plt.title("gen_cw")
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.legend()
    plt.show()
    return t, i, q

#%% Test
t, i, q = gen_cw(1, 100, 1000, 20)
t, i, q = gen_cw(2, 200, 1000, 20)
t, i, q = gen_cw(3, 300, 1000, 20)
# %%
