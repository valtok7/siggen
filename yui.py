#%% Import module
#%matplotlib inline   # Uncomment when .ipynb
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as spsig
from scipy import special as spsp
import struct
import math

#%% Power Analysis
def anal_power(iq, length, start=0, log=False):
    signal = iq[start:start+length]
    sqr = np.multiply(signal.real, signal.real) + np.multiply(signal.imag, signal.imag)
    power = np.average(sqr)
    power_db = 10 * math.log10(power)
    if log:
        print(f"power = {power}")
        print(f"power_db = {power_db}")
    return power, power_db

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

#%% Generate Cosine Wave
def gen_cos(ampl, freq, fs, length, axis, init_phase=0, log=False):
    n = np.arange(length)
    if axis == "real":
        iq = ampl * np.cos((2 * np.pi * freq * n / fs) + init_phase) + 0.j
    else:   # if axis == "imag"
        iq = 1.j * ampl * np.cos((2 * np.pi * freq * n / fs) + init_phase)
    if log:
        plt.plot(n, iq.real, label="i")
        plt.plot(n, iq.imag, label="q")
        plt.title(f"gen_cos ampl={ampl},freq={freq},fs={fs}")
        plt.xlabel("sample")
        plt.ylabel("amplitude")
        plt.legend()
        plt.show()
    return iq

#%% Generate Noise
def gen_noise(ampl, fs, length, log=False):
    iq = (np.random.randn(length) + 1j * np.random.randn(length)) / math.sqrt(2.0)
    if log:
        n = np.arange(length)
        plt.plot(n, iq.real, label="i")
        plt.plot(n, iq.imag, label="q")
        plt.title(f"gen_noise ampl={ampl},fs={fs}")
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
    """
    スペクトラム解析
    
    Parameters
    ----------
    window : str
        窓関数
        rect, blackmanharris, flattop, etc...
        see https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows

    """
    # Windowing
    signal = iq[start:start+length]
    if window != "rect":
        window_coef = spsig.windows.get_window(window, length, False)
    else:
        window_coef = np.ones(length)
    window_gain = np.average(window_coef)
    signal_windowed = signal * window_coef / window_gain

    # FFT
    spect = np.fft.fft(signal_windowed, n=length)
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
        print(f"max={np.max(spect_abs)}")

        plt.figure()
        plt.plot(spect_freq, spect_abs_db, label="spect_abs_db")
        plt.title(f"anal_spect")
        plt.xlabel("freq")
        plt.ylabel("amplitude(dB)")
        plt.legend()
        plt.show()
        print(f"max={np.max(spect_abs_db)}")

        plt.figure()
        plt.plot(spect_freq, spect_phase, label="spect_phase")
        plt.title(f"anal_spect")
        plt.xlabel("freq")
        plt.ylabel("phase(rad)")
        plt.legend()
        plt.show()
    return spect, spect_abs, spect_abs_db, spect_phase, spect_freq

#%% Resampling
def resample(iq, before_fs, after_fs, src_length, filter_length=9, log=False):
    #window_coef = spsig.windows.get_window(('kaiser', 6.0), filter_length, False)
    resample_list = []
    n = 0.0
    while (n + filter_length <= src_length):
        n_int = round(n)
        iq_input = iq[n_int:n_int+filter_length]
        n_frac = n - n_int  # 目標インデックスの端数成分 [-1/2fs,+1/2fs)
        filter_start = -n_frac - int(filter_length / 2)      # フィルタ開始インデックス
        filter_stop = filter_start + (filter_length - 1)     # フィルタ終端インデックス
        resample_filter_index = np.linspace(filter_stop, filter_start, filter_length)
        resample_filter = spsp.sinc(resample_filter_index)
        #resample_filter *= window_coef
        tmp = spsig.convolve(iq_input, resample_filter, mode = "valid")
        resample_list.append(*tmp)
        n += before_fs/after_fs
    resample_array = np.array(resample_list)
    if log:
        n = np.arange(len(resample_array))
        plt.plot(n, resample_array.real, label="i")
        plt.plot(n, resample_array.imag, label="q")
        plt.title(f"resample before_fs={before_fs},after_fs={after_fs}")
        plt.xlabel("sample")
        plt.ylabel("amplitude")
        plt.legend()
        plt.show()
    return resample_array

#%% Rate Conversion
def rate_conv(iq, up_rate, down_rate, filter_length=51, log=False):
    # Prepare a filter
    tmp = 1.0 / max(up_rate, down_rate) / 2.0
    cutoff = tmp - tmp * 0.2
    trans_width = tmp * 0.4
    taps = spsig.remez(filter_length, [0, cutoff, cutoff + trans_width, 0.5], [1, 0])

    # Filtering
    converted_iq = spsig.upfirdn(taps, iq, up_rate, down_rate) * up_rate
    start_position = filter_length

    if log:
        n = np.arange(len(converted_iq))
        plt.plot(n, converted_iq.real, label="i")
        plt.plot(n, converted_iq.imag, label="q")
        plt.title(f"rate_conv up_rate={up_rate},down_rate={down_rate}")
        plt.xlabel("sample")
        plt.ylabel("amplitude")
        plt.legend()
        plt.show()
    return converted_iq, start_position
