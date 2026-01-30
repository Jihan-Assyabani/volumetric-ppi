
import numpy as np 
import pandas as pd 

def analyze_psd2(time, signal, fs=None):
    """Compute FFT amplitude, power, and power spectral density (PSD)."""
    # Ensure datetime64 format
    t = np.array(time, dtype="datetime64[ns]")

    # Compute sampling frequency
    if fs is None:
        dt = (t[1] - t[0]) / np.timedelta64(1, "s")  # seconds
        fs = 1 / dt

    T = 1 / fs
    N = len(signal)

    # FFT
    fft_vals = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(N, T)

    # Only positive frequencies
    pos_mask = fft_freqs >= 0
    fft_freqs = fft_freqs[pos_mask]
    fft_vals = fft_vals[pos_mask]

    # Amplitude spectrum (for reference)
    amplitude = (2.0 / N) * np.abs(fft_vals)
    amplitude[0] /= 2  # DC component fix

    # --- Power spectrum ---
    # Power = |FFT|^2 normalized by N^2
    power = (np.abs(fft_vals) ** 2) / (N**2)

    # --- Power Spectral Density (PSD) ---
    # Normalize power by frequency resolution (df = fs/N)
    df = fs / N
    psd = power / df

    return fft_freqs, amplitude, power, psd

def split4fft(df, time_col="time", value_col="signal", max_gap=1.0):
    """
    Splits a time series DataFrame into contiguous signal segments
    based on gaps in the timestamp.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a time column and a signal column.
    time_col : str
        Name of the timestamp column.
    value_col : str
        Name of the signal column.
    max_gap : float
        Maximum allowed gap in seconds before splitting.

    Returns
    -------
    list of pd.DataFrame
        Each element is a contiguous segment ready for FFT.
    """
    # Ensure time is datetime or numeric
    if not np.issubdtype(df[time_col].dtype, np.number):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df["__time_s"] = (df[time_col] - df[time_col].iloc[0]).dt.total_seconds()
        time_col = "__time_s"

    # Compute gaps between consecutive points
    df["__gap"] = df[time_col].diff()

    # Mark where gaps exceed threshold
    split_indices = df.index[df["__gap"] > max_gap].tolist()

    # Always include start and end
    split_points = [df.index[0]] + split_indices + [df.index[-1]]

    # Slice into segments
    segments = []
    for i in range(len(split_points)-1):
        seg = df.loc[split_points[i]:split_points[i+1]].copy()
        if len(seg) > 1:  # keep only non-empty
            seg = seg[[time_col, value_col]]
            segments.append(seg.reset_index(drop=True))

    return segments