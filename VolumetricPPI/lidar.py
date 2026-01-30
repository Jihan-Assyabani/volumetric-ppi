import numpy as np 
import pandas as pd 
import xarray as xr

def analyze_psd(time, signal):
    """Compute FFT amplitude, power, and power spectral density (PSD)."""
    # Ensure datetime64 format
    t = np.array(time, dtype="datetime64[ns]")

    # Compute sampling frequency
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

    return t, signal, fft_freqs, amplitude, power, psd


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


def bin_altitude(cscan, col2bin:str, bin_width:int, center:int):
    """
    This function is used to create a column that indicates the altitude bin of the data
    based on the altitude column used to bin.

    Input: 
    - cscan: is the lidar scan dataframe
    - col2bin: name of column that corresponds to altitude or z values of the data
    - bin_width: width or altitude binning (in meter)
    - center: center of the binning (e.g., hub height, platform height, etc.) 

    """
    start = center - ((center - 0) // bin_width + 1) * bin_width + bin_width / 2
    end = center + ((300 - center) // bin_width + 1) * bin_width - bin_width / 2 

    # Create bins and labels
    bin_edges = np.arange(start - bin_width, end + bin_width, bin_width)
    bin_labels = [f"{int(left)}-{int(right)}" for left, right in zip(bin_edges[:-1], bin_edges[1:])]

    cscan["alt_bin"] = pd.cut(cscan[col2bin], bins=bin_edges, labels=bin_labels, include_lowest=True)

    return cscan


def filter_cscan(cscan, add_filter='both', cnr_min=-28, cnr_max=-5, vmin=-30, vmax=30, std_margin=3):
    '''
    this function is used to remove vlos values with NaNs when:
        - 'filter' value is False
        - add additional filter based on: cnr, vlos, or both
    :param cscan:
    :return: filtered cscan
    '''
    cscan['time'] = pd.to_datetime(cscan.time, unit="s")
    cscan['time'] = cscan['time'].dt.round("100ms")

    # replace vlos values with NaNs when filter = False
    cscan.loc[~cscan['filter'], 'vlos'] = np.nan

    std_vlos = cscan['vlos'].std()
    mean_vlos = cscan['vlos'].mean()

    # replace more vlos outliers that not filtered by 'filter':
    if add_filter == 'cnr':
        # use cnr range as filter
        cscan.loc[(cscan['cnr'] > cnr_max) | (cscan['cnr'] < cnr_min), 'vlos'] = np.nan
    
    elif add_filter == 'std':
        # use std range as filter
        cscan.loc[
            (cscan['vlos'] > mean_vlos + std_margin * std_vlos) | (cscan['vlos'] < mean_vlos - std_margin * std_vlos), 'vlos'
            ] = np.nan
    
    elif add_filter == 'vlos':
        # use vlos absolute value as filter
        cscan.loc[(cscan['vlos'] > vmax) | (cscan['vlos'] < vmin), 'vlos'] = np.nan
    
    elif add_filter == 'both':
        # use both cnr and std for filtering the data
        cscan.loc[(cscan['cnr'] > cnr_max) | (cscan['cnr'] < cnr_min), 'vlos'] = np.nan
        cscan.loc[(cscan['vlos'] > mean_vlos + std_margin * std_vlos) | (cscan['vlos'] < mean_vlos - std_margin * std_vlos), 'vlos'] = np.nan
    
    else:
        raise ValueError("add_filter must be 'cnr', 'vlos', 'std' or 'both'")

    return cscan


def getmainfreq(signalprop_array, freq_array, PSD_array):
    """
    Compute the frequency and signal property (amplitude, power, psd) 
    corresponding to the maximum PSD, ignoring 0 Hz.

    Args:
        signalprop_array (array): signal property values (e.g., amplitude, power)
        freq_array (array): frequency bins
        PSD_array (array): power spectral density values

    Returns:
        tuple: (freq_maxPSD, sig_maxPSD)
    """

    # Exclude the 0 Hz bin from the search
    valid_indices = np.where(freq_array != 0)[0]

    if len(valid_indices) == 0:  # edge case: only 0 Hz exists
        return None, None

    maxPSD_idx = valid_indices[np.argmax(PSD_array[valid_indices])]
    freq_maxPSD = freq_array[maxPSD_idx]
    sig_maxPSD = signalprop_array[maxPSD_idx]

    return freq_maxPSD, sig_maxPSD

def lidar_pos(azis, ranges, elevs):
    "simulating dataframe of points of scan (pos) of lidar"
    # Create meshgrid of all combinations
    azis = np.float64(azis)
    ranges = np.float64(ranges)
    elevs = np.float64(elevs)
    
    azi_grid, elev_grid, range_grid = np.meshgrid(azis, elevs, ranges, indexing='ij')

    # Flatten the grids and stack into DataFrame
    pos = pd.DataFrame({
        "azi": azi_grid.ravel(),
        "elev": elev_grid.ravel(),
        "range": range_grid.ravel()
    })

    return pos


def simulatelidar(azis, ranges, elevs, freq=5.0, start_time=0):
    """
    Simulating dataframe of lidar scan points, with time as the primary index.
    Time is assigned first, then azimuth, elevation, and range.

    Parameters
    ----------
    azis : array-like
        Azimuth angles [deg]
    ranges : array-like
        Range distances
    elevs : array-like
        Elevation angles [deg]
    freq : float, optional
        Frequency in Hz (scan steps per second). Default = 1
    start_time : float, optional
        Starting timestamp in seconds. Default = 0
    """
    azis = np.float64(azis)
    ranges = np.float64(ranges)
    elevs = np.float64(elevs)

    # Number of scan steps = number of azimuths
    n_steps = len(azis)

    # Create timestamp sequence (one per azimuth step)
    if isinstance(start_time, (int, float)):
        times = start_time + np.arange(n_steps) / freq
    elif isinstance(start_time, str):
        times = pd.date_range(start=start_time, periods=n_steps, freq=pd.Timedelta(seconds=1/freq))

    # Expand into full grid: (time, azi, elev, range)
    time_grid, elev_grid, range_grid = np.meshgrid(times, elevs, ranges, indexing='ij')
    azi_grid, _, _ = np.meshgrid(azis, elevs, ranges, indexing='ij')

    # Flatten everything
    pos = pd.DataFrame({
        "time": time_grid.ravel(),
        "azi": azi_grid.ravel(),
        "elev": elev_grid.ravel(),
        "range": range_grid.ravel()
    })

    return pos

def simulatelidar2(azis, ranges, elevs, freq=5.0, start_time=0, last_time=None):
    """
    Simulating dataframe of lidar scan points, with time as the primary index.
    Time is assigned first, then azimuth, elevation, and range.
    """

    azis = np.float64(azis)
    ranges = np.float64(ranges)
    elevs = np.float64(elevs)

    n_steps = len(azis)

    # If continuing from previous scan
    if last_time is not None:
        if isinstance(last_time, pd.Timestamp):
            start_time = last_time + pd.Timedelta(seconds=1/freq)
        else:
            start_time = float(last_time) + 1.0/freq

    # Create timestamp sequence
    if isinstance(start_time, (int, float)):
        times = start_time + np.arange(n_steps) / freq
    elif isinstance(start_time, str) or isinstance(start_time, pd.Timestamp):
        times = pd.date_range(start=pd.to_datetime(start_time), 
                              periods=n_steps, freq=pd.Timedelta(seconds=1/freq))

    # Expand into full grid
    time_grid, elev_grid, range_grid = np.meshgrid(times, elevs, ranges, indexing='ij')
    azi_grid, _, _ = np.meshgrid(azis, elevs, ranges, indexing='ij')

    pos = pd.DataFrame({
        "time": time_grid.ravel(),
        "azi": azi_grid.ravel(),
        "elev": elev_grid.ravel(),
        "range": range_grid.ravel()
    })

    return pos

def simulatelidar3(azis, ranges, elevs, freq=5.0, start_time=0, n_scans=1, delay=0.0):
    """
    Simulate lidar scan points over multiple scans, with time as the primary index.
    Each scan continues after the previous one, with optional delay between them.

    Parameters
    ----------
    azis : array-like
        Azimuth angles [deg]
    ranges : array-like
        Range distances
    elevs : array-like
        Elevation angles [deg]
    freq : float, optional
        Frequency in Hz (scan steps per second). Default = 1
    start_time : float | str | pd.Timestamp, optional
        Starting timestamp (seconds or datetime string). Default = 0
    n_scans : int, optional
        Number of repeated scans. Default = 1
    delay : float, optional
        Time gap [s] between consecutive scans. Default = 0
    """

    azis = np.float64(azis)
    ranges = np.float64(ranges)
    elevs = np.float64(elevs)

    n_steps = len(azis)

    # Handle start_time input
    if isinstance(start_time, (int, float)):
        base_time = float(start_time)
        is_datetime = False
    else:
        base_time = pd.to_datetime(start_time)
        is_datetime = True

    all_scans = []

    for i in range(n_scans):
        # Offset start time for this scan
        if is_datetime:
            t0 = base_time + pd.Timedelta(seconds=i*(n_steps/freq + delay))
            times = pd.date_range(start=t0, periods=n_steps, freq=pd.Timedelta(seconds=1/freq))
        else:
            t0 = base_time + i*(n_steps/freq + delay)
            times = t0 + np.arange(n_steps) / freq

        # Expand into full grid
        time_grid, elev_grid, range_grid = np.meshgrid(times, elevs, ranges, indexing='ij')
        azi_grid, _, _ = np.meshgrid(azis, elevs, ranges, indexing='ij')

        df = pd.DataFrame({
            "time": time_grid.ravel(),
            "azi": azi_grid.ravel(),
            "elev": elev_grid.ravel(),
            "range": range_grid.ravel()
        })

        all_scans.append(df)

    return pd.concat(all_scans, ignore_index=True)



def simulateinclinations(times, amp, freq, phaseshift=0):
    """
    Simulate pitch angle over time as a sine wave.

    Parameters
    ----------
    times : array-like
        Time values [s]
    amp : float
        Amplitude of pitch oscillation [deg]
    freq : float
        Frequency of pitch oscillation [Hz]
    phaseshift : float, optional
        Phase shift of the sine wave [deg]. Default = 0

    Returns
    -------
    inclinations : array-like
        Simulated pitch angles [deg]
    """
    # Handle pandas datetime inputs
    if isinstance(times, (pd.Series, pd.DatetimeIndex)):
        times = (times - times[0]).total_seconds().astype(float) \
            if isinstance(times, pd.DatetimeIndex) \
            else (times - times.iloc[0]).dt.total_seconds().astype(float)
        
    elif isinstance(times[0], (np.datetime64, pd.Timestamp)):
        times = (times - times[0]) / np.timedelta64(1, "s")

    inclinations = amp * np.sin(2 * np.pi * freq * times + phaseshift * np.pi / 180)
    return inclinations


def wspd2vlos(wspd, wdir, azi, elev):
    """ 
    This function converts wind speed and direction to line-of-sight velocity.
    """
    return wspd * np.cos(np.radians(elev)) * np.cos(np.radians(azi - wdir))


def getwspd4scan(scan, x, y, z, WSPD, xscan='x_recal', yscan='y_recal', zscan='z_recal'):
    """ 
    This function retrieves the wind speed for each point in the scan based on the 3D wind field.
    It uses the closest grid point to the scan points to get the wind speed.
    Can be used for both wind speed and wind direction.
    """
    wind_speeds = []
    for i in range(len(scan)):
        x_lookup = scan[xscan][i]
        y_lookup = scan[yscan][i]
        z_lookup = scan[zscan][i]
        
        # Find the closest grid point
        x_idx = np.argmin(np.abs(x - x_lookup))
        y_idx = np.argmin(np.abs(y - y_lookup))
        z_idx = np.argmin(np.abs(z - z_lookup))
        
        # Get the wind speed at that point
        wind_speed = WSPD[x_idx, y_idx, z_idx]
        wind_speeds.append(wind_speed)
    
    return np.array(wind_speeds)

# Nearest lookup function (works for both u and v)
def get_uvwLES(scan, x_vals, y_vals, z_vals, field, 
                       xscan='x_recal', yscan='y_recal', zscan='z_recal'):
    values = []
    for i in range(len(scan)):
        x_lookup = scan[xscan][i]
        y_lookup = scan[yscan][i]
        z_lookup = scan[zscan][i]
        
        x_idx = np.argmin(np.abs(x_vals - x_lookup))
        y_idx = np.argmin(np.abs(y_vals - y_lookup))
        z_idx = np.argmin(np.abs(z_vals - z_lookup))
        
        values.append(field[z_idx, y_idx, x_idx])  # careful with dim order
    return np.array(values)
    
def get_uv(ds: xr.Dataset, x: float, y: float, z: float, time_s: float,
                    tol_x: float = 40.0, tol_y: float = 40.0, tol_z: float = 40.0,
                    tol_t: float = 10.0):
    """
    Retrieve u and v wind components at a given (x, y, z, time), 
    ensuring the nearest grid point is within given tolerances.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'u' and 'v' variables.
    x, y, z : float
        Spatial coordinates of the desired point.
    time_s : float
        Time in seconds since start (since 'time' is timedelta64).
    tol_x, tol_y, tol_z : float, optional
        Maximum distance (in same units as coordinates) allowed for spatial matching.
    tol_t : float, optional
        Maximum time difference allowed in seconds.

    Returns
    -------
    dict
        {'u': float, 'v': float} if within tolerance, else {'u': None, 'v': None}.
    """

    # Convert numeric time (seconds) to timedelta64 for comparison
    time_val = np.timedelta64(int(time_s), 's')
    #print(time_s, time_val)

    # Find the nearest time in the dataset
    nearest_time = ds['time'].sel(time=time_val, method='nearest').item()
    nearest_time_s = nearest_time // 1e9
    #print(nearest_time, nearest_time_s)

    # Retrieve nearest coordinate values
    nearest_xu = ds['xu'].sel(xu=x, method='nearest').item()
    nearest_x = ds['x'].sel(x=x, method='nearest').item()
    nearest_y = ds['y'].sel(y=y, method='nearest').item()
    nearest_yv = ds['yv'].sel(yv=y, method='nearest').item()
    nearest_z = ds['zu_3d'].sel(zu_3d=z, method='nearest').item()

    # Compute differences for diagnostics
    dt = abs(time_s - nearest_time_s)
    dx = abs(x - nearest_xu)
    dy = abs(y - nearest_y)
    dz = abs(z - nearest_z)
    #print(dt, dz, dy, dx)

    # Check tolerance (optional)
    if dx > tol_x or dy > tol_y or dz > tol_z or dt > tol_t or z < 0:
         return {"u": np.NaN, "v": np.NaN}


    # Retrieve u, v using nearest neighbor (now robust)
    u_val = ds['u'].sel(
        time=np.timedelta64(int(time_s), 's'),
        zu_3d=nearest_z,
        y=nearest_y,
        xu=nearest_xu,
        method='nearest'
    ).item()

    v_val = ds['v'].sel(
        time=np.timedelta64(int(time_s), 's'),
        zu_3d=nearest_z,
        yv=nearest_yv,
        x=nearest_x,
        method='nearest'
    ).item()

    return {"u": u_val, "v": v_val}

def get_uv_recycler(
    ds: xr.Dataset, x: float, y: float, z: float, time_s: float,
    tol_x: float = 40.0, tol_y: float = 40.0, tol_z: float = 40.0,
    tol_t: float = 10.0
):
    """
    Retrieve u and v wind components at a given (x, y, z, time), 
    allowing recycled (periodic) time if dataset time span is shorter.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'u' and 'v' variables.
    x, y, z : float
        Spatial coordinates of the desired point.
    time_s : float
        Time in seconds since start (can exceed dataset duration).
    tol_x, tol_y, tol_z : float, optional
        Spatial tolerances.
    tol_t : float, optional
        Temporal tolerance (seconds).

    Returns
    -------
    dict
        {'u': float, 'v': float} if within tolerance, else {'u': NaN, 'v': NaN}.
    """

    # Compute total dataset time duration (in seconds)
    time_values = ds['time'].values
    t0 = time_values[0].astype('timedelta64[s]').astype(float)
    t_end = time_values[-1].astype('timedelta64[s]').astype(float)
    total_duration = t_end - t0

    if total_duration <= 0:
        raise ValueError("Dataset must have a positive time span.")

    # --- Time recycling (modulus wrap-around)
    # Wrap input time_s into dataset range
    wrapped_time_s = ((time_s - t0) % total_duration) + t0
    time_val = np.timedelta64(int(wrapped_time_s), 's')

    cycle_index = int((time_s - t0) // total_duration)

    # Find nearest time in dataset
    nearest_time = ds['time'].sel(time=time_val, method='nearest').item()
    nearest_time_s = nearest_time // 1e9

    # Nearest coordinates
    nearest_xu = ds['xu'].sel(xu=x, method='nearest').item()
    nearest_x = ds['x'].sel(x=x, method='nearest').item()
    nearest_y = ds['y'].sel(y=y, method='nearest').item()
    nearest_yv = ds['yv'].sel(yv=y, method='nearest').item()
    nearest_z = ds['zu_3d'].sel(zu_3d=z, method='nearest').item()

    # Differences for tolerance checks
    dt = abs(wrapped_time_s - nearest_time_s)
    dx = abs(x - nearest_xu)
    dy = abs(y - nearest_y)
    dz = abs(z - nearest_z)

    if dx > tol_x or dy > tol_y or dz > tol_z or dt > tol_t or z < 0:
        return {"u": np.NaN, "v": np.NaN, "cycle": cycle_index}

    # Retrieve u and v using nearest neighbor selection
    u_val = ds['u'].sel(
        time=time_val,
        zu_3d=nearest_z,
        y=nearest_y,
        xu=nearest_xu,
        method='nearest'
    ).item()

    v_val = ds['v'].sel(
        time=time_val,
        zu_3d=nearest_z,
        yv=nearest_yv,
        x=nearest_x,
        method='nearest'
    ).item()

    return {"u": u_val, "v": v_val, "cycle":cycle_index}

def getfromLES(ds:xr.Dataset, cscan:pd.DataFrame, xcol:str, ycol:str, zcol:str, tcol:str,
               tol_x: float = 20.0, tol_y: float = 20.0, tol_z: float = 20.0,
               tol_t: float = 10.0, 
               use_recycler: bool = False
               ):
    
    uvals = [] # u-component values
    vvals = [] # v-component values
    wvals = [] # v-component values
    cvals = [] # recycle index values
    
    for i in range(len(cscan)):
        xfind = cscan[xcol][i]
        yfind = cscan[ycol][i]
        zfind = cscan[zcol][i]
        tfind = cscan[tcol][i]

        if use_recycler:
            uv = get_uv_recycler(ds=ds, 
                    x=xfind, y=yfind, z=zfind, time_s=tfind, 
                    tol_x=tol_x, tol_y=tol_y, tol_z=tol_z, tol_t=tol_t)

        else:
            uv = get_uv(ds=ds, 
                    x=xfind, y=yfind, z=zfind, time_s=tfind, 
                    tol_x=tol_x, tol_y=tol_y, tol_z=tol_z, tol_t=tol_t)
        
        uvals.append(uv["u"])
        vvals.append(uv["v"])
        
        if use_recycler:
            cvals.append(uv["cycle"])
    
    cscan["u"] = uvals 
    cscan["v"] = vvals

    if use_recycler:
        cscan["cycle"] = cvals 

    return cscan 