import numpy as np 
import pandas as pd 
import os 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 
import cmocean 

def cos_fit (azimuth, amp, azi_max, offset):
    """
    cosine fit used for VAD fit analysis
    """
    return offset + amp * np.cos((azimuth - azi_max) * np.pi / 180)

def cosfreq_fit(azimuth, amp, azi_max, offset, freq):
    return offset + amp * np.cos((freq * azimuth - azi_max) * np.pi / 180)

def get_bounds(data, by, bounds=None):
    """
    - This function is used to get bounds that will be used to bin the dataframe.
    - The bounds defined the minimum and maximum value in each bin. 
    - If the bounds are `None`, it will use the available minimum and maximum
    value of the `by` column as the bounds. 
    - This function is used to bin both, range and altitude in the VAD function.

    Input:
    data : dataframe that contains the data to bin
    col : column name in the dataframe the binning based on 
    """
    if bounds is None or len(bounds) == 0:
        min_val = data[by].min()
        max_val = data[by].max()
        return [(min_val, max_val)]
    else:
        return list(zip(bounds[:-1], bounds[1:]))

def angular_diff(azi, wdir):
    """ 
    This function is taking the true angular separation between 
    azimuth angle of lidar measurement and the calculated wind direction
    from VAD fit. 
    """
    return np.abs((azi - wdir + 180) % 360 - 180)

def VAD_withbounds(cscan, z_col, azis_col, eles_col, ndatamin = 50, range_bounds=None, altitude_bounds=None, timestamp=None,
                   plot_VAD=False, plot_title_initial=None, save_dir=None, vmin=None, vmax=None, alpha=None,cmap='viridis', vnorm_fit=False):
    """
    This function is used to apply VAD fit to lidar data. This function is modified from 
    a script originally written by Johannes Paulsen @ForWind Oldenburg.
    """
    
    VAD_params = pd.DataFrame(columns=['range_bin', 'range_min', 'range_max', 'alt_bin','alt_lower', 'alt_upper', 
                                       'h_mean', 'n_data', 'wspd', 'wspd_std', 'wdir', 'wdir_std', 
                                       'offset', 'offset_std'])
    
    cscan['wspd'] = np.nan
    cscan['wdir'] = np.nan
    cscan['range_bin'] = np.nan 
    cscan['alt_bin'] = np.nan
    range_bins = get_bounds(data=cscan, bounds=range_bounds, by='range')
    altitude_bins = get_bounds(data=cscan, bounds=altitude_bounds, by=z_col)
    
    for range_min, range_max in range_bins:

        # Define the ranges location mask:
        range_loc = (cscan['range'] >= range_min) & (cscan['range'] <= range_max)
        
        # Modify the scan range_bin column with the looped range_bin
        cscan.loc[range_loc, 'range_bin'] = f'{range_min}-{range_max}'

        for alt_lower, alt_upper in altitude_bins:
            #print(f'altitude bounds:({alt_lower}, {alt_upper})')

            # Define the altitude location mask:
            altitude_loc = (cscan[z_col] >= alt_lower) & (cscan[z_col] <= alt_upper)

            # Modify the altitude_bin column with the looped altitude
            cscan.loc[altitude_loc, 'alt_bin'] = f'{alt_lower:.2f} - {alt_upper:.2f}'

            VAD_loc = range_loc & altitude_loc 

            azi_min = cscan[azis_col].loc[VAD_loc].min()
            azi_max = cscan[azis_col].loc[VAD_loc].max()
            azi_range = np.abs(azi_max - azi_min)
            
            # VAD is performed only when the length of data is > 10 and
            # the azimuth angles range are > 30 degrees 
            if (len(cscan.loc[VAD_loc]) > ndatamin) and (azi_range > 30):
                
                amp_guess = cscan["vlos"].iloc[np.argmax(np.abs(cscan["vlos"].loc[VAD_loc]))]
                theta_guess = cscan[azis_col].iloc[np.argmax(np.abs(cscan["vlos"].loc[VAD_loc]))]
                #print(theta_guess)
                h_mean = cscan[z_col].loc[VAD_loc].mean()
                n_data = cscan["vlos"].loc[VAD_loc].count()

                # Curve fitting:
                popt, pcov = curve_fit(f=cos_fit, 
                                    xdata=cscan[azis_col].loc[VAD_loc], 
                                    ydata=cscan["vlos"].loc[VAD_loc], #* np.cos(cscan[eles_col].loc[VAD_loc] * np.pi / 180), 
                                    # multiply vlos with cos for perfectly horizontal component
                                    p0=[amp_guess, theta_guess, 0])
                
                # extract properties:
                wspd = popt[0]
                wdir = popt[1]
                
                if wspd < 0:
                    wspd = - wspd
                    wdir = wdir + 180

                wdir = np.mod(wdir,360)
                offset = popt[2]

                # Back projection of the obtained wspd and wdir to vlos if |azimuth - wdir| <= 75° and |azimuth - wdir| >= 115°:
                azimuth_wdir_diff = angular_diff(cscan.loc[VAD_loc, azis_col], wdir)
                azimuth_loc = (azimuth_wdir_diff <= 75) | (azimuth_wdir_diff >= 115)
                back_projection_loc = VAD_loc & azimuth_loc

                cscan.loc[back_projection_loc, 'wspd'] = cscan.loc[back_projection_loc, 'vlos'] / (
                    np.cos((cscan.loc[back_projection_loc, azis_col] - wdir)*np.pi/180) * np.cos(cscan.loc[back_projection_loc, eles_col]*np.pi/180)
                    )
                
                cscan.loc[back_projection_loc, 'wdir'] = wdir

                # Second curve_fit:
                vhor = cscan.loc[VAD_loc, (azis_col, "vlos")].copy()
                vhor["vhor"] = cos_fit(azimuth=vhor[azis_col],
                                        amp = wspd, azi_max=wdir, 
                                        offset=offset)
                vhor["vnorm"] = vhor["vlos"] - vhor["vhor"]
                vhor = vhor.sort_values(by=azis_col, ascending=True)
                
                if vnorm_fit:
                    amp_guess = vhor['vnorm'].iloc[np.argmax(np.abs(vhor['vnorm']))]
                    theta_guess = vhor[azis_col].iloc[np.argmax(np.abs(vhor['vnorm']))]
                    
                    popt, _ = curve_fit(f = cosfreq_fit, 
                                        xdata = vhor[azis_col],
                                        ydata = vhor['vnorm'],
                                        p0 = [amp_guess, theta_guess, 0, 10])
                    vnorm_max = popt[0]
                    vnorm_shift = popt[1]
                    vnorm_offset = popt[2]
                    vnorm_freq = popt[3]

                if plot_VAD:
                    azi_fit = np.linspace(0,360,360)
                    vlos_fit = cos_fit(azimuth=azi_fit, 
                               amp=wspd, 
                               azi_max=wdir, 
                               offset=offset)
                    
                    vnorm_azi_fit = np.linspace(vhor[azis_col].min(), vhor[azis_col].max(), 91)
                    
                    if vnorm_fit:
                        vnorm_fit = cosfreq_fit(azimuth = vnorm_azi_fit,
                                            amp = vnorm_max,
                                            azi_max = vnorm_shift, 
                                            offset = vnorm_offset,
                                            freq = vnorm_freq)
                    
                    maintitle = f'{timestamp}, Range bin:{range_min:.0f}-{range_max:.0f}m; Alt. bin:{alt_lower:.0f}-{alt_upper:.0f}m; Data counts: {n_data}'
                    
                    fig, axes = plt.subplots(1, 2, figsize=(8,3))
                    
                    ax = axes[0]
                    sc = ax.scatter(cscan.loc[VAD_loc, azis_col], cscan.loc[VAD_loc, 'vlos'], 
                               c=cscan.loc[VAD_loc, z_col], vmin=vmin, vmax=vmax, alpha=alpha, cmap=cmap, s=5)
                    
                    #plt.colorbar(sc, ax=ax, label='h (m)')
                    
                    ax.plot(azi_fit, vlos_fit, c='r', label=f'$v_{{VAD}}$: {wspd:.2f}m/s, {wdir:.2f}°')
                    ax.axhline(offset, ls='--', c='gray', label=f'offset={offset:.2f} m/s')
                    ax.set_xlabel('Azimuth (°)')
                    ax.set_ylabel('$v_{LOS}$ (m/s)')
                    #ax.set_xlim(0,120)
                    
                    vlos_mean = cscan.loc[VAD_loc, 'vlos'].mean()
                    if vlos_mean > 0:
                        legend_loc = 'lower left'
                    else:
                        legend_loc = 'upper right'
                    ax.legend(loc=legend_loc)

                    ax = axes[1]
                    sc = ax.scatter(vhor[azis_col], vhor['vnorm'], 
                               c=cscan.loc[VAD_loc, z_col], vmin=vmin, vmax=vmax, alpha=alpha, cmap=cmap, s=5)
                    
                    if vnorm_fit:
                        ax.plot(vnorm_azi_fit, vnorm_fit, c='b', label=f'A:{vnorm_max:.2f}m/s, f: {vnorm_freq:.2f}Hz')
                    
                    
                    ax.axhline(0, ls='--', c='r', label=f'$v_{{VAD}}$={wspd:.2f} m/s')
                    ax.set_xlabel('Azimuth (°)')
                    ax.set_ylabel('$v_{LOS} - v_{VAD}$ (m/s)')
                    #ax.set_xlim(0,120)
                    ax.legend()

                    
                    
                    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.75])
                    cbar = fig.colorbar(sc, cax=cbar_ax)
                    cbar.set_label('h (m)')

                    plt.suptitle(maintitle)
                    plt.tight_layout(rect=[0, 0, 0.9, 1])

                    

                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                        
                        save_path = f"{save_dir}/{plot_title_initial}_{range_min}_{range_max}_{alt_lower}_{alt_upper}.png"
                        plt.savefig(save_path, bbox_inches='tight')
                    
                    else:
                        plt.show()

                    

                
                #print(wdir)
                #print(cscan.loc[VAD_loc])
                
                # Calculate the fitting function
                azis_fit = np.linspace(0,360,360)
                vlos_fit = cos_fit(azimuth=azis_fit, amp=wdir, azi_max=wdir, offset=offset)

                # Calculate the standard deviation
                vlos = cscan['vlos'].values
                azimuth = cscan[azis_col].values
                vlos_error = vlos - cos_fit(azimuth=azimuth, amp=wspd, azi_max=wdir, offset=offset)
                wspd_std = np.sqrt(np.mean(vlos_error**2))
                wdir_std = np.sqrt(pcov[2,2])
                offset_std = np.sqrt(pcov[0,0]) 

                params = pd.DataFrame([{'range_bin': f'{str(range_min)}-{str(range_max)}',
                                    'range_min':range_min, 
                                    'range_max':range_max,
                                    'alt_bin': f'{alt_lower:.1f}-{alt_upper:.1f}', 
                                    'alt_lower':alt_lower, 
                                    'alt_upper':alt_upper,
                                    'h_mean':h_mean, 
                                    'n_data':n_data, 
                                    'wspd':wspd, 
                                    'wspd_std':wspd_std,
                                    'wdir':wdir, 
                                    'wdir_std':wdir_std, 
                                    'offset':offset,
                                    'offset_std':offset_std}])
                VAD_params = pd.concat([VAD_params, params], ignore_index=True)

            # If length of data is not enough or <30 degrees range     
            else:
                params = pd.DataFrame([{'range_bin': f'{str(range_min)}-{str(range_max)}',
                                    'range_min':range_min, 
                                    'range_max':range_max,
                                    'alt_bin': f'{alt_lower:.1f}-{alt_upper:.1f}',
                                    'alt_lower': alt_lower,
                                    'alt_upper':alt_upper, 
                                    'h_mean':np.nan, 
                                    'n_data':len(cscan.loc[VAD_loc]),
                                    'wspd':np.nan, 
                                    'wspd_std':np.nan,
                                    'wdir':np.nan, 
                                    'wdir_std':np.nan, 
                                    'offset':np.nan,
                                    'offset_std':np.nan}])
                VAD_params = pd.concat([VAD_params, params], ignore_index=True)

    # Check the standard deviation of wspd and wdir:
    std_wspd = VAD_params['wspd'].std()
    std_wdir = VAD_params['wdir'].std()
    mean_wspd = VAD_params['wspd'].mean()
    mean_wdir = VAD_params['wdir'].mean()

    """   
    wspd_mask = ((VAD_params['wspd'] - mean_wspd).abs() > 3*std_wspd) #& (VAD_params['wspd'] < mean_wspd - std_wspd)
    wdir_mask = ((VAD_params['wdir'] - mean_wdir).abs() > 3*std_wdir) #& (VAD_params['wdir'] < mean_wdir - std_wdir)
    
    VAD_params.loc[wspd_mask, 'wspd'] = np.nan
    VAD_params.loc[wdir_mask, 'wdir'] = np.nan
    """
    # Calculate the u and v component of the windspeed:
    """
    u = -popt[1] * np.sin(popt[2] * np.pi/180)
    v = -popt[1] * np.cos(popt[2] * np.pi/180)
    #print(u, v)

    cscan["u"] = cscan["vlos"] / np.cos((cscan[azis_col] - popt[2]) * np.pi/180) * np.cos(cscan[eles_col] * np.pi/180)
    cscan["v"] = cscan["vlos"] / np.sin((cscan[azis_col] - popt[2]) * np.pi/180) * np.cos(cscan[eles_col] * np.pi/180)
    cscan["u_mean"] = np.sqrt(cscan["u"]**2 + cscan["v"]) 
    """

    return cscan, VAD_params #range, wspd, wspd_std, wdir, offset, h_mean, n_data #VAD_params



def mesh2cartesian(cscan, alt_bins, xcol='x_recal', ycol='y_recal', wspd_col='wspd', altbincol='alt_bin',
                   xgridmin=0, xgridmax=12200, ygridmin=0, ygridmax=12200, resolution=100, ti=False, tke=False):

    x = np.arange(xgridmin, xgridmax + resolution, resolution)
    y = np.arange(ygridmin, ygridmax + resolution, resolution)
    X, Y = np.meshgrid(x, y)

    x_edges = x - resolution / 2
    x_edges = np.append(x_edges, x_edges[-1] + resolution)
    y_edges = y - resolution / 2
    y_edges = np.append(y_edges, y_edges[-1] + resolution)

    WSPD = []
    TI = [] if ti else None 
    TKE = [] if tke else None 

    for alt_bin in alt_bins:
        data_at_plane = cscan[cscan[altbincol] == alt_bin].copy()

        WSPD_grid = np.full(X.shape, np.nan)
        TI_grid = np.full(X.shape, np.nan) if ti else None 
        TKE_grid = np.full(X.shape, np.nan) if tke else None 

        for yi in range(WSPD_grid.shape[0]):
            yloc_lbound = y_edges[yi]
            yloc_ubound = y_edges[yi + 1]

            for xi in range(WSPD_grid.shape[1]):
                xloc_lbound = x_edges[xi]
                xloc_ubound = x_edges[xi + 1]

                subset = data_at_plane.loc[
                    (data_at_plane[xcol] > xloc_lbound) &
                    (data_at_plane[xcol] <= xloc_ubound) &
                    (data_at_plane[ycol] > yloc_lbound) &
                    (data_at_plane[ycol] <= yloc_ubound),
                    wspd_col
                ]

                if len(subset) > 0:
                    wspd = subset.mean()
                    WSPD_grid[yi, xi] = wspd

                    if ti:
                        if wspd != 0:
                            TI_grid[yi, xi] = (subset.std() / wspd) * 100

                    if tke:
                        TKE_grid[yi, xi] = 0.5 * ((subset - wspd) ** 2).mean()

        WSPD.append(WSPD_grid)

        if ti:
            TI.append(TI_grid)

        if tke:
            TKE.append(TKE_grid)

        results = [X, Y, WSPD]
        if ti:
            results.append(TI)
        if tke:
            results.append(TKE)

    return tuple(results)

def mesh2polarplane(cscan, alt_bins, azicol='azis', rangecol='range', wspd_col='wspd', altbincol='alt_bin',
                   azigridmin=0, azigridmax=360, azires=1, rgridmin=0, rgridmax=12200, rres=100, ti=False, tke=False):

    azimuths = np.arange(azigridmin, azigridmax + azires, azires)
    ranges = np.arange(rgridmin, rgridmax + rres, rres)
    
    # Create polarlike mesh grid
    AZ, RG = np.meshgrid(np.radians (90-azimuths), ranges)
    X = RG * np.cos(AZ)
    Y = RG * np.sin(AZ)

    # Define bin edges for the data aggregation
    az_edges = np.append(azimuths - azires/2, azimuths[-1] + azires/2)
    range_edges = np.append(ranges - rres/2, ranges[-1] + rres/2, )

    last_WSPD = None 

    WSPD = []
    TI = [] if ti else None 
    TKE = [] if tke else None

    for alt_bin in alt_bins:
        data_at_plane = cscan[cscan[altbincol] == alt_bin].copy()

        WSPD_layer = np.full(X.shape, np.nan)
        TI_layer = np.full(X.shape, np.nan) if ti else None 
        TKE_layer = np.full(X.shape, np.nan) if tke else None
        
        for ai in range(len(azimuths)):
            azi_lbound = az_edges[ai]
            azi_ubound = az_edges[ai + 1]

            for ri in range(len(ranges)):
                r_lbound = range_edges[ri]
                r_ubound = range_edges[ri + 1]

                subset = data_at_plane.loc[
                    (data_at_plane[azicol] > azi_lbound) &
                    (data_at_plane[azicol] <= azi_ubound) &
                    (data_at_plane[rangecol] > r_lbound) &
                    (data_at_plane[rangecol] <= r_ubound),
                    wspd_col
                ]

                if len(subset) > 0:
                    wspd = subset.mean()
                    WSPD_layer[ri, ai] = wspd

                    if ti and wspd != 0:
                        TI_layer[ri, ai] = (subset.std() / wspd) * 100

                    if tke:
                        TKE_layer[ri, ai] = 0.5 * ((subset - wspd) **2).mean()
        
        WSPD.append(WSPD_layer)

        if ti:
            TI.append(TI_layer)
        if tke:
            TKE.append(TKE_layer)
    
    results = [X, Y, WSPD]
    if ti:
        results.append(TI)
    
    if tke:
        results.append(TKE)

    return tuple(results)

def mesh2polarrhi(cscan, elevcol='eles', rangecol='range', wspd_col='wspd', yaw_col='turb_yaw', yaw_range=2,
                   elevgridmin=-5, elevgridmax=5, elevres=0.5, rgridmin=0, rgridmax=12200, rres=100, ti=False, tke=False):

    elevations = np.arange(elevgridmin, elevgridmax + elevres, elevres)
    ranges = np.arange(rgridmin, rgridmax + rres, rres)

    # Create polar-like mesh grid (elevation in radians)
    EL, RG = np.meshgrid(np.radians(elevations), ranges)
    X = RG * np.cos(EL)
    Y = RG * np.sin(EL)

    # Bin edges
    elev_edges = np.append(elevations - elevres / 2, elevations[-1] + elevres / 2)
    range_edges = np.append(ranges - rres / 2, ranges[-1] + rres / 2)

    # Filter around yaw
    yaw_mean = cscan[yaw_col].mean()
    yaw_lbound = yaw_mean - yaw_range
    yaw_ubound = yaw_mean + yaw_range
    data_at_yaw = cscan[(cscan[yaw_col] >= yaw_lbound) & (cscan[yaw_col] <= yaw_ubound)].copy()

    # Initialize grids
    WSPD = np.full(X.shape, np.nan)
    TI = np.full(X.shape, np.nan) if ti else None
    TKE = np.full(X.shape, np.nan) if tke else None

    for ai in range(len(elevations)):
        elev_lbound = elev_edges[ai]
        elev_ubound = elev_edges[ai + 1]

        for ri in range(len(ranges)):
            r_lbound = range_edges[ri]
            r_ubound = range_edges[ri + 1]

            subset = data_at_yaw.loc[
                (data_at_yaw[elevcol] > elev_lbound) &
                (data_at_yaw[elevcol] <= elev_ubound) &
                (data_at_yaw[rangecol] > r_lbound) &
                (data_at_yaw[rangecol] <= r_ubound),
                wspd_col
            ]

            if len(subset) > 0:
                wspd = subset.mean()
                WSPD[ri, ai] = wspd

                if ti and wspd != 0:
                    TI[ri, ai] = (subset.std() / wspd) * 100

                if tke:
                    TKE[ri, ai] = 0.5 * ((subset - wspd) ** 2).mean()

    results = [X, Y, WSPD]
    if ti:
        results.append(TI)
    if tke:
        results.append(TKE)

    return tuple(results)


