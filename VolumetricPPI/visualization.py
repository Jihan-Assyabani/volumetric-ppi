import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import os 


def plot_turbines(ax, turbines, turbine_labels=None, color='r', marker='1', size=20, label_offset=5):
    """
    Plots wind turbines on the given axes.

    Parameters:
        - ax: matplotlib Axes object to plot on
        - turbines: tuple/list of (x_positions, y_positions)
        - turbine_labels: list of labels corresponding to turbines
        - color: marker color
        - marker: marker style
        - size: marker size
        - label_offset: vertical offset for label positioning
    """
    x_turbines, y_turbines = turbines

    if len(x_turbines) == 0 or len(y_turbines) == 0:
        return  # Nothing to plot

    # Plot turbines and register them for legend
    ax.scatter(x_turbines, y_turbines, color=color, marker=marker, s=size, linewidth=1, label='WTG')

    if turbine_labels:
        for x, y, label in zip(x_turbines, y_turbines, turbine_labels):
            ax.text(x, y + label_offset, label, ha='center', fontsize=8)

def plot_yawed_turbines(ax, turbines, turbine_labels=None,
                               color='k',
                               rotor_line_color='r', rotor_line_width=1,
                               label_offset=5):
    """
    Plots wind turbines and rotor diameter lines based on yaw angles.

    Parameters:
        - ax: matplotlib Axes object
        - turbines: tuple of (x_turbines, y_turbines, dia_turbines, yaw_turbines)
        - turbine_labels: list of turbine labels (optional)
        - color: turbine marker color
        - rotor_line_color: color of rotor diameter line
        - rotor_line_width: linewidth of rotor line
        - label_offset: offset (in y units) for labels above the turbine
    """
    x_turbines, y_turbines, dia_turbines, yaw_turbines = turbines

    # Add rotor lines
    for idx, (x, y, d, yaw) in enumerate(zip(x_turbines, y_turbines, dia_turbines, yaw_turbines)):
        yaw_rad = -np.deg2rad(yaw)
        dx = (d / 2) * np.cos(yaw_rad)
        dy = (d / 2) * np.sin(yaw_rad)

        # Give label only to the first line so legend shows one entry
        lbl = 'WTG' if idx == 0 else None
        ax.plot([x - dx, x + dx], [y - dy, y + dy],
                color=rotor_line_color, linewidth=rotor_line_width, zorder=2, label=lbl)

    # Add turbine labels above markers if provided
    if turbine_labels:
        for x, y, label in zip(x_turbines, y_turbines, turbine_labels):
            ax.text(x, y + label_offset, label, ha='center', fontsize=8, zorder=4)


def plot_zVSrange(cscan, zcol, rangecol, colorcols, labels, 
                  cmaps=None, alphas=None, vmins=None, vmaxs=None, s=None, 
                  turb=None, scan_dirs=None, xlim=None, ylim=None, xticks=None, yticks=None,
                  unitwidth=6, unitheight=2, plot_rejected = False,
                  xlabel='range (m)', ylabel='z (m)',
                  sup_title=None, save_path=None):
    """ 
    Plots z vs range with color-coded scatter plots and turbine overlay.
    
    Parameters:
        - cscan: DataFrame containing the data
        - zcol, rangecol: column names for z and range values
        - colorcols: list of column names for color mapping
        - labels: list of colorbar labels for each subplot
        - cmaps: list of colormaps for each subplot
        - alphas: list of alpha values for each subplot
        - vmins, vmaxs: lists of color scale limits
        - s: marker size (scalar or list)
        - turb: [rpos, hub_height, rotor_diameter, nacelle_length]
        - scan_dirs: list of 'inflow' or 'wake' for each subplot
        - unitwidth, unitheight: dimensions per subplot
        - xlabel, ylabel: axis labels
        - sup_title: figure title
        - save_path: path to save the figure (optional)
    """
    nrows = len(colorcols)

    # Handle optional lists
    if scan_dirs is None:
        scan_dirs = ['inflow'] * nrows
    if cmaps is None:
        cmaps = ['viridis'] * nrows
    if alphas is None:
        alphas = [1.0] * nrows
    if vmins is None:
        vmins = [None] * nrows
    if vmaxs is None:
        vmaxs = [None] * nrows
    if isinstance(s, list):
        sizes = s
    else:
        sizes = [s] * nrows
    if plot_rejected:
        rejects = cscan.loc[cscan['filter']==False]

    # Set figure size
    fig, axes = plt.subplots(nrows, 1, figsize=(unitwidth, unitheight * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]  # make iterable

    for i, colorcol in enumerate(colorcols):
        ax = axes[i]
        if plot_rejected:
            ax.scatter(
                rejects[rangecol], rejects[zcol],
                c='grey',
                s=sizes[i],
                alpha=0.5,
                label='rejected'
            )
        sc = ax.scatter(
            cscan[rangecol], cscan[zcol],
            c=cscan[colorcol],
            cmap=cmaps[i],
            vmin=vmins[i],
            vmax=vmaxs[i],
            s=sizes[i],
            alpha=alphas[i]
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(labels[i])



        if turb:
            rpos, zhub, turb_dia, nacelle = turb
            if scan_dirs[i] == 'inflow':
                ax.plot([rpos, rpos], [0, zhub], c='k')  # tower
                ax.plot([rpos, rpos + nacelle], [zhub, zhub], c='k')  # nacelle
                ax.plot([rpos + nacelle, rpos + nacelle],
                        [zhub - turb_dia / 2, zhub + turb_dia / 2], c='k')  # rotor
            elif scan_dirs[i] == 'wake':
                ax.plot([rpos + nacelle, rpos + nacelle], [0, zhub], c='k')  # tower
                ax.plot([rpos, rpos + nacelle], [zhub, zhub], c='k')  # nacelle
                ax.plot([rpos, rpos],
                        [zhub - turb_dia / 2, zhub + turb_dia / 2], c='k')  # rotor

            ax.axhline(zhub, linestyle='--', color='red', label='hub height')
        
        ax.axhline(0, linestyle='--', color='blue', label='sea level')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)
        
        ax.grid()
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper left', fontsize='small')

        if i == nrows - 1:
            ax.set_xlabel(xlabel)

    if sup_title:
        plt.suptitle(sup_title, y=1.02)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved {save_path}.")
        plt.close(fig)

    plt.show()


def plot_mesh(
    X, Y, VALUES, valuelabel, wdirs=None,
    layer_titles=None, cmaps=None, vmins=None, vmaxs=None, legendloc='upper left',
    xlim=None, ylim=None, xlabels=None, ylabel=None, sup_title=None,
    turbines=None, turbine_labels=None, turbine_plot_type='scatter', turbine_marker='1', marker_size=20,
    save_path=None, shared_legend=False
):
    """
    Plots one or more 2D layers using pcolormesh on Cartesian coordinates.

    Parameters:
        - X, Y: 2D meshgrid arrays
        - VALUES: list of 2D arrays for each subplot
        - valuelabel: colorbar label
        - layer_titles: titles for subplots
        - cmaps: list of colormaps
        - vmins, vmaxs: list of color scale bounds
        - xlim, ylim: axis limits
        - xlabels: list of x-axis labels
        - ylabel: y-axis label (shared)
        - sup_title: supertitle
        - turbines_pos: (x, y) tuple/list for turbine markers
        - turbine_labels: list of labels for turbines
        - save_path: if provided, saves the figure
    """
    # Ensure VALUES is a list
    if not isinstance(VALUES, (list, tuple)):
        VALUES = [VALUES]

    ncols = len(VALUES)
    cmaps = cmaps or ['viridis'] * ncols
    vmins = vmins or [val.min() for val in VALUES]
    vmaxs = vmaxs or [val.max() for val in VALUES]
    xlabels = xlabels or ['x (m)'] * ncols
    ylabel = ylabel or 'y (m)'
    turbines = turbines or ([], [])
    turbine_labels = turbine_labels or []

    if wdirs is not None and len(wdirs) > 0:
        wdirs = np.array(wdirs)
        wdirs = np.deg2rad(270 - wdirs)
        dx = np.cos(wdirs)
        dy = np.sin(wdirs)

    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), sharey=True)

    if ncols == 1:
        axes = [axes]

    for i in range(ncols):
        ax = axes[i]
        val = VALUES[i]

        pcm = ax.pcolormesh(X, Y, val, shading='gouraud',
                            cmap=cmaps[i], vmin=vmins[i], vmax=vmaxs[i])

        # Wind direction arrow
        if wdirs is not None and len(wdirs) > 0:
            ax.annotate(
                '', xy=(0.9 + dx[i] * 0.09, 0.9 + dy[i] * 0.09), xytext=(0.9, 0.9), 
                arrowprops=dict(facecolor='k', width=2, headwidth=8),
                xycoords=ax.transAxes
            )

        # Turbine plotting
        if turbine_plot_type == 'rotorline':
            plot_yawed_turbines(ax, turbines, turbine_labels)
        else:
            x_turbines, y_turbines = turbines[0], turbines[1]
            plot_turbines(ax, (x_turbines, y_turbines), turbine_labels,
                          marker=turbine_marker, size=marker_size)

        # Axis settings
        ax.set_xlabel(xlabels[i])
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_aspect('equal')

        if layer_titles:
            ax.set_title(layer_titles[i])

        if i == 0:
            ax.set_ylabel(ylabel)
            if ylim is not None:
                ax.set_ylim(ylim)

        # Per-subplot legend if not shared
        if not shared_legend:
            ax.legend(loc=legendloc)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.75])
    cbar = fig.colorbar(pcm, cax=cbar_ax)
    cbar.set_label(valuelabel)

    # Shared legend if enabled
    if shared_legend:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc=legendloc)

    if sup_title:
        plt.suptitle(sup_title, y=1.05)

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved {save_path}.")
        plt.close(fig)

    plt.show()


def plot_rhimesh(X, Y, VALUE, valuelabel, title=None, cmap=None, 
                          vmin=None, vmax=None, xlim=None, ylim=None,
                          xlabel=None, ylabel=None,
                          save_path=None):
    """
    Plots multiple 2D layers on Cartesian coordinates using pcolormesh.

    Parameters:
        - X, Y: 2D meshgrid arrays
        - VALUE: list of 2D arrays for each layer
        - label: colorbar label
        - titles: subplot titles
        - cmaps: list of colormaps
        - vmins, vmaxs: list of color scale limits
        - xlim, ylim: plot limits
        - xlabels: x-axis labels
        - ylabel: y-axis label (shared)
        - sup_title: overall figure title
        - save_path: file path to save the figure
    """

    # Handle optional parameters
    if cmap is None:
        cmap = 'viridis'
    if vmin is None:
        vmin = [val.min() for val in VALUE]
    if vmax is None:
        vmax = [val.max() for val in VALUE]
    if xlabel is None:
        xlabel = 'x (m)'
    if ylabel is None:
        ylabel = 'y (m)'

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    pcm = ax.pcolormesh(X, Y, VALUE, shading='gouraud',
                        cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_xlim(xlim)
    

    if title:
        ax.set_title(title)

    
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)

    # Colorbar (based on last pcolormesh)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.75])
    cbar = fig.colorbar(pcm, cax=cbar_ax)
    cbar.set_label(valuelabel)


    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def plot_layers_scatter(
    cscan, alt_bins, altbincol, xcol='x', ycol='y', colorcol='wspd',
    clabel=None, xlabel='x (m)', ylabel='y (m)', xlim=None, ylim=None,
    cmap='viridis', vmin=None, vmax=None, s=0.5, sup_title=None,
    turbines = None, turbine_labels=None, turbine_plot_type='scatter',
    layer_titles=None, save_path=None
):
    """
    Plots scatter layers by altitude bin on Cartesian coordinates.

    Parameters:
        - cscan: DataFrame with the data
        - alt_bins: list of altitudes to plot
        - altbincol: column name for altitude bin
        - xcol, ycol, colorcol: column names for coordinates and color
        - clabel: colorbar label
        - xlim, ylim: axis limits
        - layer_titles: optional list of subplot titles
        - save_path: optional path to save the figure
    """
    ncols = len(alt_bins)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), sharey=True)

    if ncols == 1:
        axes = [axes]

    for i, alt_bin in enumerate(alt_bins):
        ax = axes[i]
        subset = cscan[cscan[altbincol] == alt_bin]

        sc = ax.scatter(
            subset[xcol], subset[ycol],
            c=subset[colorcol],
            cmap=cmap, vmin=vmin, vmax=vmax,
            s=s
        )

        # Plot wind farms
        if turbine_plot_type == 'rotorline':
            # Expecting (x, y, dia, yaw) for rotorline plotting
            plot_yawed_turbines(ax, turbines, turbine_labels)
        else:
            # Fallback to basic marker plotting using only (x, y)
            x_turbines, y_turbines = turbines[0], turbines[1]
            plot_turbines(ax, (x_turbines, y_turbines), turbine_labels)

        if layer_titles:
            ax.set_title(layer_titles[i])

        ax.set_xlabel(xlabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')

        if i == 0:
            ax.set_ylabel(ylabel)

    # Colorbar (shared)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.75])
    cbar = plt.colorbar(sc, cax=cbar_ax)
    if clabel:
        cbar.set_label(clabel)

    if sup_title:
        plt.suptitle(sup_title, y=1.05)

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved {save_path}.")
        plt.close(fig)

    plt.show()


def plot_layers_wspd_wdir(
    cscan, alt_bins, altbincol,
    xcol='x', ycol='y', wspdcol='wspd', wdircol='wdir',
    clabels=None, xlabel='x (m)', ylabel='y (m)',
    xlim=None, ylim=None,
    cmap='viridis', vmin=None, vmax=None, s=0.5,
    layer_titles=None, save_path=None, cbar_shrink=None
):
    """
    Plots wind speed and wind direction scatter layers by altitude bin.

    Parameters:
        - cscan: DataFrame with data
        - alt_bins: list of altitude bin values
        - altbincol: column name for altitude bin
        - xcol, ycol: column names for coordinates
        - wspdcol, wdircol: column names for wind speed and direction
        - clabels: list of colorbar labels [wspd_label, wdir_label]
        - xlabel, ylabel: axis labels
        - xlim, ylim: axis limits
        - cmap: colormap
        - vmin, vmax: color scale limits (shared across all plots)
        - s: marker size
        - layer_titles: list of titles for each row
        - save_path: path to save figure
    """
    ncols = 2
    nrows = len(alt_bins)

    if clabels is None:
        clabels = ['Wind Speed (m/s)', 'Wind Direction (°)']
        

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for i, alt_bin in enumerate(alt_bins):
        subset = cscan[cscan[altbincol] == alt_bin]

        # --- Wind Speed ---
        ax_wspd = axes[i, 0]
        sc_wspd = ax_wspd.scatter(
            subset[xcol], subset[ycol],
            c=subset[wspdcol],
            cmap=cmap, vmin=vmin, vmax=vmax,
            s=s
        )
        cbar_wspd = plt.colorbar(sc_wspd, ax=ax_wspd, shrink=cbar_shrink)
        cbar_wspd.set_label(clabels[0])

        title_wspd = f"Wind Speed at {layer_titles[i]}" if layer_titles else f"Layer {alt_bin}"
        ax_wspd.set_title(title_wspd)
        ax_wspd.set_xlim(xlim)
        ax_wspd.set_ylim(ylim)
        ax_wspd.set_aspect('equal')
        if i == 0:
            ax_wspd.set_ylabel(ylabel)

        # --- Wind Direction ---
        ax_wdir = axes[i, 1]
        sc_wdir = ax_wdir.scatter(
            subset[xcol], subset[ycol],
            c=subset[wdircol],
            cmap=cmap, vmin=vmin, vmax=vmax,
            s=s
        )
        cbar_wdir = plt.colorbar(sc_wdir, ax=ax_wdir, shrink=cbar_shrink)
        cbar_wdir.set_label(clabels[1])

        title_wdir = f"Wind Direction at {layer_titles[i]}" if layer_titles else f"Layer {alt_bin}"
        ax_wdir.set_title(title_wdir)
        ax_wdir.set_xlim(xlim)
        ax_wdir.set_ylim(ylim)
        ax_wdir.set_aspect('equal')

        if i == nrows - 1:
            ax_wspd.set_xlabel(xlabel)
            ax_wdir.set_xlabel(xlabel)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved {save_path}.")
        plt.close(fig)

    plt.show()


# to do: plot vertical profile

def mesh3D_cartesian(
    cscan:pd.DataFrame, 
    xcol:str, ycol:str, zcol:str, wspd_col:str, 
    xgridmin=0, xgridmax=12200, 
    ygridmin=0, ygridmax=12200, 
    zgridmin=0, zgridmax=400, 
    xyresolution=100, zresolution=10, 
    ti=False, tke=False
):
    """
    Generate 3D gridded wind field data from Cartesian lidar scan data.

    Parameters
    ----------
    cscan : DataFrame
        Input data containing x, y, z, and wspd columns.
    xcol, ycol, zcol : str
        Column names for x, y, z coordinates.
    wspd_col : str
        Column name for wind speed.
    resolution : float
        Grid spacing in x and y directions.
    zresolution : float
        Grid spacing in z direction.
    ti, tke : bool
        Whether to compute turbulence intensity (TI) and turbulent kinetic energy (TKE).

    Returns
    -------
    tuple
        (X, Y, Z, WSPD [, TI] [, TKE])
    """

    # Create 3D grid
    x = np.arange(xgridmin, xgridmax + xyresolution, xyresolution)
    y = np.arange(ygridmin, ygridmax + xyresolution, xyresolution)
    z = np.arange(zgridmin, zgridmax + zresolution, zresolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='xy')

    # Define bin edges
    x_edges = np.append(x - xyresolution / 2, x[-1] + xyresolution / 2)
    y_edges = np.append(y - xyresolution / 2, y[-1] + xyresolution / 2)
    z_edges = np.append(z - zresolution / 2, z[-1] + zresolution / 2)

    # Initialize grids
    shape = X.shape
    WSPD = np.full(shape, np.nan)
    TI_grid = np.full(shape, np.nan) if ti else None
    TKE_grid = np.full(shape, np.nan) if tke else None

    # Populate 3D grid
    for zi in range(len(z)):
        z_lbound = z_edges[zi]
        z_ubound = z_edges[zi + 1]

        data_at_z = cscan.loc[
            (cscan[zcol] > z_lbound) & (cscan[zcol] <= z_ubound)
        ]

        for yi in range(len(y)):
            y_lbound = y_edges[yi]
            y_ubound = y_edges[yi + 1]

            for xi in range(len(x)):
                x_lbound = x_edges[xi]
                x_ubound = x_edges[xi + 1]

                subset = data_at_z.loc[
                    (data_at_z[xcol] > x_lbound) & (data_at_z[xcol] <= x_ubound) &
                    (data_at_z[ycol] > y_lbound) & (data_at_z[ycol] <= y_ubound),
                    wspd_col
                ]

                if len(subset) > 0:
                    wspd = subset.mean()
                    WSPD[yi, xi, zi] = wspd

                    if ti:
                        if wspd != 0:
                            TI_grid[yi, xi, zi] = (subset.std() / wspd) * 100

                    if tke:
                        TKE_grid[yi, xi, zi] = 0.5 * ((subset - wspd) ** 2).mean()

    results = [X, Y, Z, WSPD]
    if ti:
        results.append(TI_grid)
    if tke:
        results.append(TKE_grid)

    return tuple(results)