import numpy as np
import pandas as pd  
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def dist2water(z, elev):
    """ 
    This function is used to estimate the range of LOS to water from a given 
    negative elevation of Lidar's LOS.

    Input:
        - z : height of lidar placement
        - elev : elevation of lidar LOS (must be a negative value)
    """
    return z / np.sin(np.radians(-elev))


def earth_curv(dist):
    R = 6_378_137.0  # Earth's radius in meters
    delta = np.sqrt(R**2 + dist**2) - R
    return delta


def rotate_yaw(cscan:pd.DataFrame, xcol:str, ycol:str, zcol:str, yaw_deg:str, rotdir='CW', xnew=None, ynew=None, znew=None):
    
    if xnew is None:
        xnew = f'{xcol}Y'
    if ynew is None:
        ynew = f'{ycol}Y'
    if znew is None:
        znew = f'{zcol}Y'
        
    # if input value for pitch is a number
    if isinstance(yaw_deg, (int, float)):
        
        if rotdir == 'CW':
            yaw = -np.radians(yaw_deg) # minus for clockwise rotation due to meteorological convention
        elif rotdir == 'CCW':
            yaw = np.radians(yaw_deg) # plus for counter clockwise rotation following mathematical convention
        else:
            raise ValueError('rotdir must be CW (clockwise) of CCW (counter clockwise).')
    
    # if input value for pitch is a column name
    elif isinstance(yaw_deg, (str)):
        
        if rotdir == 'CW':
            yaw = -np.radians(cscan[yaw_deg])
        elif rotdir == 'CCW':
            yaw = np.radians(cscan[yaw_deg])
        else:
            raise ValueError('rotdir must be CW (clockwise) of CCW (counter clockwise).')
    
    else:
        raise ValueError("pitch_deg must be a float/int or a column name (string)")
    
    #yaw = -np.radians(yaw_deg) # minus for clockwise rotation due to meteorological convention
    new_cols = pd.DataFrame({
        xnew : cscan[xcol] * np.cos(yaw) - cscan[ycol] * np.sin(yaw),
        ynew : cscan[xcol] * np.sin(yaw) + cscan[ycol] * np.cos(yaw),
        znew : cscan[zcol]
    }, index=cscan.index)

    return pd.concat([cscan, new_cols], axis=1)


def rotate_roll(cscan:pd.DataFrame, xcol:str, ycol:str, zcol:str, roll_deg:str, rotdir='CCW', xnew=None, ynew=None, znew=None):
    
    if xnew is None:
        xnew = f'{xcol}R'
    if ynew is None:
        ynew = f'{ycol}R'
    if znew is None:
        znew = f'{zcol}R'

    # if input value for roll is a number
    if isinstance(roll_deg, (int, float)):
        
        if rotdir == 'CW':
            roll = -np.radians(roll_deg) # minus for clockwise rotation 
        elif rotdir == 'CCW':
            roll = np.radians(roll_deg) 
        else:
            raise ValueError('rotdir must be CW (clockwise) or CCW (counter clockwise).')
    
    # if input value for roll isa a column name
    elif isinstance(roll_deg, (str)):
        
        if rotdir == 'CW':
            roll = -np.radians(cscan[roll_deg]) # minus for clockwise rotation 
        elif rotdir == 'CCW':
            roll = np.radians(cscan[roll_deg])
        else:
            raise ValueError('rotdir must be CW (clockwise) or CCW (counter clockwise).')
    else:
        raise ValueError("roll_deg must be a float/int or a column name (string)")
    
    new_cols = pd.DataFrame({
        xnew : cscan[xcol] * np.cos(roll) + cscan[zcol] * np.sin(roll),
        ynew : cscan[ycol],
        znew : - cscan[xcol] * np.sin(roll) + cscan[zcol] * np.cos(roll)
    }, index=cscan.index)

    return pd.concat([cscan, new_cols], axis=1)


def rotate_pitch(cscan:pd.DataFrame, xcol:str, ycol:str, zcol:str, pitch_deg:str, rotdir='CCW', xnew=None, ynew=None, znew=None):
    
    if xnew is None:
        xnew = f'{xcol}P'
    if ynew is None:
        ynew = f'{ycol}P'
    if znew is None:
        znew = f'{zcol}P'

    # if input value for pitch is a number
    if isinstance(pitch_deg, (int, float)):
        
        if rotdir == 'CW':
            pitch =  - np.deg2rad(pitch_deg) # minus for clockwise rotation | + = look down, - = look up
        elif rotdir == 'CCW':
            pitch = np.deg2rad(pitch_deg) 
        else:
            raise ValueError('rotdir must be CW (clockwise) or CCW (counter clockwise).')
    
    # if input value for pitch is a column name
    elif isinstance(pitch_deg, (str)):
        
        if rotdir == 'CW':
            pitch = - np.deg2rad(cscan[pitch_deg]) # minus for clockwise rotation | + = look down, - = look up
        elif rotdir == 'CCW':
            pitch = np.deg2rad(cscan[pitch_deg])
        else:
            raise ValueError('rotdir must be CW (clockwise) of CCW (counter clockwise).')
    
    else:
        raise ValueError("pitch_deg must be a float/int or a column name (string)")
    
    new_cols = pd.DataFrame({
        xnew : cscan[xcol], 
        ynew : cscan[ycol] * np.cos(pitch) - cscan[zcol] * np.sin(pitch),
        znew : cscan[ycol] * np.sin(pitch) + cscan[zcol] * np.cos(pitch)
    }, index=cscan.index)

    return pd.concat([cscan, new_cols], axis=1)


def recalculate_xyzd(cscan:pd.DataFrame, range_col:str, elevation_col:str, azimuth_col:str, xnew='x_recal', ynew='y_recal', znew='z_recal', dnew='d_recal', z_tp=0):

    if elevation_col is not None:
        eles = cscan[elevation_col]
    else:
        eles = cscan["eles"]

    if azimuth_col is not None:
        azis = cscan[azimuth_col]
    else:
        azis = cscan["azis"]

    # Recalculate x, y, and z
    cscan[xnew] = (cscan[range_col] * np.sin((90 - eles) * np.pi / 180) * np.cos(
                    (azis - 90) * np.pi / 180))
    cscan[ynew] = (-cscan[range_col] * np.sin((90 - eles) * np.pi / 180) * np.sin(
                    (azis - 90) * np.pi / 180))
    cscan[znew] = cscan[range_col] * np.cos((90 - eles) * np.pi / 180) + z_tp
    cscan[dnew] = cscan[range_col] * np.sin((90 - eles) * np.pi / 180)

    return cscan


def rotate_RcwPcwYcw(cscan:pd.DataFrame, 
                      xcol:str, ycol:str, zcol:str, 
                      pitch_deg:str, roll_deg:str, yaw_deg:str, 
                      drop_helpcols:bool=True):
    """
    This function rotates the xyz points from Lidar frame of reference by applying
    rotation matrix in the order: 
        Roll(clockwise) -> Pitch (clockwise) -> Yaw (clockwise).

    Args:
        cscan (dataframe): dataframe contain lidar measurement
        xcol (str): column contains x values from lidar frame of reference
        ycol (str): column contains y values from lidar frame of reference
        zcol (str): column contains z values from lidar frame of refence
        pitch_deg (str): column contains pitch angles of the platform where lidar is placed
        roll_deg (str): column contains roll angles of the platform where lidar is placed
        yaw_deg (str): column contains yaw angles of the turbine
        drop_helpcols (bool): options to only get final rotation columns 
                                and drop other intermediary helping columns.

    Returns:
        dataframe: modified dataframe containing columns of the rotations
    """
    
    # Roll rotation:
    cscan = rotate_roll(cscan, 
                        xcol=xcol, ycol=ycol, zcol=zcol, 
                        xnew='x_Rcw', ynew='y_Rcw', znew='z_Rcw', 
                        roll_deg=roll_deg, rotdir='CW')

    # Pitch rotation:
    cscan = rotate_pitch(cscan, 
                        xcol='x_Rcw', ycol='y_Rcw', zcol='z_Rcw', 
                        xnew='x_RcwPcw', ynew='y_RcwPcw', znew='z_RcwPcw', 
                        pitch_deg=pitch_deg, rotdir='CW')
    
    # Yaw rotation:
    cscan = rotate_yaw(cscan, 
                        xcol ='x_RcwPcw', ycol ='y_RcwPcw', zcol ='z_RcwPcw', 
                        xnew ='x_RcwPcwYcw', ynew ='y_RcwPcwYcw', znew ='z_RcwPcwYcw', 
                        yaw_deg =yaw_deg, rotdir='CW')
    
    cscan['d_RcwPcwYcw'] =  np.sqrt(cscan['x_RcwPcwYcw']**2 + cscan['y_RcwPcwYcw']**2)
    
    # Add earth curvature correction:
    cscan['z_RcwPcwYcw'] += earth_curv(cscan['d_RcwPcwYcw'])

    # Drop unnecessary helping columns:
    if drop_helpcols:
        cscan = cscan.drop(columns=['x_Rcw', 'y_Rcw', 'z_Rcw', 
                            'x_RcwPcw', 'y_RcwPcw', 'z_RcwPcw'])
    
    return cscan 


def rotate_RcwPccwYcw(cscan:pd.DataFrame, 
                      xcol:str, ycol:str, zcol:str, 
                      pitch_deg:str, roll_deg:str, yaw_deg:str, 
                      drop_helpcols:bool=True):
    """
    This function rotates the xyz points from Lidar frame of reference by applying
    rotation matrix in the order: 
        Roll(clockwise) -> Pitch (counter-clockwise) -> Yaw (clockwise).

    Args:
        cscan (dataframe): dataframe contain lidar measurement
        xcol (str): column contains x values from lidar frame of reference
        ycol (str): column contains y values from lidar frame of reference
        zcol (str): column contains z values from lidar frame of refence
        pitch_deg (str): column contains pitch angles of the platform where lidar is placed
        roll_deg (str): column contains roll angles of the platform where lidar is placed
        yaw_deg (str): column contains yaw angles of the turbine
        drop_helpcols (bool): options to only get final rotation columns 
                                and drop other intermediary helping columns.

    Returns:
        dataframe: modified dataframe containing columns of the rotations
    """
    
    # Roll rotation:
    cscan = rotate_roll(cscan, 
                        xcol=xcol, ycol=ycol, zcol=zcol, 
                        xnew='x_Rcw', ynew='y_Rcw', znew='z_Rcw', 
                        roll_deg=roll_deg, rotdir='CW')

    # Pitch rotation:
    cscan = rotate_pitch(cscan, 
                        xcol='x_Rcw', ycol='y_Rcw', zcol='z_Rcw', 
                        xnew='x_RcwPccw', ynew='y_RcwPccw', znew='z_RcwPccw', 
                        pitch_deg=pitch_deg, rotdir='CCW')
    
    # Yaw rotation:
    cscan = rotate_yaw(cscan, 
                        xcol ='x_RcwPccw', ycol ='y_RcwPccw', zcol ='z_RcwPccw', 
                        xnew ='x_RcwPccwYcw', ynew ='y_RcwPccwYcw', znew ='z_RcwPccwYcw', 
                        yaw_deg =yaw_deg, rotdir='CW')
    
    cscan['d_RcwPccwYcw'] =  np.sqrt(cscan['x_RcwPccwYcw']**2 + cscan['y_RcwPccwYcw']**2)
    
    # Add earth curvature correction:
    cscan['z_RcwPccwYcw'] += earth_curv(cscan['d_RcwPccwYcw'])

    # Drop unnecessary helping columns:
    if drop_helpcols:
        cscan = cscan.drop(columns=['x_Rcw', 'y_Rcw', 'z_Rcw', 
                            'x_RcwPccw', 'y_RcwPccw', 'z_RcwPccw'])
    
    return cscan 


def rotate_RccwPcwYcw(cscan:pd.DataFrame, 
                      xcol:str, ycol:str, zcol:str, 
                      pitch_deg:str, roll_deg:str, yaw_deg:str, 
                      drop_helpcols:bool=True):
    """
    This function rotates the xyz points from Lidar frame of reference by applying
    rotation matrix in the order: 
        Roll(counter-clockwise) -> Pitch (clockwise) -> Yaw (clockwise).

    Args:
        cscan (dataframe): dataframe contain lidar measurement
        xcol (str): column contains x values from lidar frame of reference
        ycol (str): column contains y values from lidar frame of reference
        zcol (str): column contains z values from lidar frame of refence
        pitch_deg (str): column contains pitch angles of the platform where lidar is placed
        roll_deg (str): column contains roll angles of the platform where lidar is placed
        yaw_deg (str): column contains yaw angles of the turbine
        drop_helpcols (bool): options to only get final rotation columns 
                                and drop other intermediary helping columns.

    Returns:
        dataframe: modified dataframe containing columns of the rotations
    """
    
    # Roll rotation:
    cscan = rotate_roll(cscan, 
                        xcol=xcol, ycol=ycol, zcol=zcol, 
                        xnew='x_Rccw', ynew='y_Rccw', znew='z_Rccw', 
                        roll_deg=roll_deg, rotdir='CCW')

    # Pitch rotation:
    cscan = rotate_pitch(cscan, 
                        xcol='x_Rccw', ycol='y_Rccw', zcol='z_Rccw', 
                        xnew='x_RccwPcw', ynew='y_RccwPcw', znew='z_RccwPcw', 
                        pitch_deg=pitch_deg, rotdir='CW')
    
    # Yaw rotation:
    cscan = rotate_yaw(cscan, 
                        xcol ='x_RccwPcw', ycol ='y_RccwPcw', zcol ='z_RccwPcw', 
                        xnew ='x_RccwPcwYcw', ynew ='y_RccwPcwYcw', znew ='z_RccwPcwYcw', 
                        yaw_deg =yaw_deg, rotdir='CW')
    
    cscan['d_RccwPcwYcw'] =  np.sqrt(cscan['x_RccwPcwYcw']**2 + cscan['y_RccwPcwYcw']**2)
    
    # Add earth curvature correction:
    cscan['z_RccwPcwYcw'] += earth_curv(cscan['d_RccwPcwYcw'])

    # Drop unnecessary helping columns:
    if drop_helpcols:
        cscan = cscan.drop(columns=['x_Rccw', 'y_Rccw', 'z_Rccw', 
                            'x_RccwPcw', 'y_RccwPcw', 'z_RccwPcw'])
    
    return cscan 


def rotate_RccwPccwYcw(cscan:pd.DataFrame, 
                      xcol:str, ycol:str, zcol:str, 
                      pitch_deg:str, roll_deg:str, yaw_deg:str, 
                      drop_helpcols:bool=True):
    """
    This function rotates the xyz points from Lidar frame of reference by applying
    rotation matrix in the order: 
        Roll(counter-clockwise) -> Pitch (counter-clockwise) -> Yaw (clockwise).

    Args:
        cscan (dataframe): dataframe contain lidar measurement
        xcol (str): column contains x values from lidar frame of reference
        ycol (str): column contains y values from lidar frame of reference
        zcol (str): column contains z values from lidar frame of refence
        pitch_deg (str): column contains pitch angles of the platform where lidar is placed
        roll_deg (str): column contains roll angles of the platform where lidar is placed
        yaw_deg (str): column contains yaw angles of the turbine
        drop_helpcols (bool): options to only get final rotation columns 
                                and drop other intermediary helping columns.

    Returns:
        dataframe: modified dataframe containing columns of the rotations
    """
    
    # Roll rotation:
    cscan = rotate_roll(cscan, 
                        xcol=xcol, ycol=ycol, zcol=zcol, 
                        xnew='x_Rccw', ynew='y_Rccw', znew='z_Rccw', 
                        roll_deg=roll_deg, rotdir='CCW')

    # Pitch rotation:
    cscan = rotate_pitch(cscan, 
                        xcol='x_Rccw', ycol='y_Rccw', zcol='z_Rccw', 
                        xnew='x_RccwPccw', ynew='y_RccwPccw', znew='z_RccwPccw', 
                        pitch_deg=pitch_deg, rotdir='CCW')
    
    # Yaw rotation:
    cscan = rotate_yaw(cscan, 
                        xcol ='x_RccwPccw', ycol ='y_RccwPccw', zcol ='z_RccwPccw', 
                        xnew ='x_RccwPccwYcw', ynew ='y_RccwPccwYcw', znew ='z_RccwPccwYcw', 
                        yaw_deg =yaw_deg, rotdir='CW')
    
    cscan['d_RccwPccwYcw'] =  np.sqrt(cscan['x_RccwPccwYcw']**2 + cscan['y_RccwPccwYcw']**2)
    
    # Add earth curvature correction:
    cscan['z_RccwPccwYcw'] += earth_curv(cscan['d_RccwPccwYcw'])

    # Drop unnecessary helping columns:
    if drop_helpcols:
        cscan = cscan.drop(columns=['x_Rccw', 'y_Rccw', 'z_Rccw', 
                            'x_RccwPccw', 'y_RccwPccw', 'z_RccwPccw'])
    
    return cscan 


def rotate_PcwRcwYcw(cscan:pd.DataFrame, 
                      xcol:str, ycol:str, zcol:str, 
                      pitch_deg:str, roll_deg:str, yaw_deg:str, 
                      drop_helpcols:bool=True):
    """
    This function rotates the xyz points from Lidar frame of reference by applying
    rotation matrix in the order: 
        Pitch (clockwise) -> Roll (clockwise) -> Yaw (clockwise).

    Args:
        cscan (dataframe): dataframe contain lidar measurement
        xcol (str): column contains x values from lidar frame of reference
        ycol (str): column contains y values from lidar frame of reference
        zcol (str): column contains z values from lidar frame of refence
        pitch_deg (str): column contains pitch angles of the platform where lidar is placed
        roll_deg (str): column contains roll angles of the platform where lidar is placed
        yaw_deg (str): column contains yaw angles of the turbine
        drop_helpcols (bool): options to only get final rotation columns 
                                and drop other intermediary helping columns.

    Returns:
        dataframe: modified dataframe containing columns of the rotations
    """
    
    # Pitch rotation:
    cscan = rotate_pitch(cscan, 
                        xcol=xcol, ycol=ycol, zcol=zcol, 
                        xnew='x_Pcw', ynew='y_Pcw', znew='z_Pcw', 
                        pitch_deg=pitch_deg, rotdir='CW')

    # Roll rotation:
    cscan = rotate_roll(cscan, 
                        xcol='x_Pcw', ycol='y_Pcw', zcol='z_Pcw', 
                        xnew='x_PcwRcw', ynew='y_PcwRcw', znew='z_PcwRcw', 
                        roll_deg=roll_deg, rotdir='CW')
    
    # Yaw rotation:
    cscan = rotate_yaw(cscan, 
                        xcol ='x_PcwRcw', ycol ='y_PcwRcw', zcol ='z_PcwRcw', 
                        xnew ='x_PcwRcwYcw', ynew ='y_PcwRcwYcw', znew ='z_PcwRcwYcw', 
                        yaw_deg =yaw_deg, rotdir='CW')
    
    cscan['d_PcwRcwYcw'] =  np.sqrt(cscan['x_PcwRcwYcw']**2 + cscan['y_PcwRcwYcw']**2)
    
    # Add earth curvature correction:
    cscan['z_PcwRcwYcw'] += earth_curv(cscan['d_PcwRcwYcw'])

    # Drop unnecessary helping columns:
    if drop_helpcols:
        cscan = cscan.drop(columns=['x_Pcw', 'y_Pcw', 'z_Pcw', 
                            'x_PcwRcw', 'y_PcwRcw', 'z_PcwRcw'])
    
    return cscan 


def rotate_PcwRccwYcw(cscan:pd.DataFrame, 
                      xcol:str, ycol:str, zcol:str, 
                      pitch_deg:str, roll_deg:str, yaw_deg:str, 
                      drop_helpcols:bool=True):
    """
    This function rotates the xyz points from Lidar frame of reference by applying
    rotation matrix in the order: 
        Pitch (clockwise) -> Roll (counter-clockwise) -> Yaw (clockwise).

    Args:
        cscan (dataframe): dataframe contain lidar measurement
        xcol (str): column contains x values from lidar frame of reference
        ycol (str): column contains y values from lidar frame of reference
        zcol (str): column contains z values from lidar frame of refence
        pitch_deg (str): column contains pitch angles of the platform where lidar is placed
        roll_deg (str): column contains roll angles of the platform where lidar is placed
        yaw_deg (str): column contains yaw angles of the turbine
        drop_helpcols (bool): options to only get final rotation columns 
                                and drop other intermediary helping columns.

    Returns:
        dataframe: modified dataframe containing columns of the rotations
    """
    
    # Pitch rotation:
    cscan = rotate_pitch(cscan, 
                        xcol=xcol, ycol=ycol, zcol=zcol, 
                        xnew='x_Pcw', ynew='y_Pcw', znew='z_Pcw', 
                        pitch_deg=pitch_deg, rotdir='CW')

    # Roll rotation:
    cscan = rotate_roll(cscan, 
                        xcol='x_Pcw', ycol='y_Pcw', zcol='z_Pcw', 
                        xnew='x_PcwRccw', ynew='y_PcwRccw', znew='z_PcwRccw', 
                        roll_deg=roll_deg, rotdir='CCW')
    
    # Yaw rotation:
    cscan = rotate_yaw(cscan, 
                        xcol ='x_PcwRccw', ycol ='y_PcwRccw', zcol ='z_PcwRccw', 
                        xnew ='x_PcwRccwYcw', ynew ='y_PcwRccwYcw', znew ='z_PcwRccwYcw', 
                        yaw_deg =yaw_deg, rotdir='CW')
    
    cscan['d_PcwRccwYcw'] =  np.sqrt(cscan['x_PcwRccwYcw']**2 + cscan['y_PcwRccwYcw']**2)
    
    # Add earth curvature correction:
    cscan['z_PcwRccwYcw'] += earth_curv(cscan['d_PcwRccwYcw'])

    # Drop unnecessary helping columns:
    if drop_helpcols:
        cscan = cscan.drop(columns=['x_Pcw', 'y_Pcw', 'z_Pcw', 
                            'x_PcwRccw', 'y_PcwRccw', 'z_PcwRccw'])
    
    return cscan 


def rotate_PccwRcwYcw(cscan:pd.DataFrame, 
                      xcol:str, ycol:str, zcol:str, 
                      pitch_deg:str, roll_deg:str, yaw_deg:str, 
                      drop_helpcols:bool=True):
    """
    This function rotates the xyz points from Lidar frame of reference by applying
    rotation matrix in the order: 
        Pitch (counter-clockwise) -> Roll (clockwise) -> Yaw (clockwise).

    Args:
        cscan (dataframe): dataframe contain lidar measurement
        xcol (str): column contains x values from lidar frame of reference
        ycol (str): column contains y values from lidar frame of reference
        zcol (str): column contains z values from lidar frame of refence
        pitch_deg (str): column contains pitch angles of the platform where lidar is placed
        roll_deg (str): column contains roll angles of the platform where lidar is placed
        yaw_deg (str): column contains yaw angles of the turbine
        drop_helpcols (bool): options to only get final rotation columns 
                                and drop other intermediary helping columns.

    Returns:
        dataframe: modified dataframe containing columns of the rotations
    """
    
    # Pitch rotation:
    cscan = rotate_pitch(cscan, 
                        xcol=xcol, ycol=ycol, zcol=zcol, 
                        xnew='x_Pccw', ynew='y_Pccw', znew='z_Pccw', 
                        pitch_deg=pitch_deg, rotdir='CCW')

    # Roll rotation:
    cscan = rotate_roll(cscan, 
                        xcol='x_Pccw', ycol='y_Pccw', zcol='z_Pccw', 
                        xnew='x_PccwRcw', ynew='y_PccwRcw', znew='z_PccwRcw', 
                        roll_deg=roll_deg, rotdir='CW')
    
    # Yaw rotation:
    cscan = rotate_yaw(cscan, 
                        xcol ='x_PccwRcw', ycol ='y_PccwRcw', zcol ='z_PccwRcw', 
                        xnew ='x_PccwRcwYcw', ynew ='y_PccwRcwYcw', znew ='z_PccwRcwYcw', 
                        yaw_deg =yaw_deg, rotdir='CW')
    
    cscan['d_PccwRcwYcw'] =  np.sqrt(cscan['x_PccwRcwYcw']**2 + cscan['y_PccwRcwYcw']**2)
    
    # Add earth curvature correction:
    cscan['z_PccwRcwYcw'] += earth_curv(cscan['d_PccwRcwYcw'])

    # Drop unnecessary helping columns:
    if drop_helpcols:
        cscan = cscan.drop(columns=['x_Pccw', 'y_Pccw', 'z_Pccw', 
                            'x_PccwRcw', 'y_PccwRcw', 'z_PccwRcw'])
    
    return cscan 


def rotate_PccwRccwYcw(cscan:pd.DataFrame, 
                      xcol:str, ycol:str, zcol:str, 
                      pitch_deg:str, roll_deg:str, yaw_deg:str, 
                      drop_helpcols:bool=True):
    """
    This function rotates the xyz points from Lidar frame of reference by applying
    rotation matrix in the order: 
        Pitch (counter-clockwise) -> Roll (counter-clockwise) -> Yaw (clockwise).

    Args:
        cscan (dataframe): dataframe contain lidar measurement
        xcol (str): column contains x values from lidar frame of reference
        ycol (str): column contains y values from lidar frame of reference
        zcol (str): column contains z values from lidar frame of refence
        pitch_deg (str): column contains pitch angles of the platform where lidar is placed
        roll_deg (str): column contains roll angles of the platform where lidar is placed
        yaw_deg (str): column contains yaw angles of the turbine
        drop_helpcols (bool): options to only get final rotation columns 
                                and drop other intermediary helping columns.

    Returns:
        dataframe: modified dataframe containing columns of the rotations
    """
    
    # Roll rotation:
    cscan = rotate_pitch(cscan, 
                        xcol=xcol, ycol=ycol, zcol=zcol, 
                        xnew='x_Pccw', ynew='y_Pccw', znew='z_Pccw', 
                        pitch_deg=pitch_deg, rotdir='CCW')

    # Roll rotation:
    cscan = rotate_roll(cscan, 
                        xcol='x_Pccw', ycol='y_Pccw', zcol='z_Pccw', 
                        xnew='x_PccwRccw', ynew='y_PccwRccw', znew='z_PccwRccw', 
                        roll_deg=roll_deg, rotdir='CCW')
    
    # Yaw rotation:
    cscan = rotate_yaw(cscan, 
                        xcol ='x_PccwRccw', ycol ='y_PccwRccw', zcol ='z_PccwRccw', 
                        xnew ='x_PccwRccwYcw', ynew ='y_PccwRccwYcw', znew ='z_PccwRccwYcw', 
                        yaw_deg =yaw_deg, rotdir='CW')
    
    cscan['d_PccwRccwYcw'] =  np.sqrt(cscan['x_PccwRccwYcw']**2 + cscan['y_PccwRccwYcw']**2)
    
    # Add earth curvature correction:
    cscan['z_PccwRccwYcw'] = (cscan['z_PccwRccwYcw'] + earth_curv(cscan['d_PccwRccwYcw']))

    # Drop unnecessary helping columns:
    if drop_helpcols:
        cscan = cscan.drop(columns=['x_Pccw', 'y_Pccw', 'z_Pccw', 
                            'x_PccwRccw', 'y_PccwRccw', 'z_PccwRccw'])
    
    return cscan 


def rotate_PRY(cscan, xcol: str, ycol: str, zcol: str, pitch_deg:str, roll_deg:str, yaw_deg:str, rot_order: str, drop_helpcols=True):
    """
    This function rotates the xyz points from Lidar frame of reference by applying
    rotation matrix in the selected order.

    Args:
        cscan (DataFrame): DataFrame containing lidar measurement
        xcol (str): column contains x values from lidar frame of reference
        ycol (str): column contains y values from lidar frame of reference
        zcol (str): column contains z values from lidar frame of refence
        pitch_deg (str): column contains pitch angles of the platform where lidar is placed
        roll_deg (str): column contains roll angles of the platform where lidar is placed
        yaw_deg (str): column contains yaw angles of the turbine
        
        rot_order (str): order of the rotation operation: 
            "RcwPcwYcw", "RcwPccwYcw", "RccwPcwYcw", "RccwPccwYcw", 
            "PcwRcwYcw", "PcwRccwYcw", "PccwRcwYcw", "PccwRccwYcw" 
        
        drop_helpcols (bool, optional): Dropping helping columns. Defaults to True.

    Returns:
        DataFrame: modified DataFrame containing columns of the rotations
    """

    func_map = {
        "RcwPcwYcw": rotate_RcwPcwYcw,
        "RcwPccwYcw": rotate_RcwPccwYcw,
        "RccwPcwYcw": rotate_RccwPcwYcw,
        "RccwPccwYcw": rotate_RccwPccwYcw,
        "PcwRcwYcw": rotate_PcwRcwYcw,
        "PcwRccwYcw": rotate_PcwRccwYcw,
        "PccwRcwYcw": rotate_PccwRcwYcw,
        "PccwRccwYcw": rotate_PccwRccwYcw
    }

    if rot_order not in func_map:
        raise ValueError(f"Invalid rot_order '{rot_order}'. Must be one of {list(func_map)}")

    # Apply the selected rotation order:
    cscan = func_map[rot_order](cscan, 
                                xcol = xcol, ycol = ycol, zcol = zcol, 
                                pitch_deg = pitch_deg, roll_deg = roll_deg, yaw_deg = yaw_deg,
                                drop_helpcols = drop_helpcols)

    return cscan


def angular_PaddRadd(cscan:pd.DataFrame, 
                     elevlidar_col:str, 
                     azilidar_col:str,
                     range_col:str, 
                     pitch_deg:str, 
                     roll_deg:str, 
                     azim_offset:str, 
                     pitch_offset:float = 90, 
                     roll_offset:float = 0,
                     elevnew:str = 'elev_true',
                     azinew:str = 'azi_true',
                     xnew:str = 'x_true',
                     ynew:str = 'y_true',
                     znew:str = 'z_true',
                     dnew:str = 'd_true'):
    """
    This function is used to apply angular correction by applying addition operation
    to lidar elevation by pitch angles and roll angles where the pitches and rolls have
    a certain offset from the lidar's azimuth:

        |-------------------------------------------------------------------|
        |   elevation = lidar_elevation                                     |
        |               + pitch * cos ( pitch_offset - lidar_azimuth )      |
        |               + roll * cos ( roll_offset - lidar azimuth)         |
        |-------------------------------------------------------------------|

    Args:
        cscan (pd.DataFrame): DataFrame containing lidar measurement data.
        elevlidar_col (str): column of lidar elevation.
        azilidar_col (str): column of lidar azimuth.
        range_col (str): column of lidar range gates.
        pitch_deg (str): column of platform pitch angle (in degrees).
        roll_deg (str): column of platform roll angle (in degrees).
        azim_offset (str): column that indicating lidar's azimuth offset from meteorological north.
        pitch_offset (float, optional): Pitch orientation offset to lidar's 0° azimuth. 
            Defaults to 90, i.e., pitch's influence is maximum when lidar azimuth is 90° and 270°.
        roll_offset (float, optional): Roll orientation offset to lidar's 0° azimuth. 
            Defaults to 0, i.e., roll's influence is maximum when lidar azimuth is 0° and 180°.
        elevnew (str, optional): Column name for the corrected elevation. Defaults to 'elev_true'.
        azinew (str, optional): Column name for the corrected azimuth. Defaults to 'azi_true'.
        xnew (str, optional): Column name for the corrected x-coordinate. Defaults to 'x_true'.
        ynew (str, optional): Column name for the corrected y-coordinate. Defaults to 'y_true'.
        znew (str, optional): Column name for the corrected z-coordinate. Defaults to 'z_true'.
        dnew (str, optional): Column name for the corrected horizontal distance. Defaults to 'd_true'.

    Returns:
        pd.DataFrame: Modified dataframe.
    """

    # Elevation correction:
    cscan[elevnew] = (cscan[elevlidar_col] 
                              + (cscan[pitch_deg] 
                                 * np.cos((pitch_offset - cscan[azilidar_col]) * np.pi/180)
                                 )
                              + (cscan[roll_deg]
                                 * np.cos((roll_offset - cscan[azilidar_col]) * np.pi/180)
                                 )
                            )
    
    # Azimuth correction:
    cscan[azinew] = np.mod(cscan[azilidar_col] + cscan[azim_offset], 360)
    
    # Recalculate xyzd from the 
    cscan = recalculate_xyzd(cscan, elevation_col=elevnew, azimuth_col=azinew, range_col=range_col, z_tp=90, 
                                xnew=xnew, ynew=ynew, znew=znew, dnew=dnew)
    
    # Earth curvature correction
    cscan[znew] += earth_curv(cscan[dnew])

    return cscan


def angular_PaddRsub(cscan:pd.DataFrame, 
                     elevlidar_col:str, 
                     azilidar_col:str,
                     range_col:str, 
                     pitch_deg:str, 
                     roll_deg:str, 
                     azim_offset:str, 
                     pitch_offset:float = 90, 
                     roll_offset:float = 0,
                     elevnew:str = 'elev_true',
                     azinew:str = 'azi_true',
                     xnew:str = 'x_true',
                     ynew:str = 'y_true',
                     znew:str = 'z_true',
                     dnew:str = 'd_true'):
    """
    This function is used to apply angular correction by applying addition operation
    to lidar elevation by pitch angles and roll angles where the pitches and rolls have
    a certain offset from the lidar's azimuth:

        |-------------------------------------------------------------------|
        |   elevation = lidar_elevation                                     |
        |               + pitch * cos ( pitch_offset - lidar_azimuth )      |
        |               - roll * cos ( roll_offset - lidar azimuth)         |
        |-------------------------------------------------------------------|

    Args:
        cscan (pd.DataFrame): DataFrame containing lidar measurement data.
        elevlidar_col (str): column of lidar elevation.
        azilidar_col (str): column of lidar azimuth.
        range_col (str): column of lidar range gates.
        pitch_deg (str): column of platform pitch angle (in degrees).
        roll_deg (str): column of platform roll angle (in degrees).
        azim_offset (str): column that indicating lidar's azimuth offset from meteorological north.
        pitch_offset (float, optional): Pitch orientation offset to lidar's 0° azimuth. 
            Defaults to 90, i.e., pitch's influence is maximum when lidar azimuth is 90° and 270°.
        roll_offset (float, optional): Roll orientation offset to lidar's 0° azimuth. 
            Defaults to 0, i.e., roll's influence is maximum when lidar azimuth is 0° and 180°.
        elevnew (str, optional): Column name for the corrected elevation. Defaults to 'elev_true'.
        azinew (str, optional): Column name for the corrected azimuth. Defaults to 'azi_true'.
        xnew (str, optional): Column name for the corrected x-coordinate. Defaults to 'x_true'.
        ynew (str, optional): Column name for the corrected y-coordinate. Defaults to 'y_true'.
        znew (str, optional): Column name for the corrected z-coordinate. Defaults to 'z_true'.
        dnew (str, optional): Column name for the corrected horizontal distance. Defaults to 'd_true'.

    Returns:
        pd.DataFrame: Modified dataframe.
    """

    # Elevation correction:
    cscan[elevnew] = (cscan[elevlidar_col] 
                              + (cscan[pitch_deg] 
                                 * np.cos((pitch_offset - cscan[azilidar_col]) * np.pi/180)
                                 )
                              - (cscan[roll_deg]
                                 * np.cos((roll_offset - cscan[azilidar_col]) * np.pi/180)
                                 )
                            )
    
    # Azimuth correction:
    cscan[azinew] = np.mod(cscan[azilidar_col] + cscan[azim_offset], 360)
    
    # Recalculate xyzd from the 
    cscan = recalculate_xyzd(cscan, elevation_col=elevnew, azimuth_col=azinew, range_col=range_col, z_tp=90, 
                                xnew=xnew, ynew=ynew, znew=znew, dnew=dnew)
    
    # Earth curvature correction
    cscan[znew] += earth_curv(cscan[dnew])

    return cscan


def angular_PsubRadd(cscan:pd.DataFrame, 
                     elevlidar_col:str, 
                     azilidar_col:str,
                     range_col:str, 
                     pitch_deg:str, 
                     roll_deg:str, 
                     azim_offset:str, 
                     pitch_offset:float = 90, 
                     roll_offset:float = 0,
                     elevnew:str = 'elev_true',
                     azinew:str = 'azi_true',
                     xnew:str = 'x_true',
                     ynew:str = 'y_true',
                     znew:str = 'z_true',
                     dnew:str = 'd_true'):
    """
    This function is used to apply angular correction by applying addition operation
    to lidar elevation by pitch angles and roll angles where the pitches and rolls have
    a certain offset from the lidar's azimuth:

        |-------------------------------------------------------------------|
        |   elevation = lidar_elevation                                     |
        |               - pitch * cos ( pitch_offset - lidar_azimuth )      |
        |               + roll * cos ( roll_offset - lidar azimuth)         |
        |-------------------------------------------------------------------|

    Args:
        cscan (pd.DataFrame): DataFrame containing lidar measurement data.
        elevlidar_col (str): column of lidar elevation.
        azilidar_col (str): column of lidar azimuth.
        range_col (str): column of lidar range gates.
        pitch_deg (str): column of platform pitch angle (in degrees).
        roll_deg (str): column of platform roll angle (in degrees).
        azim_offset (str): column that indicating lidar's azimuth offset from meteorological north.
        pitch_offset (float, optional): Pitch orientation offset to lidar's 0° azimuth. 
            Defaults to 90, i.e., pitch's influence is maximum when lidar azimuth is 90° and 270°.
        roll_offset (float, optional): Roll orientation offset to lidar's 0° azimuth. 
            Defaults to 0, i.e., roll's influence is maximum when lidar azimuth is 0° and 180°.
        elevnew (str, optional): Column name for the corrected elevation. Defaults to 'elev_true'.
        azinew (str, optional): Column name for the corrected azimuth. Defaults to 'azi_true'.
        xnew (str, optional): Column name for the corrected x-coordinate. Defaults to 'x_true'.
        ynew (str, optional): Column name for the corrected y-coordinate. Defaults to 'y_true'.
        znew (str, optional): Column name for the corrected z-coordinate. Defaults to 'z_true'.
        dnew (str, optional): Column name for the corrected horizontal distance. Defaults to 'd_true'.

    Returns:
        pd.DataFrame: Modified dataframe.
    """

    # Elevation correction:
    cscan[elevnew] = (cscan[elevlidar_col] 
                              - (cscan[pitch_deg] 
                                 * np.cos((pitch_offset - cscan[azilidar_col]) * np.pi/180)
                                 )
                              + (cscan[roll_deg]
                                 * np.cos((roll_offset - cscan[azilidar_col]) * np.pi/180)
                                 )
                            )
    
    # Azimuth correction:
    cscan[azinew] = np.mod(cscan[azilidar_col] + cscan[azim_offset], 360)
    
    # Recalculate xyzd from the 
    cscan = recalculate_xyzd(cscan, elevation_col=elevnew, azimuth_col=azinew, range_col=range_col, z_tp=90, 
                                xnew=xnew, ynew=ynew, znew=znew, dnew=dnew)
    
    # Earth curvature correction
    cscan[znew] += earth_curv(cscan[dnew])

    return cscan


def angular_PsubRsub(cscan:pd.DataFrame, 
                     elevlidar_col:str, 
                     azilidar_col:str,
                     range_col:str, 
                     pitch_deg:str, 
                     roll_deg:str, 
                     azim_offset:str, 
                     pitch_offset:float = 90, 
                     roll_offset:float = 0,
                     elevnew:str = 'elev_true',
                     azinew:str = 'azi_true',
                     xnew:str = 'x_true',
                     ynew:str = 'y_true',
                     znew:str = 'z_true',
                     dnew:str = 'd_true'):
    """
    This function is used to apply angular correction by applying addition operation
    to lidar elevation by pitch angles and roll angles where the pitches and rolls have
    a certain offset from the lidar's azimuth:

        |-------------------------------------------------------------------|
        |   elevation = lidar_elevation                                     |
        |               - pitch * cos ( pitch_offset - lidar_azimuth )      |
        |               - roll * cos ( roll_offset - lidar azimuth)         |
        |-------------------------------------------------------------------|

    Args:
        cscan (pd.DataFrame): DataFrame containing lidar measurement data.
        elevlidar_col (str): column of lidar elevation.
        azilidar_col (str): column of lidar azimuth.
        range_col (str): column of lidar range gates.
        pitch_deg (str): column of platform pitch angle (in degrees).
        roll_deg (str): column of platform roll angle (in degrees).
        azim_offset (str): column that indicating lidar's azimuth offset from meteorological north.
        pitch_offset (float, optional): Pitch orientation offset to lidar's 0° azimuth. 
            Defaults to 90, i.e., pitch's influence is maximum when lidar azimuth is 90° and 270°.
        roll_offset (float, optional): Roll orientation offset to lidar's 0° azimuth. 
            Defaults to 0, i.e., roll's influence is maximum when lidar azimuth is 0° and 180°.
        elevnew (str, optional): Column name for the corrected elevation. Defaults to 'elev_true'.
        azinew (str, optional): Column name for the corrected azimuth. Defaults to 'azi_true'.
        xnew (str, optional): Column name for the corrected x-coordinate. Defaults to 'x_true'.
        ynew (str, optional): Column name for the corrected y-coordinate. Defaults to 'y_true'.
        znew (str, optional): Column name for the corrected z-coordinate. Defaults to 'z_true'.
        dnew (str, optional): Column name for the corrected horizontal distance. Defaults to 'd_true'.

    Returns:
        pd.DataFrame: Modified dataframe.
    """

    # Elevation correction:
    cscan[elevnew] = (cscan[elevlidar_col] 
                              - (cscan[pitch_deg] 
                                 * np.cos((pitch_offset - cscan[azilidar_col]) * np.pi/180)
                                 )
                              - (cscan[roll_deg]
                                 * np.cos((roll_offset - cscan[azilidar_col]) * np.pi/180)
                                 )
                            )
    
    # Azimuth correction:
    cscan[azinew] = np.mod(cscan[azilidar_col] + cscan[azim_offset], 360)
    
    # Recalculate xyzd from the 
    cscan = recalculate_xyzd(cscan, elevation_col=elevnew, azimuth_col=azinew, range_col=range_col, z_tp=90, 
                                xnew=xnew, ynew=ynew, znew=znew, dnew=dnew)
    
    # Earth curvature correction
    cscan[znew] += earth_curv(cscan[dnew])

    return cscan


def angular_addsub(cscan:pd.DataFrame,
                angular_operation:str, 
                elevlidar_col:str, 
                azilidar_col:str,
                range_col:str, 
                pitch_deg:str, 
                roll_deg:str, 
                azim_offset:str, 
                pitch_offset:float = 90, 
                roll_offset:float = 0,
                elevnew:str = 'elev_true',
                azinew:str = 'azi_true',
                xnew:str = 'x_true',
                ynew:str = 'y_true',
                znew:str = 'z_true',
                dnew:str = 'd_true'):
    """
    This function rotates the xyz points from Lidar frame of reference by applying
    rotation matrix in the selected order.

    Args:
        cscan (pd.DataFrame): DataFrame containing lidar measurement data.
        angular_operation (str): selected angular operation:
            "PaddRadd", "PaddRsub", "PsubRadd", "PsubRsub"
        elevlidar_col (str): column of lidar elevation.
        azilidar_col (str): column of lidar azimuth.
        range_col (str): column of lidar range gates.
        pitch_deg (str): column of platform pitch angle (in degrees).
        roll_deg (str): column of platform roll angle (in degrees).
        azim_offset (str): column that indicating lidar's azimuth offset from meteorological north.
        pitch_offset (float, optional): Pitch orientation offset to lidar's 0° azimuth. 
            Defaults to 90, i.e., pitch's influence is maximum when lidar azimuth is 90° and 270°.
        roll_offset (float, optional): Roll orientation offset to lidar's 0° azimuth. 
            Defaults to 0, i.e., roll's influence is maximum when lidar azimuth is 0° and 180°.
        elevnew (str, optional): Column name for the corrected elevation. Defaults to 'elev_true'.
        azinew (str, optional): Column name for the corrected azimuth. Defaults to 'azi_true'.
        xnew (str, optional): Column name for the corrected x-coordinate. Defaults to 'x_true'.
        ynew (str, optional): Column name for the corrected y-coordinate. Defaults to 'y_true'.
        znew (str, optional): Column name for the corrected z-coordinate. Defaults to 'z_true'.
        dnew (str, optional): Column name for the corrected horizontal distance. Defaults to 'd_true'.

    Returns:
        pd.DataFrame: Modified dataframe.
    """

    func_map = {
        "PaddRadd": angular_PaddRadd,
        "PaddRsub": angular_PaddRsub,
        "PsubRadd": angular_PsubRadd,
        "PsubRsub": angular_PsubRsub,
    }

    if angular_operation not in func_map:
        raise ValueError(f"Invalid rot_order '{angular_operation}'. Must be one of {list(func_map)}")

    # Apply the selected rotation order:
    cscan = func_map[angular_operation](cscan, 
                                elevlidar_col = elevlidar_col, 
                                azilidar_col = azilidar_col,
                                range_col = range_col, 
                                pitch_deg = pitch_deg, 
                                roll_deg = roll_deg, 
                                azim_offset = azim_offset, 
                                pitch_offset = pitch_offset, 
                                roll_offset = roll_offset,
                                elevnew = elevnew,
                                azinew = azinew,
                                xnew = xnew,
                                ynew = ynew,
                                znew = znew,
                                dnew = dnew)

    return cscan
