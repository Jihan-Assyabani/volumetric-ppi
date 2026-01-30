import numpy as np
import pandas as pd
import netCDF4 
import gzip 
import os 
from datetime import datetime 

def read_ppi(pathfile, z_tp:float) -> pd.DataFrame:
    """
    This function takes ppi scan with specified name and scan type, unzips the file an reads it as a netCDF Dataset. Subsequently,
    the data is stored in a pandas DataFrame and returned.


    Parameters
    ----------
    pathfile : TYPE = string
        path to the desired "*.nc.gz" file which includes the specific scan

    Returns
    -------
    cscan : TYPE = DataFrame
        DataFrame including all measurement points of the loaded scan, including
            all necessary desired parameters

    """

    cscan = pd.DataFrame(columns=["time", "azi_L", "ele_L", "range", "vlos", "filter", "x_L", "y_L", "z_L"])

    with gzip.open(pathfile) as gz:
        with netCDF4.Dataset('dummy', mode='r', memory=gz.read()) as nc:

            keys = nc.groups.keys()
            groupname = list(keys)[1]

            # print(nc.groups[groupname])

            cscan["filter"] = pd.DataFrame(nc.groups[groupname]['radial_wind_speed_status'][:],
                                           dtype='bool').to_numpy().flatten()  # Filter by Leosphere
            vlos = pd.DataFrame(nc.groups[groupname]['radial_wind_speed'][:]).to_numpy().flatten()
            cscan["vlos"] = - vlos  #minus sign so that inflow is positive and wake is negative
            cscan["cnr"] = pd.DataFrame(nc.groups[groupname]['cnr'][:]).to_numpy().flatten()

            ranges, azis = np.meshgrid(nc.groups[groupname]['range'][:], nc.groups[groupname]['azimuth'][:])
            _, eles = np.meshgrid(nc.groups[groupname]['range'][:], nc.groups[groupname]['elevation'][:])
            _, time = np.meshgrid(nc.groups[groupname]['range'][:], nc.groups[groupname]['time'][:])
            cscan["range"] = ranges.flatten()
            cscan["azi_L"] = azis.flatten()
            cscan["ele_L"] = eles.flatten()
            cscan["time"] = time.flatten()

            cscan["x_L"] = (cscan["range"] * np.sin((90 - cscan["ele_L"]) * np.pi / 180) * np.cos(
                (cscan["azi_L"] - 90) * np.pi / 180))
            cscan["y_L"] = (-cscan["range"] * np.sin((90 - cscan["ele_L"]) * np.pi / 180) * np.sin(
                (cscan["azi_L"] - 90) * np.pi / 180))
            cscan["z_L"] = cscan["range"] * np.cos((90 - cscan["ele_L"]) * np.pi / 180) + z_tp
            cscan["d_L"] = cscan["range"] * np.sin((90 - cscan["ele_L"]) * np.pi / 180)

        return cscan  ##, l_pos
    
def read_gnss(gnss_path, date, yaw_offset=0, tilt_offset=0, export_feather=False, export_path=None):
    """
    This function reads GNSS data from single csv file  and returns a dataframe.
    -----------
    Input:
        gnss_path: path to single GNSS file
    -----------
    Return:
        GNSS_df: DataFrame containing the processed GNSS data
    """
    
    GNSS_df = pd.DataFrame(columns=["time", "yaw", "tilt"])
    
    raw_df = pd.read_csv(gnss_path, sep=",", header=None)

    # Convert the time column to datetime format:
    raw_df[2] = pd.to_datetime(raw_df[2].astype(int).astype(str).str.zfill(6), format="%H%M%S").dt.time 

    #take date from file name:
    date_for_index = datetime.strptime(date, "%Y%m%d").date()
    raw_df[2] = raw_df[2].apply(lambda x: datetime.combine(date_for_index, x))

    # Store relevant columns in GNSS_df
    GNSS_df["time"] = raw_df[2]
    GNSS_df["time"] = pd.to_datetime(GNSS_df["time"], format="%Y-%m-%d %H:%M:%S.%f")
    # increase time resolution to 10 ms

    GNSS_df["yaw"] = pd.to_numeric(raw_df[3], errors="coerce") + yaw_offset
    GNSS_df["tilt"] = pd.to_numeric(raw_df[5], errors="coerce") + tilt_offset
    GNSS_df = GNSS_df.iloc[:-1]

    # export to .feather option:
    if export_feather:
        gnss_date = datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')

        if export_path is not None:
            GNSS_df.to_feather(f"{export_path}/gnss_{gnss_date}.feather")
        else:
            GNSS_df.to_feather(f"{gnss_path}/gnss_{gnss_date}.feather")
    else:
        pass 

    return GNSS_df 

def detect_separator(file_path, skiprows=14):
    with open(file_path, encoding="latin-1") as f:
        # Skip header lines
        for _ in range(skiprows):
            next(f)
        # Read the first data line
        sample_line = f.readline()
        if ";" in sample_line and "\t" not in sample_line:
            return ";"
        elif "\t" in sample_line and ";" not in sample_line:
            return "\t"
        elif ";" in sample_line and "\t" in sample_line:
            raise ValueError("Ambiguous line: contains both ';' and tabs.")
        else:
            raise ValueError("Unknown separator in file.")

def read_inclino(inclino_path, pitch_sensor_id="1128", roll_sensor_id="1150", sampling=None):
    """
        This function handle inclinometer data from a single csv file.
        -----------
        Input:
            inclino_path: path to single inclinometer file
        -----------
        Return:
            inclino_df: DataFrame containing the processed inclinometer data
    """
    inclino_df = pd.DataFrame(columns=["time", "accelX_g", "accelY_g", "accelZ_g", "pitch", "roll"])

    # Check serial number of sensor #2 and #3:
    # Serial number for pitch sensor is 1128, roll sensor is 1150
    with open(inclino_path, 'r', encoding="latin-1") as f:
        lines = f.readlines()
        for line in lines:
            if "#2_SensorTypeName:" in line:
                sensor_2 = line.split("SerialNumber:")[1].strip().split()[0]
                pass 
            if "#3_SensorTypeName:" in line:
                sensor_3 = line.split("SerialNumber:")[1].strip().split()[0]
                break
    
    sep = detect_separator(inclino_path)
    raw_df = pd.read_csv(inclino_path, sep=sep, header=None, skiprows=14, encoding="latin-1", low_memory=False)
    raw_df.columns = raw_df.iloc[0].astype(str)
    raw_df = raw_df[1:]

    # Extracting relevant variables from raw_df:
    inclino_df["time"]= pd.to_datetime(raw_df["# Data acquisition time"], format="%Y-%m-%d %H:%M:%S.%f")
    inclino_df["accelX_g"] = raw_df.loc[:, raw_df.columns.str.contains("_Acceleration X")]
    inclino_df["accelY_g"] = raw_df.loc[:, raw_df.columns.str.contains("_Acceleration Y")]
    inclino_df["accelZ_g"] = raw_df.loc[:, raw_df.columns.str.contains("_Acceleration Z")]

    # 1128 is pitch sensor; 1150 is roll sensor:
    if sensor_2 == pitch_sensor_id: 
        inclino_df["pitch"] = raw_df.loc[:, raw_df.columns.str.contains("#2_Signal1 Low-pass filter")]
        inclino_df["roll"] = raw_df.loc[:, raw_df.columns.str.contains("#3_Signal1 Low-pass filter")]
    elif sensor_3 == pitch_sensor_id:
        inclino_df["pitch"] = raw_df.loc[:, raw_df.columns.str.contains("#3_Signal1 Low-pass filter")]
        inclino_df["roll"] = raw_df.loc[:, raw_df.columns.str.contains("#2_Signal1 Low-pass filter")]
    else:
        pass
    
    # Sorting and cleaning:
    for col in inclino_df.columns[1:]:
        inclino_df[col] = pd.to_numeric(inclino_df[col], errors="coerce")
        
    inclino_df = inclino_df.sort_values(by="time", ascending=True, ignore_index=True)
    
    if sampling is not None:
        inclino_df = inclino_df.resample(sampling).mean()

    return inclino_df

def inclino_merge(path, sampling=None, export_parquet=False):
    """
        This function merges multiple inclinometer data files into a single DataFrame.
        The merged data should be stored in the same 'path' directory. 
        -----------
        Input:
            path: path to the directory containing the inclinometer files
        -----------
        Return:
            merged_inclino_df: DataFrame containing the merged inclinometer data.
    """
    inclino_files = os.listdir(path)
    # read inclino files if it contains "protocol_MULTI_SENSOR"
    inclino_files = [f for f in inclino_files if f.endswith(".csv") and "protocol_MULTI_SENSOR" in f] 
    inclino_files.sort()
    
    merged_inclino_df = pd.DataFrame()
    
    for file in inclino_files:
        print(f"Handling {file}...")
        inclino_path = os.path.join(path, file)
        inclino_df = read_inclino(inclino_path, sampling)
        merged_inclino_df = pd.concat([merged_inclino_df, inclino_df], axis=0, ignore_index=True)  

    
    if export_parquet:
        inclinodata_date_start = pd.to_datetime(merged_inclino_df["time"].iloc[0]).date()
        inclinodata_date_start = inclinodata_date_start.strftime('%Y-%m-%d')

        inclinodata_date_end = pd.to_datetime(merged_inclino_df["time"].iloc[-1]).date()
        inclinodata_date_end = inclinodata_date_end.strftime('%Y-%m-%d')

        if inclinodata_date_start == inclinodata_date_end:
            inclinodata_date = inclinodata_date_start
            merged_inclino_df.to_parquet(os.path.join(path,f"inclino_{inclinodata_date}.parquet"))
        else:
            merged_inclino_df.to_parquet(os.path.join(path,f"inclino_{inclinodata_date_start}_{inclinodata_date_end}.parquet"))

    return merged_inclino_df

def files_in_interval(scan_list, start_time, end_time, keyword=None, fmt=".parquet"):
    """
    This function filters the list of scan files based on the specified time interval.

    Parameters:
    scan_list : list
        List of scan file names.
    start_time : datetime
        Start time of the interval.
    end_time : datetime
        End time of the interval.

    Returns:
    list
        List of scan files within the specified time interval.
    """

    files_in_interval = []

    # select only files that match the specified format
    scan_list = [file for file in scan_list if file.endswith(fmt) and keyword in file]
    #print(scan_list)

    for file in scan_list:
        
        # Extract time component from file name
        # File name format: WLS400s-192_YYYY-MM-DD_HH-MM-SS_ppi_1944_150m.nc.gz
        # or               'WLS400s-192_2024-12-12_01-39-44_inflow.parquet'
        if keyword is None:
            filetime_str =file.split("_")[1] + "_" + file.split("_")[2]
        else:
            filetime_str =file.split("_")[2] + "_" + file.split("_")[3]
        
        # Convert to datetime object
        filetime = datetime.strptime(filetime_str, "%Y-%m-%d_%H-%M-%S")
        #print(filetime)
        
        if start_time <= filetime <= end_time:
            files_in_interval.append(file)
        
    return files_in_interval

