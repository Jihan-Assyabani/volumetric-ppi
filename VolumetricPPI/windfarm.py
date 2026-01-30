import numpy as np 
import pandas as pd 
from pyproj import Proj
import matplotlib.pyplot as plt 
from geopy.distance import distance

def latlong2xy(lat, long, zone, ellps='WGS84'):
    """
    Convert latitude and longitude to cartesian coordinates

    Args:
        lat (np_array): latitude (in degree)
        long (np_array): longitude (in degree)
        zone (int): UTM zone
        ellps (str): ellipsoid

    Returns:
        (np_array): x coordinate
        (np_array): y coordinates
    """
    p = Proj(proj='utm',zone=zone, ellps=ellps)
    x,y = p(long,lat)
    return x,y
