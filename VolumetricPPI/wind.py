import numpy as np 

def log_profile(vref, href, hrange, roughness):
    v = vref * np.log(hrange/roughness) / np.log(href/roughness)
    return v