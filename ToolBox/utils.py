import time
import numpy as np 
import 
#####################
#  Compute Distane  #
#####################

def minkowski_distance(x1, x2, y1, y2, p):
    delta_x = x1 - x2
    delta_y = y1 - y2
    return ((abs(delta_x) ** p) + (abs(delta_y)) ** p) ** (1 / p)

def deg2rad(coordinate):
    return coordinate * np.pi / 180
    
# convert radians into distance
def rad2dist(coordinate):
    earth_radius = 6371 # km
    return earth_radius * coordinate
    
# correct the longitude distance regarding the latitude (https://jonisalonen.com/2014/computing-distance-between-coordinates-can-be-simple-and-fast/)
def lng_dist_corrected(lng_dist, lat):
    return lng_dist * np.cos(lat)

def minkowski_distance_gps(lat1, lat2, lon1, lon2, p):
    lat1, lat2, lon1, lon2 = [deg2rad(coordinate) for coordinate in [lat1, lat2, lon1, lon2]]
    y1, y2, x1, x2 = [rad2dist(angle) for angle in [lat1, lat2, lon1, lon2]]
    x1, x2 = [lng_dist_corrected(elt['x'], elt['lat']) for elt in [{'x': x1, 'lat': lat1}, {'x': x2, 'lat': lat2}]]
    return minkowski_distance(x1, x2, y1, y2, p)

def get_manhattan_distance(lat1, lat2, lon1, lon2):
    '''
        Caculate the distance between two points measured 
        along axes at right angles. 
        Return a int/pandas Series.
    '''
    return minkowski_distance_gps(lat1, lat2, lon1, lon2, 1)

def get_euclidian_distance(lat1, lat2, lon1, lon2):
    '''
        Caculate euclidian distance between two points. 
        Return a int/pandas Series.
    '''
    return minkowski_distance_gps(lat1, lat2, lon1, lon2, 2)

def get_haversine_distance(lat1, lat2, lon1, lon2):
    """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees).
        Input/Return--pandas Series.
    """

    lat_1_rad, lon_1_rad = np.radians(lat1.astype(float)), np.radians(lon1.astype(float))
    lat_2_rad, lon_2_rad = np.radians(lat2.astype(float)), np.radians(lon2.astype(float))
    dlon = lon_2_rad - lon_1_rad
    dlat = lat_2_rad - lat_1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_1_rad) * np.cos(lat_2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    haversine_distance = 6371 * c
    return haversine_distance


################
#  DECORATORS  #
################

def simple_time_tracker(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int(te - ts)
        else:
            print(method.__name__, round(te - ts, 2))
        return result

    return timed