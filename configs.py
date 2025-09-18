import numpy as np
import datci


simu_pace = 0.1

wheelbase = 2.8
wheelbase_r = 1.6
wheelbase_f = 1.2
max_delta_f = np.deg2rad(15)

a_max, a_min = 2.00, -4.00
d_max, d_min = np.tan(max_delta_f), -np.tan(max_delta_f)
v_max, v_min = 40.0, 0.0
h_max, h_min = np.deg2rad(30), np.deg2rad(-30)


# sigma_ax = 0.3
# sigma_ay = 0.02
# risk_level = 0.3

global __risk_level, __sigma_ax, __sigma_ay
__risk_level = 0.5
__sigma_ax = 0.6
__sigma_ay = 0.01


plan_pace = 0.1
plan_horizon = round(4.0/plan_pace)


def set_risk_level(u:float):
    global __risk_level
    __risk_level = u

def get_risk_level():
    global __risk_level
    return __risk_level

def set_sigma(ax:float, ay:float):
    global __sigma_ax, __sigma_ay
    __sigma_ax = ax
    __sigma_ay = ay

def get_sigma():
    global __sigma_ax, __sigma_ay
    return __sigma_ax, __sigma_ay