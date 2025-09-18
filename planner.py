import os, sys
import numpy as np
from numpy import ndarray, array
from copy import deepcopy
from typing import Tuple, Union
from rich import print
from math import cos, sin, tan, acos, asin, atan, atan2, exp, sqrt, log
from dataclasses import dataclass
import scipy
import scipy.optimize
from cyipopt import minimize_ipopt
import time

import datci
from utils import block_matrix as bm
from utils.controller import convert_system_continuous_to_discrete, linear_mpc, linear_mpc_track, eig, discrete_algebraic_Riccati_equation, lqr
from scipy.spatial import ConvexHull
from utils.polytope import Polytope
from utils.ellipse import Ellipse2D
from utils.simple_agent import Agent, AgentTrajectory, get_agent_from_daci
from configs import *

def moment_s(k:int, E1:float, E2:float):
    Ts = plan_pace
    if k==0:
        s_E1 = 0
        s_E2 = 0
    else:
        m1 = 0.5 * Ts**2 * k**2
        m2 = k * (2*k - 1) * (2*k + 1) * Ts**4 / 12
        m3 = k * (k - 1) * (3*k**2 - k - 1) * Ts**4 / 12
        s_E1 = m1 * E1
        s_E2 = m2 * E2 + m3 * E1**2
    
    return s_E1, s_E2

def agt_radius_xy(agt:Agent):
    rx = sqrt((agt.length/2)**2 + (agt.width/2)**2)
    ry = agt.width/2
    return rx, ry

def agt_risk_radius(agt:Agent, k:int, sigma_ax:float, sigma_ay:float, risk_level:float) -> float:

    rx, ry = agt_radius_xy(agt)
    eps = Ellipse2D(agt.x, agt.y, rx, ry, agt.h)
    w11, w22, w12 = eps.rmat[0,0], eps.rmat[1,1], eps.rmat[0,1]
    sx_E1, sx_E2 = moment_s(k, 0, sigma_ax**2)
    sy_E1, sy_E2 = moment_s(k, 0, sigma_ay**2)
    Ls_E2 = w11 * sx_E2 + w22 * sy_E2 + 2 * w12 * sx_E1 * sy_E1

    return 1 + sqrt(2 / risk_level * Ls_E2)

def agt_nominal_xy(agt: Agent, k:int):
    Ts = plan_pace
    nx = agt.x + agt.v * cos(agt.h) * k * Ts
    ny = agt.y + agt.v * sin(agt.h) * k * Ts
    return (nx, ny)

def get_possible_tm_lon(veh_e:Agent, veh_p:Agent, veh_r:Agent, veh_f:Agent, use_risk:bool = True):
    Ts = plan_pace
    ts = np.arange(0, plan_horizon+1) * Ts
    risk_level = get_risk_level()
    sigma_ax, sigma_ay = get_sigma()

    pos_vp = np.array([agt_nominal_xy(veh_p, k) for k in range(len(ts))])
    rx, ry = agt_radius_xy(veh_p)
    if use_risk:
        rad_vp = np.array([rx * agt_risk_radius(veh_p, k, sigma_ax, sigma_ay, risk_level) for k in range(len(ts))])
    else:
        rad_vp = rx * np.ones_like(ts)
    rad_vp = rad_vp + veh_e.length/2

    pos_vr = np.array([agt_nominal_xy(veh_r, k) for k in range(len(ts))])
    rx, ry = agt_radius_xy(veh_r)
    if use_risk:
        rad_vr = np.array([rx * agt_risk_radius(veh_r, k, sigma_ax, sigma_ay, risk_level) for k in range(len(ts))])
    else:
        rad_vr = rx * np.ones_like(ts)
    rad_vr = rad_vr + veh_e.length/2

    pos_vf = np.array([agt_nominal_xy(veh_f, k) for k in range(len(ts))])
    rx, ry = agt_radius_xy(veh_r)
    if use_risk:
        rad_vf = np.array([rx * agt_risk_radius(veh_f, k, sigma_ax, sigma_ay, risk_level) for k in range(len(ts))])
    else:
        rad_vf = rx * np.ones_like(ts)
    rad_vf = rad_vf + veh_e.length/2

    T_acc = (v_max - veh_e.v) / a_max        
    mask_acc = ts >= T_acc
    x_ve_acc = veh_e.x + veh_e.v * ts + 0.5 * a_max * ts**2
    x_ve_acc[mask_acc] = veh_e.x + veh_e.v * T_acc + 0.5 * a_max * T_acc**2 + v_max * (ts[mask_acc] - T_acc)

    T_dec = (v_min - veh_e.v) / a_min
    mask_dec = ts >= T_dec
    x_ve_dec = veh_e.x + veh_e.v * ts + 0.5 * a_min * ts**2
    x_ve_dec[mask_dec] = veh_e.x + veh_e.v * T_dec + 0.5 * a_min * T_dec**2 + v_min * (ts[mask_dec] - T_dec)

    v1 = (x_ve_dec <= pos_vp[:,0] - rad_vp)
    v2 = (x_ve_dec <= pos_vf[:,0] - rad_vf)
    v3 = (x_ve_acc >= pos_vr[:,0] + rad_vr)
    v4 = (pos_vf[:,0] - pos_vr[:,0] >= rad_vf + rad_vr)
    v5 = (pos_vp[:,0] - pos_vr[:,0] >= rad_vp + rad_vr)

    # valid_lon = (x_ve_dec <= pos_vp[:,0] - rad_vp) * (x_ve_dec <= pos_vf[:,0] - rad_vf) * (x_ve_acc >= pos_vr[:,0] + rad_vr)
    valid_lon = v1 * v2 * v3 * v4 * v5

    return valid_lon

def get_possible_tm_lat(veh_e:Agent, ym:float):
    Ts = plan_pace
    ts = np.arange(0, plan_horizon+1) * Ts

    a1 = veh_e.v
    b1 = wheelbase_r * veh_e.v / wheelbase
    b2 = veh_e.v / wheelbase
    c0 = a1 * veh_e.h
    c_max = a1 * h_max
    s_max = (a1 * b2 + b1) * d_max
    T_str = (h_max - veh_e.h) / d_max / b2
    mask_str = ts >= T_str
    y_ve_str = veh_e.y + c0 * ts + 0.5 * s_max * ts**2
    y_ve_str[mask_str] = veh_e.y + c0 * T_str + 0.5 * s_max * T_str**2 + c_max * (ts[mask_str] - T_str)

    valid_lat = (y_ve_str >= ym)

    return valid_lat

def calc_cost_tm(veh_e:Agent, veh_p:Agent, veh_r:Agent, veh_f:Agent, tm:float, tm_min:float, tm_max:float, ym:float):
    Ts = plan_pace
    xp = veh_p.x + veh_p.v * tm
    xr = veh_r.x + veh_r.v * tm
    xf = veh_f.x + veh_f.v * tm
    xm1 = (min(xp, xf) + xr) / 2
    
    tp = plan_horizon * Ts
    xr = veh_r.x + veh_r.v * tp
    xf = veh_f.x + veh_f.v * tp
    xm2 = (xr + xf) / 2

    vm1 = (xm1 - veh_e.x) / (tm+0.01)
    vm2 = (xm2 - xm1) / (tp - tm + 0.01)
    hm = atan2(ym - veh_e.y, veh_e.v * tm)


    lon_1 = abs(vm1 - veh_e.v) / (tm+0.01)
    lon_2 = abs(vm2 - vm1)  / (tp - tm + 0.01)
    lon_3 = abs(veh_f.v - vm2)   / (tp - tm + 0.01)
    cost_lon = lon_1 + lon_2 + lon_3
    
    lat_1 = abs(hm - veh_e.h) / (tm+0.01)
    lat_2 = abs(0.0 - hm) / (tp - tm + 0.01)
    cost_lat = np.rad2deg(lat_1) + np.rad2deg(lat_2)

    tmc_1 = exp(tm / tp)
    tmc_2 = 1 / (tm - tm_min + 0.01) + 1 / (tm_max - tm + 0.01)
    cost_tmc = tmc_1 + tmc_2

    return cost_lon + cost_lat + cost_tmc

def search_optimal_tm(veh_e:Agent, veh_p:Agent, veh_r:Agent, veh_f:Agent, err:float, ym:float, use_risk:bool = True):
    Ts = plan_pace
    valid_lat = get_possible_tm_lat(veh_e, ym)
    valid_lon = get_possible_tm_lon(veh_e, veh_p, veh_r, veh_f, use_risk)
    if np.count_nonzero(valid_lat * valid_lon) == 0:
        return None, np.Inf, None
    
    idxs = np.argwhere(valid_lon * valid_lat)
    ts = np.arange(0,plan_horizon+1) * Ts
    tm_min, tm_max = ts[np.min(idxs)], ts[np.max(idxs)]

    tl, tu = tm_min, tm_max
    Nm = 3
    res_dict = {
        'veh_e': veh_e,
        'veh_p': veh_p,
        'veh_f': veh_f,
        'veh_r': veh_r,
        'tm_min': tm_min,
        'tm_max': tm_max,
        'Nm': Nm,
        'err': err,
        'reslist': [],
    }
    while tu - tl > err:
        dt = (tu - tl) / Nm
        opt_cost, opt_tm = None, None
        for i in range(Nm+1):
            tm = tl + i * dt
            tm_cost = calc_cost_tm(veh_e, veh_p, veh_r, veh_f, tm, tm_min, tm_max, ym)
            if opt_cost is None or tm_cost < opt_cost:
                opt_cost, opt_tm = tm_cost, tm
        
        res_dict['reslist'].append([opt_tm, opt_cost, tl, tu])
        
        tl = max(tm_min, opt_tm - dt)
        tu = min(tm_max, opt_tm + dt)
    
    opt_tm = (tu + tl)/2
    opt_cost = calc_cost_tm(veh_e, veh_p, veh_r, veh_f, opt_tm, tm_min, tm_max, ym)
    return opt_tm, opt_cost, res_dict

def calc_xv_from_a(a, x0, v0, Ts):
    x = x0 * np.ones(len(a)+1)
    v = v0 * np.ones(len(a)+1)
    for k in range(len(a)):
        x[k+1] = x[k] + Ts * v[k]
        v[k+1] = v[k] + Ts * a[k]
    return x, v

def calc_yh_from_w(w, y0, h0, v0, Ts):
    h = h0 * np.ones(len(w)+1) # t = 0,1,2,3,...,N
    y = y0 * np.ones(len(w)+1) # t = 0,1,2,3,...,N
    for k in range(len(w)):
        y[k+1] = y[k] + Ts * v0 * h[k] + Ts * wheelbase_r * v0 / wheelbase * w[k]
        h[k+1] = h[k] + Ts * v0 / wheelbase * w[k]

    return y, h

def optimize_longitudinal(tm:float, veh_e:Agent, veh_p:Agent, veh_r:Agent, veh_f:Agent, use_risk:bool = True) -> tuple[bool, np.ndarray]:
    """
    Optimize the longitudinal trajectory
    """
    Ts = plan_pace
    # define model
    A = np.array([[1.0, Ts], [0.0, 1.0]], float)
    B = np.array([[0.0], [Ts]], float)
    Q = np.eye(2) * 0.0
    R = np.eye(1) 

    # develop constraints
    ts = np.arange(1, plan_horizon+1) * Ts
    risk_level = get_risk_level()
    sigma_ax, sigma_ay = get_sigma()

    pos_vp = np.array([agt_nominal_xy(veh_p, k) for k in range(1, plan_horizon+1)])
    rx, ry = agt_radius_xy(veh_p)
    if use_risk:
        rad_vp = np.array([rx * agt_risk_radius(veh_p, k, sigma_ax, sigma_ay, risk_level) for k in range(1, plan_horizon+1)])
    else:
        rad_vp = rx * np.ones(plan_horizon)
    rad_vp = rad_vp + veh_e.length/2

    pos_vr = np.array([agt_nominal_xy(veh_r, k) for k in range(1, plan_horizon+1)])
    rx, ry = agt_radius_xy(veh_r)
    if use_risk:
        rad_vr = np.array([rx * agt_risk_radius(veh_r, k, sigma_ax, sigma_ay, risk_level) for k in range(1, plan_horizon+1)])
    else:
        rad_vr = rx * np.ones(plan_horizon)
    rad_vr = rad_vr + veh_e.length/2

    pos_vf = np.array([agt_nominal_xy(veh_f, k) for k in range(1, plan_horizon+1)])
    rx, ry = agt_radius_xy(veh_f)
    if use_risk:
        rad_vf = np.array([rx * agt_risk_radius(veh_f, k, sigma_ax, sigma_ay, risk_level) for k in range(1, plan_horizon+1)])
    else:
        rad_vf = rx * np.ones(plan_horizon)
    rad_vf = rad_vf + veh_e.length/2

    x_lb = veh_e.x * np.ones_like(ts)
    x_lb[ts>=tm] = pos_vr[ts>=tm, 0] + rad_vr[ts>=tm]
    x_ub = pos_vp[:,0] - rad_vp
    x_ub[ts>=tm] = pos_vf[ts>=tm, 0] - rad_vf[ts>=tm]

    v_lb = v_min * np.ones_like(ts)
    v_ub = v_max * np.ones_like(ts)
    
    # optimize
    s0 = np.array([veh_e.x, veh_e.v], float)
    s_lb = np.array([x_lb, v_lb]).transpose(1,0)
    s_ub = np.array([x_ub, v_ub]).transpose(1,0)
    u_lb = np.array([a_min])
    u_ub = np.array([a_max])
    
    a, flag = linear_mpc(plan_horizon, plan_horizon, A, B, Q, R, xt=s0, x_min=s_lb, x_max=s_ub, u_min=u_lb, u_max=u_ub)
    if not flag:
        return None, False
    a = a.reshape(-1)
    return a, flag

def get_rpi_approx(N: int, v_min:float, v_max:float, y_min:float, y_max:float, Qm:np.ndarray, Rm:np.ndarray):

    Ts = plan_pace
    set_AB = []
    for v in [v_min, v_max]:
        A = np.matrix([[1.0, Ts * v], [0.0, 1.0]], float)
        B = np.matrix([[Ts * wheelbase_r * v / wheelbase], [Ts * v / wheelbase]], float)
        set_AB.append((A, B))
    
    vm = (v_min + v_max) / 2
    Am = np.matrix([[1.0, Ts * vm], [0.0, 1.0]], float)
    Bm = np.matrix([[Ts * wheelbase_r * vm / wheelbase], [Ts * vm / wheelbase]], float)

    Km, Wm = discrete_algebraic_Riccati_equation(Am, Bm, Qm, Rm)
    Phi = Am + Bm * Km

    set_w = []
    for y in [y_min, y_max]:
        for h in [h_min, h_max]:
            for u in [d_min, d_max]:
                for (A, B) in set_AB:
                    w = (A - Am) * np.matrix([[y], [h]], float) + (B - Bm) * np.matrix([[u]], float)
                    w = np.array(w).reshape(-1)
                    set_w.append(w)
    set_w = np.array(set_w)

    cvh = ConvexHull(set_w)
    set_w = set_w[cvh.vertices]
    set_rpi = np.array([set_w.min(axis=0), set_w.max(axis=0)])
    for k in range(1,N+1):
        wk = Phi**k * np.matrix(set_w).T
        wk = np.array(wk.T)
        set_rpi = set_rpi + np.array([wk.min(axis=0), wk.max(axis=0)])
    
    set_rpi_ap = np.array([
        [set_rpi[0,0], set_rpi[0,1]],
        [set_rpi[0,0], set_rpi[1,1]],
        [set_rpi[1,0], set_rpi[1,1]],
        [set_rpi[1,0], set_rpi[0,1]]
    ])
    return Km, set_rpi_ap

def get_constraints_lateral(tm:float, v_min:float, v_max:float, y_min:float, y_max:float, ym:float, yt:float, delta_m1:float, delta_m2:float):
    Ts = plan_pace
    vm = (v_min + v_max) / 2
    Am = np.matrix([[1.0, Ts * vm], [0.0, 1.0]], float)
    Bm = np.matrix([[Ts * wheelbase_r * vm / wheelbase], [Ts * vm / wheelbase]], float)
    Qm, Rm = np.eye(2)*1e-4, np.eye(1)

    Km, set_rpi = get_rpi_approx(plan_horizon, v_min, v_max, y_min, y_max, Qm, Rm)

    rpi_min, rpi_max = set_rpi.min(0), set_rpi.max(0)

    ts = np.arange(1, plan_horizon+1) * Ts

    if yt > ym:
        y_lb = y_min * np.ones_like(ts) - rpi_min[0]
        y_lb[ts>=tm+delta_m2] = ym - rpi_min[0]
        y_ub = y_max * np.ones_like(ts) - rpi_max[0]
        y_ub[ts<=tm-delta_m1] = ym - rpi_max[0]
    else:
        y_lb = y_min * np.ones_like(ts) - rpi_min[0]
        y_lb[ts<=tm-delta_m1] = ym - rpi_min[0]

        y_ub = y_max * np.ones_like(ts) - rpi_max[0]
        y_ub[ts>=tm+delta_m2] = ym - rpi_max[0]

    h_lb = h_min * np.ones_like(ts) - rpi_min[1]
    h_ub = h_max * np.ones_like(ts) - rpi_max[1]

    s_lb = np.array([y_lb, h_lb]).transpose(1,0)
    s_ub = np.array([y_ub, h_ub]).transpose(1,0)

    du = np.array(Km * np.matrix(set_rpi).T).T
    du_min, du_max = du.min(0), du.max(0)
    u_lb = d_min * np.ones(1) - du_min
    u_ub = d_max * np.ones(1) - du_max

    return (s_lb, s_ub), (u_lb, u_ub)

def optimize_lateral(veh_e:Agent, yt:float, tm:float, v_min:float, v_max:float, y_min:float, y_max:float, ym:float, delta_m1:float, delta_m2:float):
    """
    Optimize the latitude trajectory
    """
    Ts = plan_pace
    (s_lb, s_ub), (u_lb, u_ub) = get_constraints_lateral(tm, v_min, v_max, y_min, y_max, ym, yt, delta_m1, delta_m2)

    vm = (v_min + v_max) / 2
    A = np.matrix([[1.0, Ts * vm], [0.0, 1.0]], float)
    B = np.matrix([[Ts * wheelbase_r * vm / wheelbase], [Ts * vm / wheelbase]], float)
    Q = np.eye(2) * 1e-4
    R = np.eye(1)

    s0 = np.array([veh_e.y, veh_e.h], float)
    sr = np.array([yt, 0.0], float)
    w, flag = linear_mpc_track(plan_horizon, plan_horizon, A, B, Q, R, xt=s0, xr=sr, x_min=s_lb, x_max=s_ub, u_min=u_lb, u_max=u_ub)
    if not flag:
        return None, False
    w = w.reshape(-1)
    return w, flag

def refine_trajectory(traj:AgentTrajectory) -> AgentTrajectory:
    ts, (x, y, h, v), (acc, dfw) = traj.ts, traj.states.transpose(1,0), traj.inputs.transpose(1,0)
    t0, te = ts.min(), ts.max()
    Np = round((te-t0) / simu_pace)
    if len(ts) == Np:
        return deepcopy(traj)
    
    rt = np.linspace(t0, te, Np+1)
    rx = np.interp(rt, ts, x)
    ry = np.interp(rt, ts, y)
    rh = np.interp(rt, ts, h)
    rv = np.interp(rt, ts, v)
    racc = np.interp(rt[:-1], ts[:-1], acc)
    rdfw = np.interp(rt[:-1], ts[:-1], dfw)

    return AgentTrajectory(
        rt, 
        np.vstack([rx, ry, rh, rv]).transpose(1,0),
        np.vstack([racc, rdfw]).transpose(1,0),
        traj.tm,
        traj.tm_margin
    )

class LaneKeepPlanner:
    def __init__(self) -> None:
        self.veh_f = None
        self.veh_r = None

    def run(self, EV: Agent, lane_id: str) -> AgentTrajectory:
        self.initialize(EV, lane_id)

        acc = self.get_acc()
        xe, ve = calc_xv_from_a(acc, self.veh_e.x, self.veh_e.v, plan_pace)

        vm = np.mean(ve)
        dfw = self.get_dfw(vm)
        ye, he = calc_yh_from_w(dfw, self.veh_e.y, self.veh_e.h, vm, plan_pace)

        t0 = datci.getTimestep() * datci.getSimupace()
        traj = AgentTrajectory(
            ts=np.arange(0, plan_horizon+1) * plan_pace + t0,
            states=np.vstack([xe, ye, he, ve]).transpose(1,0),
            inputs=np.vstack([acc, dfw]).transpose(1,0)
        )
        out_traj = refine_trajectory(traj)
        return out_traj
    
    def run_lateral(self, EV: Agent, lane_id: str) -> AgentTrajectory:
        self.initialize(EV, lane_id)
        self.yt = EV.y

        acc = self.get_acc()
        xe, ve = calc_xv_from_a(acc, self.veh_e.x, self.veh_e.v, plan_pace)

        vm = np.mean(ve)
        dfw = self.get_dfw(vm)
        ye, he = calc_yh_from_w(dfw, self.veh_e.y, self.veh_e.h, ve, plan_pace)

        t0 = datci.getTimestep() * datci.getSimupace()
        traj = AgentTrajectory(
            ts=np.arange(0, plan_horizon+1) * plan_pace + t0,
            states=np.vstack([xe, ye, he, ve]).transpose(1,0),
            inputs=np.vstack([acc, dfw]).transpose(1,0)
        )
        out_traj = refine_trajectory(traj)
        return out_traj

    def initialize(self, EV:Agent, lane_id:str):
        self.veh_e = EV
        self.lane_keep = lane_id

        ########## set PV & TVs ##########
        all_vehicles = datci.getFrameVehicleIds()
        self.veh_f:Agent = None
        self.veh_r:Agent = None
        for Vid in all_vehicles:
            if Vid == self.veh_e.id:
                continue

            Vlane = datci.vehicle.getLaneID(Vid)
            if Vlane == self.lane_keep:
                Vx, Vy = datci.vehicle.getPosition(Vid)
                if Vx > self.veh_e.x:
                    if self.veh_f is None or self.veh_f.x > Vx:
                        Vh = datci.angle2yaw(datci.vehicle.getAngle(Vid))
                        Vv = datci.vehicle.getSpeed(Vid)
                        Vl, Vw = datci.vehicle.getVehicleShape(Vid)
                        self.veh_f = Agent(Vid, Vx, Vy, Vh, Vv, length=Vl, width=Vw, label='veh_f')
                elif Vx < self.veh_e.x:
                    if self.veh_r is None or self.veh_r.x < Vx:
                        Vh = datci.angle2yaw(datci.vehicle.getAngle(Vid))
                        Vv = datci.vehicle.getSpeed(Vid)
                        Vl, Vw = datci.vehicle.getVehicleShape(Vid)
                        self.veh_r = Agent(Vid, Vx, Vy, Vh, Vv, length=Vl, width=Vw, label='veh_r')

        ########## set ym and yt ##########
        lane_centery = datci.lane.getLane(self.lane_keep).center_line[0][1]
        lane_width = datci.lane.getLaneWidth(self.lane_keep)

        self.yt = lane_centery

        ########## set constraints on y, h, v ##########
        self.y_min = lane_centery - lane_width/2
        self.y_max = lane_centery + lane_width/2

        # print("veh_f = %s, veh_r=%s" % (self.veh_f.id, self.veh_r.id))

    def get_acc(self) -> ndarray:
        if self.veh_f is None and self.veh_r is None:
            return np.zeros(plan_horizon, float)
        
        Ts = plan_pace
        if self.veh_f is None:
            v_ref = max(self.veh_r.v, self.veh_e.v) * np.ones(plan_horizon)
            x_ref = np.cumsum(v_ref) * Ts + self.veh_r.x + self.veh_r.length/2 + self.veh_e.length/2 + self.veh_r.v * 2.0
            s_lb = np.array([
                np.cumsum(v_ref) * Ts + self.veh_r.x + self.veh_r.length/2 + self.veh_e.length/2, 
                v_min * np.ones_like(v_ref)
            ]).transpose(1,0)
            s_ub = np.array([
                x_ref + 100.0,
                v_max * np.ones_like(v_ref)
            ]).transpose(1,0)
        elif self.veh_r is None:
            v_ref = self.veh_f.v * np.ones(plan_horizon)
            x_ref = np.cumsum(v_ref) * Ts + self.veh_f.x - self.veh_f.length/2 - self.veh_e.length/2 - self.veh_f.v * 2.0
            s_lb = np.array([
                x_ref - 100.0, 
                v_min * np.ones_like(v_ref)
            ]).transpose(1,0)
            s_ub = np.array([
                np.cumsum(v_ref) * Ts + self.veh_f.x - self.veh_f.length/2 - self.veh_e.length/2, 
                v_max * np.ones_like(v_ref)
            ]).transpose(1,0)
        else:
            v_ref = (self.veh_r.v + self.veh_f.v) / 2 * np.ones(plan_horizon)
            x_ref = np.cumsum(v_ref) * Ts + (self.veh_r.x + self.veh_f.x) / 2
            s_lb = np.array([
                self.veh_r.x + self.veh_r.v * np.arange(1,plan_horizon+1) * Ts,
                v_min * np.ones_like(v_ref)
            ]).transpose(1,0)
            s_ub = np.array([
                self.veh_f.x + self.veh_f.v * np.arange(1,plan_horizon+1) * Ts, 
                v_max * np.ones_like(v_ref)
            ]).transpose(1,0)

        A = np.array([[1.0, Ts], [0.0, 1.0]], float)
        B = np.array([[0.0], [Ts]], float)
        Q = np.eye(2)
        R = np.eye(1)

        s0 = np.array([self.veh_e.x, self.veh_e.v], float)
        sr = np.array([x_ref, v_ref], float).transpose(1,0)
        u_lb = np.array([a_min])
        u_ub = np.array([a_max])
        acc, flag = linear_mpc_track(plan_horizon, plan_horizon, A, B, Q, R, xt=s0, xr=sr, x_min=s_lb, x_max=s_ub, u_min=u_lb, u_max=u_ub)
        if not flag:
            return np.zeros(plan_horizon, float)

        acc = acc.reshape(-1)
        return acc

    def get_dfw(self, vm:float) -> ndarray:
        Ts = plan_pace

        # y0, h0, d0 = veh_E.y, veh_E.h, 0.0
        A = np.matrix([[1.0, Ts * vm], [0.0, 1.0]], float)
        B = np.matrix([[Ts * wheelbase_r * vm / wheelbase], [Ts * vm / wheelbase]], float)
        Q = np.eye(2) * 0.001
        R = np.eye(1)

        # st = np.array([self.veh_e.y, self.veh_e.h], float) - np.array([self.yt, 0.0], float)

        s0 = np.array([self.veh_e.y, self.veh_e.h], float)
        sr = np.array([self.yt, 0.0], float)
        s_lb = np.array([[self.y_min, h_min]] * plan_horizon)
        s_ub = np.array([[self.y_max, h_max]] * plan_horizon)
        u_lb = np.array([d_min])
        u_ub = np.array([d_max])
        dfw, flag = linear_mpc_track(plan_horizon, plan_horizon, A, B, Q, R, xt=s0, xr=sr, x_min=s_lb, x_max=s_ub, u_min=u_lb, u_max=u_ub)
        if not flag:
            return np.zeros(plan_horizon)
        # dfw = lqr(A, B, Q, R, st, plan_horizon).squeeze()

        dfw = dfw.reshape(-1)
        return dfw

class LaneChangePlanner_OTP:
    def __init__(self) -> None:
        self.veh_e = None
        self.veh_p = None
        self.veh_f = None
        self.veh_r = None
        self.plan_tm = None
        self.plan_tm_margin = (0, 0)

    def initialize(self, EV:Agent, lane_from:str, lane_to:str):
        self.veh_e = EV
        self.lane_from = lane_from
        self.lane_to = lane_to

        ########## set PV & TVs ##########
        all_vehicles = datci.getFrameVehicleIds()
        self.veh_p: Agent = None
        veh_now_following: Agent = None
        self.all_veh_t: list[Agent] = []
        for Vid in all_vehicles:
            if Vid == self.veh_e.id:
                continue

            Vlane = datci.vehicle.getLaneID(Vid)
            if Vlane == self.lane_from:
                Vx, Vy = datci.vehicle.getPosition(Vid)
                if Vx > self.veh_e.x:
                    if self.veh_p is None or self.veh_p.x > Vx:
                        self.veh_p = get_agent_from_daci(Vid, label='veh_p')
                else:
                    if veh_now_following is None or veh_now_following.x < Vx:
                        veh_now_following = get_agent_from_daci(Vid)

            elif Vlane == self.lane_to:
                Vx, Vy = datci.vehicle.getPosition(Vid)
                Dx = self.veh_e.v * 6.0
                if Vx > self.veh_e.x-Dx and Vx < self.veh_e.x + Dx:
                    self.all_veh_t.append(get_agent_from_daci(Vid))

        if self.veh_p is None:
            self.veh_p = Agent('-1', self.veh_e.x+100, self.veh_e.y, 0.0, self.veh_e.v, 0.0, 0.0, 4.0, 2.0, label='virtual')

        if len(self.all_veh_t) > 0:
            self.all_veh_t = sorted(self.all_veh_t, key=lambda V: V.x)
            TV = self.all_veh_t[0]
            self.all_veh_t.insert(0, Agent('-2',TV.x-100, TV.y, 0.0, TV.v, 0.0, 0.0, 4.0, 2.0, label='virtual'))
            TV = self.all_veh_t[-1]
            self.all_veh_t.append(Agent('-3',TV.x+100, TV.y, 0.0, TV.v, 0.0, 0.0, 4.0, 2.0, label='virtual'))

        
        ########## set ym and yt ##########
        self.lane_from_centery = datci.lane.getLane(self.lane_from).center_line[0][1]
        self.lane_from_width = datci.lane.getLaneWidth(self.lane_from)
        self.lane_to_centery = datci.lane.getLane(self.lane_to).center_line[0][1]
        self.lane_to_width = datci.lane.getLaneWidth(self.lane_to)

        if self.lane_to_centery > self.lane_from_centery:
            self.ym = self.lane_to_centery - self.lane_to_width/2
        else:
            self.ym = self.lane_to_centery + self.lane_to_width/2
        self.yt = self.lane_to_centery

        ########## set constraints on y, h, v ##########
        if self.yt > self.ym:
            self.y_min = self.lane_from_centery - self.lane_from_width/2
            self.y_max = self.lane_to_centery + self.lane_to_width/2
        else:
            self.y_min = self.lane_to_centery - self.lane_to_width/2
            self.y_max = self.lane_from_centery + self.lane_from_width/2
        
        self.lane_from_miny = self.lane_from_centery - self.lane_from_width/2
        self.lane_from_maxy = self.lane_from_centery + self.lane_from_width/2
        self.lane_to_miny = self.lane_to_centery - self.lane_to_width/2
        self.lane_to_maxy = self.lane_to_centery + self.lane_to_width/2
        
        self.h_min = h_min
        self.h_max = h_max
        self.v_min = v_min
        self.v_max = v_max

    def run(self, EV: Agent, lane_from: str, lane_to: str) -> Union[None, AgentTrajectory]:
        self.initialize(EV, lane_from, lane_to)

        tm_dict = self.get_optimal_tm_dict()
        if len(tm_dict) == 0: # no valid tm
            # print("No Valid Tm")
            return None
        
        tm_key = list(tm_dict.keys())[0]
        tm_val = tm_dict[tm_key]
        
        veh_r = self.all_veh_t[tm_key[0]]
        veh_f = self.all_veh_t[tm_key[1]]
        tm, tm_min, tm_max = tm_val['best'], tm_val['min'], tm_val['max']
        self.veh_r, self.veh_f = veh_r, veh_f
        # print(tm, tm_min, tm_max)


        # print("veh_f: %s, veh_r: %s, veh_p: %s" % (veh_f.id, veh_r.id, self.veh_p.id))
        traj = self.get_optimal_trajectory(tm, tm_min, tm_max, veh_r, veh_f)
        if traj is None:
            return None
        
        out_traj = refine_trajectory(traj)
        self.plan_tm = out_traj.tm

        return out_traj

    def run_replan(self, EV:Agent, lane_from: str, lane_to: str) -> Union[None, AgentTrajectory]:
        self.initialize(EV, lane_from, lane_to)

        tm_dict = self.get_optimal_tm_dict()
        if len(tm_dict) == 0: # no valid tm
            return None
        
        now_t = datci.getTimestep() * datci.getSimupace()
        now_tm = self.plan_tm - now_t

        veh_r : Agent = None
        veh_f : Agent = None
        tm_min, tm_max = None, None
        for tm_key, tm_val in tm_dict.items():
            if now_tm >= tm_val['min'] and now_tm <= tm_val['max']:
                veh_r = self.all_veh_t[tm_key[0]]
                veh_f = self.all_veh_t[tm_key[1]]
                tm_min, tm_max = now_tm - tm_val['min'], tm_val['max'] - now_tm
        
        if tm_min is None and tm_max is None:
            return None

        if veh_r is None:
            veh_r = Agent('-2', self.veh_e.x-100, self.lane_to, 0.0, self.veh_e.v, 0.0, 0.0, 4.0, 2.0, label='virtual')
        if veh_f is None:
            veh_f = Agent('-3', self.veh_e.x+100, self.lane_to, 0.0, self.veh_e.v, 0.0, 0.0, 4.0, 2.0, label='virtual')

        traj = self.get_optimal_trajectory(now_tm, tm_min, tm_max, veh_r, veh_f)
        if traj is None:
            return None
        
        out_traj = refine_trajectory(traj)
        self.veh_r, self.veh_f = veh_r, veh_f
        self.plan_tm = out_traj.tm
        return out_traj

    def get_optimal_tm_dict(self) -> dict:
        all_tm = {}
        for i in range(len(self.all_veh_t)-1):
            tm_best, tm_cost, tm_min, tm_max = self.get_local_optimal_tm(self.all_veh_t[i], self.all_veh_t[i+1])
            if tm_best is None:
                continue
            all_tm[i, i+1] = {'best': tm_best, 'cost': tm_cost, 'min': tm_min, 'max': tm_max}
        
        all_tm = dict(sorted(all_tm.items(), key=lambda d: d[1]['cost']))
        return all_tm
    
    def get_local_optimal_tm(self, veh_r:Agent, veh_f:Agent) -> tuple[float, float, float, float]:
        ts = np.arange(0,plan_horizon+1) * plan_pace

        opt_tm, opt_cost, opt_res = search_optimal_tm(self.veh_e, self.veh_p, veh_r, veh_f, err=0.05, ym=self.ym)
        if opt_tm is None:
            return None, None, None, None
        
        tm_min = opt_res['tm_min']
        tm_max = opt_res['tm_max']

        return opt_tm, opt_cost, tm_min, tm_max
    
    def get_optimal_trajectory(self, tm: float, tm_min:float, tm_max:float, veh_r:Agent, veh_f:Agent) -> AgentTrajectory:

        acc, flon = optimize_longitudinal(tm, self.veh_e, self.veh_p, veh_r, veh_f)
        if not flon:
            # print("no lon traj")
            return None
        
        xe, ve = calc_xv_from_a(acc, self.veh_e.x, self.veh_e.v, plan_pace)

        delta_m1 = (tm - tm_min) * 0.4
        delta_m2 = (tm_max - tm) * 0.4
        dfw, flat = optimize_lateral(self.veh_e, self.yt, tm, ve.min(), ve.max(), self.y_min, self.y_max, self.ym, delta_m1, delta_m2)
        if not flat:
            # print("no lat traj", ve.min(), ve.max())
            return None
                
        ye, he = calc_yh_from_w(dfw, self.veh_e.y, self.veh_e.h, np.mean(ve), plan_pace)

        t0 = datci.getTimestep() * datci.getSimupace()
        traj = AgentTrajectory(
            ts=np.arange(0, plan_horizon+1) * plan_pace + t0,
            states=np.vstack([xe, ye, he, ve]).transpose(1,0),
            inputs=np.vstack([acc, dfw]).transpose(1,0),
            tm=tm+t0,
            tm_margin=(tm_min+t0, tm_max+t0)
        )
        return traj

''' OTP w/o risk '''
class LaneChangePlanner_OTP_WOR:
    def __init__(self) -> None:
        self.veh_e : Agent = None
        self.veh_p : Agent = None
        self.veh_f : Agent = None
        self.veh_r : Agent = None
        self.plan_tm = None
        self.plan_tm_margin = (0, 0)

    def initialize(self, EV:Agent, lane_from:str, lane_to:str):
        self.veh_e = EV
        self.lane_from = lane_from
        self.lane_to = lane_to

        ########## set PV & TVs ##########
        all_vehicles = datci.getFrameVehicleIds()
        self.veh_p: Agent = None
        veh_now_following: Agent = None
        self.all_veh_t: list[Agent] = []
        for Vid in all_vehicles:
            if Vid == self.veh_e.id:
                continue

            Vlane = datci.vehicle.getLaneID(Vid)
            if Vlane == self.lane_from:
                Vx, Vy = datci.vehicle.getPosition(Vid)
                if Vx > self.veh_e.x:
                    if self.veh_p is None or self.veh_p.x > Vx:
                        self.veh_p = get_agent_from_daci(Vid, label='veh_p')
                else:
                    if veh_now_following is None or veh_now_following.x < Vx:
                        veh_now_following = get_agent_from_daci(Vid)

            elif Vlane == self.lane_to:
                Vx, Vy = datci.vehicle.getPosition(Vid)
                Dx = self.veh_e.v * 6.0
                if Vx > self.veh_e.x-Dx and Vx < self.veh_e.x + Dx:
                    self.all_veh_t.append(get_agent_from_daci(Vid))

        if self.veh_p is None:
            self.veh_p = Agent('-1', self.veh_e.x+100, self.veh_e.y, 0.0, self.veh_e.v, 0.0, 0.0, 4.0, 2.0, label='virtual')

        if len(self.all_veh_t) > 0:
            self.all_veh_t = sorted(self.all_veh_t, key=lambda V: V.x)
            TV = self.all_veh_t[0]
            self.all_veh_t.insert(0, Agent('-2',TV.x-100, TV.y, 0.0, TV.v, 0.0, 0.0, 4.0, 2.0, label='virtual'))
            TV = self.all_veh_t[-1]
            self.all_veh_t.append(Agent('-3',TV.x+100, TV.y, 0.0, TV.v, 0.0, 0.0, 4.0, 2.0, label='virtual'))

        ########## set ym and yt ##########
        self.lane_from_centery = datci.lane.getLane(self.lane_from).center_line[0][1]
        self.lane_from_width = datci.lane.getLaneWidth(self.lane_from)
        self.lane_to_centery = datci.lane.getLane(self.lane_to).center_line[0][1]
        self.lane_to_width = datci.lane.getLaneWidth(self.lane_to)

        if self.lane_to_centery > self.lane_from_centery:
            self.ym = self.lane_to_centery - self.lane_to_width/2
        else:
            self.ym = self.lane_to_centery + self.lane_to_width/2
        self.yt = self.lane_to_centery

        ########## set constraints on y, h, v ##########
        if self.yt > self.ym:
            self.y_min = self.lane_from_centery - self.lane_from_width/2
            self.y_max = self.lane_to_centery + self.lane_to_width/2
        else:
            self.y_min = self.lane_to_centery - self.lane_to_width/2
            self.y_max = self.lane_from_centery + self.lane_from_width/2
        
        self.lane_from_miny = self.lane_from_centery - self.lane_from_width/2
        self.lane_from_maxy = self.lane_from_centery + self.lane_from_width/2
        self.lane_to_miny = self.lane_to_centery - self.lane_to_width/2
        self.lane_to_maxy = self.lane_to_centery + self.lane_to_width/2
        
        self.h_min = h_min
        self.h_max = h_max
        self.v_min = v_min
        self.v_max = v_max

    def run(self, EV: Agent, lane_from: str, lane_to: str) -> Union[None, AgentTrajectory]:
        self.initialize(EV, lane_from, lane_to)

        tm_dict = self.get_optimal_tm_dict()
        if len(tm_dict) == 0: # no valid tm
            # print("No Valid Tm")
            return None
        
        tm_key = list(tm_dict.keys())[0]
        tm_val = tm_dict[tm_key]
        
        veh_r = self.all_veh_t[tm_key[0]]
        veh_f = self.all_veh_t[tm_key[1]]
        tm, tm_min, tm_max = tm_val['best'], tm_val['min'], tm_val['max']
        self.veh_r, self.veh_f = veh_r, veh_f

        traj = self.get_optimal_trajectory(tm, tm_min, tm_max, veh_r, veh_f)
        if traj is None:
            return None
        
        out_traj = refine_trajectory(traj)
        self.plan_tm = out_traj.tm

        return out_traj

    def run_replan(self, EV:Agent, lane_from: str, lane_to: str) -> Union[None, AgentTrajectory]:
        self.initialize(EV, lane_from, lane_to)

        tm_dict = self.get_optimal_tm_dict()
        if len(tm_dict) == 0: # no valid tm
            return None
        
        now_t = datci.getTimestep() * datci.getSimupace()
        now_tm = self.plan_tm - now_t

        veh_r : Agent = None
        veh_f : Agent = None
        tm_min, tm_max = None, None
        for tm_key, tm_val in tm_dict.items():
            if now_tm >= tm_val['min'] and now_tm <= tm_val['max']:
                veh_r = self.all_veh_t[tm_key[0]]
                veh_f = self.all_veh_t[tm_key[1]]
                tm_min, tm_max = now_tm - tm_val['min'], tm_val['max'] - now_tm
        
        if tm_min is None and tm_max is None:
            return None

        if veh_r is None:
            veh_r = Agent('-2', self.veh_e.x-100, self.lane_to, 0.0, self.veh_e.v, 0.0, 0.0, 4.0, 2.0, label='virtual')
        if veh_f is None:
            veh_f = Agent('-3', self.veh_e.x+100, self.lane_to, 0.0, self.veh_e.v, 0.0, 0.0, 4.0, 2.0, label='virtual')
        
        self.veh_r, self.veh_f = veh_r, veh_f

        traj = self.get_optimal_trajectory(now_tm, tm_min, tm_max, veh_r, veh_f)
        if traj is None:
            return None
        
        out_traj = refine_trajectory(traj)
        self.plan_tm = out_traj.tm
        return out_traj

    def get_optimal_tm_dict(self) -> dict:
        all_tm = {}
        for i in range(len(self.all_veh_t)-1):
            tm_best, tm_cost, tm_min, tm_max = self.get_local_optimal_tm(self.all_veh_t[i], self.all_veh_t[i+1])
            if tm_best is None:
                continue
            all_tm[i, i+1] = {'best': tm_best, 'cost': tm_cost, 'min': tm_min, 'max': tm_max}
        
        all_tm = dict(sorted(all_tm.items(), key=lambda d: d[1]['cost']))
        return all_tm
    
    def get_local_optimal_tm(self, veh_r:Agent, veh_f:Agent) -> tuple[float, float, float, float]:
        ts = np.arange(0,plan_horizon+1) * plan_pace

        opt_tm, opt_cost, opt_res = search_optimal_tm(self.veh_e, self.veh_p, veh_r, veh_f, err=0.05, ym=self.ym, use_risk=False)
        if opt_tm is None:
            return None, None, None, None
        
        tm_min = opt_res['tm_min']
        tm_max = opt_res['tm_max']

        return opt_tm, opt_cost, tm_min, tm_max

    def get_optimal_trajectory(self, tm: float, tm_min:float, tm_max:float, veh_r:Agent, veh_f:Agent) -> AgentTrajectory:

        acc, flon = optimize_longitudinal(tm, self.veh_e, self.veh_p, veh_r, veh_f, use_risk=False)
        if not flon:
            return None
        
        xe, ve = calc_xv_from_a(acc, self.veh_e.x, self.veh_e.v, plan_pace)

        delta_m1 = (tm - tm_min) * 0.4
        delta_m2 = (tm_max - tm) * 0.4
        dfw, flat = optimize_lateral(self.veh_e, self.yt, tm, ve.min(), ve.max(), self.y_min, self.y_max, self.ym, delta_m1, delta_m2)
        if not flat:
            return None
                
        ye, he = calc_yh_from_w(dfw, self.veh_e.y, self.veh_e.h, np.mean(ve), plan_pace)

        t0 = datci.getTimestep() * datci.getSimupace()
        traj = AgentTrajectory(
            ts=np.arange(0, plan_horizon+1) * plan_pace + t0,
            states=np.vstack([xe, ye, he, ve]).transpose(1,0),
            inputs=np.vstack([acc, dfw]).transpose(1,0),
            tm=tm+t0,
            tm_margin=(tm_min+t0, tm_max+t0)
        )
        return traj

''' OTP using nonlinear programing '''
class LaneChangePlanner_OTP_NLP:
    def __init__(self) -> None:
        self.veh_e : Agent = None
        self.veh_p : Agent = None
        self.veh_f : Agent = None
        self.veh_r : Agent = None
        self.plan_tm = None
        self.plan_tm_margin = (0, 0)

    def initialize(self, EV:Agent, lane_from:str, lane_to:str):
        self.veh_e = EV
        self.lane_from = lane_from
        self.lane_to = lane_to

        ########## set PV & TVs ##########
        all_vehicles = datci.getFrameVehicleIds()
        self.veh_p: Agent = None
        veh_now_following: Agent = None
        self.all_veh_t: list[Agent] = []
        for Vid in all_vehicles:
            if Vid == self.veh_e.id:
                continue

            Vlane = datci.vehicle.getLaneID(Vid)
            if Vlane == self.lane_from:
                Vx, Vy = datci.vehicle.getPosition(Vid)
                if Vx > self.veh_e.x:
                    if self.veh_p is None or self.veh_p.x > Vx:
                        self.veh_p = get_agent_from_daci(Vid, label='veh_p')
                else:
                    if veh_now_following is None or veh_now_following.x < Vx:
                        veh_now_following = get_agent_from_daci(Vid)

            elif Vlane == self.lane_to:
                Vx, Vy = datci.vehicle.getPosition(Vid)
                Dx = self.veh_e.v * 6.0
                if Vx > self.veh_e.x-Dx and Vx < self.veh_e.x + Dx:
                    self.all_veh_t.append(get_agent_from_daci(Vid))

        if self.veh_p is None:
            self.veh_p = Agent('-1', self.veh_e.x+100, self.veh_e.y, 0.0, self.veh_e.v, 0.0, 0.0, 4.0, 2.0, label='virtual')

        if len(self.all_veh_t) > 0:
            self.all_veh_t = sorted(self.all_veh_t, key=lambda V: V.x)
            TV = self.all_veh_t[0]
            self.all_veh_t.insert(0, Agent('-2',TV.x-100, TV.y, 0.0, TV.v, 0.0, 0.0, 4.0, 2.0, label='virtual'))
            TV = self.all_veh_t[-1]
            self.all_veh_t.append(Agent('-3',TV.x+100, TV.y, 0.0, TV.v, 0.0, 0.0, 4.0, 2.0, label='virtual'))

        ########## set ym and yt ##########
        self.lane_from_centery = datci.lane.getLane(self.lane_from).center_line[0][1]
        self.lane_from_width = datci.lane.getLaneWidth(self.lane_from)
        self.lane_to_centery = datci.lane.getLane(self.lane_to).center_line[0][1]
        self.lane_to_width = datci.lane.getLaneWidth(self.lane_to)

        if self.lane_to_centery > self.lane_from_centery:
            self.ym = self.lane_to_centery - self.lane_to_width/2
        else:
            self.ym = self.lane_to_centery + self.lane_to_width/2
        self.yt = self.lane_to_centery

        ########## set constraints on y, h, v ##########
        if self.yt > self.ym:
            self.y_min = self.lane_from_centery - self.lane_from_width/2
            self.y_max = self.lane_to_centery + self.lane_to_width/2
        else:
            self.y_min = self.lane_to_centery - self.lane_to_width/2
            self.y_max = self.lane_from_centery + self.lane_from_width/2
        
        self.lane_from_miny = self.lane_from_centery - self.lane_from_width/2
        self.lane_from_maxy = self.lane_from_centery + self.lane_from_width/2
        self.lane_to_miny = self.lane_to_centery - self.lane_to_width/2
        self.lane_to_maxy = self.lane_to_centery + self.lane_to_width/2
        
        self.h_min = h_min
        self.h_max = h_max
        self.v_min = v_min
        self.v_max = v_max

    def run(self, EV: Agent, lane_from: str, lane_to: str) -> Union[None, AgentTrajectory]:
        self.initialize(EV, lane_from, lane_to)

        tm_dict = self.get_optimal_tm_dict()
        if len(tm_dict) == 0: # no valid tm
            # print("No Valid Tm")
            return None
        
        tm_key = list(tm_dict.keys())[0]
        tm_val = tm_dict[tm_key]
        
        veh_r = self.all_veh_t[tm_key[0]]
        veh_f = self.all_veh_t[tm_key[1]]
        tm, tm_min, tm_max = tm_val['best'], tm_val['min'], tm_val['max']
        self.veh_r, self.veh_f = veh_r, veh_f

        traj = self.get_optimal_trajectory(tm, tm_min, tm_max, veh_r, veh_f)
        if traj is None:
            return None
        
        out_traj = refine_trajectory(traj)
        self.plan_tm = out_traj.tm

        return out_traj

    def run_replan(self, EV:Agent, lane_from: str, lane_to: str) -> Union[None, AgentTrajectory]:
        self.initialize(EV, lane_from, lane_to)

        tm_dict = self.get_optimal_tm_dict()
        if len(tm_dict) == 0: # no valid tm
            return None
        
        now_t = datci.getTimestep() * datci.getSimupace()
        now_tm = self.plan_tm - now_t

        veh_r : Agent = None
        veh_f : Agent = None
        tm_min, tm_max = None, None
        for tm_key, tm_val in tm_dict.items():
            if now_tm >= tm_val['min'] and now_tm <= tm_val['max']:
                veh_r = self.all_veh_t[tm_key[0]]
                veh_f = self.all_veh_t[tm_key[1]]
                tm_min, tm_max = now_tm - tm_val['min'], tm_val['max'] - now_tm
        
        if tm_min is None and tm_max is None:
            return None

        if veh_r is None:
            veh_r = Agent('-2', self.veh_e.x-100, self.lane_to, 0.0, self.veh_e.v, 0.0, 0.0, 4.0, 2.0, label='virtual')
        if veh_f is None:
            veh_f = Agent('-3', self.veh_e.x+100, self.lane_to, 0.0, self.veh_e.v, 0.0, 0.0, 4.0, 2.0, label='virtual')
        
        self.veh_r, self.veh_f = veh_r, veh_f

        traj = self.get_optimal_trajectory(now_tm, tm_min, tm_max, veh_r, veh_f)
        if traj is None:
            return None
        
        out_traj = refine_trajectory(traj)
        self.plan_tm = out_traj.tm
        return out_traj

    def get_optimal_tm_dict(self) -> dict:
        all_tm = {}
        for i in range(len(self.all_veh_t)-1):
            tm_best, tm_cost, tm_min, tm_max = self.get_local_optimal_tm(self.all_veh_t[i], self.all_veh_t[i+1])
            if tm_best is None:
                continue
            all_tm[i, i+1] = {'best': tm_best, 'cost': tm_cost, 'min': tm_min, 'max': tm_max}
        
        all_tm = dict(sorted(all_tm.items(), key=lambda d: d[1]['cost']))
        return all_tm
    
    def get_local_optimal_tm(self, veh_r:Agent, veh_f:Agent) -> tuple[float, float, float, float]:
        ts = np.arange(0,plan_horizon+1) * plan_pace

        opt_tm, opt_cost, opt_res = search_optimal_tm(self.veh_e, self.veh_p, veh_r, veh_f, err=0.05, ym=self.ym)
        if opt_tm is None:
            return None, None, None, None
        
        tm_min = opt_res['tm_min']
        tm_max = opt_res['tm_max']

        return opt_tm, opt_cost, tm_min, tm_max

    def get_optimal_trajectory(self, tm: float, tm_min:float, tm_max:float, veh_r:Agent, veh_f:Agent) -> AgentTrajectory:

        Ts = plan_pace
        ts = np.arange(1, plan_horizon+1) * Ts
        risk_level = get_risk_level()
        sigma_ax, sigma_ay = get_sigma()

        pos_vp = np.array([agt_nominal_xy(self.veh_p, k) for k in range(1, plan_horizon+1)])
        rx_vp, ry_vp = agt_radius_xy(self.veh_p)
        rr_vp = np.array([agt_risk_radius(self.veh_p, k, sigma_ax, sigma_ay, risk_level) for k in range(1, plan_horizon+1)])

        pos_vr = np.array([agt_nominal_xy(veh_r, k) for k in range(1, plan_horizon+1)])
        rx_vr, ry_vr = agt_radius_xy(veh_r)
        rr_vr = np.array([agt_risk_radius(veh_r, k, sigma_ax, sigma_ay, risk_level) for k in range(1, plan_horizon+1)])

        pos_vf = np.array([agt_nominal_xy(veh_f, k) for k in range(1, plan_horizon+1)])
        rx_vf, ry_vf = agt_radius_xy(veh_f)
        rr_vf = np.array([agt_risk_radius(veh_f, k, sigma_ax, sigma_ay, risk_level) for k in range(1, plan_horizon+1)])

        if self.yt > self.ym:
            y_lb = self.y_min * np.ones_like(ts)
            y_lb[ts>=tm_max] = self.ym
            y_ub = self.y_max * np.ones_like(ts)
            y_ub[ts<=tm_min] = self.ym
        else:
            y_lb = self.y_min * np.ones_like(ts)
            y_lb[ts<=tm_min] = self.ym
            y_ub = self.y_max * np.ones_like(ts)
            y_ub[ts>=tm_max] = self.ym

        def fn_model(u):
            # u = [ax] * Np + [ay] * Np
            ax = u[:plan_horizon]
            ay = u[plan_horizon:]

            vx = self.veh_e.v * cos(self.veh_e.h) + np.cumsum(ax) * Ts
            vy = self.veh_e.v * sin(self.veh_e.h) + np.cumsum(ay) * Ts

            xx = self.veh_e.x + np.cumsum(vx) * Ts
            yy = self.veh_e.y + np.cumsum(vy) * Ts

            hh = np.arctan2(vy, vx)
            ww = np.diff(hh, prepend=(self.veh_e.h,)) / Ts

            dd = ww * wheelbase / np.sqrt(vx**2 + vy**2 + 1e-4)

            return np.array([xx, yy, vx, vy, ax, ay, hh, ww, dd])
        
        def fn_cost(u):
            xx, yy, vx, vy, ax, ay, hh, ww, dd = fn_model(u)
            return 1.0 * np.sum(ax ** 2) + \
                   1.0 * np.sum(dd ** 2) + 1e-4 * np.sum((yy - self.yt) ** 2) + 1e-4 * np.sum((hh - 0.0) ** 2)

        def fn_cons(u):
            xx, yy, vx, vy, ax, ay, hh, ww, dd = fn_model(u)
            all_cons = []

            # collision
            dx = np.abs(xx - pos_vp[:,0]) - self.veh_e.length/2 - rx_vp * rr_vp
            dy = np.abs(yy - pos_vp[:,1]) - self.veh_e.width/2 - ry_vp * rr_vp
            d_ = - (dx <= 0).astype('float') * (dy <= 0).astype('float') # -1: collision, 0: no-collision
            all_cons.append(np.min(d_))

            dx = np.abs(xx - pos_vf[:,0]) - self.veh_e.length/2 - rx_vf * rr_vf
            dy = np.abs(yy - pos_vf[:,1]) - self.veh_e.width/2 - ry_vf * rr_vf
            d_ = - (dx <= 0).astype('float') * (dy <= 0).astype('float') # -1: collision, 0: no-collision
            all_cons.append(np.min(d_))

            dx = np.abs(xx - pos_vr[:,0]) - self.veh_e.length/2 - rx_vr * rr_vr
            dy = np.abs(yy - pos_vr[:,1]) - self.veh_e.width/2 - ry_vr * rr_vr
            d_ = - (dx <= 0).astype('float') * (dy <= 0).astype('float') # -1: collision, 0: no-collision
            all_cons.append(np.min(d_))

            # y-direction
            all_cons.append(np.min(yy - y_lb))
            all_cons.append(np.min(y_ub - yy))

            # vx limits
            all_cons.append(np.min(vx - v_min))
            all_cons.append(np.min(v_max - vx))

            # steer limits
            all_cons.append(np.min(dd - d_min))
            all_cons.append(np.min(d_max - dd))

            # heading limits
            all_cons.append(np.min(hh - h_min))
            all_cons.append(np.min(h_max - hh))

            return np.min(all_cons)

        def fn_initial_u():
            tt = np.arange(plan_horizon+1) * Ts
            t0, te = tt[0], tt[-1]
            y0, ye, ym = self.veh_e.y, self.yt, self.ym
            vy0, vye = self.veh_e.v * sin(self.veh_e.h), 0.0
            aye = 0.0

            A = np.matrix([
                [1.0,   t0,     t0**2,  t0**3,      t0**4,      t0**5],
                [1.0,   te,     te**2,  te**3,      te**4,      te**5],
                [1.0,   tm,     tm**2,  tm**3,      tm**4,      tm**5],
                [0.0,   1.0,    2*t0,   3*t0**2,    4*t0**3,    5*t0**4],
                [0.0,   1.0,    2*te,   3*te**2,    4*te**3,    5*te**4],
                [0.0,   0.0,    2.0,    6*te,       12*te**2,   20*te**3],
            ], float)
            B = np.matrix([
                [y0], [ye], [ym], [vy0], [vye], [aye]
            ])
            p = np.array(A.I * B).squeeze()[::-1]
        
            vy = np.polyval(np.polyder(p, 1), tt)
            ay = np.diff(vy) / Ts

            ax = np.zeros_like(ay)
            u = np.vstack([ax, ay]).reshape(-1)
            return u

        state_cons = [{'type': 'ineq', 'fun': fn_cons}]
        input_bounds = [(a_min, a_max)] * plan_horizon + [(-5.0, 5.0)] * plan_horizon
        u0 = fn_initial_u()
        res = scipy.optimize.minimize(fun=fn_cost, x0=u0, bounds=input_bounds, constraints=state_cons, options={'maxiter':100}, tol=1e-3)
        if not res.success:
            return None
        u = np.array(res.x, float)
        xx, yy, vx, vy, ax, ay, hh, ww, dd = fn_model(u)
        xx = np.insert(xx, 0, self.veh_e.x)
        yy = np.insert(yy, 0, self.veh_e.y)
        hh = np.insert(hh, 0, self.veh_e.h)
        vv = np.sqrt(vx**2 + vy**2)
        vv = np.insert(vv, 0, self.veh_e.v)
        
        t0 = datci.getTimestep() * datci.getSimupace()
        traj = AgentTrajectory(
            ts=np.arange(0, plan_horizon+1) * plan_pace + t0,
            states=np.vstack([xx, yy, hh, vv]).transpose(1,0),
            inputs=np.vstack([ax, dd]).transpose(1,0),
            tm=tm+t0,
            tm_margin=(tm_min+t0, tm_max+t0)
        )
        return traj



class LaneChangePlanner_SBF: 
    def __init__(self) -> None:
        self.veh_e = None
        self.veh_p = None
        self.veh_f = None
        self.veh_r = None

    def initialize(self, EV:Agent, lane_from:str, lane_to:str):
        self.delta = 0
        self.eta = 1
        self.y_ref = EV.y
        self.v_ref = EV.v

        self.veh_e = EV
        self.lane_from = lane_from
        self.lane_to = lane_to

        ########## set PV, FV, RV ##########
        all_vehicles = datci.getFrameVehicleIds()
        self.veh_p: Agent = None
        self.veh_f: Agent = None
        self.veh_r: Agent = None
        for Vid in all_vehicles:
            if Vid == self.veh_e.id:
                continue

            Vlane = datci.vehicle.getLaneID(Vid)
            if Vlane == self.lane_from:
                Vx, Vy = datci.vehicle.getPosition(Vid)
                if Vx > self.veh_e.x:
                    if self.veh_p is None or self.veh_p.x > Vx:
                        self.veh_p = get_agent_from_daci(Vid, label='veh_p')

            elif Vlane == self.lane_to:
                Vx, Vy = datci.vehicle.getPosition(Vid)
                if Vx > self.veh_e.x:
                    if self.veh_f is None or self.veh_f.x > Vx:
                        self.veh_f = get_agent_from_daci(Vid, label='veh_f')
                else:
                    if self.veh_r is None or self.veh_r.x < Vx:
                        self.veh_r = get_agent_from_daci(Vid, label='veh_r')
        
        ########## set ym and yt ##########
        self.lane_from_centery = datci.lane.getLane(self.lane_from).center_line[0][1]
        self.lane_from_width = datci.lane.getLaneWidth(self.lane_from)
        self.lane_to_centery = datci.lane.getLane(self.lane_to).center_line[0][1]
        self.lane_to_width = datci.lane.getLaneWidth(self.lane_to)

        if self.veh_p is None:
            self.veh_p = Agent('-1', self.veh_e.x+1000, self.veh_e.y, 0.0, self.veh_e.v, 0.0, 0.0, 4.0, 2.0, label='virtual')
        if self.veh_r is None:
            self.veh_r = Agent('-2', self.veh_e.x-1000, self.lane_to_centery, 0.0, self.veh_e.v, 0.0, 0.0, 4.0, 2.0, label='virtual')
        if self.veh_f is None:
            self.veh_f = Agent('-3', self.veh_e.x+1000, self.lane_to_centery, 0.0, self.veh_e.v, 0.0, 0.0, 4.0, 2.0, label='virtual')

        ########## set constraints on y, h, v ##########
        self.h_min = h_min
        self.h_max = h_max
        self.v_min = v_min
        self.v_max = v_max

    def run(self, EV: Agent, lane_from: str, lane_to: str) -> Union[None, AgentTrajectory]:

        # print("run LaneChangePlanner !!!!")

        self.initialize(EV, lane_from, lane_to)
        self.decider()
        if self.delta == 0:
            return None
        
        traj = self.generate_trajectory()
        if traj is None:
            return None

        out_traj = refine_trajectory(traj)
        return out_traj

    def run_replan(self, EV: Agent, lane_from: str, lane_to: str) -> Union[None, AgentTrajectory]:
        return self.run(EV, lane_from, lane_to)

    def decider(self):
        self.delta, self.eta = 1, 1
        if self.is_in_mitigation_range(self.veh_p) or self.is_in_mitigation_range(self.veh_f) or self.is_in_mitigation_range(self.veh_r):
            self.delta, self.eta = 1, 0

    def generate_trajectory(self):
        # Np, Nc, Ts = 20, 3, 0.2
        Np, Nc, Ts = plan_horizon, plan_horizon, plan_pace
        tTIV, tTTC = 1.0, 0.5
        Mm = 1e3
        q1, q2 = 1.0, 1.0
        r1, r2 = 0.1, 1.0
        kr = 1e3
        ax_min, ax_max = -4.0, 2.0
        ay_min, ay_max = -2.0, 2.0
        Delta_ax_min, Delta_ax_max = -1.5, 1.5
        Delta_ay_min, Delta_ay_max = -0.5, 0.5
        beta_max = np.deg2rad(10.0)
        lamb = exp(1.317)
        y_err = 0.01
        miu = 0.8
        Gg = 9.81
        delta_k, eta_k = self.delta, self.eta
        t_now = datci.getTimestep() * datci.getSimupace()


        rho_max = 0.85 * miu * Gg / self.veh_e.v**2
        y_lat = self.lane_to_centery - self.veh_e.y - 1.0
        z_max = sqrt(rho_max * (lamb+1)**3 / abs(y_lat) / lamb / (lamb-1))
        s_max = Np * Ts
        z_min = -2 * log(y_err / (1 - y_err)) / s_max
        Zm = (z_max + z_min) / 2

        v_ref = np.max([self.veh_p.v, self.veh_f.v, self.veh_r.v, self.veh_e.v])
        y_ref = self.lane_to_centery if delta_k > 0 else self.veh_e.y
        S_long = tTTC * v_ref
        S_f = tTIV * v_ref

        veh_p_x = [self.veh_p.x + self.veh_p.v * k*Ts for k in range(Np)]

        if self.lane_to_centery > self.lane_from_centery:
            flag_dir = 1
            y_max = self.lane_to_centery + self.lane_to_width/2
            y_min = self.lane_from_centery - self.lane_from_width/2
        else:
            flag_dir = -1
            y_max = self.lane_from_centery + self.lane_from_width/2
            y_min = self.lane_to_centery - self.lane_to_width/2

        def fn_model(u):
            ds = np.array(u[:2*Nc]).reshape(Nc, 2)

            dax = np.zeros(Np, dtype=float)
            dax[:Nc] = ds[:,0]

            day = np.zeros(Np, dtype=float)
            day[:Nc] = ds[:,1]

            ax = self.veh_e.a * cos(self.veh_e.h) + np.cumsum(dax)
            ay = self.veh_e.a * sin(self.veh_e.h) + np.cumsum(day)

            vx = self.veh_e.v * cos(self.veh_e.h) + Ts * np.cumsum(ax)
            vy = self.veh_e.v * sin(self.veh_e.h) + Ts * np.cumsum(ay)
            xx = self.veh_e.x + Ts * np.cumsum(vx)
            yy = self.veh_e.y + Ts * np.cumsum(vy)

            return np.array([xx, yy, vx, vy, ax, ay], dtype=float) #(6, Np)

        def fn_cost(u):
            # u = [d_ax, d_ay] * Nc + [rx_p] * N_p
            all_input_s = np.array(u[:2*Nc]).reshape(Nc, 2)
            all_input_r = np.array(u[2*Nc:]).reshape(Np)
            all_state = fn_model(u)
            
            state_y = all_state[1]
            state_v = all_state[2]
            input_dax = all_input_s[:,0]
            input_day = all_input_s[:,1]
            input_rx = np.reshape(all_input_r, -1)

            return  q1 * np.sum((state_y - y_ref) ** 2) + q2 * np.sum((state_v - v_ref) ** 2) + \
                    r1 * np.sum(input_dax ** 2) + r2 * np.sum(input_day ** 2) + \
                    kr * np.sum(input_rx ** 2)
        
        def fn_cons(u):
            xx, yy, vx, vy, ax, ay = fn_model(u)
            ux, uy = np.array(u[:2*Nc]).reshape(Nc, 2).transpose(1,0)
            ur = np.array(u[2*Nc:]).reshape(Np)

            all_cons = []

            all_cons.append(np.min(yy - y_min))
            all_cons.append(np.min(y_max - yy))

            all_cons.append(np.min(vx - v_min))
            all_cons.append(np.min(v_max - vx))

            vy_min = -vx*beta_max
            vy_max =  vx*beta_max
            all_cons.append(np.min(vy - vy_min))
            all_cons.append(np.min(vy_max - vy))

            all_cons.append(np.min(ax - ax_min))
            all_cons.append(np.min(ax_max - ax))

            all_cons.append(np.min(ay - ay_min))
            all_cons.append(np.min(ay_max - ay))


            # position x
            Delta_x_p = veh_p_x - xx
            all_cons.append(np.min(Delta_x_p + ur - S_long + Mm * eta_k))

            # position y
            Dy = delta_k * y_lat / (1 + np.exp(-Zm * (-Delta_x_p + S_f)))
            Delta_y_p = yy - self.veh_p.y - Dy
            all_cons.append(np.min(flag_dir* Delta_y_p))

            return np.min(all_cons)
            
        def fn_initial_u():
            tt = np.arange(Nc+1) * Ts
            t0, te = tt[0], tt[-1]
            y0, ye = self.veh_e.y, self.lane_to_centery
            vy0, vye = self.veh_e.v * sin(self.veh_e.h), 0.0
            ay0, aye = self.veh_e.a * sin(self.veh_e.h), 0.0

            A = np.matrix([
                [1.0,   t0,     t0**2,  t0**3,      t0**4,      t0**5],
                [1.0,   te,     te**2,  te**3,      te**4,      te**5],
                [0.0,   1.0,    2*t0,   3*t0**2,    4*t0**3,    5*t0**4],
                [0.0,   1.0,    2*te,   3*te**2,    4*te**3,    5*te**4],
                [0.0,   0.0,    2.0,    6*t0,       12*t0**2,   20*t0**3],
                [0.0,   0.0,    2.0,    6*te,       12*te**2,   20*te**3],
            ], float)
            B = np.matrix([
                [y0], [ye], [vy0], [vye], [ay0], [aye]
            ])
            p = np.array(A.I * B).squeeze()[::-1]
            
            x = self.v_ref * tt + self.veh_e.x
            vx = self.v_ref * np.ones_like(tt)
            ax = np.diff(vx, n=1, prepend=(self.veh_e.v,)) / Ts
            dax = np.diff(ax, n=1)

            y = np.polyval(p, tt)
            vy = np.polyval(np.polyder(p, 1), tt)
            ay = np.polyval(np.polyder(p, 2), tt)
            day = np.diff(ay, n=1)
                        
            us = np.vstack([dax, day]).transpose(1,0)
            ur = np.zeros((Np*1,), float)
            u = np.hstack([us.reshape(-1), ur.reshape(-1)])
            return u

        state_cons = [{'type': 'ineq', 'fun': fn_cons}]
        input_bounds = [(Delta_ax_min, Delta_ax_max), (Delta_ay_min, Delta_ay_max)] * Nc + [(0.0, 2.0)] * Np
        u0 = fn_initial_u()
        res = scipy.optimize.minimize(fun=fn_cost, x0=u0, bounds=input_bounds, constraints=state_cons, options={'maxiter':100}, tol=1e-3)
        # res = minimize_ipopt(fun=fn_cost, x0=u0, bounds=input_bounds, constraints=state_cons, options={'maxiter':100, 'disp': False}, tol=1e-3)
        if not res.success:
            return None
        u = np.array(res.x, float)
        xx, yy, vx, vy, ax, ay = fn_model(u)
        xx = np.insert(xx, 0, self.veh_e.x)
        yy = np.insert(yy, 0, self.veh_e.y)
        vx = np.insert(vx, 0, self.veh_e.v * cos(self.veh_e.h))
        vy = np.insert(vy, 0, self.veh_e.v * sin(self.veh_e.h))
        vv = np.sqrt(vx**2 + vy**2)
        hh = np.arctan2(vy, vx)

        traj = AgentTrajectory(
            ts=np.arange(Np+1) * Ts + t_now,
            states=np.vstack([xx, yy, hh, vv]).transpose(1,0),
            inputs=np.vstack([ax, ay]).transpose(1,0)
        )
        return traj
   
    def is_in_danger_range(self, SV:Agent):
        if SV.x < self.veh_e.x:
            safety_x = SV.v * 4.0
        else:
            safety_x = self.veh_e.v * 4.0
        
        if abs(SV.x - self.veh_e.x) > safety_x + self.veh_e.length/2 + SV.length/2:
            return False
        
        safety_y = 0.5
        if abs(SV.y - self.veh_e.y) > safety_y + self.veh_e.width/2 + SV.width/2:
            return False
        
        return True

    def is_in_mitigation_range(self, SV: Agent):
        if SV.x < self.veh_e.x:
            safety_x = SV.v * 2.0
        else:
            safety_x = self.veh_e.v * 2.0
        
        if abs(SV.x - self.veh_e.x) > safety_x + self.veh_e.length/2 + SV.length/2:
            return False
        
        safety_y = 0.1
        if abs(SV.y - self.veh_e.y) > safety_y + self.veh_e.width/2 + SV.width/2:
            return False
        
        return True

class LaneChangePlanner_RAP:
    def __init__(self) -> None:
        self.veh_e : Agent = None
        self.veh_p : Agent = None
        self.veh_f : Agent = None
        self.veh_r : Agent = None
        self.plan_tm = None
        self.plan_tm_margin = (0, 0)

    def initialize(self, EV:Agent, lane_from:str, lane_to:str):
        self.veh_e = EV
        self.lane_from = lane_from
        self.lane_to = lane_to

        ########## set PV & TVs ##########
        all_vehicles = datci.getFrameVehicleIds()
        self.veh_p: Agent = None
        veh_now_following: Agent = None
        self.all_veh_t: list[Agent] = []
        for Vid in all_vehicles:
            if Vid == self.veh_e.id:
                continue

            Vlane = datci.vehicle.getLaneID(Vid)
            if Vlane == self.lane_from:
                Vx, Vy = datci.vehicle.getPosition(Vid)
                if Vx > self.veh_e.x:
                    if self.veh_p is None or self.veh_p.x > Vx:
                        self.veh_p = get_agent_from_daci(Vid, label='veh_p')
                else:
                    if veh_now_following is None or veh_now_following.x < Vx:
                        veh_now_following = get_agent_from_daci(Vid)

            elif Vlane == self.lane_to:
                Vx, Vy = datci.vehicle.getPosition(Vid)
                Dx = self.veh_e.v * 6.0
                if Vx > self.veh_e.x-Dx and Vx < self.veh_e.x + Dx:
                    self.all_veh_t.append(get_agent_from_daci(Vid))

        if self.veh_p is None:
            self.veh_p = Agent('-1', self.veh_e.x+100, self.veh_e.y, 0.0, self.veh_e.v, 0.0, 0.0, 4.0, 2.0, label='virtual')

        if len(self.all_veh_t) > 0:
            self.all_veh_t = sorted(self.all_veh_t, key=lambda V: V.x)
            TV = self.all_veh_t[0]
            self.all_veh_t.insert(0, Agent('-2',TV.x-100, TV.y, 0.0, TV.v, 0.0, 0.0, 4.0, 2.0, label='virtual'))
            TV = self.all_veh_t[-1]
            self.all_veh_t.append(Agent('-3',TV.x+100, TV.y, 0.0, TV.v, 0.0, 0.0, 4.0, 2.0, label='virtual'))

        ########## set ym and yt ##########
        lane_from_centery = datci.lane.getLane(self.lane_from).center_line[0][1]
        lane_from_width = datci.lane.getLaneWidth(self.lane_from)
        lane_to_centery = datci.lane.getLane(self.lane_to).center_line[0][1]
        lane_to_width = datci.lane.getLaneWidth(self.lane_to)

        self.ym = (lane_from_centery + lane_to_centery) / 2
        self.yt = lane_to_centery

        ########## set constraints on y, h, v ##########
        if self.yt > self.ym:
            self.y_min = lane_from_centery - lane_from_width/2
            self.y_max = lane_to_centery + lane_to_width/2
        else:
            self.y_min = lane_to_centery - lane_to_width/2
            self.y_max = lane_from_centery + lane_from_width/2
        
        self.h_min = h_min
        self.h_max = h_max
        self.v_min = v_min
        self.v_max = v_max

    def run(self, EV: Agent, lane_from: str, lane_to: str) -> Union[None, AgentTrajectory]:
        self.initialize(EV, lane_from, lane_to)

        tm_dict = self.get_optimal_tm_dict()
        if len(tm_dict) == 0: # no valid tm
            # print("No Valid Tm")
            return None
        
        tm_key = list(tm_dict.keys())[0]
        tm_val = tm_dict[tm_key]
        
        veh_r = self.all_veh_t[tm_key[0]]
        veh_f = self.all_veh_t[tm_key[1]]
        tm, tm_min, tm_max = tm_val['best'], tm_val['min'], tm_val['max']
        self.veh_r, self.veh_f = veh_r, veh_f

        traj = self.get_optimal_trajectory(tm, tm_min, tm_max)
        if traj is None:
            return None
        
        out_traj = refine_trajectory(traj)
        self.plan_tm = out_traj.tm

        return out_traj

    def run_replan(self, EV:Agent, lane_from: str, lane_to: str) -> Union[None, AgentTrajectory]:
        self.initialize(EV, lane_from, lane_to)

        tm_dict = self.get_optimal_tm_dict()
        if len(tm_dict) == 0: # no valid tm
            return None
        
        now_t = datci.getTimestep() * datci.getSimupace()
        now_tm = self.plan_tm - now_t

        veh_r : Agent = None
        veh_f : Agent = None
        tm_min, tm_max = None, None
        for tm_key, tm_val in tm_dict.items():
            if now_tm >= tm_val['min'] and now_tm <= tm_val['max']:
                veh_r = self.all_veh_t[tm_key[0]]
                veh_f = self.all_veh_t[tm_key[1]]
                tm_min, tm_max = now_tm - tm_val['min'], tm_val['max'] - now_tm
        
        if tm_min is None and tm_max is None:
            return None

        if veh_r is None:
            veh_r = Agent('-2', self.veh_e.x-100, self.lane_to, 0.0, self.veh_e.v, 0.0, 0.0, 4.0, 2.0, label='virtual')
        if veh_f is None:
            veh_f = Agent('-3', self.veh_e.x+100, self.lane_to, 0.0, self.veh_e.v, 0.0, 0.0, 4.0, 2.0, label='virtual')
        
        self.veh_r, self.veh_f = veh_r, veh_f

        traj = self.get_optimal_trajectory(now_tm, tm_min, tm_max)
        if traj is None:
            return None
        
        out_traj = refine_trajectory(traj)
        self.plan_tm = out_traj.tm
        return out_traj

    def get_optimal_tm_dict(self) -> dict:
        all_tm = {}
        for i in range(len(self.all_veh_t)-1):
            tm_best, tm_cost, tm_min, tm_max = self.get_local_optimal_tm(self.all_veh_t[i], self.all_veh_t[i+1])
            if tm_best is None:
                continue
            all_tm[i, i+1] = {'best': tm_best, 'cost': tm_cost, 'min': tm_min, 'max': tm_max}
        
        all_tm = dict(sorted(all_tm.items(), key=lambda d: d[1]['cost']))
        return all_tm
    
    def get_local_optimal_tm(self, veh_r:Agent, veh_f:Agent) -> tuple[float, float, float, float]:
        ts = np.arange(0,plan_horizon+1) * plan_pace

        opt_tm, opt_cost, opt_res = search_optimal_tm(self.veh_e, self.veh_p, veh_r, veh_f, err=0.05, ym=self.ym, use_risk=False)
        if opt_tm is None:
            return None, None, None, None
        
        tm_min = opt_res['tm_min']
        tm_max = opt_res['tm_max']

        return opt_tm, opt_cost, tm_min, tm_max

    def get_optimal_trajectory(self, tm: float, tm_min:float, tm_max:float) -> AgentTrajectory:

        Ts = plan_pace
        ts = np.arange(1, plan_horizon+1) * Ts

        pos_vp = np.array([agt_nominal_xy(self.veh_p, k) for k in range(1, plan_horizon+1)])
        pos_vr = np.array([agt_nominal_xy(self.veh_r, k) for k in range(1, plan_horizon+1)])
        pos_vf = np.array([agt_nominal_xy(self.veh_f, k) for k in range(1, plan_horizon+1)])

        gamma_t = np.array([0.9**k for k in range(plan_horizon)], float)
        omega_thw = 1.0
        omega_ttc = 4.0

        def fn_model(u):
            # u = [ax] * Np + [ay] * Np
            ax = u[:plan_horizon]
            ay = u[plan_horizon:]

            vx = self.veh_e.v * cos(self.veh_e.h) + np.cumsum(ax) * Ts
            vy = self.veh_e.v * sin(self.veh_e.h) + np.cumsum(ay) * Ts

            xx = self.veh_e.x + np.cumsum(vx) * Ts
            yy = self.veh_e.y + np.cumsum(vy) * Ts

            hh = np.arctan2(vy, vx)
            ww = np.diff(hh, prepend=(self.veh_e.h,)) / Ts

            dd = ww * wheelbase / np.sqrt(vx**2 + vy**2 + 1e-4)

            return np.array([xx, yy, vx, vy, ax, ay, hh, ww, dd])
        
        def fn_cost(u):
            xx, yy, vx, vy, ax, ay, hh, ww, dd = fn_model(u)

            risk_vp = fn_risk(xx, yy, vx, 'veh_p')
            risk_vf = fn_risk(xx, yy, vx, 'veh_f')
            risk_vr = fn_risk(xx, yy, vx, 'veh_r')

            return 1.0 * np.sum(ax ** 2) + \
                   1.0 * np.sum(ay ** 2) + 1e-4 * np.sum((yy - self.yt) ** 2) + 1e-4 * np.sum((hh - 0.0) ** 2) + \
                   1.0 * np.sum(risk_vp + risk_vf + risk_vr) / 3
    
        def fn_risk(xx, yy, vv, sid:str):
            if sid == 'veh_p':
                epsilon = np.exp(-((yy - pos_vp[:,1]) / self.veh_p.width)**2)
                dx = pos_vp[:,0] - xx
                dv = vv - self.veh_p.v
                rx = self.veh_p.length
            elif sid == 'veh_f':
                epsilon = np.exp(-((yy - pos_vf[:,1]) / self.veh_f.width)**2)
                dx = pos_vf[:,0] - xx
                dv = vv - self.veh_f.v
                rx = self.veh_f.length
            elif sid == 'veh_r':
                epsilon = np.exp(-((yy - pos_vr[:,1]) / self.veh_r.width)**2)
                dx = xx - pos_vr[:,0]
                dv = self.veh_r.v - vv
                rx = self.veh_r.length
            else:
                raise ValueError("Invalid sid")

            ttc = epsilon * dx / np.clip(dv, 0.1, 100.0)
            thw = epsilon * dx / vv + rx / vv
            ttc = np.clip(ttc, 0.1, 10.0)
            thw = np.clip(thw, 0.1, 10.0)

            return np.exp(gamma_t * (omega_thw / thw + omega_ttc / ttc))

        def fn_cons(u):
            xx, yy, vx, vy, ax, ay, hh, ww, dd = fn_model(u)
            all_cons = []
            
            # y-direction
            all_cons.append(np.min(yy - self.y_min))
            all_cons.append(np.min(self.y_max - yy))

            # vx limits
            all_cons.append(np.min(vx - v_min))
            all_cons.append(np.min(v_max - vx))

            # steer limits
            all_cons.append(np.min(dd - d_min))
            all_cons.append(np.min(d_max - dd))

            # heading limits
            all_cons.append(np.min(hh - h_min))
            all_cons.append(np.min(h_max - hh))

            return np.min(all_cons)

        def fn_initial_u():
            tt = np.arange(plan_horizon+1) * Ts
            t0, te = tt[0], tt[-1]
            y0, ye = self.veh_e.y, self.yt
            vy0, vye = self.veh_e.v * sin(self.veh_e.h), 0.0
            ay0, aye = self.veh_e.a * sin(self.veh_e.h), 0.0

            A = np.matrix([
                [1.0,   t0,     t0**2,  t0**3,      t0**4,      t0**5],
                [1.0,   te,     te**2,  te**3,      te**4,      te**5],
                [0.0,   1.0,    2*t0,   3*t0**2,    4*t0**3,    5*t0**4],
                [0.0,   1.0,    2*te,   3*te**2,    4*te**3,    5*te**4],
                [0.0,   0.0,    2.0,    6*t0,       12*t0**2,   20*t0**3],
                [0.0,   0.0,    2.0,    6*te,       12*te**2,   20*te**3],
            ], float)
            B = np.matrix([
                [y0], [ye], [vy0], [vye], [ay0], [aye]
            ])
            p = np.array(A.I * B).squeeze()[::-1]
        
            vy = np.polyval(np.polyder(p, 1), tt)
            ay = np.diff(vy) / Ts

            ax = np.zeros_like(ay)
            u = np.vstack([ax, ay]).reshape(-1)
            return u

        state_cons = [{'type': 'ineq', 'fun': fn_cons}]
        input_bounds = [(a_min, a_max)] * plan_horizon + [(-2.0, 2.0)] * plan_horizon
        u0 = fn_initial_u()
        res = scipy.optimize.minimize(fun=fn_cost, x0=u0, bounds=input_bounds, constraints=state_cons, options={'maxiter':100}, tol=1e-3)
        if not res.success:
            return None
        u = np.array(res.x, float)
        xx, yy, vx, vy, ax, ay, hh, ww, dd = fn_model(u)
        xx = np.insert(xx, 0, self.veh_e.x)
        yy = np.insert(yy, 0, self.veh_e.y)
        hh = np.insert(hh, 0, self.veh_e.h)
        vv = np.sqrt(vx**2 + vy**2)
        vv = np.insert(vv, 0, self.veh_e.v)
        
        t0 = datci.getTimestep() * datci.getSimupace()
        traj = AgentTrajectory(
            ts=np.arange(0, plan_horizon+1) * plan_pace + t0,
            states=np.vstack([xx, yy, hh, vv]).transpose(1,0),
            inputs=np.vstack([ax, dd]).transpose(1,0),
            tm=tm+t0,
            tm_margin=(tm_min+t0, tm_max+t0)
        )
        return traj

class LaneChangePlanner_GDP:
    def __init__(self) -> None:
        self.veh_e : Agent = None
        self.veh_p : Agent = None
        self.veh_f : Agent = None
        self.veh_r : Agent = None
        self.plan_tm = None
        self.plan_tm_margin = (0, 0)

    def initialize(self, EV:Agent, lane_from:str, lane_to:str):
        self.veh_e = EV
        self.lane_from = lane_from
        self.lane_to = lane_to

        ########## set PV & TVs ##########
        all_vehicles = datci.getFrameVehicleIds()
        self.veh_p: Agent = None
        veh_now_following: Agent = None
        self.all_veh_t: list[Agent] = []
        for Vid in all_vehicles:
            if Vid == self.veh_e.id:
                continue

            Vlane = datci.vehicle.getLaneID(Vid)
            if Vlane == self.lane_from:
                Vx, Vy = datci.vehicle.getPosition(Vid)
                if Vx > self.veh_e.x:
                    if self.veh_p is None or self.veh_p.x > Vx:
                        self.veh_p = get_agent_from_daci(Vid, label='veh_p')
                else:
                    if veh_now_following is None or veh_now_following.x < Vx:
                        veh_now_following = get_agent_from_daci(Vid)

            elif Vlane == self.lane_to:
                Vx, Vy = datci.vehicle.getPosition(Vid)
                Dx = self.veh_e.v * 6.0
                if Vx > self.veh_e.x-Dx and Vx < self.veh_e.x + Dx:
                    self.all_veh_t.append(get_agent_from_daci(Vid))

        if self.veh_p is None:
            self.veh_p = Agent('-1', self.veh_e.x+100, self.veh_e.y, 0.0, self.veh_e.v, 0.0, 0.0, 4.0, 2.0, label='virtual')

        if len(self.all_veh_t) > 0:
            self.all_veh_t = sorted(self.all_veh_t, key=lambda V: V.x)
            TV = self.all_veh_t[0]
            self.all_veh_t.insert(0, Agent('-2',TV.x-100, TV.y, 0.0, TV.v, 0.0, 0.0, 4.0, 2.0, label='virtual'))
            TV = self.all_veh_t[-1]
            self.all_veh_t.append(Agent('-3',TV.x+100, TV.y, 0.0, TV.v, 0.0, 0.0, 4.0, 2.0, label='virtual'))

        ########## set ym and yt ##########
        lane_from_centery = datci.lane.getLane(self.lane_from).center_line[0][1]
        lane_from_width = datci.lane.getLaneWidth(self.lane_from)
        lane_to_centery = datci.lane.getLane(self.lane_to).center_line[0][1]
        lane_to_width = datci.lane.getLaneWidth(self.lane_to)

        self.ym = (lane_from_centery + lane_to_centery) / 2
        self.yt = lane_to_centery

        ########## set constraints on y, h, v ##########
        if self.yt > self.ym:
            self.y_min = lane_from_centery - lane_from_width/2
            self.y_max = lane_to_centery + lane_to_width/2
        else:
            self.y_min = lane_to_centery - lane_to_width/2
            self.y_max = lane_from_centery + lane_from_width/2
        
        self.h_min = h_min
        self.h_max = h_max
        self.v_min = v_min
        self.v_max = v_max

    def run(self, EV: Agent, lane_from: str, lane_to: str) -> Union[None, AgentTrajectory]:
        self.initialize(EV, lane_from, lane_to)

        tm_dict = self.get_optimal_tm_dict()
        if len(tm_dict) == 0: # no valid tm
            # print("No Valid Tm")
            return None
        
        tm_key = list(tm_dict.keys())[0]
        tm_val = tm_dict[tm_key]
        
        veh_r = self.all_veh_t[tm_key[0]]
        veh_f = self.all_veh_t[tm_key[1]]
        tm, tm_min, tm_max = tm_val['best'], tm_val['min'], tm_val['max']
        self.veh_r, self.veh_f = veh_r, veh_f

        traj = self.get_optimal_trajectory(tm, tm_min, tm_max)
        if traj is None:
            return None
        
        out_traj = refine_trajectory(traj)
        self.plan_tm = out_traj.tm

        return out_traj

    def run_replan(self, EV:Agent, lane_from: str, lane_to: str) -> Union[None, AgentTrajectory]:
        self.initialize(EV, lane_from, lane_to)

        tm_dict = self.get_optimal_tm_dict()
        if len(tm_dict) == 0: # no valid tm
            return None
        
        now_t = datci.getTimestep() * datci.getSimupace()
        now_tm = self.plan_tm - now_t

        veh_r : Agent = None
        veh_f : Agent = None
        tm_min, tm_max = None, None
        for tm_key, tm_val in tm_dict.items():
            if now_tm >= tm_val['min'] and now_tm <= tm_val['max']:
                veh_r = self.all_veh_t[tm_key[0]]
                veh_f = self.all_veh_t[tm_key[1]]
                tm_min, tm_max = now_tm - tm_val['min'], tm_val['max'] - now_tm
        
        if tm_min is None and tm_max is None:
            return None

        if veh_r is None:
            veh_r = Agent('-2', self.veh_e.x-100, self.lane_to, 0.0, self.veh_e.v, 0.0, 0.0, 4.0, 2.0, label='virtual')
        if veh_f is None:
            veh_f = Agent('-3', self.veh_e.x+100, self.lane_to, 0.0, self.veh_e.v, 0.0, 0.0, 4.0, 2.0, label='virtual')
        
        self.veh_r, self.veh_f = veh_r, veh_f

        traj = self.get_optimal_trajectory(now_tm, tm_min, tm_max)
        if traj is None:
            return None
        
        out_traj = refine_trajectory(traj)
        self.plan_tm = out_traj.tm
        return out_traj

    def get_optimal_tm_dict(self) -> dict:
        all_tm = {}
        for i in range(len(self.all_veh_t)-1):
            tm_best, tm_cost, tm_min, tm_max = self.get_local_optimal_tm(self.all_veh_t[i], self.all_veh_t[i+1])
            if tm_best is None:
                continue
            all_tm[i, i+1] = {'best': tm_best, 'cost': tm_cost, 'min': tm_min, 'max': tm_max}
        
        all_tm = dict(sorted(all_tm.items(), key=lambda d: d[1]['cost']))
        return all_tm
    
    def get_local_optimal_tm(self, veh_r:Agent, veh_f:Agent) -> tuple[float, float, float, float]:
        ts = np.arange(0,plan_horizon+1) * plan_pace

        opt_tm, opt_cost, opt_res = search_optimal_tm(self.veh_e, self.veh_p, veh_r, veh_f, err=0.05, ym=self.ym, use_risk=False)
        if opt_tm is None:
            return None, None, None, None
        
        tm_min = opt_res['tm_min']
        tm_max = opt_res['tm_max']

        return opt_tm, opt_cost, tm_min, tm_max

    def get_optimal_trajectory(self, tm: float, tm_min:float, tm_max:float) -> AgentTrajectory:

        Ts = plan_pace
        ts = np.arange(1, plan_horizon+1) * Ts
        delta_m1 = (tm - tm_min) * 0.4
        delta_m2 = (tm_max - tm) * 0.4
        sigma_ax, sigma_ay = get_sigma()

        pos_vp = np.array([agt_nominal_xy(self.veh_p, k) for k in range(1, plan_horizon+1)])
        sx_vp = np.sqrt(np.array([moment_s(k, 0, sigma_ax**2)[1] for k in range(1, plan_horizon+1)], float))
        sy_vp = np.sqrt(np.array([moment_s(k, 0, sigma_ay**2)[1] for k in range(1, plan_horizon+1)], float))
        rx_vp = self.veh_p.length/2 + self.veh_e.length/2
        ry_vp = self.veh_p.width/2 + self.veh_e.width/2

        pos_vr = np.array([agt_nominal_xy(self.veh_r, k) for k in range(1, plan_horizon+1)])
        sx_vr = np.sqrt(np.array([moment_s(k, 0, sigma_ax**2)[1] for k in range(1, plan_horizon+1)], float))
        sy_vr = np.sqrt(np.array([moment_s(k, 0, sigma_ay**2)[1] for k in range(1, plan_horizon+1)], float))
        rx_vr = self.veh_r.length/2 + self.veh_e.length/2
        ry_vr = self.veh_r.width/2 + self.veh_e.width/2

        pos_vf = np.array([agt_nominal_xy(self.veh_f, k) for k in range(1, plan_horizon+1)])
        sx_vf = np.sqrt(np.array([moment_s(k, 0, sigma_ax**2)[1] for k in range(1, plan_horizon+1)], float))
        sy_vf = np.sqrt(np.array([moment_s(k, 0, sigma_ay**2)[1] for k in range(1, plan_horizon+1)], float))
        rx_vf = self.veh_f.length/2 + self.veh_e.length/2
        ry_vf = self.veh_f.width/2 + self.veh_e.width/2

        if self.yt > self.ym:
            y_lb = self.y_min * np.ones_like(ts)
            y_lb[ts>=tm+delta_m2] = self.ym
            y_ub = self.y_max * np.ones_like(ts)
            y_ub[ts<=tm-delta_m1] = self.ym
        else:
            y_lb = self.y_min * np.ones_like(ts)
            y_lb[ts<=tm-delta_m1] = self.ym
            y_ub = self.y_max * np.ones_like(ts)
            y_ub[ts>=tm+delta_m2] = self.ym


        def fn_model(u):
            # u = [ax] * Np + [ay] * Np
            ax = u[:plan_horizon]
            ay = u[plan_horizon:]

            vx = self.veh_e.v * cos(self.veh_e.h) + np.cumsum(ax) * Ts
            vy = self.veh_e.v * sin(self.veh_e.h) + np.cumsum(ay) * Ts

            xx = self.veh_e.x + np.cumsum(vx) * Ts
            yy = self.veh_e.y + np.cumsum(vy) * Ts

            hh = np.arctan2(vy, vx)
            ww = np.diff(hh, prepend=(self.veh_e.h,)) / Ts

            dd = ww * wheelbase / np.sqrt(vx**2 + vy**2 + 1e-4)

            return np.array([xx, yy, vx, vy, ax, ay, hh, ww, dd])
        
        def fn_cost(u):
            xx, yy, vx, vy, ax, ay, hh, ww, dd = fn_model(u)

            dx = np.clip(np.abs(xx - pos_vp[:,0]) - rx_vp, 0.0, 100)
            dy = np.clip(np.abs(yy - pos_vp[:,1]) - ry_vp, 0.0, 2)
            risk_vp = 1 / (2 * np.pi * sx_vp * sy_vp) * np.exp(-0.5 * (dx**2 / sx_vp**2 + dy**2 / sy_vp**2))

            dx = np.clip(np.abs(xx - pos_vf[:,0]) - rx_vf, 0.0, 100)
            dy = np.clip(np.abs(yy - pos_vf[:,1]) - ry_vf, 0.0, 2)
            risk_vf = 1 / (2 * np.pi * sx_vf * sy_vf) * np.exp(-0.5 * (dx**2 / sx_vf**2 + dy**2 / sy_vf**2))

            dx = np.clip(np.abs(xx - pos_vr[:,0]) - rx_vr, 0.0, 100)
            dy = np.clip(np.abs(yy - pos_vr[:,1]) - ry_vr, 0.0, 2)
            risk_vr = 1 / (2 * np.pi * sx_vr * sy_vr) * np.exp(-0.5 * (dx**2 / sx_vr**2 + dy**2 / sy_vr**2))

            return 1.0 * np.sum(ax ** 2) + \
                   1.0 * np.sum(dd ** 2) + 1e-4 * np.sum((yy - self.yt) ** 2) + 1e-4 * np.sum((hh - 0.0) ** 2) + \
                   1.0 * np.sum(risk_vp + risk_vf + risk_vr) / 3
    
        def fn_cons(u):
            xx, yy, vx, vy, ax, ay, hh, ww, dd = fn_model(u)
            all_cons = []
            
            # y-direction
            all_cons.append(np.min(yy - y_lb))
            all_cons.append(np.min(y_ub - yy))

            # vx limits
            all_cons.append(np.min(vx - v_min))
            all_cons.append(np.min(v_max - vx))

            # steer limits
            all_cons.append(np.min(dd - d_min))
            all_cons.append(np.min(d_max - dd))

            # heading limits
            all_cons.append(np.min(hh - h_min))
            all_cons.append(np.min(h_max - hh))

            return np.min(all_cons)

        def fn_initial_u():
            tt = np.arange(plan_horizon+1) * Ts
            t0, te = tt[0], tt[-1]
            y0, ye = self.veh_e.y, self.yt
            vy0, vye = self.veh_e.v * sin(self.veh_e.h), 0.0
            ay0, aye = 0.0, 0.0

            A = np.matrix([
                [1.0,   t0,     t0**2,  t0**3,      t0**4,      t0**5],
                [1.0,   te,     te**2,  te**3,      te**4,      te**5],
                [0.0,   1.0,    2*t0,   3*t0**2,    4*t0**3,    5*t0**4],
                [0.0,   1.0,    2*te,   3*te**2,    4*te**3,    5*te**4],
                [0.0,   0.0,    2.0,    6*t0,       12*t0**2,   20*t0**3],
                [0.0,   0.0,    2.0,    6*te,       12*te**2,   20*te**3],
            ], float)
            B = np.matrix([
                [y0], [ye], [vy0], [vye], [ay0], [aye]
            ])
            p = np.array(A.I * B).squeeze()[::-1]
        
            vy = np.polyval(np.polyder(p, 1), tt)
            ay = np.diff(vy) / Ts

            ax = np.zeros_like(ay)
            u = np.vstack([ax, ay]).reshape(-1)
            return u

        state_cons = [{'type': 'ineq', 'fun': fn_cons}]
        input_bounds = [(a_min, a_max)] * plan_horizon + [(-5.0, 5.0)] * plan_horizon
        u0 = fn_initial_u()
        res = scipy.optimize.minimize(fun=fn_cost, x0=u0, bounds=input_bounds, constraints=state_cons, options={'maxiter':100}, tol=1e-3)
        if not res.success:
            return None
        u = np.array(res.x, float)
        xx, yy, vx, vy, ax, ay, hh, ww, dd = fn_model(u)
        xx = np.insert(xx, 0, self.veh_e.x)
        yy = np.insert(yy, 0, self.veh_e.y)
        hh = np.insert(hh, 0, self.veh_e.h)
        vv = np.sqrt(vx**2 + vy**2)
        vv = np.insert(vv, 0, self.veh_e.v)
        
        t0 = datci.getTimestep() * datci.getSimupace()
        traj = AgentTrajectory(
            ts=np.arange(0, plan_horizon+1) * plan_pace + t0,
            states=np.vstack([xx, yy, hh, vv]).transpose(1,0),
            inputs=np.vstack([ax, dd]).transpose(1,0),
            tm=tm+t0,
            tm_margin=(tm_min+t0, tm_max+t0)
        )
        return traj


