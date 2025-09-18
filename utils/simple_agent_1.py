import sys
from dataclasses import dataclass
import random
import time
import numpy as np
from numpy import ndarray
import scipy
from typing import Dict, Tuple, Optional, Union
from math import cos, sin, tan, acos, asin, atan, atan2, sqrt, exp, log2
import datci


@dataclass
class Agent:
    id: str = None
    x: float = 0.0
    y: float = 0.0
    h: float = 0.0
    v: float = 0.0
    a: float = 0.0
    w: float = 0.0
    length: float = 4.0
    width: float  = 2.0
    label: str    = None

    def moveTo(self, x, y, h, v, Ts):
        self.a = (v - self.v) / Ts
        self.w = asin(sin(h - self.h)) / Ts * 2.8 / v
        self.x, self.y, self.h, self.v = x, y, h, v

@dataclass
class AgentTrajectory:
    ts: ndarray = None
    states: ndarray = None #[x, y, h, v]
    inputs: ndarray = None #[acc, dfw]

    tm: float = None
    tm_margin: Tuple = (0, 0)
        
def get_approximated_bbox(agt: Agent) -> np.ndarray:
    rotateMat = np.array([
        [cos(agt.h), -sin(agt.h)],
        [sin(agt.h),  cos(agt.h)]
    ], float)
    vertexes = np.array([
        [[ agt.length/2], [ agt.width/2]],
        [[ agt.length/2], [-agt.width/2]],
        [[-agt.length/2], [-agt.width/2]],
        [[-agt.length/2], [ agt.width/2]]
    ], float)
    rotVertexes = np.array([np.dot(rotateMat, vex) for vex in vertexes], float)
    agt_vecs = np.array([[agt.x + rv[0], agt.y + rv[1]] for rv in rotVertexes], float)
    agt_bbox = np.array([
        np.min(agt_vecs[:,0]), 
        np.min(agt_vecs[:,1]), 
        np.max(agt_vecs[:,0]), 
        np.max(agt_vecs[:,1])
    ], float) # [min_x, min_y, max_x, max_y]

    return agt_bbox

def calc_agent_risk(agt1: Agent, agt2: Agent) -> float:
    
    if agt1.x >= agt2.x:
        VF, VR = agt1, agt2
    else:
        VF, VR = agt2, agt1
    
    VF_bbox = get_approximated_bbox(VF)
    VF_min_x, VF_min_y, VF_max_x, VF_max_y = VF_bbox
    VF_vx = VF.v * np.cos(VF.h)

    VR_bbox = get_approximated_bbox(VR)
    VR_min_x, VR_min_y, VR_max_x, VR_max_y = VR_bbox
    VR_vx = VR.v * np.cos(VR.h)

    if VR_min_y > VF_max_y:
        return 0.0
    if VR_max_y < VF_min_y:
        return 0.0
    if VR_max_x >= VF_min_x:
        return 1.0
    
    dx = VF_min_x - VR_max_x
    dv = VR_vx - VF_vx
    if dv <= 0:
        TTC = 10.0
    else:
        TTC = min(dx / dv, 10.0)
    if VR_vx <= 0:
        TIV = 2.0
    else:
        TIV = min(dx / VR_vx, 2.0)
    
    if TTC <= 1.0:
        risk_TTC = 1.0
    else:
        risk_TTC = 1.0 - (TTC - 1.0) / 9.0
    if TIV <= 0.5:
        risk_TIV = 1.0
    elif TIV <= 1.0:
        risk_TIV = 1.0 - (TIV - 0.5) / 1.0
    else:
        risk_TIV = 0.5 - (TIV - 1.0) / 2.0

    # EES & gravity
    def G(vx_r, vx_f):
        MR = VR.length * VR.width
        MF = VF.length * VF.width
        EES = 2 * MF / (MR + MF) * (vx_f - vx_r) * 3.6 # km/h
        if EES <= 0:
            return 0.0
        elif EES >= 80.0:
            return 1.0
        else:
            vs = [0,    8,      17,     33,     40,     56,     64,     80]
            gs = [0.0,  0.01,   0.08,   0.35,   0.60,   0.90,   0.99,   1.0]
            for k in range(1,len(vs)):
                if EES > vs[k-1] and EES <= vs[k]:
                    return gs[k-1] + (gs[k] - gs[k-1]) / (vs[k] - vs[k-1]) * (EES - vs[k-1])
    
    # risk = risk_TTC * G(VR_vx, VF_vx) + risk_TIV * max(G(VR_vx, VF_vx), G(VR_vx, VF_vx - 0.4 * 9.81 * TIV))
    risk = 0.5 * risk_TTC +  0.5 * risk_TIV
    return round(risk, 3)

def is_agent_collision(agt1: Agent, agt2: Agent, e:float = 0.0) -> bool:

    if agt1.x >= agt2.x:
        VF, VR = agt1, agt2
    else:
        VF, VR = agt2, agt1

    VF_bbox = get_approximated_bbox(VF)
    VF_min_x, VF_min_y, VF_max_x, VF_max_y = VF_bbox

    VR_bbox = get_approximated_bbox(VR)
    VR_min_x, VR_min_y, VR_max_x, VR_max_y = VR_bbox

    if VR_max_x < VF_min_x - e:
        return False
    if VR_min_y > VF_max_y + e:
        return False
    if VR_max_y < VF_min_y - e:
        return False

    return True

def get_agent_from_daci(id, label=None) -> Agent:
    x, y = datci.vehicle.getPosition(id)
    h = datci.angle2yaw(datci.vehicle.getAngle(id))
    v = datci.vehicle.getSpeed(id)
    a = datci.vehicle.getAccel(id)
    w = 0.0
    vL, vW = datci.vehicle.getVehicleShape(id)
    cx = x - vL/2 * cos(h)
    cy = y - vL/2 * sin(h)
    return Agent(id, cx, cy, h, v, a, w, vL, vW, label)