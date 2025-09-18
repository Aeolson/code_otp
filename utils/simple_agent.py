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

""" Bbox with rotation """
def get_rotated_bbox2(agt: Agent, ee=0) -> np.ndarray:
    rot_mat = np.array([
        [cos(agt.h), -sin(agt.h)],
        [sin(agt.h),  cos(agt.h)]
    ], float)
    l_, w_ = agt.length/2 + ee, agt.width/2 + ee
    vecs = np.array([
        [l_, -l_, -l_,   l_ ],
        [w_,  w_, -w_,  -w_ ],
    ], float)
    rot_vecs = np.dot(rot_mat, vecs).transpose(1,0) # (4, 2)
    agt_vecs = np.array([agt.x, agt.y], dtype=float) + rot_vecs
    return agt_vecs

def get_rotated_bbox(agt: Agent, ee=0) -> np.ndarray:
    rotateMat = np.array([
        [cos(agt.h), -sin(agt.h)],
        [sin(agt.h),  cos(agt.h)]
    ], float)

    l_, w_ = agt.length/2 + ee, agt.width/2 +ee
    vertexes = np.array([
        [[ l_], [ w_]],
        [[ l_], [-w_]],
        [[-l_], [-w_]],
        [[-l_], [ w_]]
    ], float)
    rotVertexes = np.array([np.dot(rotateMat, vex) for vex in vertexes], float)
    agt_vecs = np.array([[agt.x + rv[0], agt.y + rv[1]] for rv in rotVertexes], float)
    return agt_vecs

""" Approximated bbox of the rotated rectangle """
def get_approximated_bbox(agt: Agent, ee=0) -> np.ndarray:
    agt_vecs = get_rotated_bbox(agt, ee) # (4,2)
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
        ttc = 100.0
    else:
        ttc = min(dx / dv, 100.0)
    if VR_vx <= 0:
        thw = 100.0
    else:
        thw = min(dx / VR_vx, 100.0)
    
    if ttc <= 1.0:
        risk_ttc = 1.0
    elif ttc <= 10.0:
        risk_ttc = 1.0 - (ttc - 1.0) / 9.0
    else:
        risk_ttc = 0.0

    if thw <= 0.5:
        risk_thw = 1.0
    elif thw <= 1.0:
        risk_thw = 1.0 - (thw - 0.5) / 1.0
    elif thw <= 2.0:
        risk_thw = 0.5 - (thw - 1.0) / 2.0
    else:
        risk_thw = 0.0

    risk = 0.5 * risk_ttc +  0.5 * risk_thw
    return round(risk, 3)

def is_agent_collision(agt1: Agent, agt2: Agent, sat=True, ee=0) -> bool:

    # if set sat = True, using SAT check, otherwise, using AABB check
    flag = aabb_check(agt1, agt2, ee)
    if flag and sat:
        return sat_check(agt1, agt2, ee)
    else:
        return flag

# AABB check: only using the approximated bbox to check collison
def aabb_check2(agt1:np.ndarray, agt2:np.ndarray, ee=0) -> bool:
    min_x1, min_y1, max_x1, max_y1 = get_approximated_bbox(agt1, ee)
    min_x2, min_y2, max_x2, max_y2 = get_approximated_bbox(agt2, ee)
    flag = (min_x1 <= max_x2) * (max_x1 >= min_x2) * (min_y1 <= max_y2) * (max_y1 >= min_y2)
    return flag

def aabb_check(agt1:np.ndarray, agt2:np.ndarray, ee=0) -> bool:

    if agt1.x >= agt2.x:
        VF, VR = agt1, agt2
    else:
        VF, VR = agt2, agt1

    VF_bbox = get_approximated_bbox(VF, ee)
    VF_min_x, VF_min_y, VF_max_x, VF_max_y = VF_bbox

    VR_bbox = get_approximated_bbox(VR, ee)
    VR_min_x, VR_min_y, VR_max_x, VR_max_y = VR_bbox

    if VR_max_x < VF_min_x:
        return False
    if VR_min_y > VF_max_y:
        return False
    if VR_max_y < VF_min_y:
        return False

    return True

# SAT check: using the rotated bbox to check collison
def sat_check(agt1:np.ndarray, agt2:np.ndarray, ee=0) -> bool:
    v1 = get_rotated_bbox(agt1, ee)
    v2 = get_rotated_bbox(agt2, ee)

    axes = []
    for v_ in [v1, v2]:
        for i in range(len(v_)):
            dx, dy = v_[(i+1)%len(v_)] - v_[i]
            axes.append([dx, dy])
    
    axes = np.array(axes, float)
    angs = np.arctan2(axes[:,1], axes[:,0]) + np.pi/2
    unit_x = np.cos(angs)
    unit_y = np.sin(angs)
    for ux_, uy_ in zip(unit_x, unit_y):
        p1 = ux_ * v1[:,0] + uy_ * v1[:,1]
        p2 = ux_ * v2[:,0] + uy_ * v2[:,1]
        if p1.min() > p2.max() or p1.max() < p2.min():
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