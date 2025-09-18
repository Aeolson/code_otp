import sys, os
import pandas as pd
import numpy as np
from math import floor, ceil
from rich import print
import pickle
from dataclasses import dataclass
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Union, Tuple, Any
from .network import Network

@dataclass
class StateData:
    t: int = None           # timesteps, same to SUMO
    x: float = None         # x-position of the front-center on vehicle [m], same to SUMO
    y: float = None         # y-position of the front-center on vehicle [m], same to SUMO
    angle: float = None     # heading angle of the vehicle [deg], +Y is 0 deg and +X is 90 deg, same to SUMO
    speed: float = None     # speed of the vehicle [m/s], same to SUMO
    accel: float = None     # acceleration(+) or deceleration(-) of the vehicle [m/s2], same to SUMO
    s: float = None         # s-length of the vehicle along the lane centerline [m], same to SUMO
    edgeIdx: int = None     # index the edge in routes [int], same to SUMO
    laneID:  str = None     # lane ID [str], same to SUMO

@dataclass
class DatasetItem:
    id: str = None
    isControllable = False
    vType: str = None
    length: float = 5.0
    width: float = 1.8
    maxAccel: float = 3.0
    maxDecel: float = 4.5
    maxSpeed: float = 13.89
    minTimestep: int = 0
    maxTimestep: int = 0
    routes: List[str] = None            # all edges used within the route
    states: Dict[int, StateData] = None # dict of state sequence, 'key' is the timestep

def sema(x:np.ndarray, delta=5) -> np.ndarray:
    bar_x = deepcopy(x)
    for k in range(len(x)):
        idx_s = max(k-3*delta, 0)
        idx_e = min(k+3*delta+1, len(x))
        r_ = np.linspace(idx_s, idx_e, idx_e-idx_s, endpoint=False) - k
        r_ = np.exp(-np.abs(r_)/delta)

        bar_x[k] = np.sum(x[idx_s:idx_e]*r_) / np.sum(r_)
    return bar_x

def yaw2angle(yaw: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    # yaw:    rad, (-pi, pi], +X is 0 rad and +Y is +pi/2 rad
    # angle:  deg, [0, 360),  +Y is 0 deg and +X is 90 deg
    h = np.rad2deg(np.arctan2(np.sin(yaw), np.cos(yaw)))
    return np.floor((h-90)/360) * 360 - (h-90) + 360

def angle2yaw(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    # angle:  deg, [0, 360),  +Y is 0 deg and +X is 90 deg
    # yaw:    rad, (-pi, pi], +X is 0 rad and +Y is +pi/2 rad
    h = np.deg2rad(90 - angle)
    return np.arctan2(np.sin(h), np.cos(h))

class DatasetBuild:

    def __init__(self) -> None:
        self.datFile: str = None
        self.netFile: str = None
        self.sampleRate: int = None

        self.roadnet: Network = None
        self.vehicles: Dict[str, DatasetItem] = None
        self.frameVehicleIds: Dict[int, List[str]] = None
        self.allVehicleIds: List[str] = None
        self.allTimesteps: List[int] = None
        self.minTimestep: int = None
        self.maxTimestep: int = None
    
    def create(self, datFile: str, netFile: str, sampleRate: int = 10):
        self.datFile = datFile
        self.netFile = netFile
        self.sampleRate = sampleRate

        self.roadnet = Network()
        self.roadnet.create(netFile)

        self.processNGSIM_i80()
        self.allVehicleIds = list(self.vehicles.keys())
        self.allTimesteps = list(self.frameVehicleIds.keys())
        self.minTimestep = np.min(self.allTimesteps)
        self.maxTimestep = np.max(self.allTimesteps)

        t = np.array(self.allTimesteps, dtype=int)
        if np.any(t[1:] - t[:-1] != 1):
            raise ValueError("[red bold]Error !!! self.allTimesteps is not continuous !!![/red bold]")

        print('[green bold]Created replay dataset at {}.[/green bold]'.format(datetime.now().strftime('%H:%M:%S.%f')[:-3]))

    def processNGSIM_i80(self):
        df = pd.read_csv(self.datFile)
        edgeID = 'i80'
        ft2m = 0.3048
        maxSpeed = df['v_Vel'].values.astype(np.float32).max() * ft2m
        minX = 0.0
        minY = df['Local_Y'].values.astype(np.float32).min() * ft2m
        # modify the start frame = 0
        df['Frame_ID'] = df['Frame_ID'] - df['Frame_ID'].values.min()

        edge = self.roadnet.getEdge(edgeID)
        dict_lane_bounds: Dict[str, Tuple[float, float]] = {}
        max_lane_id, max_lane_y = None, None
        min_lane_id, min_lane_y = None, None
        
        for lid in edge.lanes:
            lane = self.roadnet.getLane(lid)
            l_, r_ = lane.left_bound[0][1], lane.right_bound[0][1]
            dict_lane_bounds[lid] = [l_, r_]

            if max_lane_id is None or max_lane_y < max(l_, r_):
                max_lane_id = lid
                max_lane_y = max(l_, r_)

            if min_lane_id is None or min_lane_y > min(l_, r_):
                min_lane_id = lid
                min_lane_y = min(l_, r_)

        map2vclass = {1: 'Motorcycle', 2: 'Car', 3: 'Truck'}
        grouped_df = df.groupby('Vehicle_ID', sort=True)
        self.vehicles = {}
        self.frameVehicleIds = {}
        # n_count = 100
        for vid, vdf in tqdm(grouped_df, ncols=100):
            # if n_count < 0:
            #     break
            # n_count -= 1

            id = str(vid)
            vtype = map2vclass[int(vdf['v_Class'].values[0])]
            length = float(vdf['v_Length'].values[0]) * ft2m
            width = float(vdf['v_Width'].values[0]) * ft2m
            rt = vdf['Frame_ID'].values.astype(int) * 0.1
            rx = vdf['Local_Y'].values.astype(np.float32) * ft2m - minY
            rx = sema(rx)
            ry = minX - vdf['Local_X'].values.astype(np.float32) * ft2m
            ry = sema(ry)
            rv = vdf['v_Vel'].values.astype(np.float32) * ft2m
            rv = sema(rv)

            # upsample data by interpolation, into 1000 FPS (1ms)
            t = np.linspace(rt.min(), rt.max(), num=round((rt.max()-rt.min()) * 1000)+1).round(3)
            x = np.interp(t, rt, rx).round(3)
            y = np.interp(t, rt, ry).round(3)
            speed = np.interp(t, rt, rv).round(3)
            
            # downsample data into specified sampleRate
            idxes = np.round(t * 1000) % int(1000 / self.sampleRate) == 0
            t = t[idxes]
            x = x[idxes]
            y = y[idxes]
            speed = speed[idxes]

            h = np.zeros_like(x)
            h[:-1] = np.arctan2(np.diff(y), np.diff(x))
            h[-1] = h[-2]
            h[speed < 5.0] = 0.0
            h = sema(h)
            angle = yaw2angle(h)
            accel = np.zeros_like(speed)
            accel[:-1] = np.diff(speed) / (1.0 / self.sampleRate)
            
            s = deepcopy(x)
            routes = [edgeID]
            edgeIdx = [0] * len(t)
            laneID = []
            for ex, ey in zip(x, y):
                if ey >= max_lane_y:
                    laneID.append(max_lane_id)
                elif ey <= min_lane_y:
                    laneID.append(min_lane_id)
                else:
                    for lid, bd in dict_lane_bounds.items():
                        if ey > np.min(bd) and ey <= np.max(bd):
                            laneID.append(lid)
                            break
            
            t = (t * self.sampleRate).astype(int)
            maxAccel = max(3.0,  np.max(accel))
            maxDecel = max(4.5, -np.min(accel))

            self.vehicles[id] = DatasetItem(
                id, vtype, length, width, maxAccel, maxDecel, maxSpeed, t.min(), t.max(),
                routes,
                dict((t[i], StateData(t[i], x[i], y[i], angle[i], speed[i], accel[i], s[i], edgeIdx[i], laneID[i])) for i in range(len(t)))
            )
            
            for key_t in t:
                if key_t not in self.frameVehicleIds.keys():
                    self.frameVehicleIds[key_t] = []
                self.frameVehicleIds[key_t].append(id)
            
            if np.any(t[1:] - t[:-1] != 1):
                raise ValueError("[red bold]Error !!! 't' in vehicle {} is not continuous !!![/red bold]".format(id))
    
        self.frameVehicleIds = dict((k_, self.frameVehicleIds[k_]) for k_ in sorted(self.frameVehicleIds.keys()))

    def checkOvertime(self, vid: str, t: int):
        if t < self.vehicles[vid].minTimestep or t > self.vehicles[vid].maxTimestep:
            return True
        return False

    def checkVehicleId(self, vid):
        if vid not in self.allVehicleIds:
            return False
        return True

    def getFrameVehicleIds(self, t: int):
        if t > self.maxTimestep or t < self.minTimestep:
            return None
        return self.frameVehicleIds[t]

    def getLaneInfosFromPos(self, x: float, y: float, yaw: float = None) -> Tuple[str, str, str, float, float]:
        """
        get the real lane information (edgeID, juncID, laneID, lanePos, laneOff) of vehicle 'vid' at time 't'
        """
        edgeID, juncID, laneID, lanePos, laneOff = None, None, None, None, None
        laneID = self.roadnet.getLaneInfosFromPos(x, y, yaw)
        if laneID:

            if laneID in self.roadnet.lanes.keys():
                lane = self.roadnet.getLane(laneID)
                edgeID = lane.affiliated_edge.id
                lanePos, laneOff = lane.course_spline.cartesian_to_frenet1D(x, y)
                # lanePos = lane.course_spline.find_nearest_rs(x, y)
                

            elif laneID in self.roadnet.junctionLanes.keys():
                lane = self.roadnet.getJunctionLane(laneID)
                juncID = lane.affJunc
                lanePos, laneOff = lane.course_spline.cartesian_to_frenet1D(x, y)
                # lanePos = lane.course_spline.find_nearest_rs(x, y)
                
        return edgeID, juncID, laneID, lanePos, laneOff

    def getStateAttributeAtTime(self, vid: str, attr: Union[str, Tuple[str]], t: int):
        if not self.checkVehicleId(vid):
            raise ValueError('[red blod]Error !!! The vehicle {%s} is not in dataset !!![/red blod]' % (vid))

        if self.checkOvertime(vid, t):
            return None

        if isinstance(attr, Tuple):
            return list( self.vehicles[vid].states[t].__getattribute__(a) for a in attr )
        else:
            return self.vehicles[vid].states[t].__getattribute__(attr)
    
    def getStateAttributeSequence(self, vid: str, attr: Union[str, Tuple[str]], start_t: int, end_t: int):
        if not self.checkVehicleId(vid):
            raise ValueError('[red blod]Error !!! The vehicle {%s} is not in dataset !!![/red blod]' % (vid))
        
        if start_t < self.vehicles[vid].minTimestep:
            return None
        
        if end_t > self.vehicles[vid].maxTimestep:
            end_t = self.vehicles[vid].maxTimestep + 1
        
        if isinstance(attr, Tuple):
            return list( list(self.vehicles[vid].states[t].__getattribute__(a) for a in attr) for t in range(start_t, end_t) )
        else:
            return list( self.vehicles[vid].states[t].__getattribute__(attr) for t in range(start_t, end_t) )

    def setStateAttributeAtTime(self, vid: str, attr: Union[str, Tuple[str]], val: Union[Any, Tuple[Any]], t: int) -> bool:
        if not self.checkVehicleId(vid):
            raise ValueError('[red blod]Error !!! The vehicle {%s} is not in dataset !!![/red blod]' % (vid))

        if self.checkOvertime(vid, t):
            return False
        
        if not self.vehicles[vid].isControllable:
            print("[yellow bold]The vehicle {} is not controllable.[/yellow bold]".format(vid))
            return False

        if isinstance(attr, Tuple):
            for a, v in zip(attr, val):
                self.vehicles[vid].states[t].__setattr__(a, v)
        else:
            self.vehicles[vid].states[t].__setattr__(attr, val)
        return True
    
    def setStateAttributeSequence(self, vid: str, attr: Union[str, Tuple[str]], val: Union[Any, Tuple[Any]], start_t: int, end_t: int):
        if not self.checkVehicleId(vid):
            raise ValueError('[red blod]Error !!! The vehicle {%s} is not in dataset !!![/red blod]' % (vid))
        
        if start_t < self.vehicles[vid].minTimestep:
            raise ValueError('[red blod]Error !!! The state time is {%d} smaller than the minTimestep of vehicle {%s} : {%d} !!![/red blod]' % (start_t, vid, self.vehicles[vid].minTimestep))
        
        if not self.vehicles[vid].isControllable:
            print("[yellow bold]The vehicle {} is not controllable.[/yellow bold]".format(vid))
            return False
        
        if end_t > self.vehicles[vid].maxTimestep:
            end_t = self.vehicles[vid].maxTimestep + 1
        
        if isinstance(attr, Tuple):
            for t in range(start_t, end_t):
                if self.checkOvertime(vid, t):
                    continue
                for a, v in zip(attr, val):
                    self.vehicles[vid].states[t].__setattr__(a, v)
        else:
            for t in range(start_t, end_t):
                if self.checkOvertime(vid, t):
                    continue
                self.vehicles[vid].states[t].__setattr__(attr, val)
        return True

    def getStatistics2(self):
        import matplotlib.pyplot as plt
        # 设置字体为 中文宋体 + 英文Times New Roman + latex
        from matplotlib import font_manager
        from matplotlib import rcParams
        font_path = "./times+simsun.ttf" # 加载字体
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        rcParams['font.family'] = 'sans-serif' # 使用字体中的无衬线体
        rcParams['font.sans-serif'] = prop.get_name()  # 根据名称设置字体
        rcParams['font.size'] = 10 # 设置字体大小
        rcParams['axes.unicode_minus'] = False # 使坐标轴刻度标签正常显示正负号
        rcParams['mathtext.fontset'] = 'cm' # latex 公式字体

        # accel distribution
        sta_aa, sta_ax, sta_ay = [], [], []
        for veh in self.vehicles.values():
            if veh.vType != 'Car':
                continue
            
            aa = np.array([st.accel for st in veh.states.values()])
            hh = angle2yaw(np.array([st.angle for st in veh.states.values()]))
            ax = aa * np.cos(hh)
            ay = aa * np.sin(hh)

            if np.max(aa) > 3.0 or np.min(aa) < -4.5:
                continue

            sta_aa.append(aa)
            sta_ax.append(ax)
            sta_ay.append(ay)

        
        sta_aa = np.hstack(sta_aa).reshape(-1)
        mu_aa, sigma_aa = np.mean(sta_aa), np.std(sta_aa, ddof=1)

        sta_ax = np.hstack(sta_ax).reshape(-1)
        mu_ax, sigma_ax = np.mean(sta_ax), np.std(sta_ax, ddof=1)

        sta_ay = np.hstack(sta_ay).reshape(-1)
        mu_ay, sigma_ay = np.mean(sta_ay), np.std(sta_ay, ddof=1)

        print("distribution \[aa] : mu = %.2f, sigma = %.3f" % (mu_aa, sigma_aa))
        print("distribution \[ax] : mu = %.2f, sigma = %.3f" % (mu_ax, sigma_ax))
        print("distribution \[ay] : mu = %.2f, sigma = %.3f" % (mu_ay, sigma_ay))


        plt.figure(1, figsize=(3.0,2.0), dpi=200)
        freqs, bins = np.histogram(sta_ax, bins=50)
        plt.hist(sta_ax, bins=bins, density=True, stacked=True, alpha=0.5)

        min_ax, max_ax = mu_ax - 4*sigma_ax, mu_ax + 4*sigma_ax
        ax = np.linspace(min_ax, max_ax, 1001)
        px = 1 / np.sqrt(2*np.pi) / sigma_ax * np.exp( -(ax - mu_ax)**2 / sigma_ax**2 / 2 )
        plt.plot(ax, px, c='r', linestyle='-', linewidth=2.0)
        plt.plot([mu_ax-sigma_ax,mu_ax-sigma_ax], [0, 2], c='gray', ls='--', lw=1.0)
        plt.plot([mu_ax+sigma_ax,mu_ax+sigma_ax], [0, 2], c='gray', ls='--', lw=1.0)
        plt.text(mu_ax+sigma_ax+0.1, 0.6, r'$\sigma$ = %.3f'%(sigma_ax))
        plt.xlabel(r"Longitudinal Acceleration $\omega^x  \rm [m/s^2]$")
        plt.xlim([-2, 2])
        plt.ylabel("Density")
        plt.ylim([0, 0.8])

        plt.subplots_adjust(left=0.16, bottom=0.227, right=0.971, top=0.971, wspace=0.265, hspace=0.738)
        plt.savefig("fig_rs_ax.png",dpi=600)

        plt.figure(2, figsize=(3.0,2.0), dpi=200)
        freqs, bins = np.histogram(sta_ay, bins=15, range=(-0.05, 0.05))
        plt.hist(sta_ay, bins=bins, range=(-0.05, 0.05), density=True, stacked=True, alpha=0.5)

        min_ay, max_ay = mu_ay - 4*sigma_ay, mu_ay + 4*sigma_ay
        ay = np.linspace(min_ay, max_ay, 1001)
        py = 1 / np.sqrt(2*np.pi) / sigma_ay * np.exp( -(ay - mu_ay)**2 / sigma_ay**2 / 2 )
        plt.plot(ay, py, c='r', linestyle='-', linewidth=2.0)
        plt.plot([mu_ay-sigma_ay,mu_ay-sigma_ay], [0, 50], c='gray', ls='--', lw=1.0)
        plt.plot([mu_ay+sigma_ay,mu_ay+sigma_ay], [0, 50], c='gray', ls='--', lw=1.0)
        plt.text(mu_ay+sigma_ay+0.005, 30, r'$\sigma$ = %.3f'%(sigma_ay))
        plt.xlabel(r"Lateral Acceleration $\omega^y  \rm [m/s^2]$")
        plt.xlim([-0.05, 0.05])
        plt.ylabel("Density")
        plt.ylim([0, 35])

        plt.subplots_adjust(left=0.16, bottom=0.227, right=0.971, top=0.971, wspace=0.265, hspace=0.738)
        plt.savefig("fig_rs_ay.png",dpi=600)

        
        plt.show()
        sys.exit()

    def getStatistics_Acceleration(self):
        # accel distribution
        sta_aa, sta_ax, sta_ay = [], [], []
        for veh in self.vehicles.values():
            if veh.vType != 'Car':
                continue
            
            aa = np.array([st.accel for st in veh.states.values()])
            hh = angle2yaw(np.array([st.angle for st in veh.states.values()]))
            ax = aa * np.cos(hh)
            ay = aa * np.sin(hh)

            if np.max(aa) > 3.0 or np.min(aa) < -4.5:
                continue

            sta_aa.append(aa)
            sta_ax.append(ax)
            sta_ay.append(ay)

        
        sta_aa = np.hstack(sta_aa).reshape(-1)
        sta_ax = np.hstack(sta_ax).reshape(-1)
        sta_ay = np.hstack(sta_ay).reshape(-1)

        return sta_aa, sta_ax, sta_ay