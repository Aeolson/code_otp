import os, sys
import numpy as np
from tqdm import tqdm
import random
from rich import print
from copy import deepcopy
import time
from scipy import interpolate
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pickle
import datci
from typing import Dict, Tuple, Optional, List
from math import cos, sin, tan, acos, asin, atan, atan2, sqrt, exp, log2
import dearpygui.dearpygui as dpg
import datci.globalvar
from utils.simple_agent import Agent, AgentTrajectory, calc_agent_risk, get_agent_from_daci, is_agent_collision, get_approximated_bbox
from configs import *
from gui import userGUI
from planner import *
from dlplann import LaneChangePlanner_DLP

class DatasetSim:

    def __init__(self, ds) -> None:
        self.use_ds = ds

        self.ego: Agent = None
        self.clear_all_qlist()

        self.gui = userGUI()
        self.lane_change_infos = []

        datci.createReplayDataset(ds, resampleRate=round(1.0/simu_pace))
        datci.setSimupace(simu_pace)
        self.lane_change_infos = self.load_lane_change_infos(ds)

        self.replay_id: int = None
        self.replay_scenario: dict = None
    
    def load_lane_change_infos(self, ds):
        lane_change_infos: list[dict] = []
        LC_INFO_FILE = "./LC_INFOS/LC_INFO_%s.txt" % (ds['name'])
        with open(LC_INFO_FILE, 'r') as fr:
            ls = fr.readlines()
            for l in ls:
                l = l.replace('\n','')
                lc = {}
                for s in l.split(','):
                    k_, v_ = s.split(':')
                    if k_ in ['t_cross', 't_start', 't_end']:
                        v_ = int(v_)
                    if k_ in ['lane_from_centery', 'lane_from_width', 'lane_to_centery', 'lane_to_width']:
                        v_ = float(v_)
                    if v_ == 'None':
                        v_ = None
                    lc[k_] = v_
                lane_change_infos.append(lc)
        
        return lane_change_infos

    def initilze_scenario(self, rp_id):
        if rp_id == 'random':
            rp_id = random.choice(self.lane_change_infos)

        self.replay_id = rp_id
        self.replay_scenario = self.lane_change_infos[rp_id]

        self.reset_scenario()
        self.clear_all_qlist()

    def reset_scenario(self):
        datci.setTimestep(self.replay_scenario['t_start'])
        self.ego = get_agent_from_daci(self.replay_scenario['ego'], label='veh_e')

        self.surrVehicls = []
        allVehicleIds = datci.getFrameVehicleIds()
        for vid in allVehicleIds:
            self.surrVehicls.append(get_agent_from_daci(vid))
        
        self.is_plan_success = False
        self.is_ego_collision = False
    
    def clear_all_qlist(self):
        self.qlist_timestep :   List[int] = []
        self.qlist_ego :        List[Agent] = []
        self.qlist_ego_lane:    List[str] = []
        self.qlist_traj :       List[AgentTrajectory] = []
        self.qlist_risk :       List[float] = []

        self.qlist_veh_p:       List[Agent] = []
        self.qlist_veh_b:       List[Agent] = []
        self.qlist_veh_f:       List[Agent] = []
        self.qlist_veh_r:       List[Agent] = []

        self.qlist_timecost:    List[float] = []
        self.qlist_planflag:    List[int]   = []

    def append_qlist(self, traj:AgentTrajectory, time_cost:float, plan_flag:int):
        risk_f, risk_r = self.get_risk()
        self.qlist_risk.append(deepcopy([risk_f, risk_r]))
        # return
    
        ego_lane = datci.vehicle.getLaneIDFromPos(self.ego.x, self.ego.y)

        veh_p: Agent = None
        veh_b: Agent = None
        veh_f: Agent = None
        veh_r: Agent = None
        all_vehicles = datci.getFrameVehicleIds()
        for Vid in all_vehicles:
            if Vid == self.ego.id:
                continue
            Vlane = datci.vehicle.getLaneID(Vid)
            if Vlane == self.replay_scenario['lane_from']:
                Vx, Vy = datci.vehicle.getPosition(Vid)
                if Vx >= self.ego.x:
                    if veh_p is None or veh_p.x > Vx:
                        veh_p = get_agent_from_daci(Vid)
                else:
                    if veh_b is None or veh_b.x < Vx:
                        veh_b = get_agent_from_daci(Vid)
            elif Vlane == self.replay_scenario['lane_to']:
                Vx, Vy = datci.vehicle.getPosition(Vid)
                if Vx >= self.ego.x:
                    if veh_f is None or veh_f.x > Vx:
                        veh_f = get_agent_from_daci(Vid)
                else:
                    if veh_r is None or veh_r.x < Vx:
                        veh_r = get_agent_from_daci(Vid)

        self.qlist_timestep.append(datci.getTimestep())
        self.qlist_ego.append(deepcopy(self.ego))
        self.qlist_ego_lane.append(deepcopy(ego_lane))
        self.qlist_traj.append(traj)

        self.qlist_veh_p.append(deepcopy(veh_p))
        self.qlist_veh_b.append(deepcopy(veh_b))
        self.qlist_veh_f.append(deepcopy(veh_f))
        self.qlist_veh_r.append(deepcopy(veh_r))

        self.qlist_timecost.append(deepcopy(time_cost))
        self.qlist_planflag.append(deepcopy(plan_flag))

    def run_data(self, show:bool = False):
        self.clear_all_qlist()
        self.reset_scenario()
        self.plannerLK = None
        self.plannerLC = None

        if show:
            self.gui.start(self.replay_scenario['lane_from'], self.replay_scenario['lane_to'])

        for t_now in tqdm(range(self.replay_scenario['t_start'], self.replay_scenario['t_end']), ncols=100, desc='dat'):
            datci.setTimestep(t_now)
            self.update_surrounding_vehicles()

            lane_now = datci.vehicle.getLaneIDFromPos(self.ego.x, self.ego.y, self.ego.h)

            if lane_now == self.replay_scenario['lane_to']:
                self.is_plan_success = True


            traj = self.get_trajectory_from_dataset()
            self.update_ego(traj)
            self.append_qlist(traj, 0.0, 0)

            # check collision
            for sve in self.surrVehicls:
                if is_agent_collision(self.ego, sve):
                    self.is_ego_collision = True
                    break

            if show:
                self.update_gui(traj, 'data', time_sleep=0.1)

        self.gui.destroy()

    def run_plan(self, planner:str, show:bool = False):
        self.clear_all_qlist()
        self.reset_scenario()
        self.plannerLK = LaneKeepPlanner()
        
        if planner == 'otp':
            self.plannerLC = LaneChangePlanner_OTP()
        elif planner == 'otp_nlp':
            self.plannerLC = LaneChangePlanner_OTP_NLP()
        elif planner == 'otp_wor':
            self.plannerLC = LaneChangePlanner_OTP_WOR()
        
        elif planner == 'sbf':
            self.plannerLC = LaneChangePlanner_SBF()
        elif planner == 'rap':
            self.plannerLC = LaneChangePlanner_RAP()
        elif planner == 'gdp':
            self.plannerLC = LaneChangePlanner_GDP()
        elif planner == 'dlp':
            self.plannerLC = LaneChangePlanner_DLP('aslstm.pt')
        
        else:
            raise ValueError("Invalid Planner !!!")

        if show:
            self.gui.start(self.replay_scenario['lane_from'], self.replay_scenario['lane_to'])

        replan_time = None
        plan_traj = None
        replan_frequence = 5
        for t_now in tqdm(range(self.replay_scenario['t_start'], self.replay_scenario['t_end']), ncols=100, desc=planner):
            datci.setTimestep(t_now)
            self.update_surrounding_vehicles()
            lane_now = datci.vehicle.getLaneIDFromPos(self.ego.x, self.ego.y, self.ego.h)

            if lane_now == self.replay_scenario['lane_to'] and plan_traj is not None:
                self.is_plan_success = True

            
            tc1 = time.time()

            plan_flag = 0 # 0: no plan; 1: plan LC plan;2: replan LC; 3: replan LK
            if plan_traj is None: # try at each step before the first successful planning
                if lane_now == self.replay_scenario['lane_from']:
                    plan_flag = 1
            else: # replan with a fixed frequence
                if t_now - replan_time >= replan_frequence:
                    plan_flag = 2 if lane_now == self.replay_scenario['lane_from'] else 3

            if plan_flag == 1: # plan LC
                try:
                    traj_ = self.plannerLC.run(self.ego, self.replay_scenario['lane_from'], self.replay_scenario['lane_to'])
                except:
                    traj_ = None
                if traj_ is not None:
                    plan_traj = traj_
                    replan_time = t_now

            elif plan_flag == 2: # replan LC
                try:
                    replan_traj_ = self.plannerLC.run_replan(self.ego, self.replay_scenario['lane_from'], self.replay_scenario['lane_to'])
                except:
                    replan_traj_ = None
                if replan_traj_ is not None:
                    plan_traj = replan_traj_
                    replan_time = t_now

            elif plan_flag == 3: # plan LK
                plan_traj = self.plannerLK.run(self.ego, self.replay_scenario['lane_to'])
                replan_time = t_now
            
            if plan_traj is None:
                traj_type = 'data'
                use_traj = self.get_trajectory_from_dataset()
            else:
                traj_type = 'plan'
                use_traj = self.adjust_trajectory(plan_traj)
            
            tc2 = time.time()

            self.update_ego(use_traj)
            self.append_qlist(use_traj, tc2-tc1, plan_flag)

            # check collision
            for sve in self.surrVehicls:
                if is_agent_collision(self.ego, sve):
                    self.is_ego_collision = True
                    break

            if show:
                self.update_gui(use_traj, traj_type, time_sleep=0.02)

        self.gui.destroy()

    def run_replay(self, qlist_timestep:list[int], qlist_ego:list[Agent], qlist_traj:list[AgentTrajectory]=None, 
                   show:bool=True, traj_type=None, sleep_time=0.01):
        self.reset_scenario()

        if show:
            self.gui.start(self.replay_scenario['lane_from'], self.replay_scenario['lane_to'])
            self.gui.is_running = False

        k = 0
        while k < len(qlist_timestep):
            t_now = qlist_timestep[k]
            datci.setTimestep(t_now)

            self.update_surrounding_vehicles()
            traj = qlist_traj[k] if qlist_traj is not None else None
            tego = qlist_ego[k]
            self.ego.moveTo(tego.x, tego.y, tego.h, tego.v, datci.getSimupace())
            risk_f, risk_r = self.get_risk()

            if show:
                if dpg.is_dearpygui_running():
                    info = "Risk: F = %.3f, R = %.3f" % (risk_f, risk_r)
                    veh_p, veh_b, veh_f, veh_r = self.get_vehicles_pbfr()
                    for sv in self.surrVehicls:
                        if veh_p is not None and sv.id == veh_p.id:
                            sv.label='veh_p'
                        if veh_b is not None and sv.id == veh_b.id:
                            sv.label='veh_b'
                        if veh_f is not None and sv.id == veh_f.id:
                            sv.label='veh_f'
                        if veh_r is not None and sv.id == veh_r.id:
                            sv.label='veh_r'
                    self.gui.render(datci.getTimestep()*simu_pace, self.ego, self.surrVehicls, traj, info, traj_type=traj_type)
                else:
                    self.gui.destroy()
                    sys.exit()
                
                time.sleep(sleep_time)
            
            if self.gui.is_running:
                k += 1
            elif self.gui.incre_frame:
                k += 1

        self.gui.destroy()

    def update_surrounding_vehicles(self):
        self.surrVehicls: list[Agent] = []
        allVehicleIds = datci.getFrameVehicleIds()
        for vid in allVehicleIds:
            if vid != self.ego.id:
                self.surrVehicls.append(get_agent_from_daci(vid))

    def update_ego(self, traj:AgentTrajectory):
        ts = traj.ts
        now_t = datci.getTimestep() * datci.getSimupace()
        t_idx = np.argmin(np.abs(ts - now_t))
        if t_idx >= len(traj.inputs)-1:
            x, y, h, v = traj.states[-1]
            acc, dfw = 0.0, 0.0
        else:
            x, y, h, v = traj.states[t_idx+1]
            acc, dfw = traj.inputs[t_idx+1]
        
        self.ego.moveTo(x, y, h, v, datci.getSimupace())

    def update_gui(self, traj:AgentTrajectory, traj_type:str, time_sleep=0.1):
        if self.plannerLC is not None and self.plannerLK is not None:
            for veh in self.surrVehicls:
                if self.plannerLC.veh_p is not None and veh.id == self.plannerLC.veh_p.id:
                    veh.label = 'veh_p'

                elif self.plannerLC.veh_r is not None and veh.id == self.plannerLC.veh_r.id:
                    veh.label = 'veh_r'

                elif self.plannerLC.veh_f is not None and veh.id == self.plannerLC.veh_f.id:
                    veh.label = 'veh_f'
                
                elif self.plannerLK.veh_f is not None and veh.id == self.plannerLK.veh_f.id:
                    veh.label = 'veh_f'

                elif self.plannerLK.veh_r is not None and veh.id == self.plannerLK.veh_r.id:
                    veh.label = 'veh_r'


        risk_f, risk_r = self.get_risk()
        if dpg.is_dearpygui_running():
            info = "Risk: F = %.3f, R = %.3f" % (risk_f, risk_r)
            self.gui.render(datci.getTimestep()*simu_pace, self.ego, self.surrVehicls, traj, info, traj_type)
        else:
            self.gui.destroy()
            sys.exit() 
        
        time.sleep(time_sleep)

    def get_trajectory_from_dataset(self):
        horizon = round((plan_horizon * plan_pace) / simu_pace)
        stSeq = datci.vehicle.getStateSequences(self.ego.id, attr=('x', 'y', 'angle', 'speed'), horizon=horizon)
        x, y, h, v = np.array(stSeq).transpose(1,0)
        h = datci.angle2yaw(h)

        x = x - self.ego.length/2 * np.cos(h)
        y = y - self.ego.length/2 * np.sin(h)

        t0 = datci.getTimestep() * simu_pace
        te = t0 + (len(x)-1) * simu_pace
        ts = np.linspace(t0, te, len(x))
        st = np.array([x,y,h,v],float).transpose(1,0)
        ip = np.array([np.diff(v)/simu_pace, np.diff(h)/simu_pace], float).transpose(1,0)
        traj = AgentTrajectory(ts=ts, states=st, inputs=ip)
        return traj

    def get_risk(self) -> Tuple[float, float]:
        if len(self.surrVehicls) == 0:
            return 0.0
        
        risk_f, risk_r = 0.0, 0.0
        for sV in self.surrVehicls:
            if sV.x > self.ego.x:
                risk_f = max(risk_f, calc_agent_risk(self.ego, sV))
            else:
                risk_r = max(risk_r, calc_agent_risk(self.ego, sV))

        now_lane = datci.vehicle.getLaneIDFromPos(self.ego.x, self.ego.y, self.ego.h)
        if now_lane == self.replay_scenario['lane_from']:
            risk_r = 0.0
            
        return risk_f, risk_r

    def get_vehicles_pbfr(self) -> Tuple[Agent, Agent, Agent, Agent]:
        veh_p: Agent = None
        veh_b: Agent = None
        veh_f: Agent = None
        veh_r: Agent = None
        all_vehicles = datci.getFrameVehicleIds()
        for Vid in all_vehicles:
            if Vid == self.ego.id:
                continue
            Vlane = datci.vehicle.getLaneID(Vid)
            if Vlane == self.replay_scenario['lane_from']:
                Vx, Vy = datci.vehicle.getPosition(Vid)
                if Vx >= self.ego.x:
                    if veh_p is None or veh_p.x > Vx:
                        veh_p = get_agent_from_daci(Vid)
                else:
                    if veh_b is None or veh_b.x < Vx:
                        veh_b = get_agent_from_daci(Vid)
            elif Vlane == self.replay_scenario['lane_to']:
                Vx, Vy = datci.vehicle.getPosition(Vid)
                if Vx >= self.ego.x:
                    if veh_f is None or veh_f.x > Vx:
                        veh_f = get_agent_from_daci(Vid)
                else:
                    if veh_r is None or veh_r.x < Vx:
                        veh_r = get_agent_from_daci(Vid)

        return veh_p, veh_b, veh_f, veh_r

    def adjust_trajectory(self, org_traj:AgentTrajectory) -> AgentTrajectory:
        now_t = round(datci.getTimestep() * datci.getSimupace(), 3)
        now_idx = np.argmin(np.abs(org_traj.ts - now_t))
        ts = org_traj.ts[now_idx:]
        st = org_traj.states[now_idx:]

        if len(ts) == 1:
            return AgentTrajectory(
                ts = now_t + np.arange(plan_horizon+1, dtype=float),
                states = np.repeat(st, plan_horizon+1, axis=0),
                inputs = np.zeros((plan_horizon, 2), dtype=float)
            )
        
        ex, ey, eh, ev = st[1]
        new_st1 = deepcopy(st[1:2])
        agt = Agent('test_ego', ex, ey, eh, ev, length=self.ego.length, width=self.ego.width)
        agt_min_x, agt_min_y, agt_max_x, agt_max_y = get_approximated_bbox(agt)
        for sve in self.surrVehicls:
            sve_min_x, sve_min_y, sve_max_x, sve_max_y = get_approximated_bbox(sve)
            if sve_min_y > agt_max_y:
                continue
            if sve_max_y < agt_min_y:
                continue
            if sve_min_x + sve.v * simu_pace > agt_max_x + max(agt.v * 0.2, 2.0):
                continue
            if sve_max_x + sve.v * simu_pace < agt_min_x - max(agt.v * 0.2, 2.0):
                continue

            # handle collision
            if sve.x >= agt.x:
                new_ev = max(self.ego.v + simu_pace * a_min, v_min)
                new_eh = self.ego.h
                new_ex = self.ego.x + simu_pace * (self.ego.v + new_ev) / 2 * cos(new_eh)
                new_ey = self.ego.y + simu_pace * (self.ego.v + new_ev) / 2 * sin(new_eh)
            else:
                new_ev = min(self.ego.v + simu_pace * a_max, v_max)
                new_eh = self.ego.h
                new_ex = self.ego.x + simu_pace * (self.ego.v + new_ev) / 2 * cos(new_eh)
                new_ey = self.ego.y + simu_pace * (self.ego.v + new_ev) / 2 * sin(new_eh)
            
            new_st1 = np.array([[new_ex, new_ey, new_eh, new_ev]], float)
            break
        
        new_st = st - st[1:2] + new_st1
        new_ip = np.array([np.diff(new_st[:,3])/simu_pace, np.diff(new_st[:,2])/simu_pace], float).transpose(1,0)
        for k in range(len(new_ip)):
            if new_ip[k,0] >= a_min and new_ip[k,0] <= a_max and new_ip[k,1] >= new_st[k,3]/wheelbase * d_min and new_ip[k,1] <= new_st[k,3]/wheelbase * d_max:
                continue
            # v
            new_st[k+1,3] = max(new_st[k+1,3], new_st[k,3] + simu_pace * a_min)
            new_st[k+1,3] = min(new_st[k+1,3], new_st[k,3] + simu_pace * a_max)
            new_st[k+1,3] = max(min(new_st[k+1,3], v_max), v_min)
            
            # h
            new_st[k+1,2] = max(new_st[k+1,2], new_st[k,2] + simu_pace * new_st[k,3]/wheelbase * d_min)
            new_st[k+1,2] = min(new_st[k+1,2], new_st[k,2] + simu_pace * new_st[k,3]/wheelbase * d_max)
            new_st[k+1,2] = max(min(new_st[k+1,2], h_max), h_min)

            # x
            new_st[k+1,1] = new_st[k,1] + simu_pace * (new_st[k,3] + new_st[k+1,3]) / 2 * cos(new_st[k,2])
            
            # y
            new_st[k+1,2] = new_st[k,2] + simu_pace * (new_st[k,3] + new_st[k+1,3]) / 2 * sin(new_st[k,2])
        
        new_ip = np.array([np.diff(new_st[:,3])/simu_pace, np.diff(new_st[:,2])/simu_pace], float).transpose(1,0)
        return AgentTrajectory(
            ts = now_t + np.arange(len(new_st), dtype=float),
            states = new_st,
            inputs = new_ip
        )


            
                

            
