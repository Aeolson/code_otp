import sys, os

import datci.network
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')

from dataclasses import dataclass
import random
import time
import numpy as np
import scipy
from typing import Dict, Tuple, Optional
from rich import print
from math import cos, sin
from typing import Union
import dearpygui.dearpygui as dpg
from utils.bazier_curve import BazierCurve2D
import datci
from configs import *
from utils.simple_agent import Agent, AgentTrajectory

# traj_color_dat = '#1E90FF' # blue
# traj_color_fsm = '#FFC125' # yellow
# traj_color_dlp = '#66CD00' # green
# traj_color_otp = '#FF4500' # red

traj_color_dat = (30,  144, 255) # blue
traj_color_fsm = (255, 193, 37 ) # yellow
traj_color_dlp = (102, 205, 0  ) # green
traj_color_otp = (255, 69,  0  ) # red

class CoordTF:
    # Ego is always in the center of the window
    def __init__(self, realSize: float, windowTag: str) -> None:
        self.realSize: float = realSize
        self.drawCenter: float = self.realSize / 2

        dh = float(dpg.get_item_height(windowTag))
        dw = float(dpg.get_item_width(windowTag))
        if dw <= dh:
            self.dpgDrawSize = dw
        else:
            self.dpgDrawSize = dh


        # self.dpgDrawSize: float = float(dpg.get_item_height(windowTag)) - 30
        if dw < dh:
            dy = (dh - dw) / self.dpgDrawSize / 2 * self.realSize
            self.offset = (0, dy)
        else:
            dx = (dw - dh) / self.dpgDrawSize / 2 * self.realSize
            self.offset = (dx, 0)

    @property
    def zoomScale(self) -> float:
        return self.dpgDrawSize / self.realSize

    def dpgCoord(
            self, x: float, y: float, ex: float, ey: float) -> tuple[float]:
        relx, rely = x - ex, y - ey
        return (
            self.zoomScale * (self.drawCenter + relx + self.offset[0]),
            self.zoomScale * (self.drawCenter - rely + self.offset[1])
        )

@dataclass
class GuiControl:
    is_dragging: bool = False
    old_offset: tuple[float, float] = (0, 0)
    zoom_speed: float = 1.0
    ctf: CoordTF = None

def evaluate_risk(ex, ev, sx, sv):
    """ return risk with TTC """
    threshold = 20.0 # (s), max ttc, represent infinity
    dx = sx - ex
    dv = ev - sv
    if dv == 0:
        ttc = threshold
    else:
        ttc = dx / dv
        if ttc < 0.0:
            ttc = threshold
    
    return (threshold - ttc) / threshold

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

def calc_risk(agt1: Agent, agt2: Agent) -> float:
        
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

class userGUI:
    def __init__(self) -> None:
        self.gct = GuiControl()
        self.ex, self.ey = 0.0, 0.0
        self.lane_from_centery, self.lane_from_width = 0.0, 0.0
        self.lane_to_centery, self.lane_to_width = 0.0, 0.0
        self.is_running = False
        self.incre_frame = 0

    def plotAgent(self, agt: Agent, node):
        if agt.id is None or agt.label == 'virtual':
            # virtual vehicle
            return
        
        rotateMat = np.array(
            [
                [cos(agt.h), -sin(agt.h)],
                [sin(agt.h),  cos(agt.h)]
            ]
        )
        vertexes = [
            np.array([[ agt.length/2], [ agt.width/2]]),
            np.array([[ agt.length/2], [-agt.width/2]]),
            np.array([[-agt.length/2], [-agt.width/2]]),
            np.array([[-agt.length/2], [ agt.width/2]])
        ]
        rotVertexes = [np.dot(rotateMat, vex) for vex in vertexes]
        relativeVex = [[agt.x + rv[0], agt.y + rv[1]] for rv in rotVertexes]
        drawVex = [self.gct.ctf.dpgCoord(rev[0], rev[1], self.ex, self.ey) for rev in relativeVex]

        vtag = agt.label
        if vtag == 'veh_e':
            vcolor = (211, 84, 0, 180) # orange
        elif vtag == 'veh_f':
            vcolor = (0, 180, 0, 180) # green
        elif vtag == 'veh_r':
            vcolor = (41, 128, 185, 180) # blue
        elif vtag == 'veh_p':
            vcolor = (200, 200, 0, 180) # yellow
        else:
            vcolor = (99, 110, 114, 180)
        
        dpg.draw_polygon(drawVex, fill=vcolor, thickness=0, parent=node)
        dpg.draw_text(
            self.gct.ctf.dpgCoord(agt.x, agt.y-agt.width/2, self.ex, self.ey),
            '%s:v=%.2f'%(agt.id, agt.v),
            color=(80,80,80,255),
            size=15,
            parent=node
        )

    def plotNetwork(self, node):
        allLaneIds = datci.lane.getAllLaneIDs()
        for laneID in allLaneIds:
            lane = datci.lane.getLane(laneID)
            left_bound_tf = [self.gct.ctf.dpgCoord(wp[0], wp[1], self.ex, self.ey) for wp in lane.left_bound]
            right_bound_tf = [self.gct.ctf.dpgCoord(wp[0], wp[1], self.ex, self.ey) for wp in lane.right_bound]

            left_bound_tf.reverse()
            right_bound_tf.extend(left_bound_tf)
            right_bound_tf.append(right_bound_tf[0])
            if laneID == self.lane_from_id:
                lane_color = (255, 180, 180)
            elif laneID == self.lane_to_id:
                lane_color = (180, 255, 180)
            else:
                lane_color = (220, 220, 220)
            dpg.draw_polygon(right_bound_tf, color=(160, 160, 160), thickness=5, fill=lane_color, parent=node)

    def plotTrajectory(self, t_now:float, trj: AgentTrajectory, node, traj_type):

        if trj is not None and trj.states is not None:
            if traj_type == 'dat':
                tcolor = traj_color_dat
                tlabel = 'Data'
            elif traj_type == 'otp':
                tcolor = traj_color_otp
                tlabel = 'OTP'
            elif traj_type == 'fsm':
                tcolor = traj_color_fsm
                tlabel = 'SBF'
            elif traj_type == 'dlp':
                tcolor = traj_color_dlp
                tlabel = 'DLP'
            elif traj_type == 'plan':
                tcolor = (0, 255, 255)
                tlabel = 'Plan'
            else:
                tcolor = (255, 0, 255)
                tlabel = 'Unknown'

                
            pts = [self.gct.ctf.dpgCoord(x, y, self.ex, self.ey) for (x, y) in trj.states[:,0:2]]
            st_idx = np.argwhere(np.round(trj.ts,3) >= np.round(t_now,3))[0].item()
            pts = pts[st_idx:]
            dpg.draw_polyline(pts, color=tcolor, thickness=5.0, parent=node)
            dpg.draw_text(
                (pts[0][0], pts[0][1]),
                '%s'%tlabel,
                color=(0,0,0,255),
                size=15,
                parent=node
            )
        
    def mouse_down(self):
        if not self.gct.is_dragging:
            if dpg.is_item_hovered("MainWindow"):
                self.gct.is_dragging = True
                self.gct.old_offset = self.gct.ctf.offset

    def mouse_drag(self, sender, app_data):
        if self.gct.is_dragging:
            self.gct.ctf.offset = (
                self.gct.old_offset[0] + app_data[1]/self.gct.ctf.zoomScale,
                self.gct.old_offset[1] + app_data[2]/self.gct.ctf.zoomScale
            )

    def mouse_release(self):
        self.gct.is_dragging = False

    def mouse_wheel(self, sender, app_data):
        if dpg.is_item_hovered("MainWindow"):
            self.gct.zoom_speed = 1 + 0.01*app_data
        
    def update_inertial_zoom(self, clip=0.005):
        if self.gct.zoom_speed != 1:
            self.gct.ctf.dpgDrawSize *= self.gct.zoom_speed
            self.gct.zoom_speed = 1+(self.gct.zoom_speed - 1) / 1.05
        if abs(self.gct.zoom_speed - 1) < clip:
            self.gct.zoom_speed = 1

    def start(self, lane_from_id:str, lane_to_id:str):

        self.lane_from_id, self.lane_to_id = lane_from_id, lane_to_id
        lane = datci.lane.getLane(self.lane_from_id)
        self.lane_from_centery = lane.center_line[0][1]
        self.lane_from_width = lane.width
        
        lane = datci.lane.getLane(self.lane_to_id)
        self.lane_to_centery = lane.center_line[0][1]
        self.lane_to_width = lane.width

        dpg.create_context()

        dpg.create_viewport(title="Scenario Replay", width=1040, height=400, x_pos=200, y_pos=200)
        dpg.setup_dearpygui()

        with dpg.window(
            tag='ControlWindow',
            label='Menu',
            no_close=True,
            no_resize=True,
            no_move=True,
            no_bring_to_front_on_focus=True,
        ):
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Pause", tag="PauseResumeButton",
                    callback=self.toggle
                )
                dpg.add_button(label="Next",
                    callback=self.next_frame
                )

        dpg.add_window(
            tag="MainWindow",
            label="Simulation",
            no_close=True,
            no_resize=True,
            no_move=True,
        )
        dpg.add_draw_node(tag="scene", parent="MainWindow")

        dpg.add_window(
            tag="InfoWindow",
            label="Infomation",
            no_close=True,
            no_resize=True,
            no_move=True,
        )
        dpg.add_draw_node(tag="info", parent="InfoWindow")

        dpg.set_item_width('ControlWindow', 1000)
        dpg.set_item_height('ControlWindow', 40)
        dpg.set_item_pos('ControlWindow', (10, 10))

        dpg.set_item_width("MainWindow", 1000)
        dpg.set_item_height("MainWindow", 200)
        dpg.set_item_pos("MainWindow", (10, 60))

        dpg.set_item_width("InfoWindow", 1000)
        dpg.set_item_height("InfoWindow", 150)
        dpg.set_item_pos("InfoWindow", (10, 270))

        with dpg.handler_registry():
            dpg.add_mouse_down_handler(callback=self.mouse_down)
            dpg.add_mouse_drag_handler(callback=self.mouse_drag)
            dpg.add_mouse_release_handler(callback=self.mouse_release)
            dpg.add_mouse_wheel_handler(callback=self.mouse_wheel)

        dpg.show_viewport()
        self.gct.ctf = CoordTF(20, "MainWindow")
        self.is_running = True

    def destroy(self):
        dpg.destroy_context()

    def render(self, t: float, veh_e:Agent, surrVehilces: list[Agent]=None, traj:AgentTrajectory=None, info:Union[str, dict] = None, traj_type=None):
        self.incre_frame = 0
        dpg.delete_item('scene', children_only=True)
        node = dpg.add_draw_node(parent="scene")

        self.ex, self.ey = veh_e.x, veh_e.y
        self.update_inertial_zoom()

        self.plotNetwork(node)
        self.plotAgent(veh_e, node)
        if surrVehilces is not None:
            for sV in surrVehilces:
                self.plotAgent(sV, node)
        if traj is not None:
            self.plotTrajectory(t, traj, node, traj_type)

        dpg.delete_item('info', children_only=True)
        node = dpg.add_draw_node(parent="info")

        dpg.draw_text(
            (10, 5),
            'Time: %.2f' % (t),
            color=(255, 255, 255),
            size=15,
            parent=node
        )

        lane_id = datci.vehicle.getLaneIDFromPos(veh_e.x, veh_e.y, veh_e.h)
        if lane_id == self.lane_from_id:
            lane_tag = "Original Lane"
        else:
            lane_tag = 'Target Lane'
        # if abs(veh_e.y - self.lane_from_centery) < self.lane_from_width / 2:
        #     lane_tag = "Original Lane"
        # else:
        #     lane_tag = "Target Lane"
        dpg.draw_text(
            (10, 25),
            'EV is in: %s' % (lane_tag),
            color=(102, 205, 0),
            size=15,
            parent=node
        )

        if info is not None:
            if isinstance(info, dict):
                info = ", ".join(["%s: %s" % (str(k), str(v)) for k, v in info.items()])
            else:
                info = str(info)

            dpg.draw_text(
                (10, 45),
                info,
                color=(255, 125, 0),
                size=15,
                parent=node
            )

        dpg.render_dearpygui_frame()

    def resume(self):
        self.is_running = True
        dpg.set_item_label('PauseResumeButton', 'Pause')

    def pause(self):
        self.is_running = False
        dpg.set_item_label('PauseResumeButton', 'Resume')

    def toggle(self):
        if self.is_running:
            self.pause()
        else:
            self.resume()

    def next_frame(self):
        self.incre_frame = 1