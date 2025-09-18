import os, sys
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from math import cos, sin
import datci

use_ds = datci.NGSIM_I80_0400_0415
# use_ds = datci.NGSIM_I80_0500_0515
# use_ds = datci.NGSIM_I80_0515_0530
datci.createReplayDataset(use_ds, resampleRate=10)
datci.setSimupace(0.1)
Ts = datci.getSimupace()
pre_lanechange = round(5.0/Ts)
suc_lanechange = round(2.0/Ts)

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

all_vehicle_ids = datci.getAllVehicleIds()
lane_change_infos: list[dict] = []
for ego_id in tqdm(all_vehicle_ids):
    veh_data = datci.vehicle.getVehicle(ego_id)
    if veh_data.vType not in ['Car']:
        continue
    ts = np.array([st.t for st in veh_data.states.values()], dtype=int)
    lanes = np.array([st.laneID for st in veh_data.states.values()], dtype=str)
    if len(np.unique(lanes)) == 1:
        continue # without lane change
    
    for i in range(1, len(ts)):
        if lanes[i] != lanes[i-1]:
            lane_from_id = lanes[i-1]
            lane_to_id = lanes[i]
            if datci.lane.getLaneIdx(lane_from_id) == 0 or datci.lane.getLaneIdx(lane_to_id) == 0:
                continue # on-ramp
            i_st = i - pre_lanechange
            i_ed = i + suc_lanechange-1
            if i_st < 0 or i_ed >= len(ts):
                continue
            if abs(int(lane_to_id.split('_')[-1]) - int(lane_from_id.split('_')[-1])) != 1:
                continue
            if lanes[i_st] != lane_from_id or lanes[i_ed] != lane_to_id:
                continue

            """ is a lane change instant"""
            cross_time, start_time, end_time = ts[i], ts[i_st], ts[i_ed]
            datci.setTimestep(cross_time)
            ego = get_agent_from_daci(ego_id)

            lane_from_centery = datci.lane.getLane(lane_from_id).center_line[0][1]
            lane_from_width = datci.lane.getLaneWidth(lane_from_id)

            lane_to_centery = datci.lane.getLane(lane_to_id).center_line[0][1]
            lane_to_width = datci.lane.getLaneWidth(lane_to_id)

            """ determine the veh_r, veh_p, veh_f at the lane change instant """
            frame_vehicle_ids = datci.getFrameVehicleIds()
            veh_p, veh_r, veh_f = None, None, None # preceding vehicle, rear vehicle, front vehicle
            for vid in frame_vehicle_ids:
                if vid == ego_id:
                    continue

                if datci.vehicle.getLaneID(vid) == lane_from_id:
                    sve = get_agent_from_daci(vid)
                    if sve.x >= ego.x:
                        if veh_p is None or veh_p.x > sve.x:
                            veh_p = sve          
                elif datci.vehicle.getLaneID(vid) == lane_to_id:
                    sve = get_agent_from_daci(vid)
                    if sve.x > ego.x:
                        if veh_f is None or veh_f.x > sve.x:
                            veh_f = sve
                    else:
                        if veh_r is None or veh_r.x < sve.x:
                            veh_r = sve
                
            """ check if veh_f, veh_r, veh_p meet requirments """
            if veh_p is None and veh_r is None and veh_f is None:
                continue

            datci.setTimestep(start_time)
            if veh_p is not None:
                if not datci.vehicle.isValid(veh_p.id):
                    continue
                if datci.vehicle.getLaneID(veh_p.id) != lane_from_id:
                    continue
            if veh_r is not None:
                if not datci.vehicle.isValid(veh_r.id):
                    continue
            if veh_f is not None:
                if not datci.vehicle.isValid(veh_f.id):
                    continue

            """ check if there is collision in dataset """
            is_collision = False
            for k in range(start_time, end_time+1):
                if is_collision:
                    break

                datci.setTimestep(k)
                ego = get_agent_from_daci(ego_id)
                frame_vehicle_ids = datci.getFrameVehicleIds()
                for vid in frame_vehicle_ids:
                    if vid != ego_id:
                        if datci.vehicle.getLaneID(vid) in [lane_from_id, lane_to_id]:
                            sve = get_agent_from_daci(vid)
                            if is_agent_collision(ego, sve, e=0.5):
                                is_collision = True
                                break

            if is_collision:
                continue


            lc = {
                'ego': ego.id,
                't_cross': cross_time,
                't_start': start_time,
                't_end': end_time,
                'lane_from': lane_from_id,
                'lane_from_centery': lane_from_centery,
                'lane_from_width': lane_from_width,
                'lane_to': lane_to_id,
                'lane_to_centery': lane_to_centery,
                'lane_to_width': lane_to_width,
                'veh_f': veh_f.id if veh_f is not None else 'None',
                'veh_r': veh_r.id if veh_r is not None else 'None',
                'veh_p': veh_p.id if veh_p is not None else 'None',
            }
            lane_change_infos.append(lc)

print("total number of samples in %s: %d" % (use_ds['name'], len(lane_change_infos)))

if not os.path.exists("./LC_INFOS"):
    os.makedirs("./LC_INFOS")
save_fn = './LC_INFOS/LC_INFO_%s.txt' % (use_ds['name'])
with open(save_fn, 'w') as fw:
    for lc in lane_change_infos:
        s_ = ','.join(["%s:%s"%(k_, str(v_)) for k_, v_ in lc.items()])
        fw.write(s_ + '\n')



                    




    