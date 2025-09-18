from typing import List, Union, Tuple
from . import globalvar as glv

class LaneMethods:
    def getLane(self, laneID):
        return glv.rn.getLane(laneID)
    
    def getLaneIdx(self, laneID):
        return int(laneID.split('_')[-1])
    
    def getLanePosition(self, laneID, x, y):        
        lane = glv.rn.getLane(laneID)
        return lane.course_spline.cartesian_to_frenet1D(x, y)
    
    def getLaneWidth(self, laneID):
        return glv.rn.getLane(laneID).width
    
    def getAllLaneIDs(self):
        return list(glv.rn.lanes.keys())

