from typing import List, Union, Tuple
from . import globalvar as glv

class VehicleMethods:

    def checkOvertime(self, vid):
        return glv.ds.checkOvertime(vid, glv.timeStep)

    def isValid(self, vid):
        return (glv.ds.checkVehicleId(vid)) and (not glv.ds.checkOvertime(vid, glv.timeStep))

    def setPosition(self, vid, x:float, y:float):
        glv.ds.setStateAttributeAtTime(vid, ('x', 'y'), (x, y), glv.timeStep)
    
    def setAngel(self, vid, angle:float):
        glv.ds.setStateAttributeAtTime(vid, 'angle', angle, glv.timeStep)
    
    def setSpeed(self, vid, speed:float):
        glv.ds.setStateAttributeAtTime(vid, 'speed', speed, glv.timeStep)
    
    def setAccel(self, vid, accel:float):
        glv.ds.setStateAttributeAtTime(vid, 'accel', accel, glv.timeStep)

    def getVehicle(self, vid):
        return glv.ds.vehicles[vid]

    def getPosition(self, vid) -> Tuple[float, float]:
        return glv.ds.getStateAttributeAtTime(vid, ('x', 'y'), glv.timeStep)

    def getAngle(self, vid) -> float:
        return glv.ds.getStateAttributeAtTime(vid, 'angle', glv.timeStep)
    
    def getSpeed(self, vid) -> float:
        return glv.ds.getStateAttributeAtTime(vid, 'speed', glv.timeStep)

    def getAccel(self, vid) -> float:
        return glv.ds.getStateAttributeAtTime(vid, 'accel', glv.timeStep)

    def getRouteIndex(self, vid) -> int:
        return glv.ds.getStateAttributeAtTime(vid, 'edgeIdx', glv.timeStep)

    def getRoadID(self, vid) -> str:
        eidx = self.getRouteIndex(vid)
        if eidx is None:
            return None
        else:
            return glv.ds.vehicles[vid].routes[eidx]

    def getLaneID(self, vid) -> str:
        return glv.ds.getStateAttributeAtTime(vid, 'laneID', glv.timeStep)

    def getLaneIndex(self, vid) -> int:
        lid = self.getLaneID(vid)
        if lid is None:
            return None
        else:
            return int(lid.split('_')[-1])

    def getLanePosition(self, vid) -> float:
        return glv.ds.getStateAttributeAtTime(vid, 's', glv.timeStep)

    def getLaneInfosFromPos(self, x: float, y: float, yaw: float = None):
        return glv.ds.getLaneInfosFromPos(x, y, yaw)

    def getLaneIDFromPos(self, x: float, y: float, yaw: float = None):
        edgeID, juncID, laneID, lanePos, laneOff = glv.ds.getLaneInfosFromPos(x, y, yaw)
        return laneID

    def getVehicleType(self, vid) -> str:
        return glv.ds.vehicles[vid].vType
    
    def getVehicleShape(self, vid) -> Tuple[float, float]:
        return (glv.ds.vehicles[vid].length, glv.ds.vehicles[vid].width)

    def getMaxAccel(self, vid) -> float:
        return glv.ds.vehicles[vid].maxAccel

    def getMaxDecel(self, vid) -> float:
        return glv.ds.vehicles[vid].maxDecel

    def getMaxSpeed(self, vid) -> float:
        return glv.ds.vehicles[vid].maxSpeed

    def getMinTimestep(self, vid) -> int:
        return glv.ds.vehicles[vid].minTimestep

    def getMaxTimestep(self, vid) -> int:
        return glv.ds.vehicles[vid].minTimestep

    def getRoutes(self, vid) -> List[str]:
        return glv.ds.vehicles[vid].routes

    def getStateAttributes(self, vid: str, attr: Union[str, Tuple[str]]):
        return glv.ds.getStateAttributeAtTime(vid, attr, glv.timeStep)
    
    def getStateSequences(self, vid: str, attr: Union[str, Tuple[str]], horizon: int):
        return glv.ds.getStateAttributeSequence(vid, attr, glv.timeStep, glv.timeStep+horizon)

