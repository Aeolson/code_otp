import os, sys, pickle
from rich import print
from typing import List, Union, Tuple
from enum import Enum, unique
from . import globalvar as glv
from .dataset import DatasetBuild
from .network import Network
from . import _vehicle
from . import _road
from . import _lane
from ._valid_datasets import *

glv.creat()

vehicle = _vehicle.VehicleMethods()
road = _road.RoadMethods()
lane = _lane.LaneMethods()

def loadReplayDataset(datFile: str) -> DatasetBuild:
    dsbFile = datFile.replace('.csv', '.dsb')
    if not os.path.exists(dsbFile):
        return None
    
    try:
        ds = DatasetBuild()
        with open(dsbFile, 'rb') as fr_:
            ds_dict: dict = pickle.load(fr_)
            # print(ds_dict.keys())
            for k_, v_ in ds_dict.items():
                ds.__setattr__(k_, v_)
        print('[green bold]Load datasetbuild {}.[/green bold]'.format(dsbFile))

        return ds
    
    except:
        print('[red bold]Failed to load datasetbuild {}.[/red bold]'.format(dsbFile))
        return None

def saveReplayDataset(ds: DatasetBuild, datFile: str):
    dsbFile = datFile.replace('.csv', '.dsb')
    with open(dsbFile, 'wb') as fw_:
        pickle.dump(ds.__dict__, fw_)

def createReplayDataset(ds: dict, resampleRate: int = 10):
    glv.ds = None
    datFile, netFile = ds['data_file'], ds['road_file'] 
    glv.ds = loadReplayDataset(datFile)
    
    if glv.ds is None:
        glv.ds = DatasetBuild()
        glv.ds.create(datFile, netFile, resampleRate)
        saveReplayDataset(glv.ds, datFile)
    
    glv.rn = Network()
    glv.rn.create(netFile)

def setTimestep(t: int):
    assert t >= 0
    glv.timeStep = int(t)

def getTimestep() -> int:
    return glv.timeStep

def setSimupace(p: float):
    assert p > 0.0
    glv.simuPace = float(p)

def getSimupace() -> float:
    return glv.simuPace

def convTimestepToRealtime(t: int) -> float:
    return (glv.ds.minTimestep + t) * 1.0 / glv.ds.sampleRate

def convRealtimeToTimestep(t: float):
    return round(t * glv.ds.sampleRate) - glv.ds.minTimestep

def next():
    glv.timeStep = glv.timeStep + 1

def previous():
    glv.timeStep = glv.timeStep - 1

def getAllVehicleIds():
    return list(glv.ds.vehicles.keys())

def getFrameVehicleIds(t: int = None):
    if t is None:
        t = glv.timeStep
    return glv.ds.getFrameVehicleIds(t)

def getNextFrameVehicleIds(t: int = None):
    if t is None:
        t = glv.timeStep
    return glv.ds.getFrameVehicleIds(t+1)

def getPreviousFrameVehicleIds(t: int = None):
    if t is None:
        t = glv.timeStep
    return glv.ds.getFrameVehicleIds(t-1)

def showStatistics():
    glv.ds.getStatistics2()

def getStatistics_Acceleration():
    return glv.ds.getStatistics_Acceleration()
