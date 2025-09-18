from .network import Network
from .dataset import DatasetBuild

def creat():
    global ds, rn, timeStep, simuPace

    ds = DatasetBuild()
    rn = Network()
    timeStep = 0
    simuPace = 0