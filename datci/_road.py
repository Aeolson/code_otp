from typing import List, Union, Tuple
from . import globalvar as glv

class RoadMethods:
    def getAllRoadIDs(self):
        return glv.rn.edges.keys()
    