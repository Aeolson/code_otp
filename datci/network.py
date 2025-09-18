'''
Revised by Aeolson, 2024-04-09
'''

import time
from utils.cubic_spline import Spline2D
from typing import Dict, Set
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC
import numpy as np
import xml.etree.ElementTree as ET


OVERLAP_DISTANCE = 0.1  # junction overlap distance

@dataclass
class Junction:
    id: str = None
    incoming_edges: set[str] = field(default_factory=set)
    outgoing_edges: set[str] = field(default_factory=set)
    JunctionLanes: set[str] = field(default_factory=set)
    affGridIDs: set[tuple[int]] = field(default_factory=set)
    shape: list[tuple[float]] = None

@dataclass
class Edge:
    id: str = None
    lane_num: int = 0
    lanes: Set[str] = field(default_factory=set)
    from_junction: str = None
    to_junction: str = None
    next_edge_info: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )  # next edge and the corresponding **self** normal lane
    obstacles: dict = field(default_factory=dict)
    affGridIDs: set[tuple[int]] = field(default_factory=set)

    # required to updated
    length = None
    width = None
    ref_spline: Spline2D = None

    def __hash__(self):
        return hash(self.id)

    def __repr__(self) -> str:
        return f"Edge(id={self.id})"
        # return f"Edge(id={self.id}, lane_num={len(self.lanes)}, from_junction={self.from_junction}, to_junction={self.to_junction})\n"

@dataclass
class AbstractLane(ABC):
    """
    Abstract lane class.
    """
    id: str
    width: float = 0
    speed_limit: float = 13.89
    sumo_length: float = 0
    course_spline: Spline2D = None

    @property
    def spline_length(self):
        return self.course_spline.s[-1]

    def getPlotElem(self):
        s = np.linspace(0, self.course_spline.s[-1], num=50)
        self.center_line = [
            self.course_spline.calc_position(si) for si in s
        ]
        self.left_bound = [
            self.course_spline.frenet_to_cartesian1D(si, self.width / 2) for si in s
        ]
        self.right_bound = [
            self.course_spline.frenet_to_cartesian1D(si, -self.width / 2) for si in s
        ]

@dataclass
class NormalLane(AbstractLane):
    """
    Normal lane from edge 
    """
    affiliated_edge: Edge = None
    next_lanes: Dict[str, tuple[str, str]] = field(
        default_factory=dict
    )  # next_lanes[to_lane_id: normal lane] = (via_lane_id, direction)


    def left_lane(self) -> str:
        lane_index = int(self.id.split("_")[-1])
        left_lane_id = f"{self.affiliated_edge.id}_{lane_index + 1}"
        for lane in self.affiliated_edge.lanes:
            if lane == left_lane_id:
                return left_lane_id
        # logging.error(f"cannot find left lane of {self.id}")
        return None

    def right_lane(self) -> str:
        lane_index = int(self.id.split("_")[-1])
        right_lane_id = f"{self.affiliated_edge.id}_{lane_index - 1}"
        for lane in self.affiliated_edge.lanes:
            if lane == right_lane_id:
                return right_lane_id
        # logging.error(f"cannot find right lane of {self.id}")
        return None

    def __hash__(self):
        return hash(self.id)

    def __repr__(self) -> str:
        # return f"NormalLane(id={self.id}, width = {self.width})"
        return f"NormalLane(id={self.id})"

@dataclass(unsafe_hash=True)
class JunctionLane(AbstractLane):
    """
    Junction lane in intersection 
    """
    last_lane_id: str = None
    next_lane_id: str = None  # next lane's id
    affJunc: str = None   # affiliated junction ID
    tlLogic: str = None
    tlsIndex: int = 0
    currTlState: str = None   # current traffic light phase state: r, g, y etc.
    # remain time (second)  switch to next traffic light phase.
    switchTime: float = 0.0
    nexttTlState: str = None   # next traffic light phase state: r, g, y etc.

    def __repr__(self) -> str:
        return f"JunctionLane(id={self.id} tlState={self.currTlState} switchTime={self.switchTime})"
        # return f"JunctionLane(id={self.id}, width = {self.width}, next_lane={self.next_lane})"

@dataclass
class TlLogic:
    id: str = None
    tlType: str = None   # static or actuated
    preDefPhases: list[str] = None

    def currPhase(self, currPhaseIndex: int) -> str:
        return self.preDefPhases[currPhaseIndex]

    def nextPhase(self, currPhaseIndex: int) -> str:
        if currPhaseIndex < len(self.preDefPhases)-1:
            return self.preDefPhases[currPhaseIndex+1]
        else:
            return self.preDefPhases[0]

@dataclass
class RoadGraph:
    """
    Road graph of the map
    """

    edges: Dict[str, Edge] = field(default_factory=dict)
    lanes: Dict[str, AbstractLane] = field(default_factory=dict)
    junction_lanes: Dict[str, JunctionLane] = field(default_factory=dict)

    def get_lane_by_id(self, lane_id: str) -> AbstractLane:
        if lane_id in self.lanes:
            return self.lanes[lane_id]
        elif lane_id in self.junction_lanes:
            return self.junction_lanes[lane_id]
        else:
            # logging.debug(f"cannot find lane {lane_id}")
            return None

    def get_next_lane(self, lane_id: str) -> AbstractLane:
        lane = self.get_lane_by_id(lane_id)
        if isinstance(lane, NormalLane):
            next_lanes = list(lane.next_lanes.values())
            if len(next_lanes) > 0:
                # first_next_lane = list(lane.next_lanes.values())[0][0]
                return self.get_lane_by_id(next_lanes[0][0])
            else:
                return None
        elif isinstance(lane, JunctionLane):
            return self.get_lane_by_id(lane.next_lane_id)
        return None

    def get_available_next_lane(self, lane_id: str, available_lanes: list[str]) -> AbstractLane:
        lane = self.get_lane_by_id(lane_id)
        if isinstance(lane, NormalLane):
            for next_lane_i in lane.next_lanes.values():
                if next_lane_i[0] in available_lanes:
                    return self.get_lane_by_id(next_lane_i[0])
        elif isinstance(lane, JunctionLane):
            if lane.next_lane_id in available_lanes:
                return self.get_lane_by_id(lane.next_lane_id)
        return None

    def __str__(self):
        return 'edges: {}, \nlanes: {}, \njunctions lanes: {}'.format(
            self.edges.keys(), self.lanes.keys(),
            self.junction_lanes.keys()
        )

def judge_point_in_polygon(polygon: list[tuple[float, float]], point: tuple[float, float], error: float = 1e-8):
    """
    method implemented to judge if the point is in the given polygon, by cross product
    Args:
        polygon: list of vertices for the polygon, along clockwise or anticlockwise
        point: 2D point with [x, y]
        error: the judgement error for cross product, for 'OnLine' judgement
    Return:
        1: Outside, 0: Onside, -1: Inside
    """
    polygon, point = np.array(polygon, float), np.array(point, float)
    sign_count = 0
    for i in range(len(polygon)-1):
        p1, p2 = polygon[i], polygon[i+1]
        cr = np.cross(p1-point, p2-point)
        print(cr)
        if abs(cr) <= error:
            if ( point[0] >= min(p1[0], p2[0]) and point[0] <= max(p1[0], p2[0]) ) and \
               ( point[1] >= min(p1[1], p2[1]) and point[1] <= max(p1[1], p2[1]) ):
                return 0
            else:
                sign_count += np.sign(cr.item())
        else:
            sign_count += np.sign(cr.item())

    if abs(sign_count) == len(polygon)-1:
        return -1
    else:
        return 1

class geoHash:
    def __init__(self, id: tuple[int]) -> None:
        self.id = id
        self.edges: set[str] = set()
        self.junctions: set[str] = set()

class Network:
    def __init__(self):
        self.networkFile = None
        self.edges: dict[str, Edge] = {}
        self.lanes: dict[str, NormalLane] = {}
        self.junctions: dict[str, Junction] = {}
        self.junctionLanes: dict[str, JunctionLane] = {}
        self.tlLogics: dict[str, TlLogic] = {}
        self.geoHashes: dict[tuple[int], geoHash] = {}

    def create(self, netFile: str):
        self.networkFile = netFile
        self.getData()
        self.buildTopology()

    def getEdge(self, eid: str) -> Edge:
        try:
            return self.edges[eid]
        except KeyError:
            return

    def getLane(self, lid: str) -> NormalLane:
        try:
            return self.lanes[lid]
        except KeyError:
            return

    def getJunction(self, jid: str) -> Junction:
        try:
            return self.junctions[jid]
        except KeyError:
            return

    def getJunctionLane(self, jlid: str) -> JunctionLane:
        try:
            return self.junctionLanes[jlid]
        except KeyError:
            return

    def getTlLogic(self, tlid: str) -> TlLogic:
        try:
            return self.tlLogics[tlid]
        except KeyError:
            return

    def affGridIDs(self, centerLine: list[tuple[float]]) -> set[tuple[int]]:
        affGridIDs = set()
        for poi in centerLine:
            poixhash = int(poi[0] // 100)
            poiyhash = int(poi[1] // 100)
            affGridIDs.add((poixhash, poiyhash))

        return affGridIDs

    def processRawShape(self, rawShape: str) -> list[list[float]]:
        rawList = rawShape.split(' ')
        floatShape = [list(map(float, p.split(','))) for p in rawList]
        return floatShape

    def processEdge(self, eid: str, child: ET.Element):
        if eid[0] == ':':
            # internal edge
            for gchild in child:
                ilid = gchild.attrib['id']
                try:
                    ilspeed = float(gchild.attrib['speed'])
                except:
                    ilspeed = 13.89
                try:
                    ilwidth = float(gchild.attrib['width'])
                except KeyError:
                    ilwidth = 3.2
                ilLength = float(gchild.attrib['length'])
                self.junctionLanes[ilid] = JunctionLane(
                    id=ilid, width=ilwidth, speed_limit=ilspeed,
                    sumo_length=ilLength,
                )

        else:
            # normal edge
            fromNode = child.attrib['from']
            toNode = child.attrib['to']
            edge = Edge(id=eid, from_junction=fromNode, to_junction=toNode)
            laneNumber = 0
            for gchild in child:
                if gchild.tag == 'lane':
                    lid = gchild.attrib['id']
                    try:
                        lwidth = float(gchild.attrib['width'])
                    except KeyError:
                        lwidth = 3.2
                    lspeed = float(gchild.attrib['speed'])
                    rawShape = gchild.attrib['shape']
                    lshape = self.processRawShape(rawShape)
                    llength = float(gchild.attrib['length'])
                    lane = NormalLane(id=lid, width=lwidth, speed_limit=lspeed,
                                      sumo_length=llength, affiliated_edge=edge)

                    shapeUnzip = list(zip(*lshape))

                    # interpolate shape points for better represent shape
                    shapeUnzip = [
                        np.interp(
                            np.linspace(0, len(shapeUnzip[0])-1, 50),
                            np.arange(0, len(shapeUnzip[0])),
                            shapeUnzip[i]
                        ) for i in range(2)
                    ]
                    lane.course_spline = Spline2D(shapeUnzip[0], shapeUnzip[1])
                    lane.getPlotElem()
                    self.lanes[lid] = lane
                    edge.lanes.add(lane.id)
                    laneAffGridIDs = self.affGridIDs(lane.center_line)
                    edge.affGridIDs = edge.affGridIDs | laneAffGridIDs
                    laneNumber += 1
            
            edge.lane_num = laneNumber
            
            for gridID in edge.affGridIDs:
                try:
                    geohash = self.geoHashes[gridID]
                except KeyError:
                    geohash = geoHash(gridID)
                    self.geoHashes[gridID] = geohash
                geohash.edges.add(eid)
            
            # update the length, width and reference line, by Aeolson
            # try:
            #     rawShape = gchild.attrib['shape']
            #     edeg_shape = self.processRawShape(rawShape)

            # except:
            #     nearest_lane_id = "_".join([edge.id, str(edge.lane_num-1)])
            #     lane_spl = self.lanes[nearest_lane_id].course_spline
            #     pd = self.lanes[nearest_lane_id].width / 2
            #     edeg_shape = [lane_spl.frenet_to_cartesian1D(ps, pd) for ps in lane_spl.s]
            
            nearest_lane_id = "_".join([edge.id, str(edge.lane_num-1)])
            edge_shape = self.lanes[nearest_lane_id].left_bound
            shapeUnzip = list(zip(*edge_shape))
            shapeUnzip = [
                np.interp(
                    np.linspace(0, len(shapeUnzip[0])-1, 50),
                    np.arange(0, len(shapeUnzip[0])),
                    shapeUnzip[i]
                ) for i in range(2)
            ]
            
            edge.ref_spline = Spline2D(shapeUnzip[0], shapeUnzip[1])
            edge.length = edge.ref_spline.s[-1]
            edge.length = round(edge.length, 2)
            edge.width = sum([self.lanes[lid].width for lid in edge.lanes])
            edge.width = round(edge.width, 2)

            self.edges[eid] = edge

    def processConnection(self, child: ET.Element):
        fromEdgeID = child.attrib['from']
        fromEdge = self.getEdge(fromEdgeID)
        fromLaneIdx = child.attrib['fromLane']
        fromLaneID = fromEdgeID + '_' + fromLaneIdx
        fromLane = self.getLane(fromLaneID)
        toEdgeID = child.attrib['to']
        toLaneIdx = child.attrib['toLane']
        toLaneID = toEdgeID + '_' + toLaneIdx
        toLane = self.getLane(toLaneID)
        if fromLane and toLane:
            direction = child.attrib['dir']
            junctionLaneID = child.attrib['via']
            junctionLane = self.getJunctionLane(junctionLaneID)

            if junctionLane.sumo_length < 1:
                fromLane.next_lanes[toLaneID] = (toLaneID, 's')
                fromEdge.next_edge_info[toEdgeID].add(fromLaneID)
            else:
                # junctionLane = self.getJunctionLane(junctionLaneID)
                if 'tl' in child.attrib.keys():
                    tl = child.attrib['tl']
                    linkIndex = int(child.attrib['linkIndex'])
                    junctionLane.tlLogic = tl
                    junctionLane.tlsIndex = linkIndex
                
                center_line = []
                for si in np.linspace(
                    fromLane.course_spline.s[-1] - OVERLAP_DISTANCE,
                    fromLane.course_spline.s[-1], num=20
                ):
                    center_line.append(
                        fromLane.course_spline.calc_position(si))
                for si in np.linspace(0, OVERLAP_DISTANCE, num=20):
                    center_line.append(
                        toLane.course_spline.calc_position(si)
                    )
                junctionLane.course_spline = Spline2D(
                    list(zip(*center_line))[0], list(zip(*center_line))[1]
                )
                junctionLane.getPlotElem()
                junctionLane.last_lane_id = fromLaneID
                junctionLane.next_lane_id = toLaneID
                fromLane.next_lanes[toLaneID] = (junctionLaneID, direction)
                fromEdge.next_edge_info[toEdgeID].add(fromLaneID)
                # add this junctionLane to it's parent Junction's JunctionLanes
                fromEdge = self.getEdge(fromEdgeID)
                juncID = fromEdge.to_junction
                junction = self.getJunction(juncID)
                junctionLane.affJunc = juncID
                jlAffGridIDs = self.affGridIDs(junctionLane.center_line)
                junction.affGridIDs = junction.affGridIDs | jlAffGridIDs
                junction.JunctionLanes.add(junctionLaneID)

    def getData(self):
        elementTree = ET.parse(self.networkFile)
        root = elementTree.getroot()
        for child in root:
            if child.tag == 'edge':
                eid = child.attrib['id']
                # Some useless internal lanes will be generated by the follow codes.
                self.processEdge(eid, child)
            elif child.tag == 'junction':
                jid = child.attrib['id']
                junc = Junction(jid)
                if jid[0] != ':':
                    intLanes = child.attrib['intLanes']
                    if intLanes:
                        intLanes = intLanes.split(' ')
                        for il in intLanes:
                            ilins = self.getJunctionLane(il)
                            ilins.affJunc = jid
                            junc.JunctionLanes.add(il)
                    jrawShape = child.attrib['shape']
                    juncShape = self.processRawShape(jrawShape)
                    # Add the first point to form a closed shape
                    juncShape.append(juncShape[0])
                    junc.shape = juncShape
                    self.junctions[jid] = junc

            elif child.tag == 'connection':
                # in .net.xml, the elements 'edge' come first than elements
                # 'connection', so the follow codes can work well.
                self.processConnection(child)
            elif child.tag == 'tlLogic':
                tlid = child.attrib['id']
                tlType = child.attrib['type']
                preDefPhases = []
                for gchild in child:
                    if gchild.tag == 'phase':
                        preDefPhases.append(gchild.attrib['state'])

                self.tlLogics[tlid] = TlLogic(tlid, tlType, preDefPhases)

        for junction in self.junctions.values():
            for gridID in junction.affGridIDs:
                try:
                    geohash = self.geoHashes[gridID]
                except KeyError:
                    geohash = geoHash(gridID)
                    self.geoHashes[gridID] = geohash
                geohash.junctions.add(junction.id)

        for ghid, ghins in self.geoHashes.items():
            ghx, ghy = ghid
            ghEdges = ','.join(ghins.edges)
            ghJunctions = ','.join(ghins.junctions)

    def buildTopology(self):
        for eid, einfo in self.edges.items():
            fj = self.getJunction(einfo.from_junction)
            tj = self.getJunction(einfo.to_junction)
            fj.outgoing_edges.add(eid)
            tj.incoming_edges.add(eid)

    def getGridIDFromPos(self, x: float, y: float) -> tuple[int]:
        ix = int(x // 100) 
        iy = int(y // 100) 
        return ( ix, iy )

    def getLaneInfosFromPos(self, x: float, y: float, yaw: float = None):
        """
        Return:
            laneID
        """
        gridID = self.getGridIDFromPos(x, y)
        edgeIDs = self.geoHashes[gridID].edges
        juncIDs = self.geoHashes[gridID].junctions

        for eid in edgeIDs:
            edge = self.getEdge(eid)
            s, d = edge.ref_spline.cartesian_to_frenet1D(x, y)
            s, d = [round(_, 2) for _ in (s, d)]
            if s >= -0.01 and s <= edge.length+0.01 and d <= 0.01 and d >= -edge.width-0.01:
                # print("edge = %s" % eid)
                for lid in edge.lanes:
                    lane = self.getLane(lid)
                    s, d = lane.course_spline.cartesian_to_frenet1D(x, y)
                    s, d = [round(_, 2) for _ in (s, d)]
                    # print("lane = %s, d = %f, lanewidth/2 = %f" % (lid, d, lane.width/2))
                    if s >= -0.01 and s <= lane.spline_length+0.01 and abs(d) <= lane.width / 2 + 0.01:
                        return lid
        
        junc_laneID, min_h = None, np.Inf
        for jid in juncIDs:
            junc = self.getJunction(jid)
            if junc.shape:
                is_injunc = judge_point_in_polygon(junc.shape, (x, y))
                if is_injunc <= 0:
                    for lid in junc.JunctionLanes:
                        lane = self.getLane(lid)
                        s, d = lane.course_spline.cartesian_to_frenet1D(x, y)
                        if s >= 0 and s <= lane.spline_length and abs(d) <= lane.width / 2:
                            if yaw is None:
                                return lid
                            else:
                                h = abs(lane.course_spline.calc_yaw(s) - yaw)
                                h = np.arctan2(np.sin(h), np.cos(h))
                                if h < min_h:
                                    min_h = h
                                    junc_laneID = lid
            
        return junc_laneID

    

