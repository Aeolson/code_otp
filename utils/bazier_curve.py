from typing import Union
import math
import numpy as np
from scipy.spatial import ConvexHull

class BazierCurve2D:
    def __init__(self, ctrlp: list[list[float]] = None) -> None:
        self.order: int = None
        self.ratio: list = None
        self.ctrlp: list = None
        self.convex_hull: list = None
        if ctrlp is not None:
            self.create_from_points(ctrlp)
    
    def create_from_points(self, pts: list[list[float]]):
        self.order = len(pts) - 1
        self.ratio = [ math.factorial(self.order) / (math.factorial(i) * math.factorial(self.order-i)) for i in range(self.order+1) ]
        self.ratio = np.array(self.ratio, dtype=float)

        self.ctrlp = np.array(pts, dtype=float) # (order, D)
        cvh = ConvexHull(pts)
        self.convex_hull = self.ctrlp[cvh.vertices]
    
    def calc_points(self, t: Union[float, list[float]]):
        """
        t: can be a 'float' or a 'list[float]', where 0 <= t <= 1
        """
        if self.ctrlp is None:
            raise ValueError("The Bazier curve is not created !!!")
        
        t = np.array([t], float).reshape(-1)
        coef = [self.ratio[i] * np.power(1-t, self.order-i) * np.power(t, i) for i in range(self.order+1)] # (order, N)
        coef = np.array(coef, float).transpose(1,0) # (N, order)
        pts = np.matmul(coef, self.ctrlp)
        if len(pts) == 1:
            return pts.reshape(-1)
        else:
            return pts

    def calc_sample_curve(self, n: int):
        t = np.linspace(0.0, 1.0, n)
        ps = self.calc_points(t)
        return ps

