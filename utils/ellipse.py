from typing import Tuple
import numpy as np
from math import sin, cos, asin, acos, sqrt
from sympy import solve, symbols, Eq

class Ellipse2D:
    
    def __init__(self, centerx: float, centery: float, rx: float, ry: float, rot: float = 0) -> None:
        super().__init__()

        """
        centerx:    x-position of the ellipse center
        centery:    y-position of the ellipse center
        rx:         half of the x-axis
        ry:         half of the y-axis
        rot:        rotation angle [rad] of the ellipse
        """

        self.centerx = centerx
        self.centery = centery
        self.rx = rx
        self.ry = ry
        self.rot = rot

        # x' = Rx
        self.rmat = np.matrix([
            [cos(rot), -sin(rot)],
            [sin(rot),  cos(rot)]
        ], float)

        R_ = self.rmat.I
        P_ = np.matrix([
            [1/rx**2,   0.     ],
            [0.     ,   1/ry**2]
        ], float)

        ''' x.T * P * x = 1'''
        self.P = R_.T * P_ * R_

    def value(self, x: float, y: float):
        p = np.matrix([x - self.centerx, y - self.centery], float)
        return (p * self.P * p.T).item()

    def judge_inside(self, x: float, y: float):
        p = np.matrix([x - self.centerx, y - self.centery], float)
        return np.all(p * self.P * p.T <= 1)
    
    def calc_boundary_y_from_x(self, x: float) -> list[float, float]:
        """
        a_11 * x^2 + 2*a_12 * x*y + a_22 * y^2 = 1
        """
        x = x - self.centerx
        a11_, a12_, a22_ = self.P[0,0], self.P[0,1], self.P[1,1]

        y = symbols('y', real=True)
        f = Eq(a22_ * y**2 + 2*a12_*x * y + a11_*x**2, 1)
        res = solve(f, y)
        return [self.centery + _ for _ in res]
    
    def calc_boundary_x_from_y(self, y: float) -> list[float, float]:
        """
        a_11 * x^2 + 2*a_12 * x*y + a_22 * y^2 = 1
        """
        y = y - self.centery
        a11_, a12_, a22_ = self.P[0,0], self.P[0,1], self.P[1,1]

        x = symbols('x', real=True)
        f = Eq(a11_ * x**2 + 2*a12_*y * x + a22_*y**2, 1)
        res = solve(f, x)
        return [self.centery + _ for _ in res]
    
    def calc_boundary_xy_from_angle(self, r: float) -> Tuple[float, float]:
        p = [self.rx * cos(r), self.ry * sin(r)]
        p_ = self.rmat * np.matrix(p).T

        return [
            self.centerx + p_[0,0],
            self.centery + p_[1,0]
        ]

    @property
    def max_x(self) -> float:
        return self.centerx + sqrt(self.rx**2 * cos(self.rot)**2 + self.ry**2 * sin(self.rot)**2)
    
    @property
    def min_x(self) -> float:
        return self.centerx - sqrt(self.rx**2 * cos(self.rot)**2 + self.ry**2 * sin(self.rot)**2)
    
    @property
    def max_y(self) -> float:
        return self.centery + sqrt(self.rx**2 * sin(self.rot)**2 + self.ry**2 * cos(self.rot)**2)
    
    @property
    def min_y(self) -> float:
        return self.centery - sqrt(self.rx**2 * sin(self.rot)**2 + self.ry**2 * cos(self.rot)**2)
    
    @property
    def tangent_rectangle(self) -> list[list[float, float]]:
        return [self.min_x, self.min_y, self.max_x, self.max_y]
    
    @property
    def tangent_circle(self) -> list[float, float, float]:
        x, y = self.centerx, self.centery
        radius = max(self.rx, self.ry)
        return [x, y, radius]
    
    