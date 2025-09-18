import os, sys
import numpy as np
from numpy import ndarray, array, matrix
from dataclasses import dataclass
from typing import Tuple, List
import block_matrix as bm
import cvxopt
import cvxpy
from copy import deepcopy


def ctrb(A:ndarray, B:ndarray):
    A, B = [np.matrix(_) for _ in [A, B]]
    return np.hstack([A**i * B for i in range(A.shape[0])])

def rank(A:ndarray):
    return np.linalg.matrix_rank(A)

def eig(A:ndarray):
    return np.linalg.eigvals(A)

def is_symmetry(A:ndarray):
    return np.all(A.T-A == 0)

def is_controllable(A:ndarray, B:ndarray):
    if rank(ctrb(A,B)) == A.shape[0]:
        return True
    else:
        return False

def discrete_algebraic_Riccati_equation(A:ndarray, B:ndarray, Q:ndarray, R:ndarray, max_error:float=1e-5, max_iter:int=200) -> Tuple[ndarray, ndarray]:

    A, B, Q, R = [np.matrix(_) for _ in [A, B, Q, R]]

    W = Q
    err = np.inf
    n_iter = 0
    while err > max_error and n_iter < max_iter:
        We = A.T*W*A + Q - A.T*W*B * (B.T*W*B + R).I * B.T*W*A
        err = np.linalg.norm(We - W, np.inf)
        W = We
        n_iter += 1
    
    K = -(B.T * W * B + R).I * B.T * W * A

    return np.array(K), np.array(W)

def lqr(A:ndarray, B:ndarray, Q:ndarray, R:ndarray, xt:ndarray, n:int = 1):
    """
    LQR output = K * xt
    """
    n_input = B.shape[1]
    K, W = discrete_algebraic_Riccati_equation(A, B, Q, R)
    u_out = []
    A, B, K, xt = (np.matrix(_) for _ in [A, B, K, xt])
    for i in range(n):
        ut = K * (A + B*K)**i * xt.T
        u_out.append(np.array(ut).reshape(n_input))
    return np.array(u_out)

def qp(H:ndarray, f:ndarray, Ale:ndarray=None, ble:ndarray=None, Aeq:ndarray=None, beq:ndarray=None, lb:ndarray=None, ub:ndarray=None, x0:ndarray=None):
    # min   0.5 * x.T * H * x + f.T * x
    # s.t.  Ale * x <= ble
    #       Aeq * x =  beq
    #       lb  <= x <= ub
    n = H.shape[0]

    if not is_symmetry(H):
        H = ( H.T + H ) / 2
    
    P = H
    q = f.reshape(-1,1)
    
    G, h = [], []
    if Ale is not None:
        G.append( Ale )
        h.append( ble.reshape(-1,1) )
    if lb is not None:
        G.append( -np.eye(n) )
        h.append( -lb.reshape(-1,1) )
    if ub is not None:
        G.append( np.eye(n))
        h.append( ub.reshape(-1,1) )
    
    if not G:
        G, h = None, None
    else:
        G = np.vstack(G)
        h = np.vstack(h)

    A = Aeq
    b = beq.reshape(-1,1) if beq is not None else None
    x0 = x0.reshape(-1,1) if x0 is not None else None

    # remove invalid rows in G, h
    def where(A: ndarray):
        return A.nonzero()
    
    if G is not None:
        singleton_row = np.array(np.sum(np.abs(G) > 1e-6, axis=1) == 1).flatten()
        cols = where(G[singleton_row, :])[1]
        rows = where(singleton_row)[0]
        if len(rows) > 0:
            G1, h1 = np.zeros((G.shape[1], G.shape[1])), np.ones(G.shape[1]) * np.inf
            G2, h2 = np.zeros((G.shape[1], G.shape[1])), np.ones(G.shape[1]) * np.inf
            for r, c in zip(rows, cols):
                val = h[r] / abs(G[r,c])
                if G[r,c] > 0: # G1 x <= h1
                    G1[c,c] = 1.0
                    h1[c] = min(h1[c], val)
                    # if h1[c] <= lb[c]: # infeasible
                    #     return x0, False
                else: # G2 x <= h2
                    G2[c,c] = -1.0
                    h2[c] = min(h2[c], val)
                    # if -h2[c] >= ub[c]: # infeasible
                    #     return x0, False
            m1 = np.count_nonzero(G1, axis=1) > 0
            G1, h1 = G1[m1,:], h1[m1].reshape(-1,1)
            m2 = np.count_nonzero(G2, axis=1) > 0
            G2, h2 = G2[m2,:], h2[m2].reshape(-1,1)

            G = G[np.logical_not(singleton_row),:]
            h = h[np.logical_not(singleton_row),:]
            if len(G1) > 0:
                G = np.vstack([G, G1])
                h = np.vstack([h, h1])
            if len(G2) > 0:
                G = np.vstack([G, G2])
                h = np.vstack([h, h2])

    P, q, G, h, A, b, x0 = [cvxopt.matrix(_) if _ is not None else None for _ in [P, q, G, h, A, b, x0]]
    # try:
    #     from mosek import iparam
    #     sol = cvxopt.solvers.qp(P, q, G, h, A, b, initvals=x0, solver='mosek', options={'mosek':{iparam.log:0}})
    # except:
    #     sol = cvxopt.solvers.qp(P, q, G, h, A, b, initvals=x0, options={'show_progress':False})

    sol = cvxopt.solvers.qp(P, q, G, h, A, b, initvals=x0, options={'show_progress':False})

    x = np.array(sol['x']).reshape(-1)
    flag = True if sol['status'] == 'optimal' else False
    return x, flag

def linear_mpc(n_predict: int, n_control: int, A:ndarray, B:ndarray, Q:ndarray, R:ndarray, xt:ndarray, x_min:ndarray=None, x_max:ndarray=None, u_min:ndarray=None, u_max:ndarray=None, Vt:ndarray=None, Bt:ndarray=None):
    """
    Linear MPC
    Input size:
        A:      (n_state, n_state)
        B:      (n_state, n_input)
        Q:      (n_state, n_state)
        R:      (n_input, n_input)
        xt:     (n_state,)
        x_min:  (n_predict, n_state), @t = 1,2,...,N
        x_max:  (n_predict, n_state), @t = 1,2,...,N
        u_min:  (n_input,)
        u_max:  (n_input,)
        Vt,Bt:  convex inequal constraints for the terminal state, Vt * x_N.T <= Bt
    """

    n_state, n_input = B.shape
    A, B, Q, R, xt = [matrix(_) for _ in [A, B, Q, R, xt]]

    # state-space: x_{k+1} = M * x0 + S * U
    bM = bm.BlockMatrix(shape=(n_predict, 1))
    bS = bm.BlockMatrix(shape=(n_predict, n_control))
    for i in range(n_predict):
        bM[i,0] = A ** (i+1)
        for j in range(min(i+1, n_control)):
            bS[i,j] = A ** (i-j) * B
    mM, mS = bM.matrix, bS.matrix

    # cost function: J = 0.5 * Phi.T * U * Phi + Theta.T * U
    mQ, mR = bm.eye(n_predict, Q).matrix, bm.eye(n_control, R).matrix
    mPhi = mS.T * mQ * mS + mR
    mTheta  = mS.T * mQ * mM * xt.T
    
    lb = np.array([u_min] * n_control, float).reshape(-1)
    ub = np.array([u_max] * n_control, float).reshape(-1)

    # print(n_predict)
    Ale, ble = [], []
    if x_max is not None:
        x_max = np.matrix(x_max)
        for i in range(n_predict):
            Si = mS[i*n_state:(i+1)*n_state,:]
            Mi = mM[i*n_state:(i+1)*n_state,:]
            Ale.append(  Si  )
            ble.append(  x_max[i].T - Mi * xt.T  )
    if x_min is not None:
        x_min = np.matrix(x_min)
        for i in range(n_predict):
            Si = mS[i*n_state:(i+1)*n_state,:]
            Mi = mM[i*n_state:(i+1)*n_state,:]
            Ale.append( -Si  )
            ble.append( -x_min[i].T + Mi * xt.T  )
    
    # terminal constraints
    if Vt is not None and Bt is not None:
        Vt, Bt = np.matrix(Vt), np.matrix(Bt).reshape(-1,1)
        Sn = mS[-n_state:, :]
        Mn = mM[-n_state:, :]
        Ale.append(Vt * Sn)
        ble.append(Bt - Vt * Mn * xt.T)

    if Ale and ble:
        Ale = bm.BlockMatrix(shape=(len(Ale), 1), data=Ale).matrix
        ble = bm.BlockMatrix(shape=(len(ble), 1), data=ble).matrix
    else:
        Ale, ble = None, None

    u_out, flag = qp(mPhi, mTheta, Ale=Ale, ble=ble, lb=lb, ub=ub)
    u_out = u_out.reshape(n_control, n_input)
    
    return u_out, flag

def linear_mpc_track(n_predict: int, n_control: int, A:ndarray, B:ndarray, Q:ndarray, R:ndarray, xt:ndarray, xr:ndarray=None, ur:ndarray=None, x_min:ndarray=None, x_max:ndarray=None, u_min:ndarray=None, u_max:ndarray=None,  Vt:ndarray=None, Bt:ndarray=None, u0:ndarray=None):
    """
    Linear MPC
    Input size:
        A:      (n_state, n_state)
        B:      (n_state, n_input)
        Q:      (n_state, n_state)
        R:      (n_input, n_input)
        xt:     (n_state,)
        xr:     (n_predict, n_state), @t = 1,2,...,N
        ur:     (n_control, n_input), @t = 0,1,...,Nc-1
        x_min:  (n_predict, n_state), @t = 1,2,...,N
        x_max:  (n_predict, n_state), @t = 1,2,...,N
        u_min:  (n_input,)
        u_max:  (n_input,)
        Vt,Bt:  convex inequal constraints for the terminal state, Vt * x_N.T <= Bt
        u0:     (n_control, n_input), @t = 0,1,...,Nc-1
    """

    n_state, n_input = B.shape
    if xr is None:
        xr = np.zeros((n_predict, n_state), float)
    elif xr.ndim == 1:
        xr = np.repeat(xr.reshape(1, n_state), n_predict, axis=0)

    if ur is None:
        ur = np.zeros((n_control, n_input), float)
    elif ur.ndim == 1:
        ur = np.repeat(xr.reshape(1, n_input), n_control, axis=0)

    A, B, Q, R, xt, xr, ur = [matrix(_) for _ in [A, B, Q, R, xt, xr.reshape(-1,1), ur.reshape(-1,1)]]

    # state-space: x_{k+1} = M * x0 + S * U
    bM = bm.BlockMatrix(shape=(n_predict, 1))
    bS = bm.BlockMatrix(shape=(n_predict, n_control))
    for i in range(n_predict):
        bM[i,0] = A ** (i+1)
        for j in range(min(i+1, n_control)):
            bS[i,j] = A ** (i-j) * B
    mM, mS = bM.matrix, bS.matrix

    # cost function: J = 0.5 * Phi.T * U * Phi + Theta.T * U
    mQ, mR = bm.eye(n_predict, Q).matrix, bm.eye(n_control, R).matrix
    mPhi = mS.T * mQ * mS + mR
    # print(rank(mS), mS.shape, rank(mQ), mQ.shape, rank(mR), mR.shape, rank(mPhi), mPhi.shape)
    mTheta  = mS.T * mQ * (mM * xt.T - xr) - mR * ur
    
    if u_min is not None:
        lb = np.matrix([u_min] * n_control, float).reshape(-1,1)
    else:
        lb = None
    if u_max is not None:
        ub = np.matrix([u_max] * n_control, float).reshape(-1,1)
    else:
        ub = None

    # print(n_predict)
    Ale, ble = [], []
    if x_max is not None:
        x_max = np.matrix(x_max)
        for i in range(n_predict):
            Si = mS[i*n_state:(i+1)*n_state,:]
            Mi = mM[i*n_state:(i+1)*n_state,:]
            Ale.append(  Si  )
            ble.append(  x_max[i].T - Mi * xt.T  )
    if x_min is not None:
        x_min = np.matrix(x_min)
        for i in range(n_predict):
            Si = mS[i*n_state:(i+1)*n_state,:]
            Mi = mM[i*n_state:(i+1)*n_state,:]
            Ale.append( -Si  )
            ble.append( -x_min[i].T + Mi * xt.T  )
    
    # terminal constraints
    if Vt is not None and Bt is not None:
        Vt, Bt = np.matrix(Vt), np.matrix(Bt).reshape(-1,1)
        Sn = mS[-n_state:, :]
        Mn = mM[-n_state:, :]
        Ale.append(Vt * Sn)
        ble.append(Bt - Vt * Mn * xt.T)

    if Ale and ble:
        Ale = bm.BlockMatrix(shape=(len(Ale), 1), data=Ale).matrix
        ble = bm.BlockMatrix(shape=(len(ble), 1), data=ble).matrix
    else:
        Ale, ble = None, None

    if Ale is not None and ble is not None:
        ble = ble - Ale * ur
    if lb is not None:
        lb = lb - ur
    if ub is not None:
        ub = ub - ur
    
    if u0 is not None:
        u0 = np.array(u0).reshape(-1,1)
        
    u_out, flag = qp(mPhi, mTheta, Ale=Ale, ble=ble, lb=lb, ub=ub, x0=u0)
    if not flag:
        return None, False
    u_out = u_out + np.array(ur).reshape(-1)
    u_out = u_out.reshape(n_control, n_input)
    
    return u_out, flag

def convert_system_continuous_to_discrete(Ac: ndarray, Bc: ndarray, Ts: float) -> Tuple[ndarray, ndarray]:
    Ad = np.eye(Ac.shape[0]) + Ac * Ts
    Bd = Bc * Ts
    return Ad, Bd

def check_bounded_constraints(x: ndarray, lb: ndarray = None, ub: ndarray = None) -> bool:
    if lb is not None:
        if np.any(x - lb < 0):
            return False
    if ub is not None:
        if np.any(x - ub > 0):
            return False
    return True
    