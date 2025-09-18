import numpy as np
from typing import Union, Tuple, List
import copy

class BlockMatrix(object):
    """
    block matrix class
    """
    def __init__(self, shape:Union[int, Tuple[int, int]], data:List = None, rows:List = None, cols:List = None, dtype=float) -> None:
        if isinstance(shape, int):
            self.shape = (shape, shape)
        elif isinstance(shape, Tuple):
            assert len(shape) == 2
            self.shape = shape
        else:
            raise ValueError('shape must be int or tuple[int, int]')
        
        self.data, self.idxs = [], []
        if data is not None:
            if rows is not None and cols is not None:
                if len(rows) == len(data) and len(cols) == len(data):
                    self.data = [np.matrix(d_, dtype=dtype) for d_ in data]
                    self.idxs = [(int(r_), int(c_)) for r_, c_ in zip(rows, cols)]
                else:
                    raise ValueError('rows and cols should has the same length as data')
                
            else:
                if len(data) == self.shape[0] * self.shape[1]:
                    self.data = [np.matrix(d_, dtype=dtype) for d_ in data]
                    self.idxs = [(r_, c_) for c_ in range(self.shape[1]) for r_ in range(self.shape[0])]
                else:
                    raise ValueError('length of data should be equal to total number of items in blackmatrix shape[0] * shape[1]')

        
        self.dtype = dtype
    
    @property
    def matrix(self) -> np.matrix:

        rows_, cols_ = np.zeros(self.shape[0],dtype=int), np.zeros(self.shape[1],dtype=int)
        for d_, (r_, c_), in zip(self.data, self.idxs):
            size_d = d_.shape
            if rows_[r_] == 0:
                rows_[r_] = size_d[0]
            elif rows_[r_] != size_d[0]:
                raise ValueError('blocks in row %d have multiple sizes'%r_)
            if cols_[c_] == 0:
                cols_[c_] = size_d[1]
            elif cols_[c_] != size_d[1]:
                raise ValueError('blocks in column %d have multiple sizes'%c_)
        
        total_rows, total_cols = np.sum(rows_), np.sum(cols_)
        mat = np.zeros(shape=(total_rows,total_cols), dtype=self.dtype)
        for d_, (r_, c_), in zip(self.data, self.idxs):
            mat[np.sum(rows_[:r_]):np.sum(rows_[:r_+1]), np.sum(cols_[:c_]):np.sum(cols_[:c_+1])] = d_

        return np.matrix(mat, dtype=self.dtype)

    @property
    def copy(self) -> "BlockMatrix":
        return copy.deepcopy(self)

    @property
    def T(self) -> "BlockMatrix":
        return BlockMatrix(
            shape = (self.shape[1], self.shape[0]),
            data = [d_.T for d_ in self.data],
            rows = [idx_[1] for idx_ in self.idxs],
            cols = [idx_[0] for idx_ in self.idxs],
            dtype = self.dtype
        )
    
    def __getitem__(self, idx:Tuple[int, int]):
        if idx in self.idxs:
            return self.data[self.idxs.index(idx)]
        return None

    def __setitem__(self, idx:Tuple[int, int], value):
        if idx in self.idxs:
            i_ = self.idxs.index(idx)
            if value is None:
                self.data.pop(i_)
                self.idxs.pop(i_)
            else:
                self.data[i_] = np.matrix(value, dtype=self.dtype)
        else:
            self.data.append(np.matrix(value, dtype=self.dtype))
            self.idxs.append(idx)

    def __neg__(self):
        return BlockMatrix(
            shape = self.shape,
            data = [-d_ for d_ in self.data],
            rows = [idx_[0] for idx_ in self.idxs],
            cols = [idx_[1] for idx_ in self.idxs],
            dtype = self.dtype
        )

    def __add__(self, other:"BlockMatrix") -> "BlockMatrix":
        assert self.shape == other.shape

        bmat = self.copy
        for d_, idx_ in zip(other.data, other.idxs):
            if idx_ in bmat.idxs:
                bmat.data[bmat.idxs.index(idx_)] += d_
            else:
                bmat.data.append(d_)
                bmat.idxs.append(idx_)

        return bmat
    
    def __sub__(self, other:"BlockMatrix") -> "BlockMatrix":
        return self.__add__(-other)

def eye(shape:int, data, dtype=float):
    if isinstance(data, List):
        assert len(data) == shape
        data = [np.matrix(d_, dtype) for d_ in data]
    else:
        data = [np.matrix(data)] * shape
    
    return BlockMatrix(shape, data, list(range(shape)), list(range(shape)), dtype)