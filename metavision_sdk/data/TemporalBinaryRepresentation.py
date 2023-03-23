import numpy as np
from time import time
import torch

class TemporalBinaryRepresentation:
    """
    @brief: Class for Temporal Binary Representation using N bits
    """

    def __init__(self, N: int, width: int, height: int, incremental: bool=False):
        self.N = N
        self.width = width
        self.height = height
        self._mask = np.ones((self.N, self.height, self.width))
        self.incremental = incremental

        # Build the mask
        for i in range(N):
            self._mask[i, :, :] = 2 ** i

        if self.incremental:
            self.frame_stack = torch.zeros((self.N, self.height, self.width)).cuda()
            self.index = self.N-1
            self.frame = torch.zeros((self.height, self.width)).cuda()

    def encode(self, mat: np.array) -> np.array:
        """
        @brief: Encode events using binary encoding
        @param: mat
        @return: Encoded frame
        """

        frame = np.sum((mat * self._mask), 0) / (2 ** (self.N-1))
        return frame

    def incremental_update(self, mat: np.array) -> np.array:
        """
        Incrementally updates the frame representation by shifting the N-channel tensor and adding the new Most Significant Bit
        MSB is the last channel in self.frame
        """
        assert self.incremental == True
        tt = time()

        # SOL 1
        # self.index = (self.index + 1) % self.N
        # print(f'incremental {time() - tt}')
        # self.frame[self.index] = mat*(2**(self.N-1))*2 # times 2 because I'm going to divide by 2 afterwards to perform the shift
        # print(f'incremental {time() - tt}')
        # self.frame /=2
        # print(f'incremental {time() - tt}')
        # e = np.sum(self.frame, 0) / (2 ** (self.N-1))
        # print(f'incremental {time() - tt}')

        # SOL 2
        # self.index = (self.index + 1) % self.N
        # print(f'incremental {time() - tt}')
        # self.frame_stack /= 2
        # print(f'incremental {time() - tt}')
        # self.frame_stack[self.index] = mat
        # print(f'incremental {time() - tt}')
        # e = np.sum(self.frame_stack, 0) #/ (2 ** (self.N-1))
        # print(f'incremental {time() - tt}')

        # SOL 3
        cuda_mat = torch.tensor(mat).cuda()
        self.index = (self.index + 1) % self.N
        print(f'incremental {time() - tt}')
        self.frame -= self.frame_stack[self.index]
        print(f'incremental {time() - tt}')
        self.frame /= 2
        print(f'incremental {time() - tt}')
        self.frame_stack /= 2
        print(f'incremental {time() - tt}')
        self.frame_stack[self.index] = cuda_mat
        print(f'incremental {time() - tt}')
        self.frame += cuda_mat
        print(f'incremental {time() - tt}')

        return self.frame
