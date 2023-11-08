from datetime import datetime

import pandas as pd
import torch as torch

from action import Action
from gpu_utils import profile, STFTUtils


class GpuAction(Action):
    """
    Defines a class that performs GPU computation
    """
    def __init__(self):
        self.def_params = {'bs': 32, 'sample_len': 50*1024, 'n_fft': 2048, 'win_length': 512}
        cols = ['group', 'impl'] + list(self.def_params.keys()) + ['mean', 'sd', 'min', 'max']
        self.results = pd.DataFrame(columns=cols)

    @staticmethod
    def __add_result(results, group, impl, params, times):
        """
        Add results to a pandas table
        :param results:
        :param group:
        :param impl:
        :param params:
        :param times:
        :return:
        """
        row = {'group': group, 'impl': impl, 'mean': times.mean(), 'sd': times.std(), 'min': times.min(),
               'max': times.max()}
        row.update(params)
        results.loc[len(results)] = row

    def perform(self, batch_size: int, impl: str = "GPU"):
        """
        Perform an example computation that uses GPU
        :param batch_size: Batch Size; possible values 1, 4, 8, 16, 32, 64
        :param impl: GPU implementation, Possible values GPU, GPU+copy1, GPU+copy2, CPU
        :return:
        """
        start = datetime.now()
        gpu = torch.device("cuda")
        params = self.def_params.copy()
        params['bs'] = batch_size
        if impl == "GPU":
            self.__add_result(self.results, 'bs', 'GPU', params, profile(STFTUtils.torch_stft, **params, device=gpu))
        elif impl == "GPU+copy1":
            self.__add_result(self.results, 'bs', 'GPU+copy1', params, profile(STFTUtils.torch_stft, **params,
                                                                               device=gpu, copyto=True))
        elif impl == "GPU+copy2":
            self.__add_result(self.results, 'bs', 'GPU+copy2', params,
                              profile(STFTUtils.torch_stft, **params, device=gpu, copyto=True, copyfrom=True))
        else:
            self.__add_result(self.results, 'bs', 'CPU', params, profile(STFTUtils.torch_stft, **params, device='cpu'))

        #print(f"GPU computation: {self.results.to_json()} took: {(datetime.now() - start).total_seconds()}")
