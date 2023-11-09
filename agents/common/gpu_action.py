from datetime import datetime
from random import random

import pandas as pd
import torch as torch

from .action import Action
from .gpu_utils import profile, STFTUtils


class GpuAction(Action):
    """
    Defines a class that performs GPU computation
    """

    def __init__(self):
        self.def_params = {'bs': 32, 'sample_len': 50 * 1024, 'n_fft': 2048, 'win_length': 512}
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

    def get_device(self):
        # Get the number of GPUs available
        num_gpus = torch.cuda.device_count()

        # Make sure there is at least one GPU
        if num_gpus > 0:
            # Select a random GPU
            random_gpu = random.randint(0, num_gpus - 1)
            device = torch.device(f"cuda:{random_gpu}")
            torch.cuda.set_device(device)  # Set the random GPU as current
        else:
            device = torch.device("cpu")

        print(f"Using device: {device}")
        # Now you can pass this device to your tensors or models
        return device

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

        # print(f"GPU computation: {self.results.to_json()} took: {(datetime.now() - start).total_seconds()}")
        #print(f"GPU computation: took: {(datetime.now() - start).total_seconds()}")
