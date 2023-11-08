import random

from mpi4py import MPI
from repast4py import core, space, schedule
import random as rr
import pandas as pd
import torch as torch

from gpu_utils import STFTUtils, profile


class GPUAgent(core.Agent):
    TYPE = 0

    def __init__(self, agent_id, rank):
        super().__init__(agent_id, GPUAgent.TYPE, rank)
        self.def_params = {'bs': 32, 'sample_len': 50*1024, 'n_fft': 2048, 'win_length': 512}
        cols = ['group', 'impl'] + list(self.def_params.keys()) + ['mean', 'sd', 'min', 'max']
        self.results = pd.DataFrame(columns=cols)

    @staticmethod
    def __add_result(results, group, impl, params, times):
        row = {'group': group, 'impl': impl, 'mean': times.mean(), 'sd': times.std(), 'min': times.min(),
               'max': times.max()}
        row.update(params)
        results.loc[len(results)] = row

    def update(self, batch_size: int):
        gpu = torch.device("cuda")
        print(f'Agent: {self.id} running bs={batch_size}')
        params = self.def_params.copy()
        params['bs'] = batch_size
        self.__add_result(self.results, 'bs', 'GPU', params, profile(STFTUtils.torch_stft, **params, device=gpu))
        self.__add_result(self.results, 'bs', 'GPU+copy1', params, profile(STFTUtils.torch_stft, **params,
                                                                           device=gpu, copyto=True))
        self.__add_result(self.results, 'bs', 'GPU+copy2', params,
                   profile(STFTUtils.torch_stft, **params, device=gpu, copyto=True, copyfrom=True))
        self.__add_result(self.results, 'bs', 'CPU', params, profile(STFTUtils.torch_stft, **params, device='cpu'))
        print(f'Agent: {self.id} completed bs={batch_size}')


# Define a ResourceAllocationModel
class GPUModel:
    def __init__(self):
        comm = MPI.COMM_WORLD
        # create the schedule
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(100)

        box = space.BoundingBox(xmin=0, xextent=100, ymin=0, yextent=200, zmin=0, zextent=0)
        self.space = space.SharedCSpace('ResourceSpace', bounds=box, borders=space.BorderType.Sticky,
                                        occupancy=space.OccupancyType.Multiple,
                                        buffer_size=2, comm=comm, tree_threshold=100)

        self.agents = [GPUAgent(x, rr.randint(1, 1000)) for x in range(1000)]
        for x in self.agents:
            self.space.add(x)

    def step(self):
        batch_sizes = [1, 4, 8, 16, 32, 64]
        for agent in self.agents:
            bs = random.choice(batch_sizes)
            agent.update(batch_size=bs)

    def start(self):
        self.runner.execute()


if __name__ == '__main__':

    # Run the simulation
    model = GPUModel()
    model.start()

    # Display the results
    # Display the results
    for i, agent in enumerate(model.agents):
        print(f"Agent {i} Results: {agent.results}")
