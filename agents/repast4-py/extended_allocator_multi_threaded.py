import argparse
import concurrent.futures
import time
from datetime import datetime
from typing import List

from mpi4py import MPI
from repast4py import core, random, space, schedule
import random

from agents.common.cpu_action import CpuAction
from agents.common.gpu_action import GpuAction
from agents.common.disk_action import DiskAction


# Define a Task Agent that can request different types of resources and has a priority
from agents.common.gpu_utils import STFTUtils
from agents.common.resource import Resource, ResourceType


class TaskAgent(core.Agent):
    class ResourceStatus:
        def __init__(self, allocated: bool, used: bool, resource: Resource = None):
            self.allocated = allocated
            self.used = used
            self.resource = resource

    TYPE = 0

    def __init__(self, agent_id, rank, resource_types: List[ResourceType]):
        super().__init__(agent_id, TaskAgent.TYPE, rank)
        self.priority = random.randint(1, 1000)
        self.start = None
        # Simplified single resource type
        random_element = random.choices(resource_types, [0.75, 0.25])[0]
        self.resources_needed = {
            random_element: TaskAgent.ResourceStatus(False, False)
        }

        # Multiple resources
        '''
        resource_sample = random.sample(resource_types, random.randint(1, len(resource_types)))
        for r_type in resource_sample:
            # Tuple representing allocation status and completion status
            self.resources_needed[r_type] = TaskAgent.ResourceStatus(False, False)
        '''

    def perform_task(self):
        done_count = 0
        for resource_type, resource_status in self.resources_needed.items():
            if not resource_status.allocated or resource_status.used or resource_status.resource is None:
                done_count += 1
                continue

            if self.start is None:
                self.start = datetime.now()

            if resource_type == ResourceType.GPU:
                # Perform STFT computations
                action = GpuAction()
                action.perform(batch_size=16)
            elif resource_type == ResourceType.CPU:
                # Perform Jax calculations
                action = CpuAction()
                action.perform()
            elif resource_type == ResourceType.NVME:
                # Perform disk operations
                action = DiskAction()
                action.perform()

            self.release_resource(resource_type=resource_type)
        if done_count == len(self.resources_needed):
            print(f"Task: {self.id} took time: {(datetime.now()-self.start).total_seconds()} to complete")

    def has_resources(self):
        for resource, status in self.resources_needed.items():
            if status.allocated:
                return True
        return False

    def is_done(self):
        for resource_type, status in self.resources_needed.items():
            if not status.used:
                return False
        return True

    def allocate_resource(self, resource: Resource):
        if resource.resource_type in self.resources_needed:
            self.resources_needed[resource.resource_type].allocated = True
            self.resources_needed[resource.resource_type].resource = resource
            resource.allocate(task=self)

    def release_resource(self, resource_type: ResourceType):
        if resource_type in self.resources_needed:
            self.resources_needed[resource_type].used = True
            self.resources_needed[resource_type].allocated = False
            self.resources_needed[resource_type].resource.release()
            self.resources_needed[resource_type].resource = None


class TaskExecutor:
    @staticmethod
    def run(task: TaskAgent):
        task.perform_task()


# Define a ResourceAgent
class ResourceAllocator(core.Agent):
    TYPE = 0
    MAX_THREADS = 16

    def __init__(self, model, agent_id, rank, num_cpu: int = 100, num_gpu: int = 50,
                 num_nvme: int = 10):
        super().__init__(agent_id, ResourceAllocator.TYPE, rank)
        self.model = model
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_THREADS,
                                                                 thread_name_prefix=self.__class__.__name__)

        # Create resources
        self.resources = {
            ResourceType.CPU: [Resource(ResourceType.CPU) for _ in range(num_cpu)],
            ResourceType.GPU: [Resource(ResourceType.GPU) for _ in range(num_gpu)]
        }

    def get_available_resource(self, resource_type: ResourceType):
        for resource in self.resources[resource_type]:
            if resource.is_available:
                return resource
        return None  # If no available resource is found

    def allocate_tasks(self):

        while True:
            # Sort tasks by priority (higher first)
            # Allocate tasks using a greedy algorithm based on priority

            allocated_tasks = []
            done_count = 0

            sorted_tasks = sorted(self.model.tasks, key=lambda t: t.priority, reverse=True)
            for task in sorted_tasks:
                if task.is_done():
                    done_count += 1
                    continue
                for resource_type, resource_status in task.resources_needed.items():
                    if resource_status.used or resource_status.allocated:
                        continue

                    available_resource = self.get_available_resource(resource_type=resource_type)
                    if available_resource:
                        task.allocate_resource(resource=available_resource)
                        allocated_tasks.append(task)

            # Run tasks in parallel
            for t in allocated_tasks:
                self.thread_pool.submit(TaskExecutor.run, t)

            if done_count == len(self.model.tasks):
                break

            time.sleep(10)

        self.thread_pool.shutdown(wait=True)


# Define a ResourceAllocationModel
class ResourceAllocationModel:
    def __init__(self, num_tasks: int = 100, num_allocators: int = 1, num_cpu: int = 100, num_gpu: int = 50,
                 num_nvme: int = 10, steps: int = 100):
        comm = MPI.COMM_WORLD
        # create the schedule
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(steps)

        box = space.BoundingBox(xmin=0, xextent=100, ymin=0, yextent=200, zmin=0, zextent=0)
        self.space = space.SharedCSpace('ResourceSpace', bounds=box, borders=space.BorderType.Sticky,
                                        occupancy=space.OccupancyType.Multiple,
                                        buffer_size=2, comm=comm, tree_threshold=100)

        # Create tasks with priorities
        rank = comm.Get_rank()
        self.tasks = [TaskAgent(x, rank, [ResourceType.CPU, ResourceType.GPU]) for x in range(num_tasks)]

        for x in self.tasks:
            self.space.add(x)

        self.allocated_tasks = {}

        self.allocators = [ResourceAllocator(self, x, random.randint(1, num_allocators),
                                             num_cpu=num_cpu, num_nvme=num_nvme,
                                             num_gpu=num_gpu) for x in range(num_allocators)]
        for x in self.allocators:
            self.space.add(x)

    def step(self):
        for allocators in self.allocators:
            allocators.allocate_tasks()

    def start(self):
        self.runner.execute()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resource Allocation Model Parameters')
    parser.add_argument('--num_agents', type=int, default=100, help='Number of task agents')
    parser.add_argument('--num_gpu', type=int, default=50, help='Number of GPU resources')
    parser.add_argument('--num_cpu', type=int, default=100, help='Number of CPU resources')
    parser.add_argument('--num_nvme', type=int, default=10, help='Number of NVMe resources')
    parser.add_argument('--steps', type=int, default=100, help='Number of steps to run the model')

    args = parser.parse_args()

    # Initialize the Sample cache for GPU computations
    STFTUtils.initialize(bs=16)

    start = datetime.now()
    # Run the simulation
    model = ResourceAllocationModel(num_tasks=args.num_agents, num_gpu=args.num_gpu, num_cpu=args.num_cpu,
                                    num_nvme=args.num_nvme, steps=args.steps)
    model.start()
    print(f"Total time: {(datetime.now() - start).total_seconds()}")
