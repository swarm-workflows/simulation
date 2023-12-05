import argparse
import concurrent.futures
import time
from datetime import datetime
from typing import List

import agentpy as ap
import random

from agents.common.gpu_utils import STFTUtils
from ..common.cpu_action import CpuAction
from ..common.gpu_action import GpuAction
from ..common.resource import Resource, ResourceType


# Define a Task Agent that can request different types of resources and has a priority
class TaskAgent(ap.Agent):
    class ResourceStatus:
        def __init__(self, allocated: bool, used: bool, resource: Resource = None):
            self.allocated = allocated
            self.used = used
            self.resource = resource

    def setup(self, resource_types: List[ResourceType]):
        self.start = None
        self.priority = random.randint(1, 100)  # Higher number means higher priority

        # Simplified single resource type
        random_element = random.choice(resource_types)
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
                #action = GpuAction()
                #action.perform(batch_size=64)
                print("GPU Action done")
            elif resource_type == ResourceType.CPU:
                # Perform Jax calculations
                #action = CpuAction()
                #action.perform()
                print("CPU Action done")

            self.release_resource(resource_type=resource_type)
        if done_count == len(self.resources_needed):
            print(f"Task: {self.id} took time: {(datetime.now() - self.start).total_seconds()} to complete")

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
        print(f"Performing task for :{task}")
        task.perform_task()


# Define the main Model
class ResourceAllocationModel(ap.Model):
    MAX_THREADS = 16

    def __init__(self, parameters=None, _run_id=None, num_cpu: int = 100, num_gpu: int = 50,
                 num_nvme: int = 10, num_agents: int = 100, **kwargs):
        super(ResourceAllocationModel, self).__init__(parameters, _run_id)
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.MAX_THREADS)

        # Create resources
        self.resources = {
            ResourceType.CPU: [Resource(ResourceType.CPU) for _ in range(num_cpu)],
            ResourceType.GPU: [Resource(ResourceType.GPU) for _ in range(num_gpu)],
        }

        # Create tasks with priorities
        self.tasks = [TaskAgent(self, list(self.resources.keys())) for _ in range(num_agents)]

    def get_available_resource(self, resource_type: ResourceType):
        for resource in self.resources[resource_type]:
            if resource.is_available:
                return resource
        return None  # If no available resource is found

    def setup(self):
        pass

    def step(self):

        while True:
            # Sort tasks by priority (higher first)
            # Allocate tasks using a greedy algorithm based on priority

            allocated_tasks = []
            done_count = 0

            sorted_tasks = sorted(self.tasks, key=lambda t: t.priority, reverse=True)
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
                        print(f"Allocated task: {task} resource: {available_resource}")

            futures = []
            # Run tasks in parallel
            for t in allocated_tasks:
                print(f"Triggering process for {t}")
                future = self.process_pool.submit(TaskExecutor.run, t)
                futures.append(future)

            concurrent.futures.wait(futures)
            if done_count == len(self.tasks):
                break

            time.sleep(5)

        self.process_pool.shutdown(wait=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resource Allocation Model Parameters')
    parser.add_argument('--num_agents', type=int, default=100, help='Number of task agents')
    parser.add_argument('--num_gpu', type=int, default=50, help='Number of GPU resources')
    parser.add_argument('--num_cpu', type=int, default=100, help='Number of CPU resources')
    parser.add_argument('--num_nvme', type=int, default=10, help='Number of NVMe resources')
    parser.add_argument('--steps', type=int, default=100, help='Number of steps to run the model')

    args = parser.parse_args()

    # Initialize the Sample cache for GPU computations
    STFTUtils.initialize()

    model = ResourceAllocationModel(num_cpu=args.num_cpu, num_gpu=args.num_gpu, num_nvme=args.num_nvme,
                                    num_agents=args.num_agents)
    start = datetime.now()
    results = model.run(steps=args.steps)
    print(f"Total time: {(datetime.now() - start).total_seconds()}")
    # Results analysis here
