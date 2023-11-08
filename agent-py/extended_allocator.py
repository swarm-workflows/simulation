import concurrent.futures
import enum
import time
from datetime import datetime
from enum import Enum
from typing import List

import agentpy as ap
import random

from cpu_action import CpuAction
from gpu_action import GpuAction
from disk_action import DiskAction


class ResourceType(Enum):
    GPU = enum.auto(),
    CPU = enum.auto(),
    NVME = enum.auto()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


# Define a Resource that represents CPU, GPU, or NVME
class ResourceAgent:
    def __init__(self, resource_type: ResourceType):
        self.resource_type = resource_type
        self.is_available = True
        self.task = None

    def allocate(self, task):
        self.is_available = False
        self.task = task

    def release(self):
        self.is_available = True
        self.task = None


# Define a Task Agent that can request different types of resources and has a priority
class TaskAgent(ap.Agent):
    class ResourceStatus:
        def __init__(self, allocated: bool, used: bool, resource: ResourceAgent = None):
            self.allocated = allocated
            self.used = used
            self.resource = resource

    def setup(self, resource_types: List[ResourceType]):
        self.start = None
        self.priority = random.randint(1, 100)  # Higher number means higher priority
        resource_sample = random.sample(resource_types, random.randint(1, len(resource_types)))
        self.resources_needed = {}

        for r_type in resource_sample:
            # Tuple representing allocation status and completion status
            self.resources_needed[r_type] = TaskAgent.ResourceStatus(False, False)

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
                action.perform(batch_size=64)
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

    def allocate_resource(self, resource: ResourceAgent):
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


# Define the main Model
class ResourceAllocationModel(ap.Model):
    MAX_THREADS = 10

    def get_available_resource(self, resource_type: ResourceType):
        for resource in self.resources[resource_type]:
            if resource.is_available:
                return resource
        return None  # If no available resource is found

    def setup(self):
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_THREADS,
                                                                 thread_name_prefix=self.__class__.__name__)

        # Create resources
        self.resources = {
            ResourceType.CPU: [ResourceAgent(ResourceType.CPU) for _ in range(100)],
            ResourceType.GPU: [ResourceAgent(ResourceType.GPU) for _ in range(50)],
            ResourceType.NVME: [ResourceAgent(ResourceType.NVME) for _ in range(10)]
        }

        # Create tasks with priorities
        self.tasks = [TaskAgent(self, self.resources.keys()) for _ in range(100)]

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

            # Run tasks in parallel
            for t in allocated_tasks:
                self.thread_pool.submit(TaskExecutor.run, t)

            if done_count == len(self.tasks):
                break

            time.sleep(10)

        self.thread_pool.shutdown(wait=True)


if __name__ == '__main__':
    model = ResourceAllocationModel()
    results = model.run(steps=100)
    # Results analysis here
