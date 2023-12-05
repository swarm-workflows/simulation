import argparse
import time
from datetime import datetime

from mpi4py import MPI
from repast4py import core, random, space, schedule
import random
from repast4py import context as ctx
import enum

# Define a Task Agent that can request different types of resources and has a priority
from agents.common.resource import Resource, ResourceType, ResourceTypeToActionMap

import logging
LOGGER = logging.getLogger("TEST")


class AgentType(enum.Enum):
    Task = enum.auto()
    ResourceAllocator = enum.auto()


class MessageHelper:
    @staticmethod
    def send_message(resource_type: ResourceType, task_id: int, prefix: str, dest: int):
        LOGGER.info(f"Sending Message: {prefix}:{resource_type} from {task_id} to {dest}")
        MPI.COMM_WORLD.send(f"{prefix}:{resource_type}", dest=dest, tag=task_id)

    @staticmethod
    def receive_message(source: int, task_id: int):
        status = MPI.Status()
        message = MPI.COMM_WORLD.recv(source=source, tag=task_id, status=status)
        LOGGER.info(f"Received Message: {message} from {status.source} to {task_id}")
        return status, message


class TaskAgent(core.Agent):
    def __init__(self, agent_id, rank, resource_type: ResourceType):
        super().__init__(agent_id, AgentType.Task.value, rank)
        self.priority = random.randint(1, 1000)
        self.requested_resource_type = resource_type

    def perform(self):
        if self.request_resource():
            action = ResourceTypeToActionMap[self.requested_resource_type]()
            action.perform(batch_size=64)
            self.release_resource()
        else:
            LOGGER.info("Resource could not be allocated")

    def request_resource(self, timeout: int = 5):
        MessageHelper.send_message(resource_type=self.requested_resource_type, task_id=self.id, prefix="REQUEST", dest=0)
        wait = 0
        message = None
        while wait < timeout:
            status, message = MessageHelper.receive_message(source=0, task_id=self.id)
            if message is not None:
                break
            time.sleep(1)
        if message == "ALLOCATED":
            return True
        return False

    def release_resource(self):
        MessageHelper.send_message(resource_type=self.requested_resource_type, task_id=self.id, prefix="RELEASE", dest=0)


# Define a ResourceAgent
class ResourceAllocator(core.Agent):
    def __init__(self, agent_id, rank, num_tasks: int, num_cpu: int = 100, num_gpu: int = 50, num_nvme: int = 10):
        super().__init__(agent_id, AgentType.ResourceAllocator.value, rank)

        self.num_tasks = num_tasks
        # Create resources
        self.resources = {
            ResourceType.CPU: [Resource(ResourceType.CPU) for _ in range(num_cpu)],
            ResourceType.GPU: [Resource(ResourceType.GPU) for _ in range(num_gpu)]
        }
        self.allocated_resources = {}

    def get_available_resource(self, resource_type: ResourceType):
        for resource in self.resources[resource_type]:
            if resource.is_available:
                return resource
        return None  # If no available resource is found

    def perform(self):
        messages_received = 0
        timeout = 0
        while messages_received <= self.num_tasks and timeout >= 60:
            status, message = MessageHelper.receive_message(source=MPI.ANY_SOURCE, task_id=0)
            if message is not None:
                messages_received += 1
                parts = message.split(":")
                key = f"{status.source}-{status.tag}"
                if parts[0] == "REQUEST":
                    r_type = ResourceType.string_to_type(parts[1])
                    res = self.get_available_resource(resource_type=r_type)
                    if res:
                        res.allocate(task=key)
                        MessageHelper.send_message(resource_type=r_type, prefix="ALLOCATED", task_id=status.tag,
                                                   dest=status.source)
                        self.allocated_resources[key] = res
                elif parts[0] == "RELEASE":
                    if key in self.allocated_resources:
                        self.allocated_resources[key].release()
            time.sleep(1)
            timeout += 1


# Define a ResourceAllocationModel
class ResourceAllocationModel:
    def __init__(self, num_tasks: int = 100, total_tasks: int = 1, num_cpu: int = 100, num_gpu: int = 50,
                 num_nvme: int = 10, steps: int = 100):
        comm = MPI.COMM_WORLD
        # create the schedule
        self.runner = schedule.init_schedule_runner(comm)

        box = space.BoundingBox(xmin=0, xextent=100, ymin=0, yextent=200, zmin=0, zextent=0)
        self.space = space.SharedCSpace('ResourceSpace', bounds=box, borders=space.BorderType.Sticky,
                                        occupancy=space.OccupancyType.Multiple,
                                        buffer_size=2, comm=comm, tree_threshold=100)

        self.context = ctx.SharedContext(comm)
        rank = comm.Get_rank()

        if rank == 0:
            LOGGER.info("Creating RA")
            # Initialize allocator agents on rank 0
            allocator = ResourceAllocator(0, rank, num_tasks=total_tasks, num_cpu=num_cpu, num_nvme=num_nvme,
                                          num_gpu=num_gpu)
            self.context.add(allocator)
            self.space.add(allocator)
            self.runner.schedule_repeating_event(1, 1, self.perform)
            self.runner.schedule_stop(steps)
        else:
            LOGGER.info("Creating Task")
            # Initialize task agents on other ranks
            for x in range(num_tasks):
                task = TaskAgent(x, rank, random.choice(list(ResourceType)))
                self.context.add(task)
                self.space.add(task)
            self.runner.schedule_repeating_event(1, 1, self.perform)
            self.runner.schedule_stop(steps)

    def perform(self):
        for x in self.context.agents():
            x.perform()

    def start(self):

        self.runner.execute()


if __name__ == '__main__':
    LOGGER.info("Entering main")
    parser = argparse.ArgumentParser(description='Resource Allocation Model Parameters')
    parser.add_argument('--total', type=int, default=100, help='Number of total agents across MPI')
    parser.add_argument('--num_agents', type=int, default=100, help='Number of task agents')
    parser.add_argument('--num_gpu', type=int, default=50, help='Number of GPU resources')
    parser.add_argument('--num_cpu', type=int, default=100, help='Number of CPU resources')
    parser.add_argument('--num_nvme', type=int, default=10, help='Number of NVMe resources')
    parser.add_argument('--steps', type=int, default=100, help='Number of steps to run the model')

    args = parser.parse_args()

    # Run the simulation
    model = ResourceAllocationModel(num_tasks=args.num_agents, num_gpu=args.num_gpu, num_cpu=args.num_cpu,
                                    num_nvme=args.num_nvme, steps=args.steps, total_tasks=args.total)

    start = datetime.now()
    model.start()
    LOGGER.info(f"Simulation completed in : {(datetime.now()-start).total_seconds()}")
