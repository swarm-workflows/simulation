import argparse
from datetime import datetime

from mpi4py import MPI
from repast4py import core, random, space, schedule
import random
from repast4py import context as ctx
import enum

# Define a Task Agent that can request different types of resources and has a priority
from agents.common.resource import Resource, ResourceType, ResourceTypeToActionMap


class AgentType(enum.Enum):
    Task = enum.auto()
    ResourceAllocator = enum.auto()


class TaskAgent(core.Agent):
    def __init__(self, agent_id, rank, resource_type: ResourceType):
        super().__init__(agent_id, AgentType.Task.value, rank)
        self.priority = random.randint(1, 1000)
        self.requested_resource_type = resource_type
        self.allocated_resource = None
        self.done = False

    def perform(self):
        start = datetime.now()
        action = ResourceTypeToActionMap[self.requested_resource_type]()
        action.perform(batch_size=64)
        self.release_resource()
        #print(f"Task: {self.id} took time: {(datetime.now()-start).total_seconds()} to complete")

    def has_resources(self):
        return self.allocated_resource is not None

    def is_done(self):
        return self.done

    def allocate_resource(self, resource: Resource):
        if self.done or self.requested_resource_type != resource.resource_type:
            return
        self.allocated_resource = resource
        resource.allocate(self)

    def release_resource(self):
        self.allocated_resource.release()
        self.done = True


# Define a ResourceAgent
class ResourceAllocator(core.Agent):
    def __init__(self, model, agent_id, rank, num_cpu: int = 100, num_gpu: int = 50,
                 num_nvme: int = 10):
        super().__init__(agent_id, AgentType.ResourceAllocator.value, rank)
        self.model = model

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

        # Sort tasks by priority (higher first)
        # Allocate tasks using a greedy algorithm based on priority

        allocated_tasks = []

        tasks = list(self.model.context.agents(agent_type=AgentType.Task.value))
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        for task in sorted_tasks:
            if task.is_done():
                continue

            available_resource = self.get_available_resource(resource_type=task.requested_resource_type)
            if available_resource:
                task.allocate_resource(resource=available_resource)
                allocated_tasks.append(task)

        for t in allocated_tasks:
            t.perform()


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

        self.context = ctx.SharedContext(comm)

        # Create tasks with priorities
        rank = comm.Get_rank()
        for x in range(num_tasks):
            task = TaskAgent(x, rank, random.choice(list(ResourceType)))
            self.context.add(task)
            self.space.add(task)

        for x in range(num_allocators):
            allocator = ResourceAllocator(self, x, rank, num_cpu=num_cpu, num_nvme=num_nvme, num_gpu=num_gpu)
            self.context.add(allocator)
            self.space.add(allocator)

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

    def step(self):
        #for allocators in self.context.agents(agent_type=AgentType.ResourceAllocator.value):
        #    allocators.allocate_tasks()

        allocated_tasks = []

        tasks = list(self.context.agents(agent_type=AgentType.Task.value))
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        for task in sorted_tasks:
            if task.is_done():
                continue

            available_resource = self.get_available_resource(resource_type=task.requested_resource_type)
            if available_resource:
                task.allocate_resource(resource=available_resource)
                allocated_tasks.append(task)

        for t in allocated_tasks:
            t.perform()

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

    # Run the simulation
    model = ResourceAllocationModel(num_tasks=args.num_agents, num_gpu=args.num_gpu, num_cpu=args.num_cpu,
                                    num_nvme=args.num_nvme, steps=args.steps)

    start = datetime.now()
    model.start()
    print(f"Simulation completed in : {(datetime.now()-start).total_seconds()}")
