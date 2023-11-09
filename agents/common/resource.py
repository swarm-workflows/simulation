import enum
from enum import Enum

from agents.common.cpu_action import CpuAction
from agents.common.gpu_action import GpuAction


class ResourceType(Enum):
    GPU = enum.auto(),
    CPU = enum.auto()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


# Define a Resource that represents CPU, GPU, or NVME
class Resource:
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


ResourceTypeToActionMap = {
    ResourceType.CPU: CpuAction,
    ResourceType.GPU: GpuAction
}
