import enum

from repast4py import core


class AgentType(enum.Enum):
    Resource = enum.auto()
    Leader = enum.auto()


class BaseAgent(core.Agent):
    def __init__(self, agent_id: int, agent_type: AgentType, rank: int):
        super().__init__(agent_id, agent_type.value, rank)
