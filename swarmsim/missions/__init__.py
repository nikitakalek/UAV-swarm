from swarmsim.missions.base import MissionPlugin, Task, TaskSet, TaskStatus
from swarmsim.missions.search import SearchMission

MISSIONS: dict[str, type[MissionPlugin]] = {"search": SearchMission}


def decompose(mission, world) -> TaskSet:
    """``decompose(mission, world) -> TaskSet`` для плагина, указанного в ``mission.type``."""
    return MISSIONS[mission.type]().decompose(mission, world)


__all__ = ["MissionPlugin", "Task", "TaskSet", "TaskStatus", "SearchMission", "MISSIONS", "decompose"]
