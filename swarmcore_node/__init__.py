"""SwarmCore — бортовой узел группы (swarm_node): распределение, отказоустойчивость, режимы.

Один процесс на аппарат. Подробности — ``docs/swarm_node.md``.
"""

from swarmcore_node.modes import Mode
from swarmcore_node.node import NodeConfig, SwarmNode

__all__ = ["SwarmNode", "NodeConfig", "Mode"]
