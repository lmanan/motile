from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ..variables import EdgeSelected
from .costs import Costs
from .weight import Weight

if TYPE_CHECKING:
    import networkx as nx

    from motile.solver import Solver


class EdgeDistance(Costs):
    """Costs for :class:`~motile.variables.EdgeSelected` variables.

    Costs are based on the spatial distance of the incident nodes.

    Args:
        position_attribute:
            The name of the node attribute that corresponds to the spatial
            position. Can also be given as a tuple of multiple coordinates,
            e.g., ``('z', 'y', 'x')``.

        weight:
            The weight to apply to the distance to convert it into a cost.

        constant:
            A constant cost for each selected node. Default is ``0.0``.
    """

    def __init__(
        self,
        position_attribute: str | tuple[str, ...],
        weight: float = 1.0,
        constant: float = 0.0,
        scale_factor = 1.0
    ) -> None:
        self.position_attribute = position_attribute
        self.weight = Weight(weight)
        self.constant = Weight(constant)
        self.scale_factor = scale_factor

    def apply(self, solver: Solver) -> None:
        edge_variables = solver.get_variables(EdgeSelected)
        for key, index in edge_variables.items():
            u, v = cast("tuple[int, int]", key)
            pos_u = self.__get_node_position(solver.graph, u)
            pos_v = self.__get_node_position(solver.graph, v)

            feature = np.linalg.norm(pos_u - pos_v)/self.scale_factor #TODO

            solver.add_variable_cost(index, feature, self.weight)
            solver.add_variable_cost(index, 1.0, self.constant)

    def __get_node_position(self, graph: nx.DiGraph, node: int) -> np.ndarray:
        if isinstance(self.position_attribute, tuple):
            return np.array([graph.nodes[node][p] for p in self.position_attribute])
        else:
            return np.array(graph.nodes[node][self.position_attribute])
