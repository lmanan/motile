from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Callable, Mapping, TypeVar, cast

import ilpy
import numpy as np

from .constraints import SelectEdgeNodes
from .constraints.constraint import Constraint
from .costs import Features, Weight, Weights
from .ssvm import fit_weights
from .track_graph import TrackGraph

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from motile.costs import Costs
    from motile.variables import Variable

    V = TypeVar("V", bound=Variable)


class Solver:
    """Create a solver for a given track graph.

    Args:
        track_graph:
            The :class:`~motile.TrackGraph` of objects to track over time.

        skip_core_constraints (:obj:`bool`, default=False):
            If true, add no constraints to the solver at all. Otherwise, core
            constraints that ensure consistencies between selected nodes and
            edges are added.
    """

    def __init__(
        self, track_graph: TrackGraph, skip_core_constraints: bool = False
    ) -> None:
        if not isinstance(track_graph, TrackGraph):
            import networkx as nx

            if isinstance(track_graph, nx.Graph):
                warnings.warn(
                    "Coercing networkx graph to TrackGraph with frame_attribute='t'. "
                    "To silence this warning, please pass a motile.TrackGraph instance."
                )
                track_graph = TrackGraph(track_graph)
            else:  # pragma: no cover
                raise ValueError(
                    f"Expected a TrackGraph or networkx.Graph, got {type(track_graph)}"
                )

        self.graph = track_graph
        self.variables: dict[type[Variable], Variable] = {}
        self.variable_types: dict[int, ilpy.VariableType] = {}

        self.weights = Weights()
        self.weights.register_modify_callback(self._on_weights_modified)
        self._weights_changed = True
        self.features = Features()

        self.ilp_solver: ilpy.Solver | None = None
        self.objective: ilpy.Objective | None = None
        self.constraints = ilpy.Constraints()

        self.num_variables: int = 0
        self._costs = np.zeros((0,), dtype=np.float32)
        self._costs_instances: dict[str, Costs] = {}
        self.solution: ilpy.Solution | None = None

        if not skip_core_constraints:
            self.add_constraints(SelectEdgeNodes())

    def add_costs(self, costs: Costs, name: str | None = None) -> None:
        """Add linear costs to the value of variables in this solver.

        Args:
            costs:
                The costs to add.  An instance of :class:`~motile.costs.Costs`.

            name:
                An optional name of the costs, used to refer to weights of
                costs in an unambiguous manner. Defaults to the name of the
                costs class, if not given.
        """
        # default name of costs is the class name
        if name is None:
            name = type(costs).__name__

        if name in self._costs_instances:
            raise RuntimeError(
                f"A cost instance with name '{name}' was already registered. "
                "Consider passing a different name with the 'name=' argument "
                "to Solver.add_costs"
            )

        logger.info("Adding %s costs...", name)

        self._costs_instances[name] = costs

        # fish out all weights used in this cost object
        for var_name, var in costs.__dict__.items():
            if not isinstance(var, Weight):
                continue
            self.weights.add_weight(var, (name, var_name))

        costs.apply(self)

    def add_constraints(self, constraints: Constraint) -> None:
        """Add linear constraints to the solver.

        Args:
            constraints:
                The :class:`~motile.constraints.Constraint` to add.
        """
        logger.info("Adding %s constraints...", type(constraints).__name__)

        for constraint in constraints.instantiate(self):
            self.constraints.add(constraint)

    def solve(
        self,
        timeout: float = 0.0,
        num_threads: int = 1,
        verbose: bool = False,
        backend: ilpy.Preference = ilpy.Preference.Any,
        on_event: Callable[[Mapping], None] | None = None,
    ) -> ilpy.Solution:
        """Solve the global optimization problem.

        Args:
            timeout:
                The timeout for the ILP solver, in seconds. Default (0.0) is no
                timeout. If the solver times out, the best solution encountered
                so far is returned (if any has been found at all).
            num_threads:
                The number of threads the ILP solver uses.
            verbose:
                If true, print more information from ILP solver. Defaults to False.
            backend:
                The ILP solver backend to use. Defaults to Any.
            on_event:
                A callback function that will be called when the solver emits an event.
                Should accept an event data dict. (see `ilpy.EventData` for typing info
                which may be imported inside of a TYPE_CHECKING block.)
                Defaults to None.

        Returns:
            :class:`ilpy.Solution`, a vector of variable values. Use
            :func:`get_variables` to find the indices of variables in this
            vector.
        """
        self.objective = ilpy.Objective(self.num_variables)
        for i, c in enumerate(self.costs):
            logger.debug("Setting cost of var %d to %.3f", i, c)
            self.objective.set_coefficient(i, c)

        # TODO: support other variable types
        self.ilp_solver = ilpy.Solver(
            self.num_variables,
            ilpy.VariableType.Binary,
            variable_types=self.variable_types,
            preference=backend,
        )

        self.ilp_solver.set_objective(self.objective)
        self.ilp_solver.set_constraints(self.constraints)

        self.ilp_solver.set_num_threads(num_threads)
        if timeout > 0:
            self.ilp_solver.set_timeout(timeout)

        self.ilp_solver.set_verbose(verbose)
        self.ilp_solver.set_event_callback(on_event)

        self.solution = self.ilp_solver.solve()
        if message := self.solution.get_status():
            logger.info("ILP solver returned with: %s", message)

        return self.solution

    def get_variables(self, cls: type[V]) -> V:
        """Get variables by their class name.

        If the solver does not yet contain those variables, they will be
        created.

        Args:
            cls (type of :class:`motile.variables.Variable`):
                A subclass of :class:`motile.variables.Variable`.

        Returns:
            A singleton instance of :class:`~motile.variables.Variable` (of whatever
            type was passed in as ``cls``), mimicking a dictionary that can be used to
            look up variable indices by their keys. See
            :class:`~motile.variables.Variable` for details.
        """
        if cls not in self.variables:
            self._add_variables(cls)
        return cast("V", self.variables[cls])

    def add_variable_cost(
        self, index: int | ilpy.Variable, value: float, weight: Weight
    ) -> None:
        """Add costs for an individual variable.

        To be used within implementations of :class:`motile.costs.Costs`.
        """
        variable_index = index
        feature_index = self.weights.index_of(weight)
        self.features.add_feature(variable_index, feature_index, value)

    def fit_weights(
        self,
        gt_attribute: str,
        ground_truth: np.ndarray,
        mask: np.ndarray,
        fit_subset_weights: bool = False,
        regularizer_weight: float = 0.1,
        max_iterations: int = 1000,
        eps: float = 1e-6,
    ) -> None:
        """Fit weights of ILP costs to ground truth with structured SVM.

        Updates the weights in the solver object to the found solution.

        See https://github.com/funkelab/structsvm for details.

        Args:
            gt_attribute:
                Node/edge attribute that marks the ground truth for fitting.
                `gt_attribute` is expected to be set to:

                - ``1`` for objects labeled as ground truth.
                - ``0`` for objects explicitly labeled as not part of the ground truth.
                - ``None`` or not set for unlabeled objects.

            ground_truth:
                Numpy array of the size of the number of variables.
                Set to 1 when the edge_selected/node_selected/node_appear/node_disappear
                is true, else false.

            mask:
                Numpy array of the size of the number of variables.
                It is set to 1 wherever annotation is available
                irrespective of whether it was a positive or negative
                annotation.

            fit_subset_weights: bool = False
                If set to True, then appearance and disappearance
                weights are not fit.

            regularizer_weight:
                The weight of the quadratic regularizer.

            max_iterations:
                Maximum number of gradient steps in the structured SVM.

            eps:
                Convergence threshold.
        """
        optimal_weights = fit_weights(
            self,
            gt_attribute=gt_attribute,
            regularizer_weight=regularizer_weight,
            max_iterations=max_iterations,
            eps=eps,
            ground_truth=ground_truth,
            mask=mask,
            fit_subset_weights=fit_subset_weights,
        )
        self.weights.from_ndarray(optimal_weights)

    @property
    def costs(self) -> np.ndarray:
        """Returns the costs as a :class:`numpy.ndarray`."""
        if self._weights_changed:
            self._compute_costs()
            self._weights_changed = False

        return self._costs

    def get_costs_instances(self) -> Mapping:
        # TODO
        return self._costs_instances

    def _add_variables(self, cls: type[V]) -> None:
        logger.info("Adding %s variables...", cls.__name__)

        keys = cls.instantiate(self)
        indices = self._allocate_variables(len(keys))
        variables = cls(self, dict(zip(keys, indices)))
        self.variables[cls] = variables

        for index in indices:
            self.variable_types[index] = cls.variable_type

        for constraint in cls.instantiate_constraints(self):
            self.constraints.add(constraint)

        self.features.resize(num_variables=self.num_variables)

    def _compute_costs(self) -> None:
        logger.info("Computing costs...")

        weights = self.weights.to_ndarray()
        features = self.features.to_ndarray()
        self._costs = np.dot(features, weights)

    def _allocate_variables(self, num_variables: int) -> range:
        indices = range(self.num_variables, self.num_variables + num_variables)

        self.num_variables += num_variables
        self.features.resize(num_variables=self.num_variables)

        return indices

    def _on_weights_modified(self, old_value: float | None, new_value: float) -> None:
        if old_value != new_value:
            if not self._weights_changed:
                logger.info("Weights have changed")

            self._weights_changed = True

    def get_selected_subgraph(
        self, solution: ilpy.Solution | None = None
    ) -> TrackGraph:
        """Return TrackGraph with only the selected nodes/edges from the solution.

        Args:
            solution:
                The solution to use. If not provided, the last solution is used.

        Returns:
            A new TrackGraph with only the selected nodes and edges.

        Raises:
            RuntimeError: If no solution is provided and the solver has not been solved
            yet.
        """
        from motile.variables import EdgeSelected, NodeSelected

        if solution is None:
            solution = self.solution

        # TODO:
        # in theory this could be made more efficient by using a nx.DiGraph view
        # but TrackGraph itself doesn't provide views (and isn't a subclass)
        if not solution:
            raise RuntimeError(
                "No solution available. Run solve() first or manually pass a solution."
            )

        node_selected = self.get_variables(NodeSelected)
        edge_selected = self.get_variables(EdgeSelected)
        selected_graph = TrackGraph()

        for node_id, node in self.graph.nodes.items():
            if solution[node_selected[node_id]] > 0.5:
                selected_graph.add_node(node_id, node)

        for edge_id, edge in self.graph.edges.items():
            if solution[edge_selected[edge_id]] > 0.5:
                selected_graph.add_edge(edge_id, edge)

        return selected_graph
