"""Monte Carlo Tree Search action planner for DRS-GUI."""

import base64
from dataclasses import asdict, dataclass
import io
import logging
import math
import random
import time

import numpy as np
from PIL import Image

from ui_perceptor import analyze_ui_image_simple, get_topk_semantic_matches
from .action import action_focus, action_scatter, action_shift, box_area
from .instruction_config import select_instruction
from .policy import QuestionSample as BaseQuestionSample


LOGGER = logging.getLogger(__name__)
REWARD_WEIGHTS = (0.4, 0.4, 0.2)
NON_INTERACTIVE_WEIGHT = 0.5
SEMANTIC_TEMPERATURE = 0.1


def _encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@dataclass(frozen=True)
class RewardBreakdown:
    """Individual terms of the region-quality reward."""

    relevance: float
    coverage: float
    concentration: float
    total: float
    element_count: int
    interactive_count: int

    def to_dict(self) -> dict:
        return asdict(self)


class MCTSNode:
    """A candidate image region in the perception search tree."""

    def __init__(self, state: dict, parent=None, actions: list[str] | None = None):
        self.state = state
        self.parent = parent
        self.children: dict[str, MCTSNode] = {}
        self.untried_actions = list(actions or [])
        self.visits = 0
        self.value = 0.0
        self.leaf_reward: float | None = None
        self.parsed_elements: list[dict] | None = None
        self.semantic_elements: list[dict] | None = None
        self.reward_components: RewardBreakdown | None = None
        self.failed_actions: list[str] = []
        self.action_from_parent = state.get("action_from_parent")

        x1, y1, x2, y2 = state["region_coords"]
        full_area = max(1.0, state["image_width"] * state["image_height"])
        self.area_ratio = box_area([x1, y1, x2, y2]) / full_area

    @property
    def depth(self) -> int:
        return int(self.state["depth"])

    @property
    def mean_value(self) -> float:
        return self.value / self.visits if self.visits else 0.0

    @property
    def region(self) -> list[float]:
        return list(self.state["region_coords"])

    @property
    def action_path(self) -> list[str]:
        return list(self.state.get("action_history", []))


class MCTSQuestionSample(BaseQuestionSample):
    """Run dynamic region search for one grounding sample."""

    def __init__(self, row, args, round_idx=0):
        super().__init__(row, args, round_idx)
        image = Image.open(io.BytesIO(base64.b64decode(self.image)))
        self.image_width, self.image_height = image.size
        self.max_depth = args.max_depth
        self.exploration_constant = 1.0
        self.simulation_budget = args.mcts_iterations
        self.actions = ["focus", "scatter", "shift"]
        self.action_executors = {
            "focus": self.execute_focus_action,
            "shift": self.execute_shift_action,
            "scatter": self.execute_scatter_action,
        }
        self.domain_prefix = select_instruction(row)
        self.root: MCTSNode | None = None
        self._global_parsed = None
        self.include_search_tree = getattr(args, "include_search_tree", False)
        self.action_attempts = {action: 0 for action in self.actions}
        self.action_successes = {action: 0 for action in self.actions}

    async def _parse_node(self, node: MCTSNode) -> list[dict]:
        if node.parsed_elements is None:
            _, _, node.parsed_elements = analyze_ui_image_simple(
                node.state["image"], self.som_model, self.caption_model
            )
            full_region = (0, 0, self.image_width, self.image_height)
            if tuple(node.state["region_coords"]) == full_region:
                image = Image.open(io.BytesIO(base64.b64decode(self.image))).convert("RGB")
                self._global_parsed = (image, node.parsed_elements)
        return node.parsed_elements

    def _get_global_parse(self) -> tuple[Image.Image, list[dict]]:
        if self._global_parsed is None:
            image = Image.open(io.BytesIO(base64.b64decode(self.image))).convert("RGB")
            _, _, elements = analyze_ui_image_simple(
                self.image, self.som_model, self.caption_model
            )
            self._global_parsed = (image, elements)
        return self._global_parsed

    def _make_local_child(
        self,
        node: MCTSNode,
        node_image: Image.Image,
        local_region: list[float],
        action: str,
    ) -> MCTSNode | None:
        x1, y1, x2, y2 = map(int, local_region)
        if x2 <= x1 or y2 <= y1:
            return None
        if box_area([x1, y1, x2, y2]) >= 0.999 * node_image.width * node_image.height:
            return None

        parent_x1, parent_y1, _, _ = node.state["region_coords"]
        global_region = [
            parent_x1 + x1,
            parent_y1 + y1,
            parent_x1 + x2,
            parent_y1 + y2,
        ]
        crop = node_image.crop((x1, y1, x2, y2))
        return self._new_node(node, crop, global_region, action)

    def _make_global_child(
        self,
        node: MCTSNode,
        original_image: Image.Image,
        global_region: list[float] | None,
        action: str,
    ) -> MCTSNode | None:
        if global_region is None:
            return None
        x1, y1, x2, y2 = map(int, global_region)
        if x2 <= x1 or y2 <= y1:
            return None
        if [x1, y1, x2, y2] == [int(value) for value in node.state["region_coords"]]:
            return None
        crop = original_image.crop((x1, y1, x2, y2))
        return self._new_node(node, crop, [x1, y1, x2, y2], action)

    def _new_node(
        self,
        parent: MCTSNode,
        image: Image.Image,
        region: list[float],
        action: str,
    ) -> MCTSNode:
        state = {
            "depth": parent.state["depth"] + 1,
            "image": _encode_image(image),
            "action_history": parent.state["action_history"] + [action],
            "action_from_parent": action,
            "text": parent.state["text"],
            "image_width": self.image_width,
            "image_height": self.image_height,
            "region_coords": region,
        }
        return MCTSNode(state, parent=parent, actions=self.actions)

    async def execute_focus_action(self, node: MCTSNode) -> MCTSNode | None:
        elements = await self._parse_node(node)
        if not elements:
            return None
        node_image = Image.open(io.BytesIO(base64.b64decode(node.state["image"]))).convert("RGB")
        region, _ = action_focus(
            image=node_image,
            parsed_content_list=elements,
            semantic_model=self.semantic_model,
            gui_prompt=node.state["text"],
            instruction=self.domain_prefix,
            top_percent=0.10,
            distance_threshold=1.8,
            max_area_ratio=0.7,
        )
        return self._make_local_child(node, node_image, region, "focus")

    async def execute_scatter_action(self, node: MCTSNode) -> MCTSNode | None:
        original_image, elements = self._get_global_parse()
        region, _ = action_scatter(
            image=original_image,
            parsed_content_list=elements,
            semantic_model=self.semantic_model,
            gui_prompt=node.state["text"],
            instruction=self.domain_prefix,
            parent_abs=list(node.state["region_coords"]),
            top_percent=0.08,
            max_area_ratio=1.5,
        )
        return self._make_global_child(node, original_image, region, "scatter")

    async def execute_shift_action(self, node: MCTSNode) -> MCTSNode | None:
        original_image, elements = self._get_global_parse()
        region, _ = action_shift(
            image=original_image,
            parsed_content_list=elements,
            semantic_model=self.semantic_model,
            gui_prompt=node.state["text"],
            instruction=self.domain_prefix,
            parent_abs=list(node.state["region_coords"]),
            top_percent=0.15,
            padding_ratio=0.01,
            min_distance_ratio=0.5,
            max_parent_iou=0.3,
        )
        return self._make_global_child(node, original_image, region, "shift")

    def selection(self, node: MCTSNode) -> MCTSNode:
        if node.state["depth"] >= self.max_depth or node.untried_actions or not node.children:
            return node

        def uct(child: MCTSNode) -> float:
            if child.visits == 0:
                return float("inf")
            exploitation = child.value / child.visits
            total_visits = max(1, sum(item.visits for item in node.children.values()))
            exploration = math.sqrt(
                2 * math.log(total_visits) / (child.visits + 1e-8)
            )
            return exploitation + self.exploration_constant * exploration

        return self.selection(max(node.children.values(), key=uct))

    def _valid_actions(self, node: MCTSNode) -> list[str]:
        if node.area_ratio > 0.6:
            allowed = {"focus"}
        elif node.area_ratio > 0.03:
            allowed = set(self.actions)
        else:
            allowed = {"scatter", "shift"}
        valid = [action for action in node.untried_actions if action in allowed]
        return valid or list(node.untried_actions)

    async def expansion(self, node: MCTSNode) -> MCTSNode:
        if node.state["depth"] >= self.max_depth:
            return node
        if not node.children and set(node.untried_actions) == set(self.actions):
            node.untried_actions = self._valid_actions(node)
        while node.untried_actions:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            self.action_attempts[action] += 1
            child = await self.action_executors[action](node)
            if child is not None:
                self.action_successes[action] += 1
                node.children[action] = child
                return child
            node.failed_actions.append(action)
        return node

    @staticmethod
    def _reward_breakdown(
        elements: list[dict], semantic_elements: list[dict]
    ) -> RewardBreakdown:
        if not semantic_elements:
            return RewardBreakdown(
                relevance=0.0,
                coverage=0.0,
                concentration=0.0,
                total=0.0,
                element_count=len(elements),
                interactive_count=0,
            )

        scores = np.asarray(
            [item["similarity"] for item in semantic_elements], dtype=float
        )
        interaction_weights = np.asarray(
            [
                1.0 if item.get("interactivity", False) else NON_INTERACTIVE_WEIGHT
                for item in semantic_elements
            ],
            dtype=float,
        )
        relevance = float(
            np.sum(interaction_weights * scores)
            / (np.sum(interaction_weights) + 1e-10)
        )

        coverage = 0.0
        for element in elements:
            box = element.get("bbox")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                coverage += box_area(list(box))

        logits = scores / SEMANTIC_TEMPERATURE
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= np.sum(probabilities) + 1e-10
        if len(probabilities) == 1:
            concentration = 1.0
        else:
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            concentration = float(
                1.0 - entropy / np.log(len(probabilities) + 1e-10)
            )

        alpha, beta, gamma = REWARD_WEIGHTS
        total = alpha * relevance + beta * coverage + gamma * concentration
        return RewardBreakdown(
            relevance=relevance,
            coverage=coverage,
            concentration=concentration,
            total=float(total),
            element_count=len(elements),
            interactive_count=sum(
                bool(item.get("interactivity", False)) for item in semantic_elements
            ),
        )

    async def simulation(self, node: MCTSNode) -> float:
        """Compute the three-term region quality reward from the paper."""
        elements = await self._parse_node(node)
        semantic_elements = get_topk_semantic_matches(
            self.semantic_model,
            node.state["text"],
            self.domain_prefix,
            elements,
            topk=len(elements),
        )
        node.semantic_elements = semantic_elements
        node.reward_components = self._reward_breakdown(elements, semantic_elements)
        return node.reward_components.total

    @staticmethod
    def backpropagation(node: MCTSNode, reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    async def _initialize_root(self, initial_state: dict) -> None:
        full_image = MCTSNode(initial_state, actions=self.actions)
        self.root = await self.execute_focus_action(full_image) or full_image
        self.root.parent = None
        self.root.leaf_reward = await self.simulation(self.root)

    async def single_run(self, initial_state: dict) -> float:
        if self.root is None:
            await self._initialize_root(initial_state)

        node = self.selection(self.root)
        node = await self.expansion(node)
        reward = await self.simulation(node)
        node.leaf_reward = reward
        self.backpropagation(node, reward)
        return reward

    @staticmethod
    def iter_tree_nodes(root: MCTSNode):
        """Yield every node in depth-first order."""
        pending = [root]
        while pending:
            node = pending.pop()
            yield node
            pending.extend(reversed(list(node.children.values())))

    def collect_search_summary(self, best_node: MCTSNode) -> dict:
        """Return compact search statistics suitable for benchmark output."""
        nodes = list(self.iter_tree_nodes(self.root))
        evaluated = [node for node in nodes if node.leaf_reward is not None]
        rewards = [node.leaf_reward for node in evaluated]
        action_distribution = {action: 0 for action in self.actions}
        for node in nodes:
            if node.action_from_parent in action_distribution:
                action_distribution[node.action_from_parent] += 1

        action_metrics = {}
        for action in self.actions:
            attempts = self.action_attempts[action]
            successes = self.action_successes[action]
            action_metrics[action] = {
                "attempts": attempts,
                "successes": successes,
                "failures": attempts - successes,
            }

        return {
            "simulation_budget": self.simulation_budget,
            "depth_limit": self.max_depth,
            "total_nodes": len(nodes),
            "evaluated_nodes": len(evaluated),
            "max_depth_reached": max((node.depth for node in nodes), default=0),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "max_reward": float(max(rewards)) if rewards else 0.0,
            "best_action_path": best_node.action_path,
            "action_distribution": action_distribution,
            "action_metrics": action_metrics,
        }

    def serialize_search_tree(
        self, node: MCTSNode, best_node: MCTSNode
    ) -> dict:
        """Serialize tree metadata without embedding crop image bytes."""
        record = {
            "depth": node.depth,
            "action_from_parent": node.action_from_parent,
            "action_history": node.action_path,
            "region_coords": node.region,
            "area_ratio": node.area_ratio,
            "visits": node.visits,
            "value": node.value,
            "mean_value": node.mean_value,
            "leaf_reward": node.leaf_reward,
            "reward_components": (
                node.reward_components.to_dict()
                if node.reward_components is not None
                else None
            ),
            "parsed_element_count": (
                len(node.parsed_elements) if node.parsed_elements is not None else None
            ),
            "semantic_element_count": (
                len(node.semantic_elements)
                if node.semantic_elements is not None
                else None
            ),
            "untried_actions": list(node.untried_actions),
            "failed_actions": list(node.failed_actions),
            "is_best": node is best_node,
            "children": {},
        }
        record["children"] = {
            action: self.serialize_search_tree(child, best_node)
            for action, child in node.children.items()
        }
        return record

    def format_search_tree(self, best_node: MCTSNode) -> str:
        """Format a compact human-readable tree for debugging experiments."""
        lines = []

        def visit(node: MCTSNode, prefix: str, is_last: bool) -> None:
            connector = "└── " if is_last else "├── "
            action = node.action_from_parent or "root"
            reward = "N/A" if node.leaf_reward is None else f"{node.leaf_reward:.3f}"
            marker = " *" if node is best_node else ""
            lines.append(
                f"{prefix}{connector}{action} depth={node.depth} "
                f"reward={reward} area={node.area_ratio:.3f} "
                f"visits={node.visits}{marker}"
            )
            children = list(node.children.values())
            child_prefix = prefix + ("    " if is_last else "│   ")
            for index, child in enumerate(children):
                visit(child, child_prefix, index == len(children) - 1)

        visit(self.root, "", True)
        return "\n".join(lines)

    async def search(self) -> MCTSNode:
        initial_state = {
            "depth": 0,
            "image": self.image,
            "action_history": [],
            "text": self.row["prompt_to_evaluate"],
            "image_width": self.image_width,
            "image_height": self.image_height,
            "region_coords": [0, 0, self.image_width, self.image_height],
        }
        for _ in range(self.simulation_budget):
            await self.single_run(initial_state)

        nodes = list(self.iter_tree_nodes(self.root))
        return max(
            (node for node in nodes if node.leaf_reward is not None),
            key=lambda node: node.leaf_reward,
        )

    async def _process(self) -> dict:
        start_time = time.time()
        best_node = await self.search()
        final_answer = await self.generate(
            self.row["prompt_to_evaluate"], best_node.state["image"]
        )

        point = final_answer.get("point") if isinstance(final_answer, dict) else None
        if isinstance(point, (list, tuple)) and len(point) == 2:
            x1, y1, x2, y2 = best_node.state["region_coords"]
            prediction = [
                x1 + float(point[0]) * (x2 - x1),
                y1 + float(point[1]) * (y2 - y1),
            ]
        else:
            prediction = None

        elapsed = time.time() - start_time
        LOGGER.info(
            "sample=%s depth=%d reward=%.4f area=%.2f%% elapsed=%.2fs",
            self.row["id"],
            best_node.state["depth"],
            best_node.leaf_reward,
            best_node.area_ratio * 100,
            elapsed,
        )
        search_summary = self.collect_search_summary(best_node)
        result = {
            "id": self.row["id"],
            "round_id": self.round_idx,
            "img_path": self.row["img_filename"],
            "group": self.row.get("group"),
            "platform": self.row["platform"],
            "application": self.row["application"],
            "lang": self.row["language"],
            "instruction_style": self.row["instruction_style"],
            "prompt_to_evaluate": self.row["prompt_to_evaluate"],
            "gt_type": self.row["gt_type"],
            "ui_type": self.row["ui_type"],
            "task_filename": self.row["task_filename"],
            "pred": prediction,
            "raw_response": final_answer.get("raw_response", "")
            if isinstance(final_answer, dict)
            else str(final_answer),
            "full_response": final_answer,
            "best_region": best_node.state["region_coords"],
            "best_region_reward": best_node.leaf_reward,
            "best_region_depth": best_node.state["depth"],
            "reward_components": (
                best_node.reward_components.to_dict()
                if best_node.reward_components is not None
                else None
            ),
            "action_history": best_node.state["action_history"],
            "search_summary": search_summary,
            "elapsed_seconds": elapsed,
        }
        if self.include_search_tree:
            result["search_tree"] = self.serialize_search_tree(
                self.root, best_node
            )
            result["search_tree_text"] = self.format_search_tree(best_node)
        return result
