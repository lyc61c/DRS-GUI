"""Monte Carlo Tree Search action planner for DRS-GUI."""

import base64
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

        x1, y1, x2, y2 = state["region_coords"]
        full_area = max(1.0, state["image_width"] * state["image_height"])
        self.area_ratio = box_area([x1, y1, x2, y2]) / full_area


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
            child = await self.action_executors[action](node)
            if child is not None:
                node.children[action] = child
                return child
        return node

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
        if not semantic_elements:
            return 0.0

        scores = np.asarray([item["similarity"] for item in semantic_elements], dtype=float)
        interaction_weights = np.asarray(
            [
                1.0 if item.get("interactivity", False) else NON_INTERACTIVE_WEIGHT
                for item in semantic_elements
            ],
            dtype=float,
        )
        relevance = float(
            np.sum(interaction_weights * scores) / (np.sum(interaction_weights) + 1e-10)
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
            concentration = float(1.0 - entropy / np.log(len(probabilities) + 1e-10))

        alpha, beta, gamma = REWARD_WEIGHTS
        return alpha * relevance + beta * coverage + gamma * concentration

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

        nodes = []
        pending = [self.root]
        while pending:
            node = pending.pop()
            nodes.append(node)
            pending.extend(node.children.values())
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
        return {
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
            "action_history": best_node.state["action_history"],
            "elapsed_seconds": elapsed,
        }
