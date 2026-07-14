"""Public DRS-GUI search policies."""

from .drsgui.mcts import MCTSQuestionSample

policy_map = {"drsgui.mcts": MCTSQuestionSample}
