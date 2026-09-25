"""Candidate tree (Architecture.md §3.4, §3.5, §4.2–4.4).

Worked example, tree_topk = [2, 3] (paper Fig. 2):

    node 0  = root  : the LM head's next token (depth 0, position n)
    nodes 1-2       : top-2 of Medusa head 1 (depth 1, position n+1), parent 0
    nodes 3-8       : top-3 of Medusa head 2 under each depth-1 node (depth 2, position n+2)

    Root -> A -> {C, D, E}
         -> B -> {C, D, E}         6 = 2 x 3 candidate branches (Cartesian product, §4.2)

The tree *shape* depends only on s_k, so it is built once. Each decoding step only fills in
token ids (`build_candidate_tokens`). Node i sits at flattened input position i of the
verification pass. Its logical sequence position is n + depth(i) (§4.4.3), because A and B
both stand for "the token after the root".
"""
import itertools
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import torch

from ..utils.sampling import topk_candidates


@dataclass
class TreeNode:
    index: int                    # flattened input position inside the tree block
    parent: int                   # parent node index, -1 for the root
    depth: int                    # 0 = root; depth d uses Medusa head d
    head: int                     # source head: -1 = original LM head, k = Medusa head k (1-based, paper notation)
    rank: int                     # which top-s_k prediction of that head (0 = best)
    path: Tuple[int, ...]         # ranks chosen at depths 1..depth (the branch prefix)
    children: List[int] = field(default_factory=list)
    branches: List[int] = field(default_factory=list)   # candidate-branch (leaf path) ids through this node


class MedusaTree:
    def __init__(self, topk: Sequence[int]):
        if len(topk) == 0 or any(s < 1 for s in topk):
            raise ValueError(f"tree_topk must be a non-empty list of positive ints, got {topk}")
        self.topk = list(topk)
        self.depth = len(topk)               # number of Medusa heads used by the tree
        self.max_s = max(topk)

        # ---- nodes, breadth-first by depth (so every parent precedes its children)
        self.nodes: List[TreeNode] = [TreeNode(0, -1, 0, -1, 0, ())]
        index_of = {(): 0}
        for d in range(1, self.depth + 1):
            for path in itertools.product(*[range(s) for s in self.topk[:d]]):
                parent = index_of[path[:-1]]
                node = TreeNode(len(self.nodes), parent, d, d, path[-1], path)
                index_of[path] = node.index
                self.nodes[parent].children.append(node.index)
                self.nodes.append(node)
        self.num_nodes = len(self.nodes)     # = 1 + sum_k prod_{i<=k} s_i

        # ---- candidate branches = root-to-leaf paths (Cartesian product of all heads)
        leaves = [n for n in self.nodes if not n.children]
        self.num_branches = len(leaves)      # = prod_k s_k
        retrieve = []
        for b, leaf in enumerate(leaves):
            chain = self.ancestors(leaf.index)                 # root ... leaf
            for i in chain:
                self.nodes[i].branches.append(b)
            retrieve.append(chain + [-1] * (self.depth + 1 - len(chain)))

        # ---- tensors used on the GPU hot path
        self.parents = torch.tensor([n.parent for n in self.nodes], dtype=torch.long)
        self.depths = torch.tensor([n.depth for n in self.nodes], dtype=torch.long)
        self.heads = torch.tensor([n.head for n in self.nodes], dtype=torch.long)
        self.ranks = torch.tensor([n.rank for n in self.nodes], dtype=torch.long)
        self.retrieve_indices = torch.tensor(retrieve, dtype=torch.long)       # [branches, depth+1]
        # Index into  flat = [root_token, head1_top0..head1_top(max_s-1), head2_top0, ...]
        self.token_gather_index = torch.tensor(
            [0 if n.depth == 0 else 1 + (n.depth - 1) * self.max_s + n.rank for n in self.nodes],
            dtype=torch.long,
        )
        self.ancestor_mask = self._build_ancestor_mask()                      # [N, N] bool

    # ------------------------------------------------------------------ structure
    def ancestors(self, i: int) -> List[int]:
        """Parent chain from the root down to node i (inclusive)."""
        chain = []
        while i != -1:
            chain.append(i)
            i = self.nodes[i].parent
        return chain[::-1]

    def _build_ancestor_mask(self) -> torch.Tensor:
        """§4.4.2: node i may see node j iff j is i itself or an ancestor of i.
        Built by walking each node's actual parent chain (not by assuming a layout)."""
        mask = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.bool)
        for n in self.nodes:
            for a in self.ancestors(n.index):
                mask[n.index, a] = True
        return mask

    def to(self, device) -> "MedusaTree":
        for name in ("parents", "depths", "heads", "ranks", "retrieve_indices",
                     "token_gather_index", "ancestor_mask"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def position_ids(self, past_len: int) -> torch.Tensor:
        """Logical sequence position of every node: n + depth (§4.4.3). Shape [1, N]."""
        return (self.depths + past_len).unsqueeze(0)

    # ------------------------------------------------------------------ per step
    def build_candidate_tokens(self, root_token: torch.Tensor, medusa_logits: torch.Tensor) -> torch.Tensor:
        """root_token: scalar id. medusa_logits: [K_total, V] (one row per Medusa head, for the
        root's position). Returns tree_tokens [N]; node i gets its head's top-rank_i token."""
        ids, _ = topk_candidates(medusa_logits[: self.depth], self.max_s)    # [depth, max_s]
        flat = torch.cat([root_token.reshape(1), ids.reshape(-1)])
        return flat[self.token_gather_index]

    def candidates(self, tree_tokens: torch.Tensor) -> torch.Tensor:
        """Every candidate branch as a token sequence [branches, depth+1] (root first)."""
        return tree_tokens[self.retrieve_indices]

    def describe(self, tree_tokens: torch.Tensor = None, tokenizer=None) -> str:
        lines = []
        for n in self.nodes:
            tok = ""
            if tree_tokens is not None:
                t = int(tree_tokens[n.index])
                tok = repr(tokenizer.decode([t])) if tokenizer else str(t)
            src = "LM head" if n.head == -1 else f"head {n.head} top-{n.rank + 1}"
            lines.append(f"{'  ' * n.depth}[{n.index}] depth={n.depth} {src} parent={n.parent} {tok}")
        return "\n".join(lines)
