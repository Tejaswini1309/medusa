"""Medusa tree decoding engine (Architecture.md §3.7, §4.5–4.8, §4.17).

One decoding step, batch size 1 (the paper's setting):

  state: KV cache holds x_1..x_n; `root` = next token already chosen by the LM head (not yet
         in the cache); `medusa_logits` = the K heads' predictions made at the root's parent.
  1. tree_tokens = root + top-s_k tokens of each head, arranged as the Cartesian tree.
  2. ONE forward pass over all N tree nodes: positions n+depth, tree attention mask.
  3. node i's output logits = p_original(· | context, ancestors of i, i).
     A child c of node i is accepted if it passes the acceptance test against node i's logits.
  4. Longest accepted prefix over all branches wins (root always accepted).
  5. KV cache keeps only the winning path's slots; the next root is chosen from the logits at the
     last accepted node, and the heads are re-run on that node's hidden state.
"""
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..model.medusa_model import MedusaModel
from ..utils.metrics import CudaTimer, DecodingStats
from ..utils.sampling import select_token
from .kv_cache import TreeKVCache, build_tree_attention_mask
from .tree import MedusaTree


def acceptance_threshold(logits: torch.Tensor, temperature: float, epsilon: float, delta: float):
    """§4.6/4.7 for every row of logits [N, V]:
    returns log p (temperature-scaled) [N, V] and threshold min(ε, δ·exp(-H(p))) [N]."""
    logp = F.log_softmax(logits.float() / temperature, dim=-1)
    entropy = -(logp.exp() * logp).sum(dim=-1)
    thr = torch.clamp(delta * torch.exp(-entropy), max=epsilon)
    return logp, thr


def evaluate_candidates(tree: MedusaTree, tree_tokens: torch.Tensor, logits: torch.Tensor,
                        temperature: float, epsilon: float, delta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (best branch index, number of accepted tokens beyond the root) as 0-d tensors.

    Every non-root node c with parent i is tested once (§4.8 steps 1–7):
      temperature 0: accept iff c == argmax p(·|..., i)   (the ε,δ test's T→0 limit = greedy)
      temperature>0: accept iff p(c|..., i) > min(ε, δ·exp(-H(p(·|..., i))))
    A branch's accepted length is its longest all-accepted prefix (§4.8 steps 8–9)."""
    logits = logits.float()
    par, cand = tree.parents[1:], tree_tokens[1:]
    if temperature <= 0:
        ok = logits.argmax(dim=-1)[par] == cand
    else:
        logp, thr = acceptance_threshold(logits, temperature, epsilon, delta)
        ok = logp[par, cand].exp() > thr[par]
    accept = torch.cat([ok.new_ones(1), ok])                           # root: accepted unconditionally
    prefix = torch.cumprod(accept[tree.retrieve_indices].to(torch.int32), dim=1)   # [branches, depth+1]
    lengths = prefix.sum(dim=1) - 1
    # Tie-break between equally long branches: higher log-likelihood of the accepted prefix.
    cand_lp = logits[par, cand] - torch.logsumexp(logits, dim=-1)[par]
    node_lp = torch.cat([cand_lp.new_zeros(1), cand_lp])
    score = lengths.to(torch.float32) * 1e4 + (node_lp[tree.retrieve_indices] * prefix).sum(dim=1)
    best = score.argmax()
    return best, lengths[best]


class MedusaDecoder:
    def __init__(self, model: MedusaModel, tree: MedusaTree, eos_token_id: int,
                 max_context: int = 2048, temperature: float = 0.0, top_k: int = 0, top_p: float = 1.0,
                 epsilon: float = 0.09, delta: float = 0.3, root_greedy: bool = True):
        if tree.depth > model.heads.num_heads:
            raise ValueError(f"tree depth {tree.depth} > number of Medusa heads {model.heads.num_heads}")
        self.model = model
        self.device = next(model.heads.parameters()).device
        self.dtype = model.get_lm_head().weight.dtype
        self.tree = tree.to(self.device)
        self.eos = eos_token_id
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self.epsilon, self.delta, self.root_greedy = epsilon, delta, root_greedy
        # Capacity: context + one full tree written before compaction.
        self.cache = TreeKVCache(model.config, max_context + tree.num_nodes, 1, self.device, self.dtype)
        self.max_context = max_context

    @classmethod
    def from_config(cls, model: MedusaModel, cfg, eos_token_id: int) -> "MedusaDecoder":
        g = cfg["generation"]
        return cls(model, MedusaTree(g["tree_topk"]), eos_token_id, g["max_context"],
                   g["temperature"], g["top_k"], g["top_p"], g["epsilon"], g["delta"], g["root_greedy"])

    # ------------------------------------------------------------------ helpers
    def _select_root(self, logits: torch.Tensor) -> torch.Tensor:
        if self.root_greedy or self.temperature <= 0:
            return logits.argmax(dim=-1)
        return select_token(logits, self.temperature, self.top_k, self.top_p)

    def _arange(self, start: int, n: int) -> torch.Tensor:
        return torch.arange(start, start + n, device=self.device)

    def _emit(self, new: List[int], out: List[int], max_new: int) -> bool:
        """Append tokens, truncating at EOS / max_new. Returns True when generation must stop."""
        for t in new:
            out.append(t)
            if t == self.eos or len(out) >= max_new:
                return True
        return False

    def _prefill(self, input_ids: torch.Tensor, stats: DecodingStats):
        n = input_ids.shape[1]
        if n >= self.max_context:
            raise ValueError(f"prompt has {n} tokens; max_context is {self.max_context}")
        self.cache.reset()
        stats.prompt_tokens = n
        with CudaTimer() as t:
            out = self.model(input_ids, position_ids=self._arange(0, n)[None], past_key_values=self.cache,
                             cache_position=self._arange(0, n), use_cache=True,
                             medusa_forward=False, lm_logits_to_keep=1)
            self.cache.commit()
            root = self._select_root(out.lm_logits[0, -1])
            medusa_logits = self.model.heads(out.hidden_states[0, -1])     # [K, V]
        stats.prefill_time = t.elapsed
        return root, medusa_logits

    # ------------------------------------------------------------------ Medusa decoding
    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int,
                 callback: Optional[Callable[[List[int]], None]] = None) -> Tuple[List[int], DecodingStats]:
        tree = self.tree
        stats = DecodingStats(num_heads=tree.depth)
        root, medusa_logits = self._prefill(input_ids.to(self.device), stats)
        out_tokens: List[int] = []
        with CudaTimer() as t:
            while True:
                past = self.cache.length
                if past + tree.num_nodes > self.cache.max_len:
                    break                                                    # context full
                tree_tokens = tree.build_candidate_tokens(root, medusa_logits)   # §4.2/4.3
                out = self.model(
                    tree_tokens[None],
                    attention_mask=build_tree_attention_mask(tree.ancestor_mask, past, self.dtype),  # §4.4
                    position_ids=tree.position_ids(past),
                    past_key_values=self.cache,
                    cache_position=self._arange(past, tree.num_nodes),
                    use_cache=True, medusa_forward=False, lm_logits_to_keep=0,
                )                                                            # §4.5 one verification pass
                logits = out.lm_logits[0]                                    # [N, V]
                best, n_acc = evaluate_candidates(tree, tree_tokens, logits,
                                                  self.temperature, self.epsilon, self.delta)   # §4.6–4.8
                n_acc = int(n_acc)                                           # one host sync per step
                path = tree.retrieve_indices[best, : n_acc + 1]              # node ids, root first
                self.cache.compact(past + path)                              # §4.17 step 8
                last = path[-1]
                root = self._select_root(logits[last])
                medusa_logits = self.model.heads(out.hidden_states[0, last])
                stats.record_step(n_acc)
                before = len(out_tokens)
                stop = self._emit(tree_tokens[path].tolist(), out_tokens, max_new_tokens)
                if callback:
                    callback(out_tokens[before:])
                if stop:
                    break
        stats.decode_time = t.elapsed
        stats.new_tokens = len(out_tokens)
        return out_tokens, stats

    # ------------------------------------------------------------------ baseline
    @torch.inference_mode()
    def baseline_generate(self, input_ids: torch.Tensor, max_new_tokens: int,
                          callback: Optional[Callable[[List[int]], None]] = None) -> Tuple[List[int], DecodingStats]:
        """Standard autoregressive decoding, one token per forward pass, same 4-bit backbone and the
        same preallocated KV cache, so the comparison isolates what Medusa adds (§3.11)."""
        backbone = self.model.backbone
        stats = DecodingStats(num_heads=0)
        input_ids = input_ids.to(self.device)
        n = input_ids.shape[1]
        self.cache.reset()
        stats.prompt_tokens = n
        with CudaTimer() as t:
            out = backbone(input_ids, position_ids=self._arange(0, n)[None], past_key_values=self.cache,
                           cache_position=self._arange(0, n), use_cache=True, num_logits_to_keep=1)
            self.cache.commit()
            nxt = select_token(out.logits[0, -1], self.temperature, self.top_k, self.top_p)
        stats.prefill_time = t.elapsed
        out_tokens: List[int] = []
        with CudaTimer() as t:
            while self.cache.length < self.max_context:
                past = self.cache.length
                out = backbone(nxt.view(1, 1), position_ids=self._arange(past, 1)[None],
                               past_key_values=self.cache, cache_position=self._arange(past, 1),
                               use_cache=True, num_logits_to_keep=1)
                self.cache.commit()
                tok = int(nxt)
                nxt = select_token(out.logits[0, -1], self.temperature, self.top_k, self.top_p)
                stats.record_step(0)
                stop = self._emit([tok], out_tokens, max_new_tokens)
                if callback:
                    callback([tok])
                if stop:
                    break
        stats.decode_time = t.elapsed
        stats.new_tokens = len(out_tokens)
        return out_tokens, stats
