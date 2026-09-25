"""Candidate tree (§4.2/4.3), acceptance (§4.6–4.8) and end-to-end decoding correctness."""
import itertools
import math

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from medusa.generation.decode import MedusaDecoder, acceptance_threshold, evaluate_candidates
from medusa.generation.tree import MedusaTree
from medusa.model.medusa_model import MedusaModel


def tiny_medusa(vocab=32, num_heads=3, seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512,
                      attn_implementation="sdpa")
    m = MedusaModel(LlamaForCausalLM(cfg).eval(), num_heads=num_heads)
    with torch.no_grad():                              # untrained but *different* from the LM head
        for h in m.heads.heads:
            torch.nn.init.normal_(h.blocks[0].linear.weight, std=0.3)
    return m


def test_node_and_branch_counts():
    for topk in ([2, 3], [4, 2, 2, 1, 1], [1], [3, 3, 3]):
        tree = MedusaTree(topk)
        expected_nodes = 1 + sum(math.prod(topk[: k + 1]) for k in range(len(topk)))
        assert tree.num_nodes == expected_nodes
        assert tree.num_branches == math.prod(topk)
        assert tree.retrieve_indices.shape == (math.prod(topk), len(topk) + 1)


def test_node_metadata():
    tree = MedusaTree([2, 3])
    for n in tree.nodes[1:]:
        parent = tree.nodes[n.parent]
        assert n.depth == parent.depth + 1 and n.index in parent.children
        assert n.head == n.depth and n.path[:-1] == parent.path and n.rank == n.path[-1]
    assert int(tree.position_ids(10)[0, 0]) == 10 and int(tree.position_ids(10)[0, -1]) == 12


def test_candidates_are_cartesian_product():
    tree = MedusaTree([2, 3])
    V = 20
    medusa_logits = torch.randn(2, V)
    tokens = tree.build_candidate_tokens(torch.tensor(7), medusa_logits)
    top1 = medusa_logits[0].topk(2).indices.tolist()
    top2 = medusa_logits[1].topk(3).indices.tolist()
    got = {tuple(c) for c in tree.candidates(tokens).tolist()}
    assert got == {(7, a, b) for a, b in itertools.product(top1, top2)}


def test_greedy_acceptance_picks_longest_argmax_prefix():
    tree = MedusaTree([2, 2])
    V = 6
    # tree tokens: root=0 | depth1: 1, 2 | depth2 under node1: 3, 4 ; under node2: 3, 4
    tokens = torch.tensor([0, 1, 2, 3, 4, 3, 4])
    logits = torch.zeros(tree.num_nodes, V)
    logits[0, 2] = 5.0          # after root the model wants token 2 -> node 2 accepted, node 1 rejected
    logits[2, 4] = 5.0          # after node 2 it wants 4 -> node 6 (path 1,1) accepted
    best, n_acc = evaluate_candidates(tree, tokens, logits, temperature=0.0, epsilon=0.09, delta=0.3)
    assert int(n_acc) == 2
    assert tree.retrieve_indices[best].tolist() == [0, 2, 6]


def test_typical_acceptance_threshold():
    """threshold = min(ε, δ·exp(-H)). Uniform over V -> H = log V -> δ/V."""
    V = 50
    logp, thr = acceptance_threshold(torch.zeros(1, V), 1.0, epsilon=0.09, delta=0.3)
    assert abs(thr.item() - 0.3 / V) < 1e-6
    peaked = torch.full((1, V), -30.0)
    peaked[0, 0] = 30.0                                       # H ~ 0 -> threshold = min(0.09, 0.3) = 0.09
    _, thr = acceptance_threshold(peaked, 1.0, 0.09, 0.3)
    assert abs(thr.item() - 0.09) < 1e-6


def test_typical_acceptance_rejects_improbable_token():
    tree = MedusaTree([1])
    logits = torch.full((2, 10), -10.0)
    logits[0, 3] = 10.0                                       # p(3) ~ 1, p(others) ~ 2e-9
    ok, n_ok = evaluate_candidates(tree, torch.tensor([0, 3]), logits, 0.7, 0.09, 0.3)
    bad, n_bad = evaluate_candidates(tree, torch.tensor([0, 5]), logits, 0.7, 0.09, 0.3)
    assert int(n_ok) == 1 and int(n_bad) == 0


def test_greedy_medusa_equals_greedy_baseline():
    """With greedy acceptance the output must be token-for-token identical to plain greedy decoding,
    whatever the heads predict. A mismatch means a mask, position, or KV-cache bug."""
    model = tiny_medusa()
    tree = MedusaTree([8, 4, 2])                             # large tree over V=32 -> some acceptances
    dec = MedusaDecoder(model, tree, eos_token_id=-1, max_context=300, temperature=0.0)
    total_accepted = 0
    for seed in range(4):
        g = torch.Generator().manual_seed(seed)
        prompt = torch.randint(0, 32, (1, 6), generator=g)
        base, _ = dec.baseline_generate(prompt, 60)
        med, stats = dec.generate(prompt, 60)
        assert med == base, f"seed {seed}: medusa {med} != baseline {base}"
        total_accepted += sum(stats.accept_lengths)
        assert stats.steps < 60 or sum(stats.accept_lengths) == 0
    assert total_accepted > 0, "test did not exercise the accept>0 / compaction path"


def test_eos_and_max_tokens_stop():
    model = tiny_medusa()
    dec = MedusaDecoder(model, MedusaTree([2, 2]), eos_token_id=-1, max_context=300)
    out, _ = dec.generate(torch.tensor([[1, 2, 3]]), 17)
    assert len(out) == 17
    eos = out[5]
    dec.eos = eos
    out2, _ = dec.generate(torch.tensor([[1, 2, 3]]), 17)
    assert out2[-1] == eos and len(out2) == out.index(eos) + 1
