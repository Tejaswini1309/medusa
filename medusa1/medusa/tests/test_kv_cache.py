"""Tree attention mask (§4.4) and KV-cache behaviour (§3.6) against brute-force recomputation."""
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from medusa.generation.kv_cache import TreeKVCache, build_tree_attention_mask
from medusa.generation.tree import MedusaTree


def tiny_backbone(vocab=64, seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      attn_implementation="sdpa")
    return LlamaForCausalLM(cfg).eval()


def full_logits(model, ids):
    """Reference: plain causal forward, no cache."""
    with torch.no_grad():
        return model(torch.tensor([ids])).logits[0]


def prefill(model, cache, ids):
    n = len(ids)
    with torch.no_grad():
        out = model(torch.tensor([ids]), position_ids=torch.arange(n)[None], past_key_values=cache,
                    cache_position=torch.arange(n), use_cache=True)
    cache.commit()
    return out


def tree_pass(model, cache, tree, tree_tokens):
    past = cache.length
    with torch.no_grad():
        return model(tree_tokens[None],
                     attention_mask=build_tree_attention_mask(tree.ancestor_mask, past, torch.float32),
                     position_ids=tree.position_ids(past), past_key_values=cache,
                     cache_position=torch.arange(past, past + tree.num_nodes), use_cache=True).logits[0]


def test_ancestor_mask_example_from_architecture():
    """Root -> {A, B} -> {C, D, E}: C under A sees Root, A, C and nothing on the B branch."""
    tree = MedusaTree([2, 3])
    m = tree.ancestor_mask
    a, b = 1, 2
    c_under_a = next(n.index for n in tree.nodes if n.path == (0, 0))
    c_under_b = next(n.index for n in tree.nodes if n.path == (1, 0))
    assert set(torch.nonzero(m[c_under_a]).flatten().tolist()) == {0, a, c_under_a}
    assert not m[c_under_a, b] and not m[c_under_a, c_under_b]
    assert m.diagonal().all()
    # causal inside a branch: an ancestor never sees its descendant
    assert not m[a, c_under_a]


def test_mask_shape_and_context_visibility():
    tree = MedusaTree([2, 3])
    mask = build_tree_attention_mask(tree.ancestor_mask, past_len=5, dtype=torch.float16)
    assert mask.shape == (1, 1, tree.num_nodes, 5 + tree.num_nodes)
    assert (mask[0, 0, :, :5] == 0).all()
    blocked = mask[0, 0, :, 5:] != 0
    assert torch.equal(blocked, ~tree.ancestor_mask)


def test_tree_verification_matches_sequential_forward():
    """Every node's logits from ONE tree pass == logits of context + its branch run causally."""
    model = tiny_backbone()
    tree = MedusaTree([3, 2, 2])
    ctx = [5, 17, 3, 42, 9]
    cache = TreeKVCache(model.config, max_len=64, dtype=torch.float32)
    prefill(model, cache, ctx)
    torch.manual_seed(1)
    tree_tokens = torch.randint(0, 64, (tree.num_nodes,))
    logits = tree_pass(model, cache, tree, tree_tokens)
    for node in tree.nodes:
        branch = [int(tree_tokens[i]) for i in tree.ancestors(node.index)]
        ref = full_logits(model, ctx + branch)[-1]
        torch.testing.assert_close(logits[node.index], ref, rtol=1e-4, atol=1e-4)


def test_compaction_keeps_only_accepted_path():
    """After keeping one branch, the cache must behave as if context+branch had been prefilled."""
    model = tiny_backbone()
    tree = MedusaTree([2, 3, 2])
    ctx = [1, 2, 3, 4]
    cache = TreeKVCache(model.config, max_len=64, dtype=torch.float32)
    prefill(model, cache, ctx)
    tree_tokens = torch.randint(0, 64, (tree.num_nodes,))
    tree_pass(model, cache, tree, tree_tokens)
    path = tree.retrieve_indices[4, :3]          # root + 2 accepted tokens of branch 4
    cache.compact(len(ctx) + path)
    assert cache.length == len(ctx) + 3
    accepted = tree_tokens[path].tolist()

    nxt = 11
    past = cache.length
    with torch.no_grad():
        got = model(torch.tensor([[nxt]]), position_ids=torch.tensor([[past]]), past_key_values=cache,
                    cache_position=torch.tensor([past]), use_cache=True).logits[0, -1]
    ref = full_logits(model, ctx + accepted + [nxt])[-1]
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


def test_cache_overflow_raises():
    model = tiny_backbone()
    cache = TreeKVCache(model.config, max_len=4, dtype=torch.float32)
    try:
        prefill(model, cache, [1, 2, 3, 4, 5])
    except RuntimeError as e:
        assert "overflow" in str(e)
    else:
        raise AssertionError("expected overflow error")
