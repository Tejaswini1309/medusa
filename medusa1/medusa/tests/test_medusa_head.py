"""Head shapes, initialisation (§4.1.1), independence, and loss alignment (§4.9). CPU-only, tiny model."""
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from medusa.model.medusa_head import MedusaHeads
from medusa.model.medusa_model import MedusaModel
from medusa.train.loss import IGNORE_INDEX, head_loss_weights, medusa1_loss


def tiny_model(num_heads=3, vocab=64, seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
                      attn_implementation="sdpa")
    return MedusaModel(LlamaForCausalLM(cfg).eval(), num_heads=num_heads, num_layers=1)


def test_output_shapes():
    m = tiny_model(num_heads=3)
    ids = torch.randint(0, 64, (2, 7))
    out = m(ids)
    assert out.lm_logits.shape == (2, 7, 64)
    assert out.hidden_states.shape == (2, 7, 32)
    assert out.medusa_logits.shape == (3, 2, 7, 64)


def test_init_matches_lm_head():
    """W_1 = 0 => SiLU(0) + h = h => head logits == original LM logits, exactly."""
    m = tiny_model()
    out = m(torch.randint(0, 64, (1, 9)))
    for k in range(3):
        torch.testing.assert_close(out.medusa_logits[k], out.lm_logits, rtol=0, atol=1e-6)
    for head in m.heads.heads:
        assert torch.count_nonzero(head.blocks[0].linear.weight) == 0


def test_heads_are_independent_parameters():
    m = tiny_model()
    ptrs = {h.lm_head.weight.data_ptr() for h in m.heads.heads}
    assert len(ptrs) == 3 and m.get_lm_head().weight.data_ptr() not in ptrs
    with torch.no_grad():
        m.heads.heads[0].lm_head.weight.add_(1.0)
    assert not torch.equal(m.heads.heads[0].lm_head.weight, m.heads.heads[1].lm_head.weight)


def test_resblock_formula():
    heads = MedusaHeads(hidden_size=8, vocab_size=5, num_heads=1)
    torch.nn.init.normal_(heads.heads[0].blocks[0].linear.weight)
    h = torch.randn(3, 8)
    w1 = heads.heads[0].blocks[0].linear.weight
    w2 = heads.heads[0].lm_head.weight
    expected = (torch.nn.functional.silu(h @ w1.T) + h) @ w2.T
    torch.testing.assert_close(heads(h)[0], expected)


def test_loss_targets_are_shifted_by_k_plus_2():
    """Head k (0-based) at position t must be scored against labels[t+k+2]."""
    V, T = 10, 8
    labels = torch.arange(T).unsqueeze(0)                    # token at position t is t
    logits = torch.full((2, 1, T, V), -1e4)
    for k in range(2):
        for t in range(T):
            if t + k + 2 < T:
                logits[k, 0, t, t + k + 2] = 1e4                 # perfect prediction of y_{t+k+2}
    loss, logs = medusa1_loss(logits, labels, head_loss_weights(2))
    assert loss.item() < 1e-4 and logs["medusa0_top1"] == 1.0 and logs["medusa1_top1"] == 1.0
    # A mis-shifted (k+1) prediction must give a large loss.
    bad = torch.roll(logits, shifts=1, dims=2)
    assert medusa1_loss(bad, labels, head_loss_weights(2))[0].item() > 100


def test_loss_ignores_masked_labels():
    labels = torch.full((1, 6), IGNORE_INDEX)
    logits = torch.randn(1, 1, 6, 10)
    loss, logs = medusa1_loss(logits, labels, [1.0])
    assert loss.item() == 0.0 and logs == {}


def test_lambda_weights():
    w = head_loss_weights(5, 0.8)
    assert abs(w[0] - 0.8) < 1e-12 and abs(w[4] - 0.8 ** 5) < 1e-12
