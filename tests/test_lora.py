import timm
import torch
import torch.nn as nn

from petslice.models.classifier import AttentionPool, SliceClassifier
from petslice.models.encoders import TIMM_VIT_LORA, build_encoder
from petslice.models.lora import LoRAConv2d, LoRALinear, inject_lora


def tiny_encoder(name="dinov3", size=32):
    encoder = build_encoder(name, size, pretrained=False)
    return encoder


def test_lora_is_identity_at_initialization():
    base = nn.Linear(6, 4)
    wrapped = LoRALinear(base, rank=2, alpha=4.0)
    x = torch.randn(3, 6)
    assert torch.equal(wrapped(x), base(x))
    conv = nn.Conv2d(3, 8, 4, 4)
    wrapped_conv = LoRAConv2d(conv, rank=2, alpha=4.0)
    image = torch.randn(1, 3, 16, 16)
    assert torch.equal(wrapped_conv(image), conv(image))


def test_inject_lora_freezes_everything_but_the_adapters():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    wrapped = inject_lora(model, lambda name: name == "2", rank=2, alpha=4.0)
    assert wrapped == ["2"]
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"2.lora_a.weight", "2.lora_b.weight"}


def test_lora_targets_match_the_paper_on_a_vit():
    # Patch embedding plus the self-attention projections of each of the 24 blocks.
    model = timm.create_model("vit_large_patch16_dinov3", pretrained=False, num_classes=0)
    targets = [n for n, _ in model.named_modules() if TIMM_VIT_LORA.match(n)]
    assert len(targets) == 1 + 24 * 2
    assert "patch_embed.proj" in targets and not any("mlp" in t for t in targets)


def test_full_model_trains_only_about_one_million_parameters():
    classifier = SliceClassifier(tiny_encoder(), n_outputs=1)
    assert 1.1e6 < classifier.n_trainable() < 1.3e6
    assert all("lora" in n for n, p in classifier.encoder.named_parameters() if p.requires_grad)


def test_encoder_stays_in_eval_mode_during_training():
    classifier = SliceClassifier(tiny_encoder(), n_outputs=1).train()
    assert not classifier.encoder.training and classifier.head.training


def test_attention_pool_matches_equation_five():
    pool = AttentionPool(4)
    tokens = torch.randn(2, 5, 4)
    pooled, weights = pool(tokens)
    expected = torch.softmax(tokens @ pool.query / 2.0, dim=1)
    assert torch.allclose(weights, expected)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))
    assert torch.allclose(pooled, (expected[..., None] * tokens).sum(dim=1))


def test_late_fusion_concatenates_pet_and_ct_tokens_per_slice():
    classifier = SliceClassifier(tiny_encoder(), n_outputs=4, late_fusion=True).eval()
    with torch.no_grad():
        logits, weights = classifier(torch.rand(1, 3, 6, 32, 32))
    assert logits.shape == (1, 4) and weights.shape == (1, 3)
    assert classifier.pool.query.numel() == 2 * classifier.encoder.dim


def test_class_token_is_used_rather_than_the_default_average():
    encoder = tiny_encoder().eval()
    images = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        features = encoder.model.forward_features((images - encoder.mean) / encoder.std)
        assert torch.equal(encoder(images), features[:, 0])
