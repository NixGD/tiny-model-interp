"""Load a trained GPT model from checkpoint and enable cache for analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from lxt.efficient import monkey_patch

from tiny_model.model import GPT, CacheKey, GPTConfig
from tiny_model.tokenize import CharTokenizer
from tiny_model.utils import REPO_ROOT

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path: str | Path) -> GPT:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model_args = checkpoint["model_args"]
    model = GPT(GPTConfig(**model_args))
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    return model


model_path = REPO_ROOT / "out-wiki-char/ckpt.pt"
model = load_model(str(model_path))
model.eval()

monkey_patch(model)


# %%
tokenizer = CharTokenizer(vocab_path="data/wiki_char/vocab.json")
prompt = tokenizer.encode("Hello, world!")
print(prompt)

# %%
idx = torch.tensor([prompt]).to(DEVICE)
output = model(idx, enable_cache=True)
cache = output.cache

print(f"✓ Generated output shape: {output.logits.shape}")
print(f"✓ Cache enabled: {cache.enabled}")
print(f"✓ Number of cached activations: {len(cache.keys())}")

# Display all cache keys with their shapes
cache_keys = cache.keys()
if cache.enabled and cache_keys:
    print("\nCached activations:")
    for key in cache_keys:
        value = cache.get_value(key)
        print(f"  {key.key} @ {key.layer}:\t {list(value.shape)}")


# %%
def get_loss_fn(pos_class_chars: str):
    multiplier = torch.full((tokenizer.vocab_size,), -1.0)
    for char in pos_class_chars:
        multiplier[tokenizer.stoi[char]] = 1.0

    def loss_fn(logits: torch.Tensor) -> torch.Tensor:
        return (logits * multiplier).sum(dim=-1)

    return loss_fn


eos_punctuation_loss = get_loss_fn(".!?")
loss = eos_punctuation_loss(output.logits)
loss[0, -1].backward()


layer = 1
mlp_acts = cache.get_value(CacheKey("mlp_pre_act", layer))
mlp_grads = cache.get_grad(CacheKey("mlp_pre_act", layer))
assert mlp_grads is not None

# %%
plt.scatter(mlp_acts[0, -1, :].cpu().numpy(), mlp_grads[0, -1, :].cpu().numpy())
plt.xlabel("MLP activations")
plt.ylabel("MLP gradients")
plt.title("MLP activations vs gradients")
plt.show()


# %%
