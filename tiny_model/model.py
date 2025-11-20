"""Adapted from nanoGPT: https://github.com/karpathy/nanoGPT."""

import inspect
import math
from collections.abc import Callable
from typing import NamedTuple, cast

import torch
import torch.nn as nn
from lxt.efficient.rules import divide_gradient, identity_rule_implicit
from pydantic import BaseModel
from torch.nn import functional as F

# Type aliases for hook management
Activations = torch.Tensor
HookFn = Callable[[Activations], Activations]


class GPTConfig(BaseModel):
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


def lxt_divide_gradient(lxt_enabled: bool, x: torch.Tensor) -> torch.Tensor:
    if lxt_enabled:
        return cast(torch.Tensor, divide_gradient(x, 2))
    return x


def lxt_identity(lxt_enabled: bool, fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    if lxt_enabled:
        return cast(torch.Tensor, identity_rule_implicit(fn, x))
    else:
        return fn(x)


class CacheKey(NamedTuple):
    key: str
    layer: int | None


class AlphaCache:
    _cache_enabled: bool = False
    _alphas_enabled: bool = False
    _cache: dict[CacheKey, torch.Tensor]
    _alphas: dict[CacheKey, torch.Tensor]
    _hooks: dict[CacheKey, HookFn]

    def __init__(
        self,
        cache_enabled: bool = False,
        alphas_enabled: bool = False,
        hooks: dict[CacheKey, HookFn] | None = None,
    ):
        self._cache_enabled = cache_enabled
        self._alphas_enabled = alphas_enabled
        self._cache = {}
        self._alphas = {}
        self._hooks = hooks if hooks is not None else {}

    def wrap(self, key: CacheKey, x: torch.Tensor) -> torch.Tensor:
        """Wraps an activation, returning identical values but storing value & gradient of that activation."""
        # Apply hook if one exists for this key
        if key in self._hooks:
            x = self._hooks[key](x)

        if self._cache_enabled:
            self._cache[key] = x.detach()

        if self._alphas_enabled:
            alpha = torch.zeros_like(x, requires_grad=True)
            self._alphas[key] = alpha
            return alpha + x

        return x

    def get_grad(self, key: CacheKey) -> torch.Tensor | None:
        return self._alphas[key].grad

    def get_value(self, key: CacheKey) -> torch.Tensor:
        return self._cache[key]

    def keys(self) -> list[CacheKey]:
        return list(self._cache.keys())


class CausalSelfAttention(nn.Module):
    freqs_cis: torch.Tensor
    bias: torch.Tensor

    def __init__(self, config: GPTConfig, layer: int):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.layer = layer
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.sinks = nn.Parameter(torch.zeros(config.n_head))

        # RoPE embeddings
        head_dim = config.n_embd // config.n_head
        freqs_cis = self._precompute_freqs_cis(head_dim, config.block_size)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.lxt_enabled = False

    def forward(self, x: torch.Tensor, cache: AlphaCache) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # apply RoPE
        q, k = self._apply_rotary_emb(q, k, self.freqs_cis)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = lxt_divide_gradient(self.lxt_enabled, att)

        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))  # type: ignore
        sinks = self.sinks.view(1, self.n_head, 1, 1).expand(B, self.n_head, T, 1)
        att = torch.cat([sinks, att], dim=-1)  # add sinks
        att = cache.wrap(CacheKey("attn_scores", self.layer), att)
        att = F.softmax(att, dim=-1)
        att = att[:, :, :, 1:]  # remove sinks
        att = self.attn_dropout(att)

        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = lxt_divide_gradient(self.lxt_enabled, y)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

    @staticmethod
    def _precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
        """Precompute the frequency tensor for complex exponentials (RoPE)."""
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    @staticmethod
    def _apply_rotary_emb(
        xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors."""
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = freqs_cis[: xq_.shape[1]]
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, T, 1, dim//2)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def set_lxt_enabled(self, lxt_enabled: bool):
        self.lxt_enabled = lxt_enabled


class MLP(nn.Module):
    def __init__(self, config: GPTConfig, layer: int):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.layer = layer
        self.lxt_enabled = False

    def forward(self, x: torch.Tensor, cache: AlphaCache) -> torch.Tensor:
        x = self.c_fc(x)
        x = cache.wrap(CacheKey("mlp_pre_act", self.layer), x)
        x = lxt_identity(self.lxt_enabled, self.relu, x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

    def set_lxt_enabled(self, lxt_enabled: bool):
        self.lxt_enabled = lxt_enabled


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer)
        self.mlp = MLP(config, layer)
        self.layer = layer

    def forward(self, x: torch.Tensor, cache: AlphaCache) -> torch.Tensor:
        x = cache.wrap(CacheKey("resid_pre", self.layer), x)
        x = x + self.attn(x, cache)
        x = cache.wrap(CacheKey("resid_mid", self.layer), x)
        x = x + self.mlp(x, cache)
        return x

    def set_lxt_enabled(self, lxt_enabled: bool):
        self.attn.set_lxt_enabled(lxt_enabled)
        self.mlp.set_lxt_enabled(lxt_enabled)


class Out(NamedTuple):
    logits: torch.Tensor
    loss: torch.Tensor | None
    cache: AlphaCache


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks: nn.ModuleList[Block] = nn.ModuleList([Block(config, layer) for layer in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters
        print(f"number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model.
        For non-embedding count (default), the token embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embed.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:  # type: ignore
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache_enabled: bool = False,
        alphas_enabled: bool = False,
        hooks: dict[CacheKey, HookFn] | None = None,
    ) -> Out:
        seqlen = idx.size(1)
        assert seqlen <= self.config.block_size, (
            f"Cannot forward sequence of length {seqlen}, block size is only {self.config.block_size}"
        )

        cache = AlphaCache(cache_enabled=cache_enabled, alphas_enabled=alphas_enabled, hooks=hooks)

        tok_emb = self.token_embed(idx)  # token embeddings of shape (b, t, n_embd)
        tok_emb = cache.wrap(CacheKey("tok_emb", None), tok_emb)
        x = self.dropout(tok_emb)
        for block in self.blocks:
            x = block(x, cache)

        x = cache.wrap(CacheKey("resid_final", None), x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            loss = None

        return Out(logits=logits, loss=loss, cache=cache)

    def __call__(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache_enabled: bool = False,
        alphas_enabled: bool = False,
        hooks: dict[CacheKey, HookFn] | None = None,
    ) -> Out:
        return self.forward(idx, targets, cache_enabled, alphas_enabled, hooks)

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, betas: tuple[float, float], device_type: str
    ) -> torch.optim.AdamW:
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None
    ) -> torch.Tensor:
        """Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            # forward the model to get the logits for the index in the sequence
            out = self(idx_cond)
            logits = out.logits
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def set_lxt_enabled(self, lxt_enabled: bool):
        for block in self.blocks:
            block.set_lxt_enabled(lxt_enabled)
