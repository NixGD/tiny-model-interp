from jaxtyping import Float
from torch import Tensor
from collections.abc import Callable

from tiny_model.tokenizer.char_tokenizer import CharTokenizer

LogitTensor = Float[Tensor, "... vocab_size"]
LogitLossFn = Callable[[LogitTensor], Float[Tensor, "..."]]


def get_logit_diff_loss(
    tokenizer: CharTokenizer, positive_tokens: list[str], negative_tokens: list[str]
) -> LogitLossFn:
    positive_token_ids = [tokenizer.encode_one(token) for token in positive_tokens]
    negative_token_ids = [tokenizer.encode_one(token) for token in negative_tokens]

    def loss_fn(logits: LogitTensor) -> Float[Tensor, "..."]:
        positive_logits = logits[..., positive_token_ids].mean(dim=-1)
        negative_logits = logits[..., negative_token_ids].mean(dim=-1)
        return positive_logits - negative_logits

    return loss_fn
