from typing import cast

from datasets import IterableDataset, load_dataset, logging

logging.set_verbosity_debug()

dataset = cast(IterableDataset, load_dataset("HuggingFaceFW/finewiki", "en", split="train", streaming=True))
dataset = dataset.take(2)
texts = [d["text"] for d in dataset]
print(texts[0])
