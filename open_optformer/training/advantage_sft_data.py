"""Plain (unpacked) dataset/DataModule for advantage-reweighted SFT.

This deliberately bypasses litdata's `TokensLoader`/`StreamingDataset` packing
(as used by `SyneTuneData` for pretraining): packing many tokenized prompts
into fixed-size raw token blocks discards trajectory boundaries, so a
per-position weight array built from per-trial advantages cannot be sliced
back into alignment after packing. For a bounded fine-tuning run, this trades
streaming throughput for correctness by keeping one trajectory (with its
aligned weight array) per dataset item, tokenized on the fly and padded/
truncated to the training block size.

This module is separate from `syne_tune_data.py` on purpose: `SyneTuneData`
and its vocab-level `loss_weights` mechanism are left untouched.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset

from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer

from weighting import segments_to_token_weights


class AdvantageWeightedDataset(Dataset):
    """One trajectory (prompt segments + per-trial advantage weights) per item.

    Each JSONL line is expected to have a "segments" field: a list of
    [text_chunk, weight] pairs, as produced by
    `History.get_prompt_with_weights`. The shared header segment carries a
    neutral weight; each trial's `{config}*{y}|` span carries that trial's
    advantage.
    """

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        tokenizer: Tokenizer,
        block_size: int,
        ignore_index: int = -100,
        pad_id: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.ignore_index = ignore_index
        self.pad_id = pad_id

        with open(jsonl_path, "r") as f:
            self.examples = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        segments = self.examples[idx]["segments"]
        token_ids, token_weights = segments_to_token_weights(segments, self.tokenizer)

        # +1 so the last input position still has a next-token target, matching
        # SyneTuneData's `seq_length = max_seq_length + 1` convention.
        token_ids = token_ids[: self.block_size + 1]
        token_weights = token_weights[: self.block_size + 1]

        input_ids = token_ids[: self.block_size]
        targets = token_ids[1 : self.block_size + 1]
        sample_weights = token_weights[1 : self.block_size + 1]

        pad_len = self.block_size - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_id] * pad_len
        target_pad_len = self.block_size - len(targets)
        if target_pad_len > 0:
            targets = targets + [self.ignore_index] * target_pad_len
            sample_weights = sample_weights + [0.0] * target_pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
            "sample_weights": torch.tensor(sample_weights, dtype=torch.float),
        }


@dataclass
class AdvantageWeightedData(DataModule):
    """DataModule wrapping `AdvantageWeightedDataset` for advantage-reweighted SFT.

    Reads plain JSONL files (produced by `compile_data.py --emit_advantage_weighted`)
    rather than a litdata-preprocessed/packed directory.
    """

    train_jsonl: Union[str, Path] = Path("data/advantage_train.jsonl")
    val_jsonl: Union[str, Path] = Path("data/advantage_valid.jsonl")
    seed: int = 42
    num_workers: int = 8

    batch_size: int = field(init=False, repr=False, default=1)
    block_size: int = field(init=False, repr=False, default=2048)

    tokenizer: Optional[Tokenizer] = field(init=False, repr=False, default=None)
    train_dataset: Optional[AdvantageWeightedDataset] = field(init=False, repr=False, default=None)
    val_dataset: Optional[AdvantageWeightedDataset] = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        super().__init__()

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.block_size = max_seq_length

    def setup(self, stage: str = "") -> None:
        self.train_dataset = AdvantageWeightedDataset(self.train_jsonl, self.tokenizer, self.block_size)
        self.val_dataset = AdvantageWeightedDataset(self.val_jsonl, self.tokenizer, self.block_size)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
