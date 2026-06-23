from __future__ import annotations

import random

from .base_augmenter import AugmentedSample, BaseAugmenter


VIETNAMESE_FILLERS = [
    "thật sự",
    "rất",
    "khá",
    "này",
    "đó",
    "luôn",
]


class EDAAugmenter(BaseAugmenter):
    method_name = "eda"

    def __init__(
        self,
        operations: list[str] | None = None,
        alpha: float = 0.1,
        num_aug: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.operations = operations or ["random_swap", "random_deletion", "random_insertion"]
        self.alpha = alpha
        self.num_aug = num_aug
        self.random = random.Random(self.random_state)

    def augment_one(self, text: str, label: str) -> list[AugmentedSample]:
        samples: list[AugmentedSample] = []
        for idx in range(self.num_aug):
            operation = self.operations[idx % len(self.operations)]
            augmented = self._apply(operation, text)
            samples.append(
                AugmentedSample(
                    original_text=text,
                    text=augmented,
                    label=label,
                    method=self.method_name,
                    metadata=f"operation={operation};alpha={self.alpha}",
                )
            )
        return samples

    def _apply(self, operation: str, text: str) -> str:
        words = text.split()
        if len(words) < 2:
            return text
        if operation == "random_swap":
            return self._random_swap(words)
        if operation == "random_deletion":
            return self._random_deletion(words)
        if operation == "random_insertion":
            return self._random_insertion(words)
        raise ValueError(f"Unsupported EDA operation: {operation}")

    def _random_swap(self, words: list[str]) -> str:
        words = words.copy()
        swaps = max(1, int(self.alpha * len(words)))
        for _ in range(swaps):
            i, j = self.random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
        return " ".join(words)

    def _random_deletion(self, words: list[str]) -> str:
        kept = [word for word in words if self.random.random() > self.alpha]
        if not kept:
            kept = [self.random.choice(words)]
        return " ".join(kept)

    def _random_insertion(self, words: list[str]) -> str:
        words = words.copy()
        inserts = max(1, int(self.alpha * len(words)))
        for _ in range(inserts):
            idx = self.random.randrange(0, len(words) + 1)
            words.insert(idx, self.random.choice(VIETNAMESE_FILLERS))
        return " ".join(words)
