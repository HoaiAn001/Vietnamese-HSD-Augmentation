from __future__ import annotations

from deep_translator import GoogleTranslator

from .base_augmenter import AugmentedSample, BaseAugmenter


class BackTranslationAugmenter(BaseAugmenter):
    method_name = "back_translation"

    def __init__(
        self,
        source_language: str = "vi",
        intermediate_language: str = "en",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.source_language = source_language
        self.intermediate_language = intermediate_language

    def augment_one(self, text: str, label: str) -> list[AugmentedSample]:
        forward = GoogleTranslator(
            source=self.source_language, target=self.intermediate_language
        ).translate(text)
        backward = GoogleTranslator(
            source=self.intermediate_language, target=self.source_language
        ).translate(forward)
        return [
            AugmentedSample(
                original_text=text,
                text=backward,
                label=label,
                method=self.method_name,
                metadata=f"via={self.intermediate_language}",
            )
        ]
