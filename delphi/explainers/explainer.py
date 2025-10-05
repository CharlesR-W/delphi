import json
import os
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import aiofiles

from delphi import logger

from ..clients.client import Client, Response
from ..latents.latents import LatentRecord


class ExplainerResult(NamedTuple):
    record: LatentRecord
    """Latent record passed through to scorer."""

    explanation: str
    """Generated explanation for latent."""

    explanation_id: Optional[int] = 0
    """The id of the explanation; used for explainers which produce multiple explanations."""


@dataclass
class Explainer(ABC):
    """
    Abstract base class for explainers.
    """

    client: Client
    """Client to use for explanation generation. """
    verbose: bool = False
    """Whether to print verbose output."""
    threshold: float = 0.3
    """The activation threshold to select tokens to highlight."""
    temperature: float = 0.0
    """The temperature for explanation generation."""
    generation_kwargs: dict = field(default_factory=dict)
    """Additional keyword arguments for the generation client."""

    async def __call__(self, record: LatentRecord) -> ExplainerResult:
        messages = self._build_prompt(record)
        response = await self.client.generate(
            messages, temperature=self.temperature, **self.generation_kwargs
        )
        assert isinstance(response, Response)

        try:
            explanation = self.parse_explanation(response.text)
            if self.verbose:
                logger.info(f"Explanation: {explanation}")
                # Show only first 500 chars of last message to avoid huge logs
                last_msg = messages[-1]["content"]
                msg_preview = (
                    last_msg[:500] + "..." if len(last_msg) > 500 else last_msg
                )
                logger.info(f"Messages (preview): {msg_preview}")
                logger.info(f"Response: {response}")

            # Use explanation_id from record if it exists
            explanation_id = getattr(record, "explanation_id", None)
            return ExplainerResult(
                record=record, explanation=explanation, explanation_id=explanation_id
            )
        except Exception as e:
            logger.error(f"Explanation parsing failed: {repr(e)}")
            explanation_id = getattr(record, "explanation_id", None)
            return ExplainerResult(
                record=record,
                explanation="Explanation could not be parsed.",
                explanation_id=explanation_id,
            )

    def parse_explanation(self, text: str) -> str:
        try:
            # Allow optional whitespace before the colon to be more robust
            match = re.search(r"\[EXPLANATION\]\s*:\s*(.*)", text, re.DOTALL)
            if match:
                return match.group(1).strip()
            else:
                return "Explanation could not be parsed."
        except Exception as e:
            logger.error(f"Explanation parsing regex failed: {repr(e)}")
            raise

    def _highlight(self, str_toks: list[str], activations: list[float]) -> str:
        result = ""
        threshold = max(activations) * self.threshold

        def check(i):
            return activations[i] > threshold

        i = 0
        while i < len(str_toks):
            if check(i):
                result += "<<"

                while i < len(str_toks) and check(i):
                    result += str_toks[i]
                    i += 1
                result += ">>"
            else:
                result += str_toks[i]
                i += 1

        return "".join(result)

    def _join_activations(
        self,
        str_toks: list[str],
        token_activations: list[float],
        normalized_activations: list[float],
    ) -> str:
        acts = ""
        activation_count = 0
        for str_tok, token_activation, normalized_activation in zip(
            str_toks, token_activations, normalized_activations
        ):
            if token_activation > max(token_activations) * self.threshold:
                # TODO: for each example, we only show the first 10 activations
                # decide on the best way to do this
                if activation_count > 10:
                    break
                acts += f'("{str_tok}" : {int(normalized_activation)}), '
                activation_count += 1

        return "Activations: " + acts

    @abstractmethod
    def _build_prompt(self, record: LatentRecord) -> list[dict]:
        pass


async def explanation_loader(
    record: LatentRecord, explanation_dir: str
) -> ExplainerResult:
    try:
        async with aiofiles.open(f"{explanation_dir}/{record.latent}.txt", "r") as f:
            explanation = json.loads(await f.read())
        return ExplainerResult(record=record, explanation=explanation)
    except FileNotFoundError:
        logger.info(f"No explanation found for {record.latent}")
        return ExplainerResult(record=record, explanation="No explanation found")


async def random_explanation_loader(
    record: LatentRecord, explanation_dir: str
) -> ExplainerResult:
    explanations = [f for f in os.listdir(explanation_dir) if f.endswith(".txt")]
    if str(record.latent) in explanations:
        explanations.remove(str(record.latent))
    random_explanation = random.choice(explanations)
    async with aiofiles.open(f"{explanation_dir}/{random_explanation}", "r") as f:
        explanation = json.loads(await f.read())

    return ExplainerResult(record=record, explanation=explanation)
