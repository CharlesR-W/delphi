import asyncio
import random
from dataclasses import dataclass
from typing import NamedTuple, Sequence

from delphi.latents.latents import ActivatingExample, NonActivatingExample

from ...latents import Example, LatentRecord
from ..scorer import Scorer, ScorerResult


@dataclass
class EmbeddingOutput:
    text: str
    """The text that was used to evaluate the similarity"""

    distance: float | int
    """Quantile or neighbor distance"""

    similarity: float = 0
    """What is the similarity of the example to the explanation"""

    activating: bool = False
    """Whether the example is activating or not"""


class Sample(NamedTuple):
    text: str
    activations: list[float]
    data: EmbeddingOutput


class EmbeddingScorer(Scorer):
    name = "embedding"

    def __init__(
        self,
        model,
        verbose: bool = False,
        **generation_kwargs,
    ):
        self.model = model
        self.verbose = verbose
        self.generation_kwargs = generation_kwargs

    async def __call__(
        self,
        record: LatentRecord,
    ) -> ScorerResult:
        import time
        start_time = time.time()
        samples = self._prepare(record)

        random.shuffle(samples)
        results = self._query(
            record.explanation,
            samples,
        )
        end_time = time.time()
        duration = end_time - start_time

        return ScorerResult(record=record, score=results, duration=duration)

    def call_sync(self, record: LatentRecord) -> ScorerResult:
        return asyncio.run(self.__call__(record))

    def _prepare(self, record: LatentRecord) -> list[Sample]:
        """
        Prepare and shuffle a list of samples for classification.
        Uses the standard activating/non-activating pools supplied on the record.
        """
        if not record.not_active:
            raise ValueError(
                "EmbeddingScorer requires non-activating examples (record.not_active)."
            )
        if not record.test:
            raise ValueError("EmbeddingScorer requires activating test examples.")

        samples = []
        samples.extend(examples_to_samples(record.not_active))
        samples.extend(examples_to_samples(record.test))
        return samples

    def _query(self, explanation: str, samples: list[Sample]) -> list[EmbeddingOutput]:
        explanation_string = (
            "Instruct: Retrieve sentences that could be related to the explanation."
            "\nQuery:"
        )
        explanation_prompt = explanation_string + explanation
        samples_text = [sample.text for sample in samples]

        query_embeding = self.model.encode(explanation_prompt)
        samples_text = [sample.text for sample in samples]

        sample_embedings = self.model.encode(samples_text)
        similarity = self.model.similarity(query_embeding, sample_embedings)[0]

        results = []
        for i in range(len(samples)):
            samples[i].data.similarity = similarity[i].item()
            results.append(samples[i].data)
        return results


def examples_to_samples(
    examples: Sequence[Example],
) -> list[Sample]:
    samples = []
    for example in examples:
        assert isinstance(example, ActivatingExample) or isinstance(
            example, NonActivatingExample
        )
        assert example.str_tokens is not None
        text = "".join(str(token) for token in example.str_tokens)
        activations = example.activations.tolist()
        samples.append(
            Sample(
                text=text,
                activations=activations,
                data=EmbeddingOutput(
                    text=text,
                    distance=(
                        example.quantile
                        if isinstance(example, ActivatingExample)
                        else example.distance
                    ),
                    activating=isinstance(example, ActivatingExample),
                ),
            )
        )

    return samples
