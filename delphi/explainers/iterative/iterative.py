import random
from dataclasses import dataclass
from typing import TypeVar

import torch
import time

from delphi.latents import (
    ActivatingExample,
    Example,
    LatentRecord,
    NonActivatingExample,
)
from delphi.scorers.classifier.classifier import Classifier, ScorerResult
from delphi.scorers.classifier.sample import ClassifierOutput

from ..default.prompt_builder import build_prompt as default_prompt
from ..explainer import Explainer
from .prompt_builder import build_prompt

# we use this type variable to ensure that the examples are either
# ActivatingExample or NonActivatingExample
Examples = TypeVar("Examples", bound=Example)


@dataclass
class IterativeExplainer(Explainer):
    activations: bool = True
    """Whether to show activations to the explainer."""

    def _to_string_examples(
        self, examples: list[Examples], show_activations: bool
    ) -> str:
        highlighted_examples = []

        for i, example in enumerate(examples):
            str_toks = example.str_tokens
            activations = example.activations.tolist()
            highlighted_examples.append("Example " + str(i) + ": \n" + self._highlight(str_toks, activations))

            if show_activations:
                assert (
                    example.normalized_activations is not None
                ), "Normalized activations are required for activations in explainer"
                normalized_activations = example.normalized_activations.tolist()
                highlighted_examples.append(
                    self._join_activations(
                        str_toks, activations, normalized_activations
                    )
                )

        highlighted_examples = "\n".join(highlighted_examples)

        return highlighted_examples

    def _get_false_positives_and_negatives(
        self, examples: list[Example]
    ) -> tuple[list[Example], list[Example]]:
        false_positives = []
        false_negatives = []
        for example in examples:
            if example.activations.max() > 0:
                false_negatives.append(example)
            else:
                false_positives.append(example)
        return false_positives, false_negatives

    def _build_prompt(self, record: LatentRecord) -> list[dict]:
        examples = record.train
        if record.explanation == "":
            # If there is no explanation, we use the default prompt
            highlighted_examples = self._to_string_examples(examples, self.activations)
            return default_prompt(highlighted_examples, self.activations)
        else:
            # If there is explanation, use the explanation, the normal examples
            # and the extra examples
            normal_examples = self._to_string_examples(examples, self.activations)

            false_positives, false_negatives = self._get_false_positives_and_negatives(
                record.extra_examples
            )
            # we show at most 20 extra examples of each type

            number_extra_false_positives = min(20, len(false_positives))
            number_extra_false_negatives = min(20, len(false_negatives))

            false_positives_examples = self._to_string_examples(
                false_positives[:number_extra_false_positives], False
            )
            false_negatives_examples = self._to_string_examples(
                false_negatives[:number_extra_false_negatives], False
            )

            return build_prompt(
                record.explanation,
                normal_examples,
                false_positives_examples,
                false_negatives_examples,
            )


@dataclass
class HillClimbing:
    scorers: list[Classifier]
    """Scorers to use for explanation generation."""

    explainer: IterativeExplainer
    """Explainer to use for explanation generation."""

    n_loops: int = 1
    """Number of loops to run the explanation generation."""

    def _compute_score(self, results: list[list[ClassifierOutput]]) -> float:
        f1_scores = []
        for result in results:
            # Score should be f1 score
            tp = 0
            fp = 0
            fn = 0
            for i, sample in enumerate(result):
                if sample.correct:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)
        print(f"F1 scores: {f1_scores}")

        return sum(f1_scores) / len(f1_scores)

    def _get_wrong_examples(
        self, results: list[list[ClassifierOutput]]
    ) -> list[Example]:
        wrong_examples = []
        for result in results:
            for i, sample in enumerate(result):
                if not sample.correct:
                    # Create a extra example
                    if sample.activating:
                        new_example = ActivatingExample(
                            tokens=torch.tensor(0),  # this does not matter
                            activations=torch.tensor(sample.activations),
                            str_tokens=sample.str_tokens,
                            normalized_activations=torch.tensor(sample.activations),
                        )
                    else:
                        new_example = NonActivatingExample(
                            tokens=torch.tensor(0),  # this does not matter
                            activations=torch.tensor(sample.activations),
                            str_tokens=sample.str_tokens,
                        )
                    # Check if there's no existing example with same str_tokens
                    if not any(ex.str_tokens == new_example.str_tokens for ex in wrong_examples):
                        wrong_examples.append(new_example)
        return wrong_examples

    async def __call__(self, record: LatentRecord) -> list[ScorerResult] | None:

        train_examples = record.train
        activating_examples = record.test
        non_activating_examples = record.not_active

        # shuffle the examples
        random.shuffle(activating_examples)
        random.shuffle(non_activating_examples)

        first_generation_examples = train_examples
        held_out_set_size = max(50, len(record.test) // 3)
        held_out_activating_examples = activating_examples[:held_out_set_size]
        held_out_non_activating_examples = non_activating_examples[:held_out_set_size]
        train_test_activating_examples = activating_examples[held_out_set_size:]
        train_test_non_activating_examples = non_activating_examples[held_out_set_size:]

        start_time = time.time()
        first_explanation = await self.explainer(record)
        record.explanation = first_explanation.explanation
        end_time = time.time()
        print(f"Time taken for first explanation: {end_time - start_time} seconds")

        print("----- First explanation ------")
        print(first_explanation.explanation)
        if first_explanation.explanation == "Explanation could not be parsed.":
            # TODO: Do I want this?
            return None

        test_record = LatentRecord(
            latent=record.latent,
            train=first_generation_examples,
            not_active=held_out_non_activating_examples,
            test=held_out_activating_examples,
            explanation=first_explanation.explanation,
        )
        results = []
        scores = []
        start_time = time.time()
        for scorer in self.scorers:
            result = await scorer(test_record)
            results.append(result)
            scores.append(result.score)
        end_time = time.time()
        print(f"Time taken for holdout score: {end_time - start_time} seconds")
        #print("----- Holdout score ------")
        holdout_score = self._compute_score(scores)

       
        for i in range(self.n_loops):
            #print(f"----- Loop {i} ------")

            random.shuffle(train_test_non_activating_examples)
            random.shuffle(train_test_activating_examples)

            new_record = LatentRecord(
                latent=record.latent,
                train=first_generation_examples,
                not_active=train_test_non_activating_examples[:15],
                test=train_test_activating_examples[:15],
                explanation=first_explanation.explanation,
            )
            scores = []
            start_time = time.time()
            for scorer in self.scorers:
                result = await scorer(new_record)
                scores.append(result.score)
            end_time = time.time()
            print(f"Time taken for train score: {end_time - start_time} seconds")
            #print("----- Train score ------")
            _ = self._compute_score(scores)
            # get the wrong examples
            wrong_examples = self._get_wrong_examples(scores)
            # update the record
            record.extra_examples.extend(wrong_examples)
            # update the explanation
            start_time = time.time()
            new_explanation = await self.explainer(record)
            end_time = time.time()
            print(f"Time taken for new explanation: {end_time - start_time} seconds")
            if new_explanation.explanation == "Explanation could not be parsed.":
                print("Error generating explanation")
                pass  # we do not update the explanation

            #print("----- New explanation ------")
            #print(new_explanation.explanation)
            # compute the score in the held out set
            test_record.explanation = new_explanation.explanation
            scores = []
            start_time = time.time()
            for scorer in self.scorers:
                result = await scorer(test_record)
                scores.append(result.score)
                results.append(result)
            end_time = time.time()
            print(f"Time taken for holdout score: {end_time - start_time} seconds")
            #print("----- Holdout score ------")
            final_score = self._compute_score(scores)
            record.explanation = new_explanation.explanation
            first_explanation = new_explanation
            #if new_holdout_score > holdout_score:
            #    holdout_score = new_holdout_score
            #    record.explanation = new_explanation.explanation
            #    first_explanation = new_explanation
        print("Initial score: ", holdout_score)
        print("Last explanation: ", record.explanation)
        print("Final score: ", final_score)
        return results
