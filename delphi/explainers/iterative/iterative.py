import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, TypeVar

import torch

from delphi.explainers.explainer import ExplainerResult
from delphi.latents import (
    ActivatingExample,
    Example,
    LatentRecord,
    NonActivatingExample,
)
from delphi.scorers.classifier.classifier import Classifier
from delphi.scorers.classifier.sample import ClassifierOutput
from delphi.scorers.scorer import ScorerResult

from ..default.prompt_builder import build_prompt as default_prompt
from ..explainer import Explainer
from .prompt_builder import iterative_build_prompt

# we use this type variable to ensure that the examples are either
# ActivatingExample or NonActivatingExample
Examples = TypeVar("Examples", bound=Example)


@dataclass
class IterativeExplainer(Explainer):
    activations: bool = True
    """Whether to show activations to the explainer."""

    # Caps for extra examples when refining
    iterative_max_num_false_positives: int = 20
    """Maximum number of extra false positives to include in refinement prompts."""

    iterative_max_num_false_negatives: int = 20
    """Maximum number of extra false negatives to include in refinement prompts."""

    append_round_to_prompt: bool = False
    """If True, append the round number to the prompt to encourage diversity."""

    # Optional extra examples beyond FPs/FNs
    iterative_max_num_true_positives: int = 0
    """Maximum number of extra true positives to include in refinement prompts."""

    iterative_max_num_true_negatives: int = 0
    """Maximum number of extra true negatives to include in refinement prompts."""

    # Optional history/score controls
    show_score_to_explainer: bool = False
    """If True, include per-round score(s) alongside history when present."""

    history_only: bool = False
    """If True, only show previous explanations (and scores if enabled), no examples."""

    allow_tp_examples: bool = True
    """If False, suppress TP examples even when available."""

    def _to_string_examples(
        self, examples: list[Examples], show_activations: bool
    ) -> str:
        highlighted_examples = []

        for i, example in enumerate(examples):
            str_toks = example.str_tokens
            activations = example.activations.tolist()
            highlighted_examples.append(
                "Example " + str(i) + ": \n" + self._highlight(str_toks, activations)
            )

            if show_activations:
                assert example.normalized_activations is not None, (
                    "Normalized activations are required for activations in explainer"
                )
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

    def _build_prompt(
        self, record: LatentRecord
    ) -> list[dict]:  # iterative, not final, single result
        # History-only mode: only show previous explanations (and their scores)
        if getattr(self, "history_only", False):
            history_lines: list[str] = []
            prev_explanations = getattr(record, "previous_explanations", []) or []
            prev_scores = getattr(record, "previous_test_f1_scores", []) or []
            for i, exp in enumerate(prev_explanations):
                score_str = ""
                if self.show_score_to_explainer and i < len(prev_scores):
                    score_str = f" (F1={prev_scores[i]:.3f})"
                history_lines.append(f"Round {i}: [EXPLANATION]: {exp}{score_str}")
            history_text = (
                "\n".join(history_lines) if history_lines else "(no prior explanations)"
            )
            if getattr(self, "verbose", False):
                print(
                    "[IterativeExplainer] history_only=True; building prompt with only prior explanations"
                )
            return [
                {
                    "role": "system",
                    "content": "You are refining a concise, faithful explanation for a latent feature.",
                },
                {
                    "role": "user",
                    "content": f"Prior explanations and scores (if any):\n{history_text}\nReturn ONLY a new [EXPLANATION]: ...",
                },
            ]

        examples = record.train
        # Treat missing or unparsable explanations as empty to force an initial prompt
        explanation_text = record.explanation
        is_unparsed = "could not be parsed" in explanation_text.lower()

        if is_unparsed:
            print(f"[IterativeExplainer] Unparsed explanation: {explanation_text}")
        if explanation_text == "" or is_unparsed:
            # If there is no explanation, we use the default prompt
            highlighted_examples = self._to_string_examples(examples, self.activations)
            if getattr(self, "verbose", False):
                print(
                    f"[IterativeExplainer] Building initial prompt \
                    with {len(examples)} examples"
                )
            messages = default_prompt(highlighted_examples, self.activations)
        else:
            # If there is explanation, use the explanation, the normal examples
            # and the extra examples
            normal_examples = self._to_string_examples(examples, self.activations)

            extra_examples_list = (record.extra_examples or [])[
                : self.iterative_max_num_false_positives
                + self.iterative_max_num_false_negatives
            ]
            false_positives, false_negatives = self._get_false_positives_and_negatives(
                extra_examples_list
            )
            # we show at most iterative_max_num_false_positives and
            # iterative_max_num_false_negatives extra examples of each type

            number_extra_false_positives = min(
                self.iterative_max_num_false_positives, len(false_positives)
            )
            number_extra_false_negatives = min(
                self.iterative_max_num_false_negatives, len(false_negatives)
            )

            false_positives_examples = self._to_string_examples(
                false_positives[:number_extra_false_positives], False
            )
            false_negatives_examples = self._to_string_examples(
                false_negatives[:number_extra_false_negatives], self.activations
            )

            # Optionally include TP/TN examples attached to the record
            tp_examples_list = (getattr(record, "tp_examples_for_prompt", []) or [])[
                : self.iterative_max_num_true_positives
            ]
            tn_examples_list = (getattr(record, "tn_examples_for_prompt", []) or [])[
                : self.iterative_max_num_true_negatives
            ]
            if not self.allow_tp_examples:
                tp_examples_list = []
            tp_count = min(self.iterative_max_num_true_positives, len(tp_examples_list))
            tn_count = min(self.iterative_max_num_true_negatives, len(tn_examples_list))
            true_positive_examples = (
                self._to_string_examples(tp_examples_list[:tp_count], self.activations)
                if tp_count > 0
                else ""
            )
            true_negative_examples = (
                self._to_string_examples(tn_examples_list[:tn_count], False)
                if tn_count > 0
                else ""
            )

            if True:  # getattr(self, "verbose", False):
                print(
                    f"[IterativeExplainer] Refining explanation; showing \
                        FP={number_extra_false_positives}, \
                            FN={number_extra_false_negatives}"
                )

            # Optionally prepend prior history and scores
            history_prefix = ""
            if getattr(self, "show_score_to_explainer", False):
                prev_explanations = getattr(record, "previous_explanations", []) or []
                prev_scores = getattr(record, "previous_test_f1_scores", []) or []
                history_lines: list[str] = []
                for i, exp in enumerate(prev_explanations):
                    score_str = (
                        f" (F1={prev_scores[i]:.3f})" if i < len(prev_scores) else ""
                    )
                    history_lines.append(f"Round {i}: [EXPLANATION]: {exp}{score_str}")
                if history_lines:
                    history_prefix = (
                        "Prior rounds:\n" + "\n".join(history_lines) + "\n\n"
                    )

            # Merge optional TP/TN blocks after the standard FP/FN sections
            augmented_normals = history_prefix + normal_examples
            if true_positive_examples:
                augmented_normals += "\n\nTrue Positives:\n" + true_positive_examples
            if true_negative_examples:
                augmented_normals += "\n\nTrue Negatives:\n" + true_negative_examples

            messages = iterative_build_prompt(
                record.explanation,
                augmented_normals,
                false_positives_examples,
                false_negatives_examples,
            )

        # Optionally append the round number to the prompt as a diversity tag
        round_idx = getattr(record, "explanation_id", None)
        if self.append_round_to_prompt and round_idx is not None:
            try:
                # find the last user message to append the tag
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        messages[i]["content"] = (
                            messages[i].get("content", "") + f"\nRound: {round_idx}"
                        )
                        if getattr(self, "verbose", False):
                            print(
                                f"[IterativeExplainer] Appended round tag to prompt: Round {round_idx}"
                            )
                        break
            except Exception:
                pass
        if getattr(self, "verbose", False):
            print(f"[IterativeExplainer] Built prompt with {len(messages)} messages")
        return messages


@dataclass
class HillClimbing:
    scorers_with_paths: list[tuple[Classifier, Path]]
    """Scorers to use for explanation generation."""

    scorer_postprocess: Callable
    """Function to postprocess the scorer results."""

    explainer: IterativeExplainer
    """Explainer to use for explanation generation."""

    iterative_num_rounds: int = 3
    """Number of loops to run the explanation generation."""

    # Ratios used to split available examples
    iterative_holdout_ratio_of_total: float = 0.1
    """Ratio of total available examples to hold out for final evaluation."""

    iterative_test_ratio_of_nonholdout: float = 0.1
    """Ratio of total available examples to use as per-round test set."""

    judge_scorer_index: int = 0
    """Index of the scorer to use for selecting the best explanation."""

    select_strategy: Literal["best", "last"] = "last"
    """Strategy for selecting the final explanation after all rounds. 
    'best' selects the explanation with the highest score; 'last' selects the 
    explanation from the final round."""

    always_new_train_examples: bool = False
    """If True, resample train/test subsets each round from the pools."""

    def _compute_f1_score(self, results: list[ClassifierOutput]) -> float:
        tp = 0
        fp = 0
        fn = 0
        for i, sample in enumerate(results):
            if sample.correct:
                tp += 1
            else:
                fp += 1
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _get_wrong_examples(self, results: list[ScorerResult]) -> list[Example]:
        wrong_examples = []
        for result in results:
            for i, sample in enumerate(result.score):
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
                    if not any(
                        ex.str_tokens == new_example.str_tokens for ex in wrong_examples
                    ):
                        wrong_examples.append(new_example)
        return wrong_examples

    def _split_train_test_holdout(
        self, record: LatentRecord
    ) -> tuple[
        list[Example],
        list[Example],
        list[Example],
        list[Example],
        list[Example],
        list[Example],
    ]:
        """Split pre-sampled pools into train/test and holdout.
        Policy: prefer sampler-driven sizes if available on record, else ratio.
        Expects that upstream sampling produced sufficient pools in record.*
        """
        all_train_examples = record.train
        all_activating_test_examples = record.test
        all_non_activating_test_examples = record.not_active

        # Shuffle pools for randomness
        random.shuffle(all_activating_test_examples)
        random.shuffle(all_non_activating_test_examples)

        # Determine holdout size from configured ratio
        held_out_set_size = max(
            0,
            int(
                len(all_activating_test_examples)
                * self.iterative_holdout_ratio_of_total
            ),
        )
        holdout_activating_examples = all_activating_test_examples[:held_out_set_size]
        holdout_non_activating_examples = all_non_activating_test_examples[
            :held_out_set_size
        ]

        train_test_activating_examples = all_activating_test_examples[
            held_out_set_size:
        ]
        train_test_non_activating_examples = all_non_activating_test_examples[
            held_out_set_size:
        ]

        # Determine per-round test size from configured ratio
        train_size = len(train_test_activating_examples) * (
            1 - self.iterative_test_ratio_of_nonholdout
        )
        train_size = int(max(20, train_size))
        train_activating_examples = train_test_activating_examples[train_size:]
        train_non_activating_examples = train_test_non_activating_examples[train_size:]
        test_activating_examples = train_test_activating_examples[:train_size]
        test_non_activating_examples = train_test_non_activating_examples[:train_size]
        return (
            train_activating_examples,
            train_non_activating_examples,
            test_activating_examples,
            test_non_activating_examples,
            holdout_activating_examples,
            holdout_non_activating_examples,
        )

    async def _run_scorers(
        self,
        train_test_record: LatentRecord,
        round_idx: int,
        holdout_activating_examples: list[Example],
        holdout_non_activating_examples: list[Example],
        test_activating_examples: list[Example],
        test_non_activating_examples: list[Example],
    ) -> tuple[list[ScorerResult], list[ScorerResult]]:
        test_scorer_results = []
        holdout_scorer_results = []
        for scorer_idx, scorer_with_path in enumerate(self.scorers_with_paths):
            scorer, score_dir = scorer_with_path
            test_scorer_results.append(
                await scorer(train_test_record)
            )  # TODO: see which this uses to score?  need to make sure it uses test, not holdout

            # temporarily use holdout
            train_test_record.test = holdout_activating_examples
            train_test_record.not_active = holdout_non_activating_examples
            holdout_scorer_results.append(await scorer(train_test_record))
            # use holdout results for writing:
            self.scorer_postprocess(
                holdout_scorer_results[-1], score_dir=score_dir, round_idx=round_idx
            )
            # revert
            train_test_record.test = test_activating_examples
            train_test_record.not_active = test_non_activating_examples
        return test_scorer_results, holdout_scorer_results

    async def _run_round(
        self,
        round_idx: int,
        record: LatentRecord,
        train_activating_examples: list[Example],
        train_non_activating_examples: list[Example],
        test_activating_examples: list[Example],
        test_non_activating_examples: list[Example],
        holdout_activating_examples: list[Example],
        holdout_non_activating_examples: list[Example],
        extra_examples: Optional[list[Example]] = None,
    ) -> tuple[
        float, list[Example], ExplainerResult, list[ScorerResult], list[ScorerResult]
    ]:
        train_test_record = LatentRecord(
            latent=record.latent,
            train=train_activating_examples,
            not_active=test_non_activating_examples,
            test=test_activating_examples,
            explanation=record.explanation,
            extra_examples=extra_examples,
        )
        # Carry forward optional history and TP/TN example lists
        try:
            setattr(
                train_test_record,
                "previous_explanations",
                getattr(record, "previous_explanations", []),
            )
            setattr(
                train_test_record,
                "previous_test_f1_scores",
                getattr(record, "previous_test_f1_scores", []),
            )
            if hasattr(record, "tp_examples_for_prompt"):
                setattr(
                    train_test_record,
                    "tp_examples_for_prompt",
                    getattr(record, "tp_examples_for_prompt"),
                )
            if hasattr(record, "tn_examples_for_prompt"):
                setattr(
                    train_test_record,
                    "tn_examples_for_prompt",
                    getattr(record, "tn_examples_for_prompt"),
                )
        except Exception:
            pass
        # Attach round index BEFORE prompting so it can be used to build the prompt
        try:
            setattr(train_test_record, "explanation_id", round_idx)
            if getattr(self, "verbose", False) and self.append_round_to_prompt:
                print(
                    f"[IterativeExplainer] Set explanation_id (round) prior to prompting: {round_idx}"
                )
        except Exception:
            pass
        start_time = time.time()
        explanation = await self.explainer(train_test_record)
        train_test_record.explanation = explanation.explanation
        # Retry once if the explanation could not be parsed; then continue gracefully
        exp_text = (explanation.explanation or "").strip()
        if "could not be parsed" in exp_text.lower():
            try:
                if getattr(self, "verbose", False):
                    print(
                        f"[IterativeExplainer] Unparsed explanation in round {round_idx}; "
                        f"retrying once"
                    )
                # Force an initial-style prompt on retry
                train_test_record.explanation = ""
                retry_explanation = await self.explainer(train_test_record)
                retry_text = (retry_explanation.explanation or "").strip()
                if "could not be parsed" not in retry_text.lower():
                    explanation = retry_explanation
                    train_test_record.explanation = explanation.explanation
                else:
                    # Keep explanation empty going forward to avoid refining on invalid text
                    train_test_record.explanation = ""
            except Exception:
                # On any retry error, proceed with empty explanation
                train_test_record.explanation = ""
        end_time = time.time()
        print(
            f"Latent: {train_test_record.latent}; Round: {round_idx}; Time taken for "
            f"explanation: {end_time - start_time} seconds"
            f"Explanation: {train_test_record.explanation}"
        )

        start_time = time.time()
        # Attach round index for downstream logging/postprocessing to distinguish files
        try:
            setattr(train_test_record, "explanation_id", round_idx)
        except Exception:
            pass

        test_scorer_results, holdout_scorer_results = await self._run_scorers(
            train_test_record,
            round_idx,
            holdout_activating_examples,
            holdout_non_activating_examples,
            test_activating_examples,
            test_non_activating_examples,
        )

        end_time = time.time()

        # print("----- Holdout score ------")
        judge_holdout_f1_score = self._compute_f1_score(
            holdout_scorer_results[self.judge_scorer_index].score
        )
        print(
            f"Latent: {train_test_record.latent}; Round: {round_idx}; Time\
                taken for score: {end_time - start_time} seconds"
            f"Judge holdout score: {judge_holdout_f1_score}"
        )
        wrong_examples = self._get_wrong_examples(test_scorer_results)
        # Attach TP/TN for optional prompting
        try:
            tp_examples: list[Example] = []
            tn_examples: list[Example] = []
            for sr in test_scorer_results:
                for sample in sr.score:
                    if sample.correct:
                        if sample.activating:
                            tp_examples.append(
                                ActivatingExample(
                                    tokens=torch.tensor(0),
                                    activations=torch.tensor(sample.activations),
                                    str_tokens=sample.str_tokens,
                                    normalized_activations=torch.tensor(
                                        sample.activations
                                    ),
                                )
                            )
                        else:
                            tn_examples.append(
                                NonActivatingExample(
                                    tokens=torch.tensor(0),
                                    activations=torch.tensor(sample.activations),
                                    str_tokens=sample.str_tokens,
                                )
                            )
            # Deduplicate by str_tokens
            dedup = {}
            for ex in tp_examples + tn_examples:
                dedup[tuple(ex.str_tokens)] = ex
            train_test_record.tp_examples_for_prompt = [
                ex for ex in dedup.values() if isinstance(ex, ActivatingExample)
            ]
            train_test_record.tn_examples_for_prompt = [
                ex for ex in dedup.values() if isinstance(ex, NonActivatingExample)
            ]
        except Exception:
            pass
        return (
            judge_holdout_f1_score,
            wrong_examples,
            explanation,
            holdout_scorer_results,
            test_scorer_results,
        )

    def _select_best_explanation(
        self, scorer_results: list[ScorerResult]
    ) -> ExplainerResult:
        f1_scores: list[float] = []
        for score_result in scorer_results:
            f1_score: float = self._compute_f1_score(score_result.score)
            f1_scores.append(f1_score)
        best_pair = max(zip(f1_scores, scorer_results), key=lambda x: x[0])
        _, best_result = best_pair

        # call scorer_postprocess to save the best score - best is written to special dictory; all other
        # scores are written by the normal postprocess fns
        _, score_dir = self.scorers_with_paths[self.judge_scorer_index]
        self.scorer_postprocess(
            best_result, score_dir=score_dir, is_final=True, round_idx=None
        )

        return ExplainerResult(
            record=best_result.record,
            explanation=best_result.record.explanation,
        )

    async def __call__(
        self, record: LatentRecord
    ) -> tuple[list[ExplainerResult], ExplainerResult]:
        (
            train_activating_examples,
            train_non_activating_examples,
            test_activating_examples,
            test_non_activating_examples,
            holdout_activating_examples,
            holdout_non_activating_examples,
        ) = self._split_train_test_holdout(record)

        wrong_examples = None
        all_holdout_scorer_results: list[list[ScorerResult]] = []
        explanations: list[ExplainerResult] = []
        final_explanation = None  # best or last according to select_strategy
        previous_test_f1_scores: list[float] = []
        all_test_scorer_results: list[list[ScorerResult]] = []
        for lv in range(self.iterative_num_rounds):
            # Control whether to reuse same train/test subsets or resample each round
            if self.always_new_train_examples and lv > 0:
                (
                    train_activating_examples,
                    train_non_activating_examples,
                    test_activating_examples,
                    test_non_activating_examples,
                    holdout_activating_examples,
                    holdout_non_activating_examples,
                ) = self._split_train_test_holdout(record)
            round_results = await self._run_round(
                round_idx=lv,
                record=record,
                train_activating_examples=train_activating_examples,
                train_non_activating_examples=train_non_activating_examples,
                test_activating_examples=test_activating_examples,
                test_non_activating_examples=test_non_activating_examples,
                holdout_activating_examples=holdout_activating_examples,
                holdout_non_activating_examples=holdout_non_activating_examples,
                extra_examples=wrong_examples if wrong_examples is not None else None,
            )

            # if round_results is None:
            #    print(f"Round {lv} failed to generate an explanation")
            #    #continue
            # else:
            (
                holdout_f1_score,
                wrong_examples,
                explanation,
                round_holdout_scorer_results,
                round_test_scorer_results,
            ) = round_results

            explanations.append(explanation)
            all_holdout_scorer_results.append(round_holdout_scorer_results)
            all_test_scorer_results.append(round_test_scorer_results)
            previous_test_f1_scores.append(
                self._compute_f1_score(
                    round_test_scorer_results[self.judge_scorer_index].score
                )
            )
            # Carry forward the latest explanation to inform the next round
            # Strategy: carry forward last or best-so-far (by test F1)
            carry_strategy = getattr(self, "select_strategy", "last")
            if carry_strategy == "best" and len(explanations) > 0:
                scores = [
                    self._compute_f1_score(r[self.judge_scorer_index].score)
                    for r in all_test_scorer_results
                ]
                best_idx = max(range(len(scores)), key=lambda i: scores[i])
                latest_text = explanations[best_idx].explanation or ""
            else:
                latest_text = explanation.explanation or ""
            if "could not be parsed" in latest_text.lower():
                latest_text = ""
            record.explanation = latest_text
            # Save history for optional prompting next round
            record.previous_explanations = [e.explanation for e in explanations]
            record.previous_test_f1_scores = previous_test_f1_scores

        if self.select_strategy == "best":
            # Compute best round according to judge scorer's f1
            judge_holdout_results: list[ScorerResult] = [
                round_results[self.judge_scorer_index]
                for round_results in all_holdout_scorer_results
                if len(round_results) > self.judge_scorer_index
            ]
            final_explanation = self._select_best_explanation(judge_holdout_results)
        else:
            final_explanation = explanations[-1]
            final_score: ScorerResult = all_holdout_scorer_results[-1][
                self.judge_scorer_index
            ]
            _, score_dir = self.scorers_with_paths[self.judge_scorer_index]
            self.scorer_postprocess(
                final_score, score_dir=score_dir, is_final=True, round_idx=None
            )
        return explanations, final_explanation
