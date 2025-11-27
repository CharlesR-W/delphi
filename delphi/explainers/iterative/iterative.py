import random
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
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
from delphi.scorers.embedding.embedding import EmbeddingOutput
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

    iterative_append_round_to_prompt: bool = False
    """If True, append the round number to the prompt to encourage diversity."""

    # Optional extra examples beyond FPs/FNs
    iterative_max_num_true_positives: int = 0
    """Maximum number of extra true positives to include in refinement prompts."""

    iterative_max_num_true_negatives: int = 0
    """Maximum number of extra true negatives to include in refinement prompts."""

    # Optional history/score controls
    iterative_show_score_to_explainer: bool = False
    """If True, include per-round score(s) alongside history when present."""

    iterative_history_only: bool = False
    """If True, only show previous explanations (and scores if enabled), no examples."""

    iterative_allow_tp_examples: bool = True
    """If False, suppress TP examples even when available."""
    
    iterative_num_train_examples_per_round: int = 20
    """Number of train examples to randomly sample and show each round."""

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
        self, wrong_examples: list[Example]
    ) -> tuple[list[Example], list[Example]]:
        false_positives = []
        false_negatives = []
        for example in wrong_examples:
            if example.activations.max() > 0:
                false_negatives.append(example)
            else:
                false_positives.append(example)
        return false_positives, false_negatives

    def _build_prompt(
        self, record: LatentRecord
    ) -> list[dict]:  # iterative, not final, single result
        # Debug: Print current configuration values that affect prompt building
        print(
            f"[IterativeExplainer] Config: activations={self.activations}, "
            f"history_only={getattr(self, 'iterative_history_only', False)}, "
            f"show_score_to_explainer={getattr(self, 'iterative_show_score_to_explainer', False)}, "
            f"append_round_to_prompt={getattr(self, 'iterative_append_round_to_prompt', False)}, "
            f"allow_tp_examples={getattr(self, 'iterative_allow_tp_examples', True)}"
        )

        # History-only mode: only show previous explanations (and their scores)
        if getattr(self, "iterative_history_only", False) and record.explanation != "":
            history_lines: list[str] = []
            prev_explanations = getattr(record, "previous_explanations", []) or []
            prev_scores = getattr(record, "previous_test_f1_scores", []) or []
            for i, exp in enumerate(prev_explanations):
                score_str = ""
                if self.iterative_show_score_to_explainer and i < len(prev_scores):
                    score_str = f" (F1 score={prev_scores[i]:.3f})"
                history_lines.append(f"Round {i}: [EXPLANATION]: {exp}{score_str}")
            history_text = (
                "\n".join(history_lines) if history_lines else "(no prior explanations)"
            )
            print(
                f"[IterativeExplainer] History-only mode: showing {len(history_lines)} prior explanations"
            )
            if getattr(self, "verbose", False):
                print(
                    "[IterativeExplainer] iterative_history_only=True; building prompt with only prior explanations"
                )
            return [
                {
                    "role": "system",
                    "content": 'Below are a set of proposed explanations of a particular hidden pattern (you are not shown examples of this pattern).  Based on the content of these explanations and their scores, please propose a new explanation which you think will score even better, based on the relative commonalities and differences amongst those shown.  Reason concisely.  The final line of your answer MUST be the string "[EXPLANATION]: " followed by your proposed explanation.',
                },
                {
                    "role": "user",
                    "content": f"Prior explanations and scores (if any):\n{history_text}\n",
                },
            ]

        examples = record.train
        # Treat missing or unparsable explanations as empty to force an initial prompt

        if (
            record.explanation is None
            or "could not be parsed" in record.explanation.lower()
        ):
            # If there is no explanation, we use the default prompt
            highlighted_examples = self._to_string_examples(examples, self.activations)
            print(
                f"[IterativeExplainer] Building initial prompt with {len(examples)} examples, "
                f"activations={self.activations}"
            )
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
            print(
                f"[IterativeExplainer] Refining existing explanation: '{record.explanation[:100]}{'...' if len(record.explanation) > 100 else ''}'"
            )

            extra_examples_list = (record.extra_examples or [])
            false_positives, false_negatives = self._get_false_positives_and_negatives(
                extra_examples_list
            )
            # After classifying all, cap what we show for each type
            false_positives = false_positives[: self.iterative_max_num_false_positives]
            false_negatives = false_negatives[: self.iterative_max_num_false_negatives]

            number_extra_false_positives = min(
                self.iterative_max_num_false_positives, len(false_positives)
            )
            number_extra_false_negatives = min(
                self.iterative_max_num_false_negatives, len(false_negatives)
            )

            print(
                f"[IterativeExplainer] Extra examples: FP={len(false_positives)} (showing {number_extra_false_positives}), "
                f"FN={len(false_negatives)} (showing {number_extra_false_negatives})"
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
            if not self.iterative_allow_tp_examples:
                tp_examples_list = []
            tp_count = min(self.iterative_max_num_true_positives, len(tp_examples_list))
            tn_count = min(self.iterative_max_num_true_negatives, len(tn_examples_list))

            print(
                f"[IterativeExplainer] TP/TN examples: TP={len(tp_examples_list)} (showing {tp_count}), "
                f"TN={len(tn_examples_list)} (showing {tn_count}), allow_tp_examples={self.iterative_allow_tp_examples}"
            )

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

            print(
                f"[IterativeExplainer] Refining explanation; showing extra FP={number_extra_false_positives}, FN={number_extra_false_negatives}, TP={tp_count}, TN={tn_count}"
            )

            # Optionally prepend prior history and scores
            history_prefix = ""
            if getattr(self, "iterative_show_score_to_explainer", False):
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
                    print(
                        f"[IterativeExplainer] Added history prefix with {len(history_lines)} prior rounds"
                    )

            # Merge optional TP/TN blocks after the standard FP/FN sections
            augmented_normals = history_prefix + normal_examples
            if true_positive_examples:
                augmented_normals += "\n\nTrue Positives:\n" + true_positive_examples
            if true_negative_examples:
                augmented_normals += "\n\nTrue Negatives:\n" + true_negative_examples

            print(
                f"[IterativeExplainer] Building refinement prompt with history_prefix={len(history_prefix) > 0}, "
                f"TP={len(true_positive_examples) > 0}, TN={len(true_negative_examples) > 0}"
            )

            messages = iterative_build_prompt(
                record.explanation,
                augmented_normals,
                false_positives_examples,
                false_negatives_examples,
                self.activations,
            )

        # Optionally append the round number to the prompt as a diversity tag
        round_idx = getattr(record, "explanation_id", None)
        if self.iterative_append_round_to_prompt and round_idx is not None:
            try:
                # find the last user message to append the tag
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        original_content = messages[i].get("content", "")
                        messages[i]["content"] = (
                            original_content + f"\nRound: {round_idx}"
                        )
                        print(
                            f"[IterativeExplainer] Appended round tag {round_idx} to prompt (length: {len(original_content)} -> {len(messages[i]['content'])})"
                        )
                        if getattr(self, "verbose", False):
                            print(
                                f"[IterativeExplainer] Appended round tag to prompt: Round {round_idx}"
                            )
                        break
            except Exception as e:
                print(f"[IterativeExplainer] Failed to append round tag: {e}")
                pass

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

    explainer_postprocess: Callable
    """Function to postprocess the explainer results."""

    iterative_num_rounds: int = 3
    """Number of loops to run the explanation generation."""

    judge_scorer_index: int = 0
    """Index of the scorer to use for selecting the best explanation."""

    iterative_carryforward_strategy: Literal["best", "last"] = "last"
    """Strategy for selecting the final explanation after all rounds. 
    'best' selects the explanation with the highest score; 'last' selects the 
    explanation from the final round."""

    iterative_always_new_train_examples: bool = False
    """If True, resample train/test subsets each round from the pools."""

    def _compute_f1_score(self, results: list[ClassifierOutput]) -> float:
        tp = 0
        fp = 0
        fn = 0
        for sample in results:
            if sample.correct:
                if sample.activating:
                    tp += 1
                else:
                    # True negative, not used in F1
                    pass
            else:
                if sample.activating:
                    fn += 1
                else:
                    fp += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _compute_embedding_score(self, results: list[EmbeddingOutput]) -> float:
        if not results:
            return float("-inf")
        pos = [sample.similarity for sample in results if sample.activating]
        neg = [sample.similarity for sample in results if not sample.activating]
        if not pos or not neg:
            return float("-inf")
        return fmean(pos) - fmean(neg)

    def _score_judge_result(self, results: list) -> float:
        if results and isinstance(results[0], EmbeddingOutput):
            return self._compute_embedding_score(results)
        return self._compute_f1_score(results)

    def _get_wrong_examples(self, results: list[ScorerResult]) -> list[Example]:
        wrong_examples = []
        for result in results:
            if not result.score:
                continue
            first_sample = result.score[0]
            if not hasattr(first_sample, "correct"):
                continue
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
    ]:
        """Use record.train for training+test and record.test for holdout.
        
        Simplified design: No re-splitting, no ratios.
        - Train pool: record.train (activating examples to sample FROM for prompting)
        - Test: record.train (activating for FP/FN collection) + record.not_active (non-activating)
        - Holdout: record.test (activating for final eval) + record.not_active (non-activating)
        
        Returns:
          (train_activating, test_activating, test_non_activating,
           holdout_activating, holdout_non_activating)
        """
        train_activating_examples = list(record.train)
        
        test_activating_examples = list(record.train)
        test_non_activating_examples = list(record.not_active)
        
        holdout_activating_examples = list(record.test)
        holdout_non_activating_examples = list(record.not_active)

        print(f"[Iterative] Using: train+test={len(train_activating_examples)} activating + {len(test_non_activating_examples)} non-activating (FP/FN collection), "
              f"holdout={len(holdout_activating_examples)} activating + {len(holdout_non_activating_examples)} non-activating (final eval)")

        return (
            train_activating_examples,
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
            test_scorer_results.append(await scorer(train_test_record))

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
        test_activating_examples: list[Example],
        test_non_activating_examples: list[Example],
        holdout_activating_examples: list[Example],
        holdout_non_activating_examples: list[Example],
        extra_examples: Optional[list[Example]] = None,
    ) -> tuple[
        float, list[Example], ExplainerResult, list[ScorerResult], list[ScorerResult]
    ]:
        # Sample subset from train pool for this round
        # If always_new_train is False, use seed based on latent to get deterministic samples
        # If always_new_train is True, resample randomly each round
        num_to_show = self.explainer.iterative_num_train_examples_per_round
        if num_to_show < len(train_activating_examples):
            if not self.iterative_always_new_train_examples:
                # Use deterministic seed so same examples each round for this latent
                rng_state = random.getstate()
                random.seed(hash(str(record.latent)))
                sampled_train = random.sample(train_activating_examples, num_to_show)
                random.setstate(rng_state)
                print(f"[Iterative Round {round_idx}] Using deterministic {num_to_show} from train pool")
            else:
                sampled_train = random.sample(train_activating_examples, num_to_show)
                print(f"[Iterative Round {round_idx}] Resampled {num_to_show} from train pool (always_new=True)")
        else:
            sampled_train = train_activating_examples
            print(f"[Iterative Round {round_idx}] Using all {len(train_activating_examples)} train examples")
        
        train_test_record = LatentRecord(
            latent=record.latent,
            train=sampled_train,
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
        print(
            f"[IterativeExplainer] Explanation (direct from explainer() call): {explanation.explanation}"
        )
        self.explainer_postprocess(explanation, is_final=False)
        train_test_record.explanation = explanation.explanation
        
        # Retry once if the explanation could not be parsed; then continue gracefully
        exp_text = (explanation.explanation).strip()
        if "could not be parsed" in exp_text.lower():
            try:
                print(
                    f"[IterativeExplainer] Unparsed explanation in round {round_idx}; "
                    f"retrying once"
                )
                # Force an initial-style prompt on retry
                train_test_record.explanation = ""
                retry_explanation = await self.explainer(train_test_record)
                self.explainer_postprocess(retry_explanation, is_final=False)
                retry_text = (retry_explanation.explanation).strip()
                if "could not be parsed" not in retry_text.lower():
                    explanation = retry_explanation
                    train_test_record.explanation = explanation.explanation
                else:
                    # Keep explanation empty going forward to avoid refining on invalid text
                    train_test_record.explanation = "[explanation could not be parsed]"
            except Exception as e:
                # On any retry error, proceed with empty explanation
                print(f"[IterativeExplainer] Error on retry: {e}")
                train_test_record.explanation = "[explanation could not be parsed]"
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
        judge_holdout_f1_score = self._score_judge_result(
            holdout_scorer_results[self.judge_scorer_index].score
        )
        print(
            f"Latent: {train_test_record.latent}; Round: {round_idx}; Time\
                taken for score: {end_time - start_time} seconds\n"
            f"Judge holdout score: {judge_holdout_f1_score}\n"
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
        self,
        all_holdout_scorer_results: list[list[ScorerResult]],
        explanations: list[ExplainerResult],
    ) -> ExplainerResult:
        """Select best explanation based on judge scorer and write final scores for all scorers.

        Args:
            all_holdout_scorer_results: List of scorer results for each round (one list per round)
            explanations: List of explanations for each round

        Returns:
            ExplainerResult for the best explanation
        """
        # Extract judge scorer results from each round
        judge_holdout_results: list[ScorerResult] = [
            round_results[self.judge_scorer_index]
            for round_results in all_holdout_scorer_results
            if len(round_results) > self.judge_scorer_index
        ]

        # Find best round based on judge scorer F1
        judge_scores: list[float] = []
        for score_result in judge_holdout_results:
            score_value: float = self._score_judge_result(score_result.score)
            judge_scores.append(score_value)

        best_round_idx = max(range(len(judge_scores)), key=lambda i: judge_scores[i])

        # Write final scores for ALL scorers from the best round
        for scorer_idx, (_, score_dir) in enumerate(self.scorers_with_paths):
            final_score = all_holdout_scorer_results[best_round_idx][scorer_idx]
            self.scorer_postprocess(
                final_score, score_dir=score_dir, is_final=True, round_idx=None
            )

        return explanations[best_round_idx]

    async def __call__(
        self, record: LatentRecord
    ) -> tuple[ExplainerResult, list[ExplainerResult]]:
        # Debug: Print HillClimbing configuration
        print(
            f"[HillClimbing] Config: rounds={self.iterative_num_rounds}, "
            f"strategy={getattr(self, 'iterative_carryforward_strategy', 'last')}, "
            f"judge_scorer_idx={getattr(self, 'judge_scorer_index', 0)}, "
            f"always_new_train={getattr(self, 'iterative_always_new_train_examples', False)}"
        )

        (
            train_activating_examples,
            test_activating_examples,
            test_non_activating_examples,
            holdout_activating_examples,
            holdout_non_activating_examples,
        ) = self._split_train_test_holdout(record)

        print(
            f"[HillClimbing] Data split: train={len(train_activating_examples)}, "
            f"test=({len(test_activating_examples)}, {len(test_non_activating_examples)}), "
            f"holdout=({len(holdout_activating_examples)}, {len(holdout_non_activating_examples)})"
        )

        wrong_examples = None
        all_holdout_scorer_results: list[list[ScorerResult]] = []
        explanations: list[ExplainerResult] = []
        final_explanation = None  # best or last according to select_strategy
        previous_test_f1_scores: list[float] = []
        all_test_scorer_results: list[list[ScorerResult]] = []

        # Initialize TP/TN example lists if they don't exist
        if not hasattr(record, "tp_examples_for_prompt"):
            record.tp_examples_for_prompt = []
        if not hasattr(record, "tn_examples_for_prompt"):
            record.tn_examples_for_prompt = []

        for lv in range(self.iterative_num_rounds):
            print(f"[HillClimbing] Starting round {lv}/{self.iterative_num_rounds}")

            # NOTE: iterative_always_new_train_examples controls train pool resampling per-round
            # If False: same subset from train pool each round (deterministic)
            # If True: resample different subset from train pool each round (more exploration)
            # Train/test/holdout pools themselves stay fixed - only the sampling changes

            round_results = await self._run_round(
                round_idx=lv,
                record=record,
                train_activating_examples=train_activating_examples,
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
                newest_explanation,
                round_holdout_scorer_results,
                round_test_scorer_results,
            ) = round_results

            explanations.append(newest_explanation)
            all_holdout_scorer_results.append(round_holdout_scorer_results)
            all_test_scorer_results.append(round_test_scorer_results)
            previous_test_f1_scores.append(
                self._score_judge_result(
                    round_test_scorer_results[self.judge_scorer_index].score
                )
            )
            # Carry forward the latest explanation to inform the next round
            # Strategy: carry forward last or best-so-far (by test F1)
            carry_strategy = getattr(self, "iterative_carryforward_strategy", "last")
            if carry_strategy == "best" and len(explanations) > 0:
                scores = [
                    self._score_judge_result(r[self.judge_scorer_index].score)
                    for r in all_test_scorer_results
                ]
                best_idx = max(range(len(scores)), key=lambda i: scores[i])
                latest_text = explanations[best_idx].explanation
            else:
                latest_text = newest_explanation.explanation
            if "could not be parsed" in latest_text.lower():
                latest_text = "[explanation could not be parsed]"
            record.explanation = newest_explanation.explanation
            # Save history for optional prompting next round
            record.previous_explanations = [e.explanation for e in explanations]
            record.previous_test_f1_scores = previous_test_f1_scores

            # Carry forward TP/TN examples from the latest round for use in next rounds
            # Only use examples from the immediate previous round, don't accumulate
            try:
                if hasattr(newest_explanation, "record") and hasattr(
                    newest_explanation.record, "tp_examples_for_prompt"
                ):
                    record.tp_examples_for_prompt = getattr(
                        newest_explanation.record, "tp_examples_for_prompt", []
                    )
                if hasattr(newest_explanation, "record") and hasattr(
                    newest_explanation.record, "tn_examples_for_prompt"
                ):
                    record.tn_examples_for_prompt = getattr(
                        newest_explanation.record, "tn_examples_for_prompt", []
                    )
            except Exception:
                pass

        if self.iterative_carryforward_strategy == "best":
            print(
                f"[HillClimbing] Selecting best explanation from {len(all_holdout_scorer_results)} rounds"
            )
            final_explanation = self._select_best_explanation(
                all_holdout_scorer_results, explanations
            )
            print(
                f"[HillClimbing] Selected best explanation with strategy='{self.iterative_carryforward_strategy}'"
            )
        else:
            print(
                f"[HillClimbing] Using last explanation with strategy='{self.iterative_carryforward_strategy}'"
            )
            final_explanation = newest_explanation
            # Write final scores for ALL scorers, not just the judge
            for scorer_idx, (_, score_dir) in enumerate(self.scorers_with_paths):
                final_score: ScorerResult = all_holdout_scorer_results[-1][scorer_idx]
                self.scorer_postprocess(
                    final_score, score_dir=score_dir, is_final=True, round_idx=None
                )

        print(
            f"[HillClimbing] Final explanation: '{final_explanation.explanation[:100]}{'...' if len(final_explanation.explanation) > 100 else ''}'"
        )

        # Call explainer_postprocess on the final explanation to write it to the main directory
        self.explainer_postprocess(final_explanation, is_final=True)

        return final_explanation, explanations
