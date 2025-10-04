import asyncio
import re
from functools import partial
from pathlib import Path
from typing import Callable

from delphi import logger
from delphi.explainers.default.default import DefaultExplainer
from delphi.explainers.default.prompts import SYSTEM_BESTOFK
from delphi.explainers.explainer import ExplainerResult
from delphi.latents.latents import LatentRecord
from delphi.pipeline import Pipe, Pipeline, process_wrapper
from delphi.scorers.classifier.classifier import ClassifierOutput
from delphi.scorers.scorer import Scorer, ScorerResult


class BestOfKExplainer(DefaultExplainer):
    def __init__(
        self,
        client,
        bestofk_num_explanations=5,
        scorers_with_paths: list[tuple[Scorer, Path]] = [],
        judge_scorer_index: int = 0,
        scorer_postprocess: Callable | None = None,
        scorer_preprocess: Callable | None = None,
        temperature: float = 0.0,
        return_only_best: bool = True,
        run_all_scorers: bool = True,
        generation_kwargs: dict | None = None,
        is_multishot: bool = True,
    ):
        super().__init__(client)
        self.bestofk_num_explanations: int = bestofk_num_explanations
        """The number of explanations to generate."""
        self.client = client
        """The client to use for explanation generation."""
        self.is_multishot: bool = is_multishot
        """Whether to generate multiple explanations from a single prompt (oneshot) or multiple prompts (multishot)."""
        self.scorers_with_paths = scorers_with_paths
        """The scorers to use for judging the explanations."""
        self.judge_scorer_index: int = judge_scorer_index
        """scorers[judge_scorer_index] is used to determine which is the 'best' explanation."""
        self.scorer_postprocess = scorer_postprocess or (lambda result, **_: result)
        """Callback applied after scorer execution."""
        self.scorer_preprocess = scorer_preprocess
        """Callback applied before scorer execution."""
        self.temperature = temperature
        """Sampling temperature used when generating explanations."""
        self.return_only_best: bool = return_only_best
        """if True, only the best explanation is returned, else a list of ExplainerResults is returned."""
        self.run_all_scorers: bool = run_all_scorers
        """if True, all scorers are run, else only the judge scorer is run."""
        self.generation_kwargs = generation_kwargs or {}
        """Extra keyword arguments passed to the generation client."""

    async def __call__(
        self, record: LatentRecord
    ) -> ExplainerResult | tuple[ExplainerResult, list[ExplainerResult]]:
        print(f"[BestOfK] Starting explanation generation for latent {record.latent}")
        messages = self._build_prompt(record)
        if self.is_multishot:
            print(
                f"[BestOfK] Generating {self.bestofk_num_explanations} explanations (multishot)"
            )
            tasks = []
            for _ in range(self.bestofk_num_explanations):
                tasks.append(
                    self.client.generate(
                        messages, temperature=self.temperature, **self.generation_kwargs
                    )
                )
            responses = await asyncio.gather(*tasks)
            # For multishot, we need to combine all responses into a single text
            combined_text = "\n".join([response.text for response in responses])
            explanations = self.parse_multiple_explanations(combined_text)
        else:
            print(
                f"[BestOfK] Generating {self.bestofk_num_explanations} explanations (oneshot)"
            )
            response = await self.client.generate(
                messages, temperature=self.temperature, **self.generation_kwargs
            )
            # Parse multiple explanations from the single response
            explanations = self.parse_multiple_explanations(response.text)

        print(f"[BestOfK] Parsed {len(explanations)} explanations")
        if len(explanations) < self.bestofk_num_explanations:
            print(
                f"[BestOfK] WARNING: Got {len(explanations)} explanations but expected {self.bestofk_num_explanations}"
            )

        try:
            # Convert explanations to ExplainerResult objects
            explainer_results = []
            for lv, explanation in enumerate(explanations):
                print(f"[BestOfK] Processing explanation {lv}: {explanation[:100]}...")
                explainer_results.append(
                    ExplainerResult(
                        record=record, explanation=explanation, explanation_id=lv
                    )
                )
        except Exception as e:
            logger.error(f"[bestofk.py:__call__] Explanation parsing failed: {repr(e)}")
            # Create empty results if parsing fails
            explainer_results = []

        print(f"[BestOfK] Created {len(explainer_results)} ExplainerResult objects")

        """if self.verbose:
            logger.info(f"[bestofk.py:__call__] Explanations: {explanations}")
            logger.info(f"[bestofk.py:__call__] Messages: {messages[-1]['content']}")
            logger.info(f"[bestofk.py:__call__] Response: {response}")"""

        # runs scorer pipeline, writes results to files according to postprocess
        print(f"[BestOfK] Starting scorer pipeline for latent {record.latent}")
        scorer_results: list[list[ScorerResult]] = await self.run_scorers(
            explainer_results
        )
        print(f"[BestOfK] Completed scorer pipeline for latent {record.latent}")

        judge_scorer_results: list[ScorerResult] = scorer_results[
            self.judge_scorer_index
        ]

        best_explanation_idx = self._select_best_explanation_idx(
            judge_scorer_results, explainer_results
        )
        print(f"[BestOfK] Selected best explanation for latent {record.latent}")

        # save the best score for all scorers
        for scorer_idx, scorer_result in enumerate(scorer_results):
            _, score_dir = self.scorers_with_paths[scorer_idx]
            self.scorer_postprocess(scorer_result, score_dir=score_dir, is_final=True)

        best_explanation = explainer_results[best_explanation_idx]
        if not self.return_only_best:
            return best_explanation, explainer_results[best_explanation_idx]
        else:
            return best_explanation

    async def run_scorers(
        self, explanations: list[ExplainerResult]
    ) -> list[list[ScorerResult]]:
        print(f"[BestOfK] run_scorers: Processing {len(explanations)} explanations")
        # if run_all_scorers is False, still returns a list of ScorerResults, just with one element.
        wrapped_scorers = []
        if self.run_all_scorers:
            # print(
            # f"[BestOfK] run_scorers: Running all {len(self.scorers_with_paths)} scorers"
            # )
            for scorer_with_path in self.scorers_with_paths:
                scorer, score_dir = scorer_with_path
                wrapped_scorer = process_wrapper(
                    scorer,
                    preprocess=self.scorer_preprocess,
                    postprocess=partial(self.scorer_postprocess, score_dir=score_dir),
                )
                wrapped_scorers.append(wrapped_scorer)
        else:
            # print(
            # f"[BestOfK] run_scorers: Running only judge scorer (index {self.judge_scorer_index})"
            # )
            scorer, score_dir = self.scorers_with_paths[self.judge_scorer_index]
            wrapped_scorer = process_wrapper(
                scorer,
                preprocess=self.scorer_preprocess,
                postprocess=partial(self.scorer_postprocess, score_dir=score_dir),
            )
            wrapped_scorers.append(wrapped_scorer)

        # print(f"[BestOfK] run_scorers: Created {len(wrapped_scorers)} wrapped scorers")

        async def explanation_async_iter():
            # print(
            # f"[BestOfK] explanation_async_iter: Starting to yield {len(explanations)} explanations"
            # )
            for i, explanation in enumerate(explanations):
                # print(
                # f"[BestOfK] explanation_async_iter: Yielding explanation {i + 1}/{len(explanations)}"
                # )
                yield explanation
            # print(
            # "[BestOfK] explanation_async_iter: Finished yielding all explanations"
            # )
            # )

        # print("[BestOfK] run_scorers: Creating pipeline")
        pipeline = Pipeline(
            explanation_async_iter(),
            Pipe(*wrapped_scorers),
        )
        print("[BestOfK] run_scorers: Starting pipeline.run()")
        result: list[list[ScorerResult]] = await pipeline.run()
        print(
            f"[BestOfK] run_scorers: Pipeline completed, returning {len(result)} results"
        )
        print(f"[BestOfK] run_scorers: result len: {len(result)}")
        print(f"[BestOfK] run_scorers: result[0] len: {len(result[0])}")
        return result  # outer index is explanation, inner index is scorer

    def _compute_f1_score(self, results: list[ClassifierOutput]) -> float:
        # Score should be f1 score
        tp = 0
        fp = 0
        fn = 0
        for i, sample in enumerate(results):
            if sample.correct:
                tp += 1
            else:
                fp += 1
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        print(f"F1 score: {f1} (tp={tp}, fp={fp}, fn={fn})")

        return f1

    def _select_best_explanation_idx(
        self,
        scorer_results: list[ScorerResult],
        explainer_results: list[ExplainerResult],
    ) -> int:
        if not explainer_results:
            logger.error(
                "[BestOfK] _select_best_explanation called with no explainer_results"
            )
            # This should not happen as we check earlier, but handle it gracefully
            raise ValueError("No explainer results available to select from")

        if not scorer_results:
            logger.error(
                "[BestOfK] _select_best_explanation called with no scorer_results, "
                "returning first explainer result"
            )
            # If we have explanations but no scores, just return the first one
            return 0

        f1_scores: list[float] = []
        for score_result in scorer_results:
            f1_score: float = self._compute_f1_score(score_result.score)
            f1_scores.append(f1_score)
        best_idx = max(range(len(f1_scores)), key=lambda i: f1_scores[i])
        # best_scorer_result = scorer_results[best_idx]
        # best_explainer_result = explainer_results[best_idx]

        # Return the original ExplainerResult which has the explanation_id preserved
        return best_idx

    def _build_prompt(self, record: LatentRecord) -> list[dict[str, str]]:
        if not self.is_multishot:
            prompt: list[dict[str, str]] = super()._build_prompt(record)
            prompt[0]["content"] = SYSTEM_BESTOFK
            prompt.append(
                {
                    "role": "user",
                    "content": f"The number of explanations to generate is: {self.bestofk_num_explanations}.",
                }
            )
        else:
            prompt: list[dict[str, str]] = super()._build_prompt(record)
        return prompt

    def parse_single_explanation(self, text: str) -> str:
        try:
            match = re.search(r"\[EXPLANATION\]:\s*(.*)", text, re.DOTALL)
            if match:
                return match.group(1).strip()
            else:
                return "Explanation could not be parsed."
        except Exception as e:
            logger.error(
                f"[bestofk.py:parse_explanation] Explanation parsing regex failed: {repr(e)}"
            )
            raise

    def parse_multiple_explanations(self, text: str) -> list[str]:
        print(f"[BestOfK] DEBUG: Parsing explanations from text of length: {len(text)}")
        print(f"[BestOfK] DEBUG: First 500 chars of text: {text[:500]}")
        try:
            explanations = re.findall(
                r"\[EXPLANATION\]:\s*(.*?)(?=\[EXPLANATION\]:|$)", text, re.DOTALL
            )
            if explanations:
                # Clean up and return only non-empty explanations
                cleaned = [exp.strip() for exp in explanations if exp.strip()]
                return cleaned if cleaned else ["Explanation could not be parsed."]
            else:
                return ["Explanation could not be parsed."]
        except Exception as e:
            logger.error(
                f"[bestofk.py:parse_multiple_explanations] Explanation parsing regex failed: {repr(e)}"
            )
            return ["Explanation could not be parsed."]
