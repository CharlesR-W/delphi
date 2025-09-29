import asyncio
import re
from functools import partial
from pathlib import Path
from typing import Literal

from delphi import logger
from delphi.clients.client import Response
from delphi.explainers.default.prompts import SYSTEM, SYSTEM_BESTOFK
from delphi.explainers.explainer import Explainer, ExplainerResult
from delphi.latents.latents import LatentRecord
from delphi.pipeline import Pipe, Pipeline, process_wrapper
from delphi.scorers.classifier.classifier import ClassifierOutput
from delphi.scorers.scorer import Scorer, ScorerResult


class BestOfKExplainer(Explainer):
    def __init__(
        self,
        client,
        bestofk_num_explanations=5,
        scorers_with_paths: list[tuple[Scorer, Path]] = [],
        judge_scorer_index: int = 0,
    ):
        super().__init__(client)
        self.bestofk_num_explanations: int = bestofk_num_explanations
        """The number of explanations to generate."""
        self.client = client
        """The client to use for explanation generation."""
        self.is_multishot: Literal["oneshot", "multishot"] = "multishot"
        """Whether to generate multiple explanations from a single prompt (oneshot) or multiple prompts (multishot)."""
        self.scorers_with_paths = scorers_with_paths
        """The scorers to use for judging the explanations."""
        self.judge_scorer_index: int = judge_scorer_index
        """scorers[judge_scorer_index] is used to determine which is the 'best' explanation."""
        self.return_only_best: bool = True
        """if True, only the best explanation is returned, else a list of ExplainerResults is returned."""
        self.run_all_scorers: bool = True
        """if True, all scorers are run, else only the judge scorer is run."""

    async def __call__(
        self, record: LatentRecord
    ) -> ExplainerResult | tuple[list[ExplainerResult], ExplainerResult]:
        messages = self._build_prompt(record)
        if self.is_multishot:
            tasks = []
            for _ in range(self.bestofk_num_explanations):
                tasks.append(
                    self.client.generate(
                        messages, temperature=self.temperature, **self.generation_kwargs
                    )
                )
            responses = await asyncio.gather(*tasks)
        else:
            response = await self.client.generate(
                messages, temperature=self.temperature, **self.generation_kwargs
            )
        assert isinstance(response, Response) | list[Response]
        try:
            if self.is_multishot:
                explanations = self.parse_multiple_explanations(response.text)
            else:
                explanations = [
                    self.parse_single_explanation(response.text)
                    for response in responses
                ]

            for lv, explanation in enumerate(explanations):
                explanations.append(
                    ExplainerResult(
                        record=record, explanation=explanation, explanation_id=lv
                    )
                )
        except Exception as e:
            logger.error(f"[bestofk.py:__call__] Explanation parsing failed: {repr(e)}")

        """if self.verbose:
            logger.info(f"[bestofk.py:__call__] Explanations: {explanations}")
            logger.info(f"[bestofk.py:__call__] Messages: {messages[-1]['content']}")
            logger.info(f"[bestofk.py:__call__] Response: {response}")"""

        # runs scorer pipeline, writes results to files according to postprocess
        scorer_results_list: list[list[ScorerResult]] = await self.run_scorers(
            explanations
        )

        judge_scorer_results = scorer_results_list[self.judge_scorer_index]

        # note, this return is only used to invoke explainer_postprocess; scoring has already occurred.
        best_explanation = self._select_best_explanation(judge_scorer_results)
        if not self.return_only_best:
            return best_explanation, explanations
        else:
            return best_explanation

    async def run_scorers(
        self, explanations: list[ExplainerResult]
    ) -> list[list[ScorerResult]]:
        # if run_all_scorers is False, still returns a list of ScorerResults, just with one element.
        wrapped_scorers = []
        if self.run_all_scorers:
            for scorer_with_path in self.scorers_with_paths:
                scorer, score_dir = scorer_with_path
                wrapped_scorer = process_wrapper(
                    scorer,
                    preprocess=self.scorer_preprocess,
                    postprocess=partial(self.scorer_postprocess, score_dir=score_dir),
                )
                wrapped_scorers.append(wrapped_scorer)
        else:
            scorer, score_dir = self.scorers_with_paths[self.judge_scorer_index]
            wrapped_scorer = process_wrapper(
                scorer,
                preprocess=self.scorer_preprocess,
                postprocess=partial(self.scorer_postprocess, score_dir=score_dir),
            )
            wrapped_scorers.append(wrapped_scorer)

        pipeline = Pipeline(
            explanations,
            Pipe(*wrapped_scorers),
        )
        return await pipeline.run()  # TODO: check which list is inner vs outer; one is over scorers, other over explanations

    def _compute_f1_score(self, results: list[list[ClassifierOutput]]) -> float:
        f1_scores: list[float] = []
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

    def _select_best_explanation(
        self, scorer_results: list[ScorerResult]
    ) -> ExplainerResult:
        f1_scores: list[float] = []
        for score_result in scorer_results:
            f1_score: float = self._compute_f1_score(score_result.score)
            f1_scores.append(f1_score)
        best_result = max(zip(f1_scores, scorer_results), key=lambda x: x[0])

        # call scorer_postprocess to save the best score - best is written to special dictory; all other
        # scores are written by the normal postprocess fns
        _, score_dir = self.scorers_with_paths[self.judge_scorer_index]
        self.scorer_postprocess(best_result, score_dir=score_dir, best=True)

        best_explanation = best_result.record.explanation  # TODO: make sure that each explanation gets attached to a separate record, not overwriting
        # score is list[ClassifierOutput]

        return best_explanation

    def _build_prompt(self, record: LatentRecord) -> list[dict[str, str]]:
        if not self.is_multishot:
            prompt: list[dict[str, str]] = self.__super()._build_prompt(
                record, system_prompt=SYSTEM_BESTOFK
            )
            prompt.append(
                {
                    "role": "user",
                    "content": f"The number of explanations to generate is: {self.bestofk_num_explanations}.",
                }
            )
        else:
            prompt: list[dict[str, str]] = self.__super()._build_prompt(
                record, system_prompt=SYSTEM
            )
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
        try:
            explanations = re.findall(r"\[EXPLANATION\]:\s*(.*)", text, re.DOTALL)
            if explanations:
                return explanations
            else:
                return "Explanations could not be parsed."
        except Exception as e:
            logger.error(
                f"[bestofk.py:parse_multiple_explanations] Explanation parsing regex failed: {repr(e)}"
            )
            raise
