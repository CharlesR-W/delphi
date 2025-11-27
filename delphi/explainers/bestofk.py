import asyncio
import random
import re
from functools import partial
from pathlib import Path
from statistics import fmean
from typing import Callable

import numpy as np
import orjson

from delphi import logger
from delphi.explainers.default.default import DefaultExplainer
from delphi.explainers.default.prompts import SYSTEM_BESTOFK
from delphi.explainers.explainer import ExplainerResult
from delphi.latents.latents import LatentRecord
from delphi.pipeline import Pipe, Pipeline, process_wrapper
from delphi.scorers.classifier.classifier import ClassifierOutput
from delphi.scorers.embedding.embedding import EmbeddingOutput
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
        bestofk_num_train_examples: int | None = None,
        use_random_baseline: bool = False,
        random_baseline_source_run: str | None = None,
        embedding_prefilter_enabled: bool = False,
        embedding_prefilter_top_k: int = 10,
        embedding_is_judge: bool = False,
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
        self.bestofk_num_train_examples: int | None = bestofk_num_train_examples if bestofk_num_train_examples is not None else 20
        """Number of train examples to show. Default: 20. Can be increased to 40 if needed."""
        self.use_random_baseline: bool = use_random_baseline
        """If True, load explanation from a different latent instead of generating."""
        self.random_baseline_source_run: str | None = random_baseline_source_run
        """Name of the run to load random baseline explanations from (under results/)."""
        self.embedding_prefilter_enabled: bool = embedding_prefilter_enabled
        self.embedding_prefilter_top_k: int = max(1, embedding_prefilter_top_k)
        self.embedding_is_judge: bool = embedding_is_judge
        self.embedding_scorer_index: int | None = self._find_embedding_scorer_index()

        if self.embedding_prefilter_enabled and self.embedding_scorer_index is None:
            print(
                "[BestOfK] Embedding prefilter requested but no embedding scorer is configured; disabling prefilter."
            )
            self.embedding_prefilter_enabled = False

        if self.embedding_is_judge and self.embedding_scorer_index is not None:
            self.judge_scorer_index = self.embedding_scorer_index

    def _find_embedding_scorer_index(self) -> int | None:
        for idx, (scorer, _) in enumerate(self.scorers_with_paths):
            if getattr(scorer, "name", "") == "embedding":
                return idx
        return None

    async def __call__(
        self, record: LatentRecord
    ) -> ExplainerResult | tuple[ExplainerResult, list[ExplainerResult]]:
        print(f"[BestOfK] Starting explanation generation for latent {record.latent}")
        
        # Split record into clean train/test pools upfront
        # (unless using random baseline, which doesn't need this)
        if not (self.use_random_baseline and self.random_baseline_source_run):
            train_pool, test_activating, test_non_activating = self._split_train_test(record)
            # Create a record with the clean test split for scoring
            clean_record = LatentRecord(
                latent=record.latent,
                train=train_pool,  # Will be resampled in _build_prompt
                test=test_activating,
                not_active=test_non_activating,
                explanation=record.explanation,
            )
        else:
            clean_record = record
        
        # Random baseline mode: load explanation from a DIFFERENT latent
        if self.use_random_baseline and self.random_baseline_source_run:
            try:
                # Find the source run's explanations directory
                if self.scorers_with_paths:
                    current_run_root = self.scorers_with_paths[0][1].parent.parent
                    results_root = current_run_root.parent
                else:
                    results_root = Path.cwd() / "results"
                
                prior_dir = (results_root / self.random_baseline_source_run / "explanations").resolve()
                if not prior_dir.exists():
                    raise FileNotFoundError(f"Random baseline source not found: {prior_dir}")
                
                safe_latent = str(record.latent).replace("/", "--")
                all_explanation_files = list(prior_dir.glob("*.txt"))
                # Exclude current latent - want explanation from DIFFERENT latent
                candidates = [f for f in all_explanation_files if not f.name.startswith(f"{safe_latent}_") and f.name != f"{safe_latent}.txt"]
                
                if not candidates:
                    raise ValueError(f"No other latent explanations found for random baseline")
                
                # Pick random explanation from different latent
                pick = np.random.choice(candidates)
                explanation_text = orjson.loads(pick.read_bytes())
                print(f"[BestOfK Random Baseline] Using explanation from {pick.name} for latent {record.latent}")
                
                # Create single explainer result
                from dataclasses import replace
                record_copy = replace(record)
                explainer_results = [ExplainerResult(record=record_copy, explanation=explanation_text, explanation_id=0)]
                
            except Exception as e:
                logger.error(f"[BestOfK] Random baseline failed: {repr(e)}")
                from dataclasses import replace
                record_copy = replace(record)
                explainer_results = [ExplainerResult(record=record_copy, explanation="[random baseline failed]", explanation_id=0)]
        else:
            # Normal explanation generation (use clean_record with split pools)
            messages = self._build_prompt(clean_record)
            # Store prompts and responses for spot_check
            prompts_and_responses = []
            
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
                # Store each prompt/response pair
                for i, response in enumerate(responses):
                    prompts_and_responses.append({
                        "prompt": messages,
                        "response": response.text,
                        "index": i
                    })
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
                # Store single prompt/response
                prompts_and_responses.append({
                    "prompt": messages,
                    "response": response.text,
                    "index": 0
                })
                # Parse multiple explanations from the single response
                explanations = self.parse_multiple_explanations(response.text)

            print(f"[BestOfK] Parsed {len(explanations)} explanations")
            if len(explanations) < self.bestofk_num_explanations:
                print(
                    f"[BestOfK] WARNING: Got {len(explanations)} explanations but expected {self.bestofk_num_explanations}"
                )
            
            # Enforce hard cap: never process more than requested
            if len(explanations) > self.bestofk_num_explanations:
                print(
                    f"[BestOfK] LIMITING: Truncating {len(explanations)} explanations to {self.bestofk_num_explanations}"
                )
                explanations = explanations[:self.bestofk_num_explanations]

            try:
                # Convert explanations to ExplainerResult objects
                # IMPORTANT: Create a copy of clean_record for each explanation to avoid
                # race conditions when setting _explanation_id in scorer_preprocess
                # Use clean_record so scorers get the clean test split
                from dataclasses import replace

                explainer_results: list[ExplainerResult] = []
                for lv, explanation in enumerate(explanations):
                    print(f"[BestOfK] Processing explanation {lv}: {explanation[:100]}...")
                    # Create a shallow copy of the clean_record for this explanation
                    record_copy = replace(clean_record)
                    explainer_results.append(
                        ExplainerResult(
                            record=record_copy, explanation=explanation, explanation_id=lv
                        )
                    )
            except Exception as e:
                logger.error(f"[bestofk.py:__call__] Explanation parsing failed: {repr(e)}")
                # Create empty results if parsing fails
                explainer_results: list[ExplainerResult] = []

            print(f"[BestOfK] Created {len(explainer_results)} ExplainerResult objects")
            
            # Sanity check: ensure we never have more than requested
            assert len(explainer_results) <= self.bestofk_num_explanations, (
                f"BUG: Created {len(explainer_results)} explainer results but only expected "
                f"{self.bestofk_num_explanations}. This should never happen!"
            )

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

        judge_scorer_results: list[ScorerResult] = [
            s[self.judge_scorer_index] for s in scorer_results
        ]

        best_explanation_idx = self._select_best_explanation_idx(
            judge_scorer_results,
            explainer_results,
        )
        print(f"[BestOfK] Selected best explanation for latent {record.latent}")

        # save the best score for all scorers
        for scorer_idx, (scorer, score_dir) in enumerate(self.scorers_with_paths):
            best_score = scorer_results[best_explanation_idx][scorer_idx]
            if best_score is None:
                continue
            self.scorer_postprocess(
                best_score,
                score_dir=score_dir,
                is_final=True,
            )

        best_explanation = explainer_results[best_explanation_idx]
        if not self.return_only_best:
            return best_explanation, explainer_results
        else:
            return best_explanation

    async def run_scorers(
        self, explanations: list[ExplainerResult]
    ) -> list[list[ScorerResult]]:
        print(f"[BestOfK] run_scorers: Processing {len(explanations)} explanations")
        # if run_all_scorers is False, still returns a list of ScorerResults, just with one element.
        num_scorers = len(self.scorers_with_paths)
        all_results: list[list[ScorerResult | None]] = [
            [None] * num_scorers for _ in range(len(explanations))
        ]
        all_indices = list(range(len(explanations)))

        def make_wrapper(scorer_idx: int):
            scorer, score_dir = self.scorers_with_paths[scorer_idx]
            return process_wrapper(
                scorer,
                preprocess=self.scorer_preprocess,
                postprocess=partial(self.scorer_postprocess, score_dir=score_dir),
            )

        async def run_for_indices(
            target_indices: list[int], scorer_indices: list[int]
        ) -> None:
            if not target_indices or not scorer_indices:
                return

            wrappers = [make_wrapper(idx) for idx in scorer_indices]

            async def generator():
                for idx in target_indices:
                    yield explanations[idx]

            pipeline = Pipeline(
                generator(),
                Pipe(*wrappers),
            )
            subset_results: list[list[ScorerResult]] = await pipeline.run()
            for pos, scorer_list in enumerate(subset_results):
                exp_idx = target_indices[pos]
                for local_idx, scorer_result in enumerate(scorer_list):
                    global_idx = scorer_indices[local_idx]
                    all_results[exp_idx][global_idx] = scorer_result

        if self.embedding_prefilter_enabled and self.embedding_scorer_index is not None:
            print(
                "[BestOfK] Embedding prefilter enabled: running embedding scorer on all explanations"
            )
            await run_for_indices(all_indices, [self.embedding_scorer_index])

            embedding_scores = [
                self._score_judge_result(all_results[idx][self.embedding_scorer_index])
                if all_results[idx][self.embedding_scorer_index] is not None
                else float("-inf")
                for idx in all_indices
            ]
            top_k = min(self.embedding_prefilter_top_k, len(explanations))
            ranked = sorted(
                all_indices, key=lambda i: embedding_scores[i], reverse=True
            )
            filtered_indices = ranked[:top_k]
            other_scorer_indices = [
                idx for idx in range(num_scorers) if idx != self.embedding_scorer_index
            ]
            print(
                f"[BestOfK] Embedding prefilter selected top-{top_k} indices: {filtered_indices}"
            )
            await run_for_indices(filtered_indices, other_scorer_indices)
        else:
            if self.run_all_scorers:
                scorer_indices = list(range(num_scorers))
            else:
                scorer_indices = [self.judge_scorer_index]
            await run_for_indices(all_indices, scorer_indices)

        return all_results  # outer index is explanation, inner index aligns with scorers

    def _compute_f1_score(self, results: list[ClassifierOutput]) -> float:
        # Score should be f1 score
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
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        print(f"F1 score: {f1} (tp={tp}, fp={fp}, fn={fn})")

        return f1

    def _compute_embedding_score(self, results: list[EmbeddingOutput]) -> float:
        if not results:
            return float("-inf")
        pos = [sample.similarity for sample in results if sample.activating]
        neg = [sample.similarity for sample in results if not sample.activating]
        if not pos or not neg:
            return float("-inf")
        return fmean(pos) - fmean(neg)

    def _score_judge_result(self, scorer_result: ScorerResult) -> float:
        samples = scorer_result.score or []
        if samples and isinstance(samples[0], EmbeddingOutput):
            return self._compute_embedding_score(samples)
        return self._compute_f1_score(samples)

    def _select_best_explanation_idx(
        self,
        scorer_results: list[ScorerResult],
        explainer_results: list[ExplainerResult],
    ) -> int:
        judge_scores: list[float] = []
        for score_result in scorer_results:
            judge_score: float = self._score_judge_result(score_result)
            judge_scores.append(judge_score)
        best_idx = max(range(len(judge_scores)), key=lambda i: judge_scores[i])

        return best_idx

    def _split_train_test(self, record: LatentRecord) -> tuple[list, list, list]:
        """Use record.train for training and record.test for scoring.
        
        Simplified design: No re-splitting, no ratios.
        - Train: record.train (activating examples for prompting)
        - Test: record.test (activating examples for scoring) + record.not_active (non-activating)
        
        Returns:
          (train_activating, test_activating, test_non_activating)
        """
        train_examples = list(record.train)
        test_activating_examples = list(record.test)
        test_non_activating_examples = list(record.not_active)
        
        print(f"[BestOfK] Using: train={len(train_examples)} activating, "
              f"test={len(test_activating_examples)} activating + {len(test_non_activating_examples)} non-activating")
        
        return train_examples, test_activating_examples, test_non_activating_examples
    
    def _build_prompt(self, record: LatentRecord) -> list[dict[str, str]]:
        """Build prompt for explanation generation.
        
        Note: record should already have clean train/test split from __call__.
        This method optionally samples a subset from record.train.
        """
        train_pool = record.train
        
        # Sample subset from train pool if limit is set
        if self.bestofk_num_train_examples is not None and self.bestofk_num_train_examples < len(train_pool):
            sampled_train = random.sample(train_pool, self.bestofk_num_train_examples)
            print(f"[BestOfK] Sampled {self.bestofk_num_train_examples} from train pool of {len(train_pool)}")
        else:
            sampled_train = train_pool
            print(f"[BestOfK] Using all {len(train_pool)} train examples")
        
        # Create record with sampled train examples (test stays the same)
        sampled_record = LatentRecord(
            latent=record.latent,
            train=sampled_train,
            test=record.test,  # Already clean from __call__
            not_active=record.not_active,
            explanation=record.explanation,
        )
        
        if not self.is_multishot:
            prompt: list[dict[str, str]] = super()._build_prompt(sampled_record)
            prompt[0]["content"] = SYSTEM_BESTOFK
            prompt.append(
                {
                    "role": "user",
                    "content": f"The number of explanations to generate is: {self.bestofk_num_explanations}.",
                }
            )
        else:
            prompt: list[dict[str, str]] = super()._build_prompt(sampled_record)
        
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
                
                # IMPORTANT: Cap at the requested number of explanations
                if len(cleaned) > self.bestofk_num_explanations:
                    print(
                        f"[BestOfK] WARNING: Parsed {len(cleaned)} explanations but only expected "
                        f"{self.bestofk_num_explanations}. Keeping first {self.bestofk_num_explanations}."
                    )
                    cleaned = cleaned[:self.bestofk_num_explanations]
                
                return cleaned if cleaned else ["Explanation could not be parsed."]
            else:
                return ["Explanation could not be parsed."]
        except Exception as e:
            logger.error(
                f"[bestofk.py:parse_multiple_explanations] Explanation parsing regex failed: {repr(e)}"
            )
            return ["Explanation could not be parsed."]
