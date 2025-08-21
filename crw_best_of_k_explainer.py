import re
from typing import AsyncIterable, Callable, List

import orjson

from crw_cls_overwrites import IntegratedExplainerScorer
from delphi import logger
from delphi.explainers.default.default import DefaultExplainer
from delphi.explainers.default.prompts import SYSTEM_BEST_OF_K_ONESHOT
from delphi.explainers.explainer import ExplainerResult
from delphi.latents.latents import ActivatingExample, LatentRecord, NonActivatingExample


class BestOfKExplainerScorer(IntegratedExplainerScorer):
    """No changes needed vs parent; kept for symmetry/extension."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def explainer_postprocess(
        self,
        result: list[ExplainerResult],
    ) -> list[ExplainerResult]:
        assert isinstance(result, list)

        path = self.integrated_explainer_scorer_cfg.explainer_cfg.explanations_path
        for expl_result in result:
            file_path = path / f"{expl_result.record.latent}.txt"
            with open(file_path, "ab") as f:
                f.write(orjson.dumps(expl_result.explanation))
        return result

    async def __call__(self, source: AsyncIterable | Callable) -> None:
        """Run explainer → scorers as a single pipeline.
        If one shot, it returns a list which must be unpacked
        """

        explainer_pl = self._explainer_pipeline(source)
        # Run it to get results
        explainer_results = await explainer_pl.run()

        # Flatten results - handle both single ExplainerResult
        #  and lists of ExplainerResult
        flattened_results = []
        for result in explainer_results:
            if isinstance(result, list):
                # If result is a list of ExplainerResult, extend flattened_results
                flattened_results.extend(result)
            else:
                # If result is a single ExplainerResult, append it
                flattened_results.append(result)

        # Create async iterable from flattened results
        async def results_iter():
            for result in flattened_results:
                yield result

        # Create the scorer pipeline with the flattened results as source
        scorer_pl = self._scorer_pipeline(results_iter)
        return await scorer_pl.run()


class BestOfKExplainer(DefaultExplainer):
    num_explanations: int = 3
    is_one_shot: bool = True  # if False: run K independent requests/contexts

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_explanations = self.generation_kwargs.get("num_explanations", 3)
        self.is_one_shot = self.generation_kwargs.get("is_one_shot", True)

    async def __call__(self, record: LatentRecord) -> list[ExplainerResult]:
        """Return K candidate explanations (one-shot or K independent calls)."""
        if self.is_one_shot:
            # Single request that asks for K explanations and parses them
            messages = self._build_prompt(record.train)
            resp = await self.client.generate(
                messages, temperature=self.temperature, **self.generation_kwargs
            )
            text = getattr(resp, "text", str(resp))
            explanations = self.parse_explanations(text)[: self.num_explanations] or [
                "No explanation parsed."
            ]
            return [ExplainerResult(record=record, explanation=e) for e in explanations]

        # Non-one-shot: K *independent* contexts (requests), default prompt per call
        results: list[ExplainerResult] = []
        old_k = self.num_explanations
        try:
            # Temporarily ask for exactly one explanation per request
            self.num_explanations = 1
            for _ in range(old_k):
                # Reuse your builder but effectively request 1 explanation
                messages = self._build_prompt(record.train)
                resp = await self.client.generate(
                    messages, temperature=self.temperature, **self.generation_kwargs
                )
                text = getattr(resp, "text", str(resp))
                parsed = self.parse_explanations(text)
                explanation = (
                    parsed[0] if parsed else text.strip()
                ) or "No explanation parsed."
                results.append(ExplainerResult(record=record, explanation=explanation))
        finally:
            self.num_explanations = old_k

        return results

    def _build_prompt(  # type: ignore; mostly copied from DefaultExplainer
        self, examples: list[ActivatingExample | NonActivatingExample]
    ) -> list[dict]:
        """Build a prompt labeling activating vs non-activating examples."""
        parts: list[str] = []

        activating = [ex for ex in examples if isinstance(ex, ActivatingExample)]
        non_activating = [
            ex for ex in examples if not isinstance(ex, ActivatingExample)
        ]

        if activating:
            parts.append("ACTIVATING EXAMPLES:")
            for i, ex in enumerate(activating, 1):
                toks = ex.str_tokens
                acts = ex.activations.tolist()
                s = self._highlight(toks, acts).strip().replace("\n", "")
                parts.append(f"Example {i}: {s}")
                if self.activations and ex.normalized_activations is not None:
                    parts.append(
                        self._join_activations(
                            toks, acts, ex.normalized_activations.tolist()
                        )
                    )

        if non_activating:
            parts.append("\nNON-ACTIVATING EXAMPLES:")
            for i, ex in enumerate(non_activating, 1):
                toks = ex.str_tokens
                acts = ex.activations.tolist()
                s = self._highlight(toks, acts).strip().replace("\n", "")
                parts.append(f"Example {i}: {s}")

        highlighted = "\n".join(parts)
        system_prompt = (
            SYSTEM_BEST_OF_K_ONESHOT
            + f"\nThe number of explanations you are asked to \
                generate is: {self.num_explanations}."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": highlighted},
        ]

    def parse_explanations(self, text: str) -> List[str]:
        """
        Extract lines like: `[EXPLANATION]: ...` (case- and whitespace-insensitive).
        Returns however many are found; caller decides how many to keep.
        """
        try:
            pattern = re.compile(
                r"^\s*\[\s*EXPLANATION\s*\]\s*:\s*(.+?)\s*$",
                re.IGNORECASE | re.MULTILINE,
            )
            matches = [m.strip() for m in pattern.findall(text)]
            if not matches and self.verbose:
                logger.debug("BestOfK: no [EXPLANATION]: lines found.")
            # Do not truncate here; caller may choose policy.
            return matches
        except Exception as e:
            logger.warning("BestOfK: regex parse error: %r", e)
            return []
