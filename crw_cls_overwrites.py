from functools import partial
from pathlib import Path
from typing import AsyncIterable, Callable, Literal, Optional

import orjson
from pydantic import BaseModel, PrivateAttr

from delphi.clients.offline import Offline
from delphi.config import CacheConfig, ConstructorConfig, SamplerConfig
from delphi.explainers.default.default import DefaultExplainer
from delphi.explainers.explainer import Explainer, ExplainerResult
from delphi.latents.latents import LatentRecord
from delphi.pipeline import Pipe, Pipeline, process_wrapper
from delphi.scorers.scorer import Scorer, ScorerResult


class OfflineClientConfig(BaseModel):
    model_name: str
    max_memory: float = 0.9
    max_model_len: int = 4096
    num_gpus: int = 2
    statistics: bool = True
    number_tokens_to_generate: int = 500
    batch_size: int = 100


class ExplainerConfig(BaseModel):
    explainer_cls: type = DefaultExplainer
    highlight_threshold: float = 0.3
    verbose: bool = True
    temperature: float = 0.0
    explanations_path: Path


class ScorerConfig(BaseModel):
    scorer_cls: type
    scorer_log_prob: bool = True
    scorer_n_examples_shown: int = 5
    scorer_temperature: float = 0.0
    verbose: bool = True
    scores_path: Path


class IntegratedExplainerScorerConfig(BaseModel):
    client_cfg: OfflineClientConfig
    explainer_cfg: ExplainerConfig
    scorer_cfgs: list[ScorerConfig]


def instantiate_scorer(scorer_cfg: ScorerConfig, client) -> Scorer:
    return scorer_cfg.scorer_cls(
        client,
        n_examples_shown=scorer_cfg.scorer_n_examples_shown,
        verbose=scorer_cfg.verbose,
        log_prob=scorer_cfg.scorer_log_prob,
        temperature=scorer_cfg.scorer_temperature,
    )


def instantiate_explainer(explainer_cfg: ExplainerConfig, client) -> Explainer:
    return explainer_cfg.explainer_cls(
        client,
        threshold=explainer_cfg.highlight_threshold,
        verbose=explainer_cfg.verbose,
        temperature=explainer_cfg.temperature,
    )


class IntegratedExplainerScorer(BaseModel):
    """One vLLM engine (via Delphi's Offline) shared by explainer + all scorers."""

    integrated_explainer_scorer_cfg: IntegratedExplainerScorerConfig

    # runtime-only attributes (not part of the pydantic schema)
    _client: Offline = PrivateAttr()
    _explainer: Explainer = PrivateAttr()
    _scorers: list[Scorer] = PrivateAttr()
    _explainer_pipeline: Callable = PrivateAttr()
    _scorer_pipeline: Callable = PrivateAttr()

    def __init__(
        self, integrated_explainer_scorer_cfg: IntegratedExplainerScorerConfig
    ) -> None:
        super().__init__(
            integrated_explainer_scorer_cfg=integrated_explainer_scorer_cfg
        )
        cfg = self.integrated_explainer_scorer_cfg

        # SINGLE engine: one Offline client reused everywhere
        self._client = Offline(
            cfg.client_cfg.model_name,
            max_memory=cfg.client_cfg.max_memory,
            max_model_len=cfg.client_cfg.max_model_len,
            num_gpus=cfg.client_cfg.num_gpus,
            statistics=cfg.client_cfg.statistics,
        )

        # Wire explainer & scorers to the same client
        self._explainer = instantiate_explainer(cfg.explainer_cfg, self._client)
        self._scorers = [
            instantiate_scorer(s_cfg, self._client) for s_cfg in cfg.scorer_cfgs
        ]

        self._explainer_pipeline = self._form_explainer_partial_pipeline()
        self._scorer_pipeline = self._form_scorer_partial_pipeline()

    def _form_explainer_partial_pipeline(
        self,
    ) -> Callable[[AsyncIterable | Callable], Pipeline]:
        def explainer_postprocess(result: ExplainerResult) -> ExplainerResult:
            with open(
                self.integrated_explainer_scorer_cfg.explainer_cfg.explanations_path
                / f"{result.record.latent}.txt",
                "wb",
            ) as f:
                f.write(orjson.dumps({"explanation": result.explanation}))
            return result

        def explainer_preprocess(result: ExplainerResult) -> ExplainerResult:
            return result

        wrapped = process_wrapper(
            self._explainer,
            preprocess=explainer_preprocess,
            postprocess=explainer_postprocess,
        )

        # Create a function that takes a source and creates
        #  a Pipeline with source as loader and wrapped as pipe
        def create_pipeline(source):
            return Pipeline(source, wrapped)

        return create_pipeline

    def _form_scorer_partial_pipeline(
        self,
    ) -> Callable[[AsyncIterable | Callable], Pipeline]:
        def scorer_preprocess(result: ExplainerResult) -> LatentRecord:
            record = result.record
            record.explanation = result.explanation
            return record

        def scorer_postprocess(result: ScorerResult, score_dir: Path) -> None:
            safe = str(result.record.latent).replace("/", "--")
            with open(score_dir / f"{safe}.txt", "wb") as f:
                f.write(orjson.dumps(result.score))

        wrapped = [
            process_wrapper(
                scorer,
                preprocess=scorer_preprocess,
                postprocess=partial(
                    scorer_postprocess,
                    score_dir=self.integrated_explainer_scorer_cfg.scorer_cfgs[
                        i
                    ].scores_path,
                ),
            )
            for i, scorer in enumerate(self._scorers)
        ]
        # Compose all scorer wrappers into a single Pipe, then create
        #  a function that creates a Pipeline
        scorer_pipe = Pipe(*wrapped)

        def create_pipeline(source):
            return Pipeline(source, scorer_pipe)

        return create_pipeline

    async def __call__(self, source: AsyncIterable | Callable) -> None:
        """Run explainer → scorers as a single pipeline."""
        explainer_pl = self._explainer_pipeline(source)
        # Run it to get results
        explainer_results = await explainer_pl.run()

        # Create a proper async iterable from the results
        class ResultsAsyncIterable:
            def __init__(self, results):
                self.results = results
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.results):
                    raise StopAsyncIteration
                result = self.results[self.index]
                self.index += 1
                return result

        # Create the scorer pipeline with the results as source
        scorer_pl = self._scorer_pipeline(ResultsAsyncIterable(explainer_results))
        return await scorer_pl.run()


class MyRunConfig(BaseModel):
    integrated_explainer_scorer_cfg: IntegratedExplainerScorerConfig
    cache_cfg: CacheConfig
    constructor_cfg: ConstructorConfig
    sampler_cfg: SamplerConfig
    model: str
    sparse_model: str
    hookpoints: list[str]
    explainer_model: str
    explainer_model_max_len: int
    name: str = ""
    max_latents: int | None = None
    filter_bos: bool = False
    num_gpus: int
    overwrite: list[Literal["cache", "neighbours", "scores"]]
    base_path: Path

    # (optionals unchanged)
    skip_generate_cache_if_exists: Optional[bool] = None
    explainer_provider: Optional[str] = None
    explainer: Optional[str] = None
    num_explanations: Optional[int] = None
    log_probs: Optional[bool] = None
    load_in_8bit: Optional[bool] = None
    hf_token: Optional[str] = None
    pipeline_num_proc: Optional[int] = None
    seed: Optional[int] = None
    verbose: Optional[bool] = None
    num_examples_per_scorer_prompt: Optional[int] = None
