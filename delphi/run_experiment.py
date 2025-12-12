import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path

# Respect an existing CUDA device selection; default to GPUs 3 and 4 only if unset.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
from functools import partial
from typing import Any, Callable, Tuple

import orjson
import torch
from torch import Tensor
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from delphi import logger
from delphi.clients import Offline, OpenRouter
from delphi.config import CacheConfig, ConstructorConfig, RunConfig, SamplerConfig
from delphi.explainers import (
    BestOfKExplainer,
    ContrastiveExplainer,
    DefaultExplainer,
    IterativeExplainer,
    NoOpExplainer,
)
from delphi.explainers.explainer import ExplainerResult
from delphi.explainers.iterative.iterative import HillClimbing
from delphi.latents import LatentCache, LatentDataset, LatentRecord
from delphi.latents.neighbours import NeighbourCalculator
from delphi.log.result_analysis import log_results
from delphi.pipeline import Pipe, Pipeline, process_wrapper
from delphi.scorers import (
    DetectionScorer,
    EmbeddingScorer,
    FuzzingScorer,
    OpenAISimulator,
    Scorer,
)
from delphi.scorers.scorer import ScorerResult
from delphi.server_utils import report_vllm_gpu_utilization, start_server_if_not_running
from delphi.sparse_coders import load_hooks_sparse_coders, load_sparse_coders
from delphi.utils import TimingAggregator, assert_type, load_tokenized_data


def load_artifacts(run_cfg: RunConfig):
    if run_cfg.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    model = AutoModel.from_pretrained(
        run_cfg.model,
        device_map={"": "cuda"},
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=run_cfg.load_in_8bit)
            if run_cfg.load_in_8bit
            else None
        ),
        torch_dtype=dtype,
        token=run_cfg.hf_token,
    )

    hookpoint_to_sparse_encode, transcode = load_hooks_sparse_coders(
        model,
        run_cfg,
        compile=True,
    )

    return (
        list(hookpoint_to_sparse_encode.keys()),
        hookpoint_to_sparse_encode,
        model,
        transcode,
    )


def create_neighbours(
    run_cfg: RunConfig,
    latents_path: Path,
    neighbours_path: Path,
    hookpoints: list[str],
):
    """
    Creates a neighbours file for the given hookpoints.
    """
    neighbours_path.mkdir(parents=True, exist_ok=True)

    constructor_cfg = run_cfg.constructor_cfg
    saes = (
        load_sparse_coders(run_cfg, device="cpu")
        if constructor_cfg.neighbours_type != "co-occurrence"
        else {}
    )

    for hookpoint in hookpoints:
        if constructor_cfg.neighbours_type == "co-occurrence":
            neighbour_calculator = NeighbourCalculator(
                cache_dir=latents_path / hookpoint, number_of_neighbours=250
            )

        elif constructor_cfg.neighbours_type == "decoder_similarity":
            neighbour_calculator = NeighbourCalculator(
                autoencoder=saes[hookpoint].to("cuda"), number_of_neighbours=250
            )

        elif constructor_cfg.neighbours_type == "encoder_similarity":
            neighbour_calculator = NeighbourCalculator(
                autoencoder=saes[hookpoint].to("cuda"), number_of_neighbours=250
            )
        else:
            raise ValueError(
                f"Neighbour type {constructor_cfg.neighbours_type} not supported"
            )

        neighbour_calculator.populate_neighbour_cache(constructor_cfg.neighbours_type)
        neighbour_calculator.save_neighbour_cache(f"{neighbours_path}/{hookpoint}")


async def process_cache(
    run_cfg: RunConfig,
    latents_path: Path,
    explanations_path: Path,
    scores_path: Path,
    hookpoints: list[str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    latent_range: Tensor | None,
):
    """
    Converts SAE latent activations in on-disk cache in the `latents_path` directory
    to latent explanations in the `explanations_path` directory and explanation
    scores in the `scores_path` directory.
    """
    explanations_path.mkdir(parents=True, exist_ok=True)
    # For multi-explainer modes (e.g., bestofk), ensure subdir exists
    (explanations_path / "multi_explanations").mkdir(parents=True, exist_ok=True)
    if run_cfg.verbose:
        print(f"[process_cache] Writing explanations under: {explanations_path}")

    if latent_range is None:
        latent_dict = None
    else:
        latent_dict = {
            hook: latent_range for hook in hookpoints
        }  # The latent range to explain

    dataset = LatentDataset(
        raw_dir=latents_path,
        sampler_cfg=run_cfg.sampler_cfg,
        constructor_cfg=run_cfg.constructor_cfg,
        modules=hookpoints,
        latents=latent_dict,
        tokenizer=tokenizer,
    )

    timing_aggregator = TimingAggregator()
    scorer_dir_to_name: dict[str, str] = {}

    def record_duration(bucket: str, name: str, value):
        key = f"{bucket}:{name}"

        def _record(entry):
            if entry is None:
                return
            if isinstance(entry, (list, tuple)):
                for item in entry:
                    _record(item)
            else:
                timing_aggregator.add(key, getattr(entry, "duration", None))

        _record(value)

    def write_timing_summary():
        log_dir = scores_path.parent / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        summary = timing_aggregator.as_dict()
        payload: dict[str, dict[str, dict[str, float]]] = {"explainers": {}, "scorers": {}}
        for key, stats in summary.items():
            bucket, _, name = key.partition(":")
            if bucket == "explainer":
                payload["explainers"][name] = stats
            elif bucket == "scorer":
                payload["scorers"][name] = stats
            else:
                payload.setdefault("other", {})[name] = stats
        timings_path = log_dir / "timings.json"
        with open(timings_path, "wb") as f:
            f.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2))

    if run_cfg.explainer_provider == "offline":
        llm_client = Offline(
            run_cfg.explainer_model,
            max_memory=0.9,
            # Explainer models context length - must be able to accommodate the longest
            # set of examples
            max_model_len=run_cfg.explainer_model_max_len,
            num_gpus=run_cfg.num_gpus,
            statistics=run_cfg.verbose,
            server_port=run_cfg.server_port,
            # temperature=run_cfg.explainer_temperature,
        )
    elif run_cfg.explainer_provider == "openrouter":
        if (
            "OPENROUTER_API_KEY" not in os.environ
            or not os.environ["OPENROUTER_API_KEY"]
        ):
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. Set "
                "`--explainer-provider offline` to use a local explainer model."
            )

        llm_client = OpenRouter(
            run_cfg.explainer_model,
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    else:
        raise ValueError(
            f"Explainer provider {run_cfg.explainer_provider} not supported"
        )

    # Builds the record from result returned by the pipeline
    # Store explanation_id in the record object temporarily for retrieval later
    def scorer_preprocess(result: ExplainerResult) -> LatentRecord:
        # Stash the explanation_id on the record object (not the latent, which is shared!)
        if result.explanation_id is not None:
            result.record._explanation_id = result.explanation_id
        # Set the explanation on the record so the scorer can access it
        result.record.explanation = result.explanation
        return result.record

    # Saves the score to a file
    def scorer_postprocess(  # Writes per-round scores for bestofk and iterative
        result: ScorerResult | list[ScorerResult],
        score_dir,  # path to write scores to
        is_final: bool = False,  # passed only for explainers which produce multiple scores; final is either best or last, according to flag
        round_idx: int
        | None = None,  # passed only for iterative, non-final result - iterative produces multiple scores,but only one at a time
    ):
        dir_path = Path(score_dir)
        dir_key = str(dir_path.resolve())
        scorer_label = scorer_dir_to_name.get(dir_key, dir_path.name)

        if not is_final:
            record_duration("scorer", scorer_label, result)

        tmp = result if isinstance(result, list) else [result]
        safe_latent_name = str(tmp[0].record.latent).replace("/", "--")

        # For bestofk and iterative (per-round), save all scores to multi_scores folder
        if run_cfg.explainer == "bestofk" and not is_final:
            explanation_id = getattr(tmp[0].record, "_explanation_id", 0)
            out_path = (
                dir_path
                / "multi_scores"
                / f"{safe_latent_name}_{explanation_id}.txt"
            )
            with open(out_path, "wb") as f:
                f.write(orjson.dumps(result.score))

            if run_cfg.verbose:
                print(
                    f"[scorer_postprocess] Wrote BestOfK candidate score {explanation_id}: {out_path}"
                )
        elif run_cfg.explainer == "iterative":
            if is_final:
                assert not isinstance(result, list)  # is_final mustnt give a list
                out_path = dir_path / f"{safe_latent_name}.txt"
                with open(out_path, "wb") as f:
                    f.write(orjson.dumps(tmp[0].score))

                if run_cfg.verbose:
                    print(
                        f"[scorer_postprocess] Wrote iterative FINAL score: {out_path}"
                    )
            else:
                if isinstance(result, list):
                    for round_idx, res in enumerate(result):
                        out_path = (
                            dir_path
                            / "multi_scores"
                            / f"{safe_latent_name}_{round_idx}.txt"
                        )
                        with open(out_path, "wb") as f:
                            f.write(orjson.dumps(res.score))

                        if run_cfg.verbose:
                            print(
                                f"[scorer_postprocess] Wrote iterative multi-score: {out_path}"
                            )
                else:
                    assert round_idx is not None
                    out_path = (
                        dir_path
                        / "multi_scores"
                        / f"{safe_latent_name}_{round_idx}.txt"
                    )
                    with open(out_path, "wb") as f:
                        f.write(orjson.dumps(result.score))

                    if run_cfg.verbose:
                        print(
                            f"[scorer_postprocess] Wrote iterative multi-score: {out_path}"
                        )
        else:  # not bestofk or iterative
            assert not isinstance(result, list)
            out_path = dir_path / f"{safe_latent_name}.txt"
            with open(out_path, "wb") as f:
                f.write(orjson.dumps(result.score))

            if run_cfg.verbose:
                print(f"[scorer_postprocess] Wrote score: {out_path}")

        # for bestofk, return the first (or list) for upstream use; others ignore
        return result if run_cfg.explainer == "bestofk" else None



    wrapped_scorers: list[
        Any
    ] = []  # contains wrapped scorers; most pipelines will use this
    scorers_with_paths: list[
        tuple[Scorer, Path]
    ] = []  # contains scorers and their paths, for bestofk
    embedding_model = None
    for scorer_name in run_cfg.scorers:
        scorer_path = scores_path / scorer_name
        scorer_path.mkdir(parents=True, exist_ok=True)
        # For multi-explainer modes (e.g., bestofk), ensure subdir exists
        (scorer_path / "multi_scores").mkdir(parents=True, exist_ok=True)
        scorer_dir_to_name[str(scorer_path.resolve())] = scorer_name
        if run_cfg.verbose:
            print(
                f"[process_cache] Scorer '{scorer_name}' outputs under: {scorer_path}"
            )

        if scorer_name == "simulation":
            if isinstance(llm_client, Offline):
                scorer = OpenAISimulator(
                    llm_client, tokenizer=tokenizer, all_at_once=True
                )
            else:
                scorer = OpenAISimulator(
                    llm_client, tokenizer=tokenizer, all_at_once=False
                )
        elif scorer_name == "fuzz":
            scorer = FuzzingScorer(
                llm_client,
                n_examples_shown=run_cfg.num_examples_per_scorer_prompt,
                verbose=run_cfg.verbose,
                log_prob=run_cfg.log_probs,
            )
        elif scorer_name == "detection":
            scorer = DetectionScorer(
                llm_client,
                n_examples_shown=run_cfg.num_examples_per_scorer_prompt,
                verbose=run_cfg.verbose,
                log_prob=run_cfg.log_probs,
            )
        elif scorer_name == "embedding":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for embedding scorer. "
                    "Install with: pip install sentence-transformers"
                )
            if embedding_model is None:
                # Use GPU if available, otherwise CPU
                device = "cuda" if torch.cuda.is_available() else "cpu"
                embedding_model = SentenceTransformer(
                    run_cfg.embedding_model, 
                    device=device,
                    trust_remote_code=True,
                )
                if run_cfg.verbose:
                    print(f"[process_cache] Initialized embedding model on device: {device}")
            scorer = EmbeddingScorer(
                embedding_model,
                verbose=run_cfg.verbose,
            )
        else:
            raise ValueError(f"Scorer {scorer_name} not supported")

        wrapped_scorer = process_wrapper(
            scorer,
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir=scorer_path),
        )
        scorers_with_paths.append(
            (scorer, scorer_path)
        )  # for bestofk, this is used to run the scorers
        wrapped_scorers.append(wrapped_scorer)

    if not run_cfg.explainer == "none":

        def explainer_postprocess(
            explainer_results: ExplainerResult
            | Tuple[ExplainerResult, list[ExplainerResult]],
            is_final: bool = False,
        ):
            def write_explanation(
                explainer_result: ExplainerResult,
                filename: str,
                subdir: str | None = None,
            ):
                path = (
                    explanations_path / filename
                    if subdir is None
                    else explanations_path / subdir / filename
                )
                with open(path, "wb") as f:
                    f.write(orjson.dumps(explainer_result.explanation))

                if run_cfg.verbose:
                    print(f"[explainer_postprocess] Wrote explanation: {path}")

            if isinstance(explainer_results, ExplainerResult):
                explainer_result = explainer_results  # single explanation
                explanation_id = getattr(explainer_result, "explanation_id", None)

                # For BestOfK, if it's a single result, it's the best one and should be saved as final
                # For other explainers, save as final if is_final=True or explanation_id is None
                if run_cfg.explainer == "bestofk" or is_final or explanation_id is None:
                    filename = f"{explainer_result.record.latent}.txt"
                    write_explanation(explainer_result, filename)
                else:
                    filename = f"{explainer_result.record.latent}_{explanation_id}.txt"
                    write_explanation(
                        explainer_result, filename, subdir="multi_explanations"
                    )

                record_duration("explainer", run_cfg.explainer, explainer_result)
                return explainer_results
            else:
                explainer_result, all_explanations = explainer_results
                # Save all explanations to multi_explanations
                for round_idx, explanation in enumerate(all_explanations):
                    filename = f"{explanation.record.latent}_{round_idx}.txt"
                    write_explanation(
                        explanation, filename, subdir="multi_explanations"
                    )
                # Save the selected explanation as the final one
                filename = f"{explainer_result.record.latent}.txt"
                write_explanation(explainer_result, filename)
                record_duration("explainer", run_cfg.explainer, [explainer_result, *all_explanations])
                return explainer_results

        if run_cfg.constructor_cfg.non_activating_source == "FAISS":
            explainer = ContrastiveExplainer(
                llm_client,
                threshold=0.3,
                verbose=run_cfg.verbose,
                temperature=run_cfg.explainer_temperature,
            )
        elif run_cfg.explainer == "bestofk":
            explainer = BestOfKExplainer(
                llm_client,
                bestofk_num_explanations=run_cfg.bestofk_num_explanations,
                scorers_with_paths=scorers_with_paths,
                scorer_postprocess=scorer_postprocess,
                scorer_preprocess=scorer_preprocess,
                temperature=run_cfg.explainer_temperature,
                judge_scorer_index=getattr(
                    run_cfg, "bestofk_judge_scorer_index", run_cfg.judge_scorer_index
                ),
                return_only_best=run_cfg.bestofk_return_only_best,
                run_all_scorers=run_cfg.bestofk_run_all_scorers,
                is_multishot=run_cfg.bestofk_is_multishot,
                bestofk_num_train_examples=run_cfg.bestofk_num_train_examples,
                use_random_baseline=run_cfg.use_random_baseline,
                random_baseline_source_run=run_cfg.random_baseline_source_run,
                embedding_prefilter_enabled=run_cfg.bestofk_embedding_prefilter_enabled,
                embedding_prefilter_top_k=run_cfg.bestofk_embedding_prefilter_top_k,
                embedding_is_judge=run_cfg.bestofk_embedding_use_as_judge,
            )
            # Spot check logging disabled for performance
            # setattr(explainer, "spot_check_dir", (scores_path.parent / "spot_check").resolve())
            # setattr(explainer, "spot_check_mod", 100)
        elif run_cfg.explainer == "iterative":
            # Iterative hill-climbing orchestrates explanation + scoring internally
            iterative_explainer = IterativeExplainer(
                llm_client,
                verbose=run_cfg.verbose,
                iterative_max_num_false_positives=run_cfg.iterative_max_num_false_positives,
                iterative_max_num_false_negatives=run_cfg.iterative_max_num_false_negatives,
                iterative_append_round_to_prompt=run_cfg.iterative_append_round_to_prompt,
                temperature=run_cfg.explainer_temperature,
                iterative_max_num_true_positives=getattr(
                    run_cfg, "iterative_max_num_true_positives", 0
                ),
                iterative_max_num_true_negatives=getattr(
                    run_cfg, "iterative_max_num_true_negatives", 0
                ),
                iterative_show_score_to_explainer=getattr(
                    run_cfg, "iterative_show_score_to_explainer", False
                ),
                iterative_history_only=getattr(
                    run_cfg, "iterative_history_only", False
                ),
                iterative_allow_tp_examples=getattr(
                    run_cfg, "iterative_allow_tp_examples", True
                ),
                iterative_num_train_examples_per_round=run_cfg.iterative_num_train_examples_per_round,
            )
            explainer = HillClimbing(
                scorers_with_paths=scorers_with_paths,
                scorer_postprocess=scorer_postprocess,
                explainer_postprocess=explainer_postprocess,
                explainer=iterative_explainer,
                iterative_num_rounds=run_cfg.iterative_num_rounds,
                judge_scorer_index=run_cfg.judge_scorer_index,
                iterative_carryforward_strategy=getattr(
                    run_cfg, "iterative_carryforward_strategy", "last"
                ),
                iterative_always_new_train_examples=getattr(
                    run_cfg, "iterative_always_new_train_examples", False
                ),
            )
            explainer_postprocess = (
                None  # since handled internally, already passed to HillClimbing above
            )
        else:
            explainer = DefaultExplainer(
                llm_client,
                threshold=0.3,
                verbose=run_cfg.verbose,
                temperature=run_cfg.explainer_temperature,
            )

        explainer_pipe = Pipe(
            process_wrapper(explainer, postprocess=explainer_postprocess)
        )
        if run_cfg.verbose:
            print(f"[process_cache] Explainer selected: {run_cfg.explainer}")
    else:

        def none_postprocessor(result):
            # Load the explanation from disk
            explanation_path = explanations_path / f"{result.record.latent}.txt"
            if not explanation_path.exists():
                raise FileNotFoundError(
                    f"Explanation file {explanation_path} does not exist. "
                    "Make sure to run an explainer pipeline first."
                )

            with open(explanation_path, "rb") as f:
                return ExplainerResult(
                    record=result.record,
                    explanation=orjson.loads(f.read()),
                )

        explainer_pipe = Pipe(
            process_wrapper(
                NoOpExplainer(),
                postprocess=none_postprocessor,
            )
        )

    # Build pipeline depending on explainer
    if run_cfg.explainer == "bestofk":
        # BestOfK handles scoring internally
        pipeline = Pipeline(
            dataset,
            explainer_pipe,
        )
    elif run_cfg.explainer == "iterative":
        # Iterative hill-climbing handles scoring internally to pick best/last explanation
        pipeline = Pipeline(
            dataset,
            explainer_pipe,
        )
    else:
        # Default/contrastive/none: run scorers as separate pipe
        pipeline = Pipeline(
            dataset,
            explainer_pipe,
            Pipe(*wrapped_scorers),
        )

    if run_cfg.pipeline_num_proc > 1 and run_cfg.explainer_provider == "openrouter":
        print(
            "OpenRouter does not support multiprocessing,"
            " setting pipeline_num_proc to 1"
        )
        run_cfg.pipeline_num_proc = 1

    if run_cfg.verbose:
        print(
            f"[process_cache] Starting pipeline with concurrency={run_cfg.pipeline_num_proc}"
        )
    await pipeline.run(run_cfg.pipeline_num_proc)
    write_timing_summary()
    if run_cfg.verbose:
        print("[process_cache] Pipeline completed")


def populate_cache(
    run_cfg: RunConfig,
    model: PreTrainedModel,
    hookpoint_to_sparse_encode: dict[str, Callable],
    latents_path: Path,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    transcode: bool,
):
    """
    Populates an on-disk cache in `latents_path` with SAE latent activations.
    """
    latents_path.mkdir(parents=True, exist_ok=True)

    # Create a log path within the run directory
    log_path = latents_path.parent / "log"
    log_path.mkdir(parents=True, exist_ok=True)

    cache_cfg = run_cfg.cache_cfg
    tokens = load_tokenized_data(
        cache_cfg.cache_ctx_len,
        tokenizer,
        cache_cfg.dataset_repo,
        cache_cfg.dataset_split,
        cache_cfg.dataset_name,
        cache_cfg.dataset_column,
        run_cfg.seed,
    )

    if run_cfg.filter_bos:
        if tokenizer.bos_token_id is None:
            print("Tokenizer does not have a BOS token, skipping BOS filtering")
        else:
            flattened_tokens = tokens.flatten()
            mask = ~torch.isin(flattened_tokens, torch.tensor([tokenizer.bos_token_id]))
            masked_tokens = flattened_tokens[mask]
            truncated_tokens = masked_tokens[
                : len(masked_tokens) - (len(masked_tokens) % cache_cfg.cache_ctx_len)
            ]
            tokens = truncated_tokens.reshape(-1, cache_cfg.cache_ctx_len)

    cache = LatentCache(
        model,
        hookpoint_to_sparse_encode,
        batch_size=cache_cfg.batch_size,
        transcode=transcode,
        log_path=log_path,
    )
    cache.run(cache_cfg.n_tokens, tokens)

    if run_cfg.verbose:
        cache.generate_statistics_cache()

    cache.save_splits(
        # Split the activation and location indices into different files to make
        # loading faster
        n_splits=cache_cfg.n_splits,
        save_dir=latents_path,
    )

    cache.save_config(save_dir=latents_path, cfg=cache_cfg, model_name=run_cfg.model)


def non_redundant_hookpoints(
    hookpoint_to_sparse_encode: dict[str, Callable] | list[str],
    results_path: Path,
    overwrite: bool,
) -> dict[str, Callable] | list[str]:
    """
    Returns a list of hookpoints that are not already in the cache.
    """
    if overwrite:
        print("Overwriting results from", results_path)
        return hookpoint_to_sparse_encode
    in_results_path = [x.name for x in results_path.glob("*")]
    if isinstance(hookpoint_to_sparse_encode, dict):
        non_redundant_hookpoints = {
            k: v
            for k, v in hookpoint_to_sparse_encode.items()
            if k not in in_results_path
        }
    else:
        non_redundant_hookpoints = [
            hookpoint
            for hookpoint in hookpoint_to_sparse_encode
            if hookpoint not in in_results_path
        ]
    if not non_redundant_hookpoints:
        print(f"Files found in {results_path}, skipping...")
    return non_redundant_hookpoints


async def run(
    run_cfg: RunConfig,
    base_results_dir: Path | None = None,
):
    base_path = base_results_dir or Path.cwd() / "results"
    if run_cfg.name:
        base_path = base_path / run_cfg.name

    base_path.mkdir(parents=True, exist_ok=True)

    run_cfg.save_json(base_path / "run_config.json", indent=4)

    if (
        run_cfg.explainer_provider == "offline"
        and run_cfg.server_port is not None
    ):
        print(
            f"[run_experiment.py:run] Ensuring vLLM server is running on port {run_cfg.server_port} before preprocessing"
        )
        start_server_if_not_running(run_cfg.server_port, run_cfg)

    latents_path = base_path / "latents"
    explanations_path = base_path / "explanations"
    scores_path = base_path / "scores"
    neighbours_path = base_path / "neighbours"
    visualize_path = base_path / "visualize"

    latent_range = torch.arange(run_cfg.max_latents) if run_cfg.max_latents else None

    hookpoints, hookpoint_to_sparse_encode, model, transcode = load_artifacts(run_cfg)
    tokenizer = AutoTokenizer.from_pretrained(run_cfg.model, token=run_cfg.hf_token)

    nrh = assert_type(
        dict,
        non_redundant_hookpoints(
            hookpoint_to_sparse_encode, latents_path, "cache" in run_cfg.overwrite
        ),
    )
    if nrh:
        populate_cache(
            run_cfg,
            model,
            nrh,
            latents_path,
            tokenizer,
            transcode,
        )

    del model, hookpoint_to_sparse_encode
    if run_cfg.constructor_cfg.non_activating_source == "neighbours":
        nrh = assert_type(
            list,
            non_redundant_hookpoints(
                hookpoints, neighbours_path, "neighbours" in run_cfg.overwrite
            ),
        )
        if nrh:
            create_neighbours(
                run_cfg,
                latents_path,
                neighbours_path,
                nrh,
            )
    else:
        print("Skipping neighbour creation")

    nrh = assert_type(
        list,
        non_redundant_hookpoints(
            hookpoints, scores_path, "scores" in run_cfg.overwrite
        ),
    )
    if nrh:
        await process_cache(
            run_cfg,
            latents_path,
            explanations_path,
            scores_path,
            nrh,
            tokenizer,
            latent_range,
        )

    report_vllm_gpu_utilization(run_cfg)

    if run_cfg.verbose:
        log_results(
            scores_path,
            visualize_path,
            run_cfg.hookpoints,
            run_cfg.scorers,
            image_formats=["png", "pdf"],
        )


def start_server_if_not_running(server_port: int, run_cfg: RunConfig):
    # This function has been moved to delphi.server_utils
    # Keeping this stub here in case it's called from elsewhere, though it shouldn't be
    from delphi.server_utils import start_server_if_not_running as _start
    return _start(server_port, run_cfg)


class ExperimentRunner:
    def __init__(self, base_results_dir: Path | None = None) -> None:
        self.base_results_dir = base_results_dir

    async def run_async(self, run_cfg: RunConfig) -> None:
        await run(run_cfg, base_results_dir=self.base_results_dir)

    def run(self, run_cfg: RunConfig) -> None:
        asyncio.run(self.run_async(run_cfg))


def default_run_config() -> RunConfig:
    cache_cfg = CacheConfig(
        dataset_repo="EleutherAI/SmolLM2-135M-10B",
        cache_ctx_len=32 * 3,
        n_tokens=10_000_000,
    )

    constructor_cfg = ConstructorConfig(
        example_ctx_len=32,
        min_examples=200,  # gate only
        n_non_activating=350,
    )

    sampler_cfg = SamplerConfig(
        # NOTE: Data flow for BestOfK and Iterative explainers:
        #
        # 1. Constructor creates:
        #    - record.examples: ALL activating examples (from constructor_cfg.min_examples, e.g., 200)
        #    - record.not_active: non-activating examples (from constructor_cfg.n_non_activating, e.g., 350)
        #
        # 2. Sampler populates record.train and record.test with subsets from record.examples:
        #    - record.train: n_examples_train activating examples (with str_tokens populated)
        #    - record.test: n_examples_test activating examples (with str_tokens populated)
        #    - record.not_active: remains unchanged (str_tokens added by constructor)
        #
        # 3. BestOfK uses record.train and record.test DIRECTLY (no re-splitting):
        #    - Train: record.train (for prompting)
        #    - Test: record.test (activating) + record.not_active (non-activating) for scoring
        #    - Example: n_train=20 → 20 train, n_test=150 → 150 test + 350 non-activating
        #
        # 4. Iterative uses record.train for training+test, record.test for holdout:
        #    - Train+Test: record.train (sample from for prompting, score against for FP/FN collection)
        #    - Holdout: record.test (activating) + record.not_active (non-activating) for final eval
        #    - Example: n_train=100 → sample 20 for prompt, score against 100 + 350 non-activating
        #               n_test=150 → 150 activating + 350 non-activating for holdout
        #
        # KEY PARAMETERS:
        #   - n_examples_train: For BestOfK prompting (20), or Iterative train+test pool (100)
        #   - n_examples_test: For scoring (150 activating examples)
        #   - constructor_cfg.min_examples: Total activating examples (200)
        #   - constructor_cfg.n_non_activating: Total non-activating pool (350)
        n_examples_train=20,  # Sampler populates this many in record.train
        n_examples_test=150,  # Sampler populates this many in record.test
        n_quantiles=5,
    )

    return RunConfig(
        cache_cfg=cache_cfg,
        constructor_cfg=constructor_cfg,
        sampler_cfg=sampler_cfg,
        # model="EleutherAI/pythia-70m",
        # sparse_model="EleutherAI/sae-pythia-70m-32k",
        # hookpoints=["layers.5"],
        # model="meta-llama/Meta-Llama-3-8B",
        # sparse_model="EleutherAI/sae-llama-3-8b-32x",
        model="EleutherAI/pythia-160m",
        sparse_model="EleutherAI/Pythia-160m-SST-k32-32k",
        hookpoints=["layers.5.mlp"],
        # explainer_model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        explainer_model="Qwen/Qwen3-32B",
        explainer_model_max_len=8192*4,
        explainer_provider="offline",
        server_port=8000,  # if set, assumes local server is running.  'None' will start server
        explainer="iterative",
        scorers=["fuzz", "detection", "embedding"],
        name="pythia-160m-iterative-100-latents",
        max_latents=100,
        filter_bos=True,
        num_gpus=2,
        verbose=True,
        num_examples_per_scorer_prompt=5,
        explainer_temperature=0.7,
        enforce_eager=True,  # Disable CUDA graphs (required for stable tensor-parallel inference)
        judge_scorer_index=1,  # used for both bestofk and iterative
        # bestofk only
        bestofk_num_explanations=3,
        bestofk_return_only_best=False,
        bestofk_run_all_scorers=True,
        bestofk_is_multishot=True,
        # iterative only
        iterative_num_rounds=5,
        #iterative_holdout_ratio_of_total=0.1,
        #iterative_test_ratio_of_nonholdout=0.75,
        iterative_max_num_false_positives=10,
        iterative_max_num_false_negatives=10,
        iterative_append_round_to_prompt=True,
        # Iterative carry-forward and visibility controls
        iterative_carryforward_strategy="last",  # or "best"
        iterative_show_score_to_explainer=False,
        iterative_history_only=False,
        iterative_always_new_train_examples=False,
        # Optional TP/TN in prompts
        iterative_max_num_true_positives=0,
        iterative_max_num_true_negatives=0,
        iterative_allow_tp_examples=True,
    )


if __name__ == "__main__":
    logger.setLevel(logging.WARNING)
    file_handler = logging.FileHandler("delphi.log")
    file_handler.setLevel(logging.WARNING)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Use absolute path to ensure consistent results location regardless of cwd
    # This resolves to the repo root's results/ directory
    results_dir = (Path(__file__).parent.parent / "results").resolve()
    runner = ExperimentRunner(base_results_dir=results_dir)
    runner.run(default_run_config())
