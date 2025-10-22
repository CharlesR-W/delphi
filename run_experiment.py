import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
from functools import partial
from typing import Any, Callable, Tuple

import orjson
import requests
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
from delphi.scorers import DetectionScorer, FuzzingScorer, OpenAISimulator, Scorer
from delphi.scorers.scorer import ScorerResult
from delphi.sparse_coders import load_hooks_sparse_coders, load_sparse_coders
from delphi.utils import assert_type, load_tokenized_data


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

    if run_cfg.explainer_provider == "offline":
        # Start server if server_port is specified
        if run_cfg.server_port is not None:
            print(
                f"[run_experiment.py:run] Starting server on port {run_cfg.server_port} (unless already running, then will do nothing)"
            )
            start_server_if_not_running(run_cfg.server_port, run_cfg)

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
        tmp = result if isinstance(result, list) else [result]
        safe_latent_name = str(tmp[0].record.latent).replace("/", "--")

        # For bestofk and iterative (per-round), save all scores to multi_scores folder
        if run_cfg.explainer == "bestofk" and not is_final:
            # BestOfK can return either a single ScorerResult or a list
            if isinstance(result, list):
                for round_idx, res in enumerate(result):
                    out_path = (
                        score_dir
                        / "multi_scores"
                        / f"{safe_latent_name}_{round_idx}.txt"
                    )
                    with open(out_path, "wb") as f:
                        f.write(orjson.dumps(res.score))
                    if run_cfg.verbose:
                        print(f"[scorer_postprocess] Wrote multi-score: {out_path}")
            else:
                # Single result - extract explanation_id that was stashed in scorer_preprocess
                # The explanation_id tells us which round this is
                explanation_id = getattr(tmp[0].record, "_explanation_id", 0)
                out_path = (
                    score_dir
                    / "multi_scores"
                    / f"{safe_latent_name}_{explanation_id}.txt"
                )
                with open(out_path, "wb") as f:
                    f.write(orjson.dumps(result.score))
                if run_cfg.verbose:
                    print(
                        f"[scorer_postprocess] Wrote single multi-score round {explanation_id}: {out_path}"
                    )
        elif run_cfg.explainer == "iterative":
            if is_final:
                assert not isinstance(result, list)  # is_final mustnt give a list
                # For the selected best explanation, also emit a top-level score file
                out_path = score_dir / f"{safe_latent_name}.txt"
                with open(out_path, "wb") as f:
                    f.write(orjson.dumps(tmp[0].score))
                if run_cfg.verbose:
                    print(
                        f"[scorer_postprocess] Wrote iterative FINAL score: {out_path}"
                    )
            else:  # iterative, not final, multiple results - set up to write either a single result or a list; idk which it returns
                if isinstance(result, list):
                    for round_idx, res in enumerate(result):
                        out_path = (
                            score_dir
                            / "multi_scores"
                            / f"{safe_latent_name}_{round_idx}.txt"
                        )
                        with open(out_path, "wb") as f:
                            f.write(orjson.dumps(res.score))
                        if run_cfg.verbose:
                            print(
                                f"[scorer_postprocess] Wrote iterative multi-score: {out_path}"
                            )
                else:  # iterative, not final, single result
                    assert round_idx is not None
                    out_path = (
                        score_dir
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
            out_path = score_dir / f"{safe_latent_name}.txt"
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
    for scorer_name in run_cfg.scorers:
        scorer_path = scores_path / scorer_name
        scorer_path.mkdir(parents=True, exist_ok=True)
        # For multi-explainer modes (e.g., bestofk), ensure subdir exists
        (scorer_path / "multi_scores").mkdir(parents=True, exist_ok=True)
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
                judge_scorer_index=run_cfg.judge_scorer_index,
                return_only_best=run_cfg.bestofk_return_only_best,
                run_all_scorers=run_cfg.bestofk_run_all_scorers,
                is_multishot=run_cfg.bestofk_is_multishot,
                bestofk_num_train_examples=getattr(run_cfg, "bestofk_num_train_examples", None),
            )
            # Configure spot-check logging
            setattr(explainer, "spot_check_dir", (scores_path.parent / "spot_check").resolve())
            setattr(explainer, "spot_check_mod", 100)
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
            )
            explainer = HillClimbing(
                scorers_with_paths=scorers_with_paths,
                scorer_postprocess=scorer_postprocess,
                explainer_postprocess=explainer_postprocess,
                explainer=iterative_explainer,
                iterative_num_rounds=run_cfg.iterative_num_rounds,
                iterative_holdout_ratio_of_total=run_cfg.iterative_holdout_ratio_of_total,
                iterative_test_ratio_of_nonholdout=run_cfg.iterative_test_ratio_of_nonholdout,
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

    if run_cfg.verbose:
        log_results(
            scores_path,
            visualize_path,
            run_cfg.hookpoints,
            run_cfg.scorers,
            image_formats=["png", "pdf"],
        )


def check_other_vllm_servers():
    """Check for other vLLM servers running and warn user"""
    try:
        # Use pgrep to find vLLM processes
        result = subprocess.run(
            ["pgrep", "-f", "vllm.*serve"], capture_output=True, text=True, check=False
        )

        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            print(
                f"⚠️  WARNING: Found {len(pids)} other vLLM server process(es) running:"
            )
            for pid in pids:
                if pid.strip():
                    # Get more details about the process
                    try:
                        ps_result = subprocess.run(
                            ["ps", "-p", pid.strip(), "-o", "pid,ppid,cmd"],
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        if ps_result.returncode == 0:
                            cmd_line = ps_result.stdout.strip().split("\n")[-1]
                            print(f"   PID {pid.strip()}: {cmd_line}")
                    except Exception:
                        print(f"   PID {pid.strip()}")
            print("   Consider stopping other servers to avoid resource conflicts.")
            print()
    except Exception as e:
        print(f"Could not check for other vLLM servers: {e}")


def start_server_if_not_running(server_port: int, run_cfg: RunConfig):
    # Check for other vLLM servers first
    check_other_vllm_servers()

    try:
        response = requests.get(f"http://localhost:{server_port}/v1/models")
        if response.status_code == 200:
            print(
                f"[run_experiment.py:start_server_if_not_running] Server is already running on port {server_port}"
            )
            return
    except requests.exceptions.RequestException:
        print(
            f"[run_experiment.py:start_server_if_not_running] Server is not running on port {server_port}; attempting to start server"
        )
        # Build command with conditional boolean flags
        cmd = [
            "vllm",
            "serve",
            getattr(
                run_cfg,
                "explainer_model",
                "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
            ),  # positional model argument
            "--host",
            "0.0.0.0",
            "--max-model-len",
            str(getattr(run_cfg, "explainer_model_max_len", 5120)),
            "--tensor-parallel-size",
            str(getattr(run_cfg, "num_gpus", torch.cuda.device_count())),
            "--gpu-memory-utilization",
            str(getattr(run_cfg, "gpu_memory_utilization", 0.9)),
            "--port",
            str(getattr(run_cfg, "server_port", 8000)),
            "--uvicorn-log-level",
            "warning",
            # "--disable-log-requests",
            # "--no-access-log",
        ]

        # Add boolean flags only if True (don't pass False values)
        if getattr(run_cfg, "enable_prefix_caching", True):
            cmd.append("--enable-prefix-caching")
        if getattr(run_cfg, "enforce_eager", True):
            cmd.append("--enforce-eager")

        model_name_lower = str(getattr(run_cfg, "explainer_model", "")).lower()
        if "qwen" in model_name_lower:
            # cmd.append("--disable-think")
            pass

        # Reduce vLLM logging verbosity
        os.environ["VLLM_LOGGING_LEVEL"] = os.environ.get(
            "VLLM_LOGGING_LEVEL", "WARNING"
        )
        os.environ["VLLM_CONFIGURE_LOGGING"] = os.environ.get(
            "VLLM_CONFIGURE_LOGGING", "1"
        )

        server_process = subprocess.Popen(
            cmd,
            # Don't redirect stdout/stderr initially - let server logs show
            start_new_session=True,
        )
        # server_process is the process - return so we can shut down later
    with open(f"/tmp/vllm_{run_cfg.server_port}.pid", "w") as f:
        f.write(str(server_process.pid))
    # kill with:
    # pid = int(open("/tmp/vllm_8000.pid").read())
    # os.killpg(pid, signal.SIGTERM)

    for i in range(10):  # 10 * 30 seconds = 5 minutes
        try:
            response = requests.get(f"http://localhost:{server_port}/v1/models")
            if response.status_code == 200:
                print(
                    f"[run_experiment.py:start_server_if_not_running] Server is running on port {server_port}"
                )
                # Now detach the server process so it continues after script ends
                print(
                    "[run_experiment.py:start_server_if_not_running] Detaching server process..."
                )
                return server_process
        except requests.exceptions.RequestException:
            # print(f"Server is not running on port {server_port}")
            print(
                f"[run_experiment.py:start_server_if_not_running] \
                Have waited {i / 2} minutes.  Will wait another {5 - i / 2} minutes"
            )
            time.sleep(30)  # 30 seconds

    print(
        f"Server did not start on port {server_port} after 5 minutes; giving up; terminating + killing"
    )
    server_process.terminate()
    server_process.kill()

    # Wait a moment for process to actually terminate
    try:
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # Force kill if it doesn't terminate gracefully
        server_process.kill()
        server_process.wait(timeout=5)

    # Assert that the server process was successfully aborted
    assert server_process.poll() is not None, (
        f"Failed to abort server process on port {server_port}; pid: {server_process.pid}"
    )
    print("Server process successfully aborted")
    return server_process


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
        n_non_activating=60,
    )

    sampler_cfg = SamplerConfig(
        # NB: for now, this split does NOT work for iterative - iterative manages its own train/test/holdout splits
        n_examples_train=50,  # 200
        n_examples_test=100,  # 100
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
        explainer_model="Qwen/Qwen2.5-32B-Instruct",
        explainer_model_max_len=8192 * 4,
        explainer_provider="offline",
        server_port=8000,  # if set, assumes local server is running.  'None' will start server
        explainer="iterative",
        scorers=["fuzz", "detection"],
        name="pythia-160m-iterative-100-latents",
        max_latents=100,
        filter_bos=True,
        num_gpus=2,
        verbose=True,
        num_examples_per_scorer_prompt=5,
        explainer_temperature=0.7,
        judge_scorer_index=1,  # used for both bestofk and iterative
        # bestofk only
        bestofk_num_explanations=3,
        bestofk_return_only_best=False,
        bestofk_run_all_scorers=True,
        bestofk_is_multishot=True,
        # iterative only
        iterative_num_rounds=5,
        iterative_holdout_ratio_of_total=0.1,
        iterative_test_ratio_of_nonholdout=0.75,
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
