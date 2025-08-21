import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import asyncio
from pathlib import Path
from typing import Callable

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

from crw_best_of_k_explainer import BestOfKExplainer, BestOfKExplainerScorer

# Import the new classes from crw_cls_overwrites.py
from crw_cls_overwrites import (
    ExplainerConfig,
    IntegratedExplainerScorer,
    IntegratedExplainerScorerConfig,
    MyRunConfig,
    OfflineClientConfig,
    ScorerConfig,
)
from delphi.config import (
    CacheConfig,
    ConstructorConfig,
    RunConfig,
    SamplerConfig,
)
from delphi.latents import LatentCache, LatentDataset  # , LatentRecord
from delphi.latents.neighbours import NeighbourCalculator
from delphi.log.result_analysis import log_results
from delphi.scorers import DetectionScorer, FuzzingScorer
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

    return run_cfg.hookpoints, hookpoint_to_sparse_encode, model, transcode


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
                autoencoder=saes[hookpoint].cuda(), number_of_neighbours=250
            )

        elif constructor_cfg.neighbours_type == "encoder_similarity":
            neighbour_calculator = NeighbourCalculator(
                autoencoder=saes[hookpoint].cuda(), number_of_neighbours=250
            )
        else:
            raise ValueError(
                f"Neighbour type {constructor_cfg.neighbours_type} not supported"
            )

        neighbour_calculator.populate_neighbour_cache(constructor_cfg.neighbours_type)
        neighbour_calculator.save_neighbour_cache(f"{neighbours_path}/{hookpoint}")


async def process_cache(
    run_cfg: MyRunConfig,
    latents_path: Path,
    neighbours_path: Path,
    explanations_path: Path,
    scores_path: Path,
    hookpoints: list[str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    latent_range: Tensor | None,
):
    """
    Converts SAE latent activations in on-disk cache in the `latents_path` directory
    to latent explanations in the `explanations_path` directory and explanation
    scores in the `scores_path` directory using the IntegratedExplainer.
    """
    explanations_path.mkdir(parents=True, exist_ok=True)
    scores_path.mkdir(parents=True, exist_ok=True)

    if latent_range is None:
        latent_dict = None
    else:
        latent_dict = {
            hook: latent_range for hook in hookpoints
        }  # The latent range to explain

    dataset = LatentDataset(
        raw_dir=str(latents_path),
        sampler_cfg=run_cfg.sampler_cfg,
        constructor_cfg=run_cfg.constructor_cfg,
        modules=hookpoints,
        latents=latent_dict,
        tokenizer=tokenizer,
        neighbours_path=str(neighbours_path),
    )

    # Create integrated explainer-scorer
    integrated_explainer = (
        run_cfg.integrated_explainer_scorer_cfg.integrated_explainer_scorer_cls(
            integrated_explainer_scorer_cfg=run_cfg.integrated_explainer_scorer_cfg,
        )
    )

    # Process all records using the integrated explainer
    results = await integrated_explainer(dataset)

    return results


def populate_cache(
    run_cfg: MyRunConfig,
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
    run_cfg: MyRunConfig,
    # start_latent: int,
    # non_active_to_show: int,
):
    base_path = run_cfg.base_path
    run_cfg_json_path = base_path / "run_config.json"
    # run_cfg_json_path.write_text(run_cfg.model_dump_json(indent=4))
    run_cfg_json_path.write_text("Skipping json dump for testing")

    # All latents will be in the first part of the name

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
            hookpoint_to_sparse_encode,
            latents_path,
            tokenizer,
            transcode,
        )

    del model, hookpoint_to_sparse_encode
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

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
            neighbours_path,
            explanations_path,
            scores_path,
            nrh,
            tokenizer,
            latent_range,
            # non_active_to_show,
        )

    if run_cfg.verbose:
        log_results(scores_path, visualize_path, run_cfg.hookpoints, run_cfg.scorers)


if __name__ == "__main__":
    # --- Make multiprocessing safe for CUDA before anything else touches CUDA ---
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    """    # Optional but often helpful on multi-GPU nodes; comment out if not needed.
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SHM_DISABLE", "1")"""

    print("Creating cache config")
    cache_cfg = CacheConfig(
        dataset_repo="EleutherAI/SmolLM2-135M-10B",
        cache_ctx_len=32,
        n_tokens=10_000_00,
    )

    print("Creating constructor config")
    constructor_cfg = ConstructorConfig(
        example_ctx_len=32,
        min_examples=20,
        n_non_activating=10,
    )

    print("Creating sampler config")
    sampler_cfg = SamplerConfig(
        n_examples_train=20,
        n_examples_test=10,
        n_quantiles=5,
    )

    print("Creating run config")
    VERBOSE = True
    EXPLAINER_CLS = BestOfKExplainer  # DefaultExplainer
    SCORER_CLSs = [DetectionScorer, FuzzingScorer]
    NUM_EXAMPLES_PER_SCORER_PROMPT = 5
    NAME = "pythia-70m-smoketest"
    EXPLAINER_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
    EXPLAINER_MODEL_MAX_LEN = 5120
    NUM_GPUS = 2
    ANALYZED_MODEL_NAME = "EleutherAI/pythia-70m"
    SPARSE_MODEL = "EleutherAI/sae-pythia-70m-32k"
    HOOKPOINTS = ["layers.5"]
    MAX_LATENTS = 10

    if EXPLAINER_CLS == BestOfKExplainer:
        INTEGRATED_EXPLAINER_SCORER_CLS = BestOfKExplainerScorer
        EXPLAINER_KWARGS = {"num_explanations": 3, "is_one_shot": True}
    else:
        INTEGRATED_EXPLAINER_SCORER_CLS = IntegratedExplainerScorer
        EXPLAINER_KWARGS = {}

    # typically not changed
    FILTER_BOS = True
    OVERWRITE = ["scores"]
    EXPLAINER_HIGHLIGHT_THRESHOLD = 0.3
    SCORER_LOG_PROB = True
    SCORER_TEMPERATURE = 0.0
    EXPLAINER_TEMPERATURE = 0.0
    CLIENT_MAX_MEMORY = 0.9

    base_path = Path.cwd().parent / "results" / NAME
    base_path.mkdir(parents=True, exist_ok=True)
    explanations_path = base_path / "explanations"
    explanations_path.mkdir(parents=True, exist_ok=True)
    scores_path = base_path / "scores"
    scores_path.mkdir(parents=True, exist_ok=True)

    # Shared client config for vLLM
    client_cfg = OfflineClientConfig(
        model_name=EXPLAINER_MODEL,
        max_memory=CLIENT_MAX_MEMORY,
        max_model_len=EXPLAINER_MODEL_MAX_LEN,
        num_gpus=NUM_GPUS,
        statistics=VERBOSE,
    )

    # Explainer config
    explainer_cfg = ExplainerConfig(
        explainer_cls=EXPLAINER_CLS,
        highlight_threshold=EXPLAINER_HIGHLIGHT_THRESHOLD,
        verbose=VERBOSE,
        temperature=EXPLAINER_TEMPERATURE,
        explanations_path=explanations_path,
        explainer_kwargs=EXPLAINER_KWARGS,
    )

    # Scorer configs
    scorer_cfgs = []
    for scorer_cls in SCORER_CLSs:
        scorer_path = scores_path / scorer_cls.__name__
        scorer_path.mkdir(parents=True, exist_ok=True)
        scorer_cfg = ScorerConfig(
            scorer_cls=scorer_cls,
            scorer_log_prob=SCORER_LOG_PROB,
            scorer_n_examples_shown=NUM_EXAMPLES_PER_SCORER_PROMPT,
            scorer_temperature=SCORER_TEMPERATURE,
            verbose=VERBOSE,
            scores_path=scorer_path,
        )
        scorer_cfgs.append(scorer_cfg)

    integrated_explainer_scorer_cfg = IntegratedExplainerScorerConfig(
        client_cfg=client_cfg,
        explainer_cfg=explainer_cfg,
        scorer_cfgs=scorer_cfgs,
        integrated_explainer_scorer_cls=INTEGRATED_EXPLAINER_SCORER_CLS,
    )

    run_cfg = MyRunConfig(
        cache_cfg=cache_cfg,
        constructor_cfg=constructor_cfg,
        sampler_cfg=sampler_cfg,
        integrated_explainer_scorer_cfg=integrated_explainer_scorer_cfg,
        model=ANALYZED_MODEL_NAME,
        sparse_model=SPARSE_MODEL,
        hookpoints=HOOKPOINTS,
        explainer_model=EXPLAINER_MODEL,
        explainer_model_max_len=EXPLAINER_MODEL_MAX_LEN,
        name=NAME,
        max_latents=MAX_LATENTS,
        filter_bos=FILTER_BOS,
        num_gpus=NUM_GPUS,
        overwrite=OVERWRITE,
        base_path=base_path,
    )

    asyncio.run(run(run_cfg))
