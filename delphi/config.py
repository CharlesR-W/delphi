from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Literal

import torch
from simple_parsing import Serializable, field, list_field


@dataclass
class SamplerConfig(Serializable):
    n_examples_train: int = 20
    """Number of activating examples in record.train.
    - BestOfK: Uses record.train directly for prompting (samples subset if needed)
    - Iterative: Uses record.train as pool to sample from each round for prompting + FP/FN collection"""

    n_examples_test: int = 50
    """Number of activating examples in record.test.
    - BestOfK: Used for scoring (along with record.not_active)
    - Iterative: Used as holdout set for final evaluation (along with record.not_active)"""

    n_quantiles: int = 10
    """Number of latent activation quantiles to sample."""

    train_type: Literal["top", "random", "quantiles", "mix"] = "quantiles"
    """Strategy to build the train/test activating pool."""

    test_type: Literal["quantiles"] = "quantiles"
    """Strategy to build the holdout activating pool."""

    ratio_top: float = 0.2
    """Ratio of top examples to use for training, if using mix."""


@dataclass
class ConstructorConfig(Serializable):
    faiss_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    """Embedding model to use for FAISS index."""

    faiss_embedding_cache_dir: str = ".embedding_cache"
    """Directory to store cached embeddings for FAISS similarity search."""

    faiss_embedding_cache_enabled: bool = True
    """Whether to cache embeddings for FAISS similarity search."""

    example_ctx_len: int = 32
    """Length of each sampled example sequence. Longer sequences
    reduce detection scoring performance in weak models.
    Has to be a multiple of the cache context length."""

    min_examples: int = 200
    """Minimum number of activating examples to generate for a single latent.
    If the number of examples is less than this, the
    latent will not be explained and scored."""

    n_non_activating: int = 50
    """Number of non-activating examples to be constructed."""

    center_examples: bool = True
    """Whether to center the examples on the latent activation.
    If True, the examples will be centered on the latent activation.
    Otherwise, windows will be used, and the activating example can be anywhere
    window."""

    non_activating_source: Literal["random", "neighbours", "FAISS"] = "random"
    """Source of non-activating examples. Random uses non-activating contexts
    sampled from any non activating window. Neighbours uses actvating contexts
    from pre-computed latent neighbours. FAISS uses semantic similarity search
    to find hard negatives that are semantically similar to activating examples
    but don't activate the latent."""

    neighbours_type: Literal[
        "co-occurrence", "decoder_similarity", "encoder_similarity"
    ] = "co-occurrence"
    """Type of neighbours to use. Only used if non_activating_source is 'neighbours'."""


@dataclass
class CacheConfig(Serializable):
    dataset_repo: str = "EleutherAI/SmolLM2-135M-10B"
    """Dataset repository to use for generating latent activations."""

    dataset_split: str = "train[:1%]"
    """Dataset split to use for generating latent activations."""

    dataset_name: str = ""
    """Dataset name to use."""

    dataset_column: str = "text"
    """Dataset row to use."""

    batch_size: int = 32
    """Number of sequences to process in a batch."""

    cache_ctx_len: int = 256
    """Context length for caching latent activations.
    Each batch is shape (batch_size, ctx_len).
    Must be divisible by ConstructorConfig.example_ctx_len for windowing."""

    n_tokens: int = 10_000_000
    """Number of tokens to cache."""

    n_splits: int = 5
    """Number of splits to divide .safetensors into."""


@dataclass
class RunConfig(Serializable):
    cache_cfg: CacheConfig

    constructor_cfg: ConstructorConfig

    sampler_cfg: SamplerConfig

    model: str = field(
        default="meta-llama/Meta-Llama-3-8B",
        positional=True,
    )
    """Name of the model to explain."""

    sparse_model: str = field(
        default="EleutherAI/sae-llama-3-8b-32x",
        positional=True,
    )
    """Name of sparse models associated with the model to explain, or path to
    directory containing their weights. Models must be loadable with sparsify
    or gemmascope."""

    hookpoints: list[str] = list_field()
    """list of model hookpoints to attach sparse models to."""

    explainer_model: str = field(
        default="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    )
    """Name of the model to use for explanation and scoring."""

    explainer_model_max_len: int = field(
        default=5120,
    )
    """Maximum length of the explainer model context window. For simulation scoring
    this length should be increased."""

    explainer_provider: str = field(
        default="offline",
    )
    """Provider to use for explanation and scoring. Options are 'offline' for local
    models and 'openrouter' for API calls."""

    explainer: str = field(
        choices=["default", "none", "bestofk", "iterative"],
        default="default",
    )
    """Explainer to use for generating explanations. Options are 'default' for
    the default single token explainer, and 'none' for no explanation generation."""

    scorers: list[str] = list_field(
        choices=[
            "fuzz",
            "detection",
            "simulation",
            "embedding",
        ],
        default=[
            "fuzz",
            "detection",
            "embedding",
        ],
    )
    """Scorer methods to score latent explanations. Options are 'fuzz', 'detection', and
    'simulation'."""

    embedding_model: str = field(
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    """SentenceTransformer model name/path to use for the embedding scorer."""

    explainer_temperature: float = field(default=0.0)
    """Temperature for generation."""

    name: str = ""
    """The name of the run. Results are saved in a directory with this name."""

    max_latents: int | None = None
    """Maximum number of features to explain for each sparse model."""

    filter_bos: bool = False
    """Whether to filter out BOS tokens from the cache."""

    log_probs: bool = False
    """Whether to attempt to gather log probabilities for each scorer prompt."""

    load_in_8bit: bool = False
    """Load the model in 8-bit mode."""

    # Use a dummy encoding function to prevent the token from being saved
    # to disk in plain text
    hf_token: str | None = field(default=None, encoding_fn=lambda _: None)
    """Huggingface API token for downloading models."""

    pipeline_num_proc: int = field(
        default_factory=lambda: cpu_count() // 2,
    )
    """Number of processes to use for preprocessing data"""

    num_gpus: int = field(
        default=torch.cuda.device_count(),
    )
    """Number of GPUs to use for explanation and scoring."""

    max_memory_utilization: float = field(
        default=0.9,
    )
    """Maximum memory utilization for the vLLM server. 0.0-1.0.  Used to avoid OOM errors."""

    seed: int = field(
        default=22,
    )
    """Seed for the random number generator."""

    verbose: bool = field(
        default=True,
    )
    """Whether to log summary statistics and results of the run."""

    num_examples_per_scorer_prompt: int = field(
        default=5,
    )
    """Number of examples to use for each scorer prompt. Using more than 1 improves
    scoring speed but can leak information to the fuzzing and detection scorer,
    as well as increasing the scorer LLM task difficulty."""

    overwrite: list[Literal["cache", "neighbours", "scores"]] = list_field(
        choices=["cache", "neighbours", "scores"],
        default=[],
    )
    """List of run stages to recompute. This is a debugging tool
    and may be removed in the future."""

    server_port: int | None = field(
        default=None,
    )
    """Port to use for the vLLM server."""

    server_metrics_port: int | None = field(default=None)
    """Optional Prometheus metrics port for the vLLM server. Defaults to server_port + 1."""

    enable_prefix_caching: bool = field(default=True)
    """Enable prefix caching in vLLM server for improved performance."""

    enforce_eager: bool = field(default=False)
    """Enforce eager execution in vLLM server (disables CUDA graphs)."""

    use_random_baseline: bool = field(default=False)
    """If True, Best-of-K/Iterative will load explanations from a prior run at random."""

    random_baseline_source_run: str | None = field(default=None)
    """When using the random baseline explainer, read explanations from this prior run name."""

    # BestOfK-specific configuration
    bestofk_num_explanations: int = field(default=3)
    """Number of explanations to generate when using the BestOfK explainer."""

    bestofk_judge_scorer_index: int = field(default=0)
    """Index of the scorer to use for selecting the best explanation in BestOfK."""

    bestofk_return_only_best: bool = field(default=True)
    """Whether to return only the best explanation or all explanations."""

    bestofk_run_all_scorers: bool = field(default=True)
    """Whether to run all scorers on all explanations or just the judge scorer."""

    bestofk_is_multishot: bool = field(default=True)
    """Whether to generate multiple explanations from multiple prompts (multishot) 
    or parse multiple from a single prompt (oneshot)."""
    
    bestofk_num_train_examples: int = field(default=20)
    """Number of train examples to show to the model in BestOfK. Default 20, can use 40."""

    bestofk_embedding_prefilter_enabled: bool = field(default=False)
    """If True, use embedding scorer to rank all explanations and only run other scorers on top-K."""

    bestofk_embedding_prefilter_top_k: int = field(default=10)
    """How many embedding-ranked explanations to score with expensive scorers."""

    bestofk_embedding_use_as_judge: bool = field(default=False)
    """If True, force the embedding scorer to act as the judge for BestOfK selection."""

    # Iterative-specific configuration
    iterative_num_rounds: int = field(default=3)
    """Number of iterative refinement rounds for the iterative explainer."""

    iterative_max_num_false_positives: int = field(default=20)
    """Maximum number of false positive extra examples to include when refining prompts."""

    iterative_max_num_false_negatives: int = field(default=20)
    """Maximum number of false negative extra examples to include when refining prompts."""

    iterative_max_num_true_positives: int = field(default=0)
    """Maximum number of true positive extra examples to include when refining prompts."""

    iterative_max_num_true_negatives: int = field(default=0)
    """Maximum number of true negative extra examples to include when refining prompts."""

    iterative_carryforward_strategy: Literal["best", "last"] = field(default="last")
    """When carrying forward explanation text to the next round, use the
    best-so-far (judged on test set) or the last round's explanation."""

    iterative_allow_tp_examples: bool = field(default=True)
    """If False, TP examples are omitted from iterative refinement prompts."""

    iterative_show_score_to_explainer: bool = field(default=False)
    """If True, include the previous round's score in the explainer prompt."""

    iterative_history_only: bool = field(default=False)
    """If True, show only prior explanations (and scores if enabled) to the
    explainer; do not show examples. Overrides other flags."""

    iterative_always_new_train_examples: bool = field(default=False)
    """If True, sample new train/test subsets from the pools each round;
    if False, reuse the same subsets across rounds."""

    iterative_append_round_to_prompt: bool = field(default=False)
    """Whether to append the round number to the prompt to encourage diversity."""
    
    iterative_num_train_examples_per_round: int = field(default=20)
    """Number of train examples to show per round in Iterative. Default 20, can use 40."""

    judge_scorer_index: int = field(default=0)
    """Index of the scorer to use for judging iterative explanations and selecting 
    the best explanation across rounds."""
