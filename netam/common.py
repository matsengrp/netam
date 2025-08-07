from collections import Counter
import inspect
import resource
import subprocess
from tqdm import tqdm
from functools import wraps
from itertools import islice, repeat

import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
import multiprocessing as mp


BIG = 1e9
SMALL_PROB = 1e-10


def zap_predictions_along_diagonal(predictions, aa_parents_idxs, fill=-BIG):
    """Set the diagonal (i.e. no amino acid change) of the predictions tensor to -BIG,
    except where aa_parents_idxs >= 20, which indicates no update should be done."""

    device = predictions.device
    batch_size, L, _ = predictions.shape
    batch_indices = torch.arange(batch_size, device=device)[:, None].expand(-1, L)
    sequence_indices = torch.arange(L, device=device)[None, :].expand(batch_size, -1)

    # Create a mask for valid positions (where aa_parents_idxs is less than 20)
    valid_mask = aa_parents_idxs < 20

    # Only update the predictions for valid positions
    predictions[
        batch_indices[valid_mask],
        sequence_indices[valid_mask],
        aa_parents_idxs[valid_mask],
    ] = fill

    return predictions


def combine_and_pad_tensors(first, second, padding_idxs, fill=float("nan")):
    res = torch.full(
        (first.shape[0] + second.shape[0] + len(padding_idxs),) + first.shape[1:], fill
    )
    mask = torch.full((res.shape[0],), True, dtype=torch.bool)
    if len(padding_idxs) > 0:
        mask[torch.tensor(padding_idxs)] = False
    res[mask] = torch.concat([first, second], dim=0)
    return res


def force_spawn():
    """Force the spawn start method for multiprocessing.

    This is necessary to avoid conflicts with the internal OpenMP-based thread pool in
    PyTorch.
    """
    mp.set_start_method("spawn", force=True)


def informative_site_count(seq_str):
    return sum(c != "N" for c in seq_str)


def clamp_probability(x: Tensor) -> Tensor:
    return torch.clamp(x, min=SMALL_PROB, max=(1.0 - SMALL_PROB))


def clamp_probability_above_only(x: Tensor) -> Tensor:
    return torch.clamp(x, max=(1.0 - SMALL_PROB))


def clamp_log_probability(x: Tensor) -> Tensor:
    return torch.clamp(x, max=np.log(1.0 - SMALL_PROB))


def print_parameter_count(model):
    total = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only count parameters in leaf modules
            num_params = sum(p.numel() for p in module.parameters())
            print(f"{name}: {num_params} parameters")
            total += num_params
    print("-----")
    print(f"total: {total} parameters")


def parameter_count_of_model(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def stack_heterogeneous(tensors, pad_value=0.0):
    """Stack an iterable of 1D or 2D torch.Tensor objects of different lengths along the
    first dimension into a single tensor.

        black --check netam tests
    Args:
        tensors (iterable): An iterable of 1D or 2D torch.Tensor objects with variable lengths in the first dimension.
        pad_value (number): The value used for padding shorter tensors. Default is 0.

    Returns:
        torch.Tensor: A stacked tensor with all input tensors padded to the length of the longest tensor in the first dimension.
    """
    if tensors is None or len(tensors) == 0:
        return torch.Tensor()  # Return an empty tensor if no tensors are provided

    dim = tensors[0].dim()
    if dim not in [1, 2]:
        raise ValueError("This function only supports 1D or 2D tensors.")

    max_length = max(tensor.size(0) for tensor in tensors)

    if dim == 1:
        # If 1D, simply pad the end of the tensor.
        padded_tensors = [
            torch.nn.functional.pad(
                tensor, (0, max_length - tensor.size(0)), value=pad_value
            )
            for tensor in tensors
        ]
    else:
        # If 2D, pad the end of the first dimension (rows); the argument to pad
        # is a tuple of (padding_left, padding_right, padding_top,
        # padding_bottom)
        padded_tensors = [
            torch.nn.functional.pad(
                tensor, (0, 0, 0, max_length - tensor.size(0)), value=pad_value
            )
            for tensor in tensors
        ]

    return torch.stack(padded_tensors)


def optimizer_of_name(optimizer_name, model_parameters, **kwargs):
    """Build a torch.optim optimizer from a string name and model parameters.

    Use a SGD optimizer with momentum if the optimizer_name is "SGDMomentum".
    """
    if optimizer_name == "SGDMomentum":
        optimizer_name = "SGD"
        kwargs["momentum"] = 0.9
    try:
        optimizer_class = getattr(optim, optimizer_name)
        return optimizer_class(model_parameters, **kwargs)
    except AttributeError:
        raise ValueError(
            f"Optimizer '{optimizer_name}' is not recognized in torch.optim"
        )


def find_least_used_cuda_gpu(mem_round_val=1300):
    """Determine the least used CUDA GPU based on utilization, then allocated memory,
    then number of running processes.

    If all GPUs are idle, return None. When choosing the GPU by memory, memory usage is
    rounded to the nearest multiple of mem_round_val.
    """
    # Query GPU utilization and memory usage in a single call
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=gpu_uuid,utilization.gpu,memory.used",
            "--format=csv,nounits,noheader",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print("Error running nvidia-smi.")
        return None

    lines = result.stdout.strip().split("\n")
    gpu_data = [line.split(", ") for line in lines]
    uuids = [gpu[0] for gpu in gpu_data]

    utilization = [int(gpu[1]) for gpu in gpu_data]
    memory_used = [
        int(gpu[2]) // mem_round_val for gpu in gpu_data
    ]  # Round memory usage

    # Query process count in a single call
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,name",
            "--format=csv,nounits,noheader",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print("Error running nvidia-smi.")
        return None

    process_entries = [
        line.split(", ") for line in result.stdout.strip().split("\n") if line
    ]

    # Count the number of processes per GPU
    gpu_counts = Counter({uuid: 0 for uuid in uuids})
    gpu_counts.update(
        [uuid for uuid, proc_name in process_entries if proc_name != "[Not Found]"]
    )

    # Map UUIDs to GPU indices
    uuid_to_index = {uuid: idx for idx, uuid in enumerate(uuids)}

    # Check prioritization order:
    if max(utilization) > 0:
        print("GPU chosen via utilization")
        return utilization.index(min(utilization))  # Least utilized GPU

    if max(memory_used) > 0:
        print("GPU chosen via memory")
        return memory_used.index(min(memory_used))  # Least memory used GPU

    if len(set(gpu_counts.values())) > 1:
        print("GPU chosen via process count")
        return min(
            uuid_to_index[uuid]
            for uuid, count in gpu_counts.items()
            if count == min(gpu_counts.values())
        )

    return None  # All GPUs are idle


def pick_device(gpu_preference=None):
    """Pick a device for PyTorch to use.

    If gpu_preference is a string, use the device with that name. This is considered a
    strong preference from a user who knows what they are doing.

    If gpu_preference is an integer, this is a weak preference for a numbered GPU.  If
    CUDA is available, use the least used GPU, and if all are idle use the gpu_index
    modulo the number of GPUs. If gpu_index is None, then use a random GPU.
    """

    # Strong preference for a specific device.
    if gpu_preference is not None and isinstance(gpu_preference, str):
        return torch.device(gpu_preference)

    # else weak preference for a numbered GPU.

    # check that CUDA is usable
    def check_CUDA():
        try:
            torch._C._cuda_init()
            return True
        except Exception:
            return False

    if torch.backends.cudnn.is_available() and check_CUDA():
        which_gpu = find_least_used_cuda_gpu()
        if which_gpu is None:
            if gpu_preference is None:
                which_gpu = np.random.randint(torch.cuda.device_count())
            else:
                which_gpu = gpu_preference % torch.cuda.device_count()
        print(f"Using CUDA GPU {which_gpu}")
        dev = torch.device(f"cuda:{which_gpu}")
        torch.ones(1).to(dev)
        return dev
    elif torch.backends.mps.is_available():
        print("Using Metal Performance Shaders")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")


def print_tensor_devices(scope="local"):
    """Print the devices of all PyTorch tensors in the given scope.

    Args:
        scope (str): 'local' for local scope, 'global' for global scope.
    """
    if scope == "local":
        frame = inspect.currentframe()
        variables = frame.f_back.f_locals
    elif scope == "global":
        variables = globals()
    else:
        raise ValueError("Scope must be 'local' or 'global'.")

    for var_name, var_value in variables.items():
        if isinstance(var_value, torch.Tensor):
            print(f"{var_name}: {var_value.device}")


def get_memory_usage_mb():
    # Returns the peak memory usage in MB
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024  # Convert from KB to MB


def tensor_to_np_if_needed(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        assert isinstance(x, np.ndarray)
        return x


def linear_bump_lr(epoch, warmup_epochs, total_epochs, max_lr, min_lr):
    """Linearly increase the learning rate from min_lr to max_lr over warmup_epochs,
    then linearly decrease the learning rate from max_lr to min_lr.

    See https://github.com/matsengrp/netam/pull/41 for more details.

    Example:
    .. code-block:: python
        pd.Series([linear_bump_lr(epoch, warmup_epochs=20, total_epochs=200, max_lr=0.01, min_lr=1e-5) for epoch in range(200)]).plot()
    """
    if epoch < warmup_epochs:
        lr = min_lr + ((max_lr - min_lr) / warmup_epochs) * epoch
    else:
        lr = max_lr - ((max_lr - min_lr) / (total_epochs - warmup_epochs)) * (
            epoch - warmup_epochs
        )
    return lr


# from https://docs.python.org/3.11/library/itertools.html#itertools-recipes
# avoiding walrus:
def chunked(iterable, n):
    "Chunk data into lists of length n. The last chunk may be shorter."
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            return
        yield chunk


def heavy_chain_shim(paired_evaluator):
    """Returns a function that evaluates only heavy chains given a paired evaluator."""

    def evaluate_heavy_chains(sequences):
        paired_seqs = [[h, ""] for h in sequences]
        paired_outputs = paired_evaluator(paired_seqs)
        return [output[0] for output in paired_outputs]

    return evaluate_heavy_chains


def light_chain_shim(paired_evaluator):
    """Returns a function that evaluates only light chains given a paired evaluator."""

    def evaluate_light_chains(sequences):
        paired_seqs = [["", light] for light in sequences]
        paired_outputs = paired_evaluator(paired_seqs)
        return [output[1] for output in paired_outputs]

    return evaluate_light_chains


def chunk_function(
    first_chunkable_idx=0, default_chunk_size=2048, progress_bar_name=None
):
    """Decorator to chunk the input to a function.

    Expects that all positional arguments are iterables of the same length,
    and that outputs are tuples of tensors whose first dimension
    corresponds to the first dimension of the input iterables.

    If function returns just one item, it must not be a tuple.

    Chunking is done along the first dimension of all inputs.

    Args:
        default_chunk_size: The default chunk size. The decorated function can
            also automatically accept a `default_chunk_size` keyword argument.
        progress_bar_name: The name of the progress bar. If None, no progress bar is shown.
    """

    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            if "chunk_size" in kwargs:
                chunk_size = kwargs.pop("chunk_size")
            else:
                chunk_size = default_chunk_size
            pre_chunk_args = args[:first_chunkable_idx]
            chunkable_args = args[first_chunkable_idx:]

            results = []
            if progress_bar_name is None:
                progargs = {"disable": True}
            else:
                progargs = {"desc": progress_bar_name}
            bar = tqdm(total=len(chunkable_args[0]), delay=2.0, **progargs)
            for chunked_args in zip(
                *(chunked(arg, chunk_size) for arg in chunkable_args)
            ):
                bar.update(len(chunked_args[0]))
                results.append(function(*pre_chunk_args, *chunked_args, **kwargs))
            if isinstance(results[0], tuple):
                return tuple(torch.cat(tensors) for tensors in zip(*results))
            else:
                return torch.cat(results)

        return wrapper

    return decorator


def _apply_args_and_kwargs(func, pre_chunk_args, chunked_args, kwargs):
    return func(*pre_chunk_args, *chunked_args, **kwargs)


def parallelize_function(
    function,
    first_chunkable_idx=0,
    max_workers=10,
    min_chunk_size=1000,
):
    """Function to parallelize another function's application with multiprocessing.

    This is intentionally not designed to be used with decorator syntax because it should only
    be used when the function it is applied to will be run on the CPU.

    Expects that all positional arguments are iterables of the same length,
    and that outputs are tuples of tensors whose first dimension
    corresponds to the first dimension of the input iterables.

    If function returns just one item, it must not be a tuple.

    Division between processes is done along the first dimension of all inputs.
    The wrapped function will be endowed with the parallelize keyword
    argument, so that parallelization can be turned on or off at each invocation.

    Args:
        function: The function to be parallelized.
        first_chunkable_idx: The index of the first argument to be chunked.
            All positional arguments after this index will be chunked.
        max_workers: The maximum number of processes to use.
        min_chunk_size: The minimum chunk size for input data. The number of
            workers is adjusted to ensure that the chunk size is at least this.
    """

    max_worker_count = min(mp.cpu_count() // 2, max_workers)
    if max_worker_count <= 1:
        return function
    force_spawn()

    @wraps(function)
    def wrapper(*args, **kwargs):
        if len(args) <= first_chunkable_idx:
            raise ValueError(
                f"Function {function.__name__} cannot be parallelized without chunkable arguments"
            )
        pre_chunk_args = args[:first_chunkable_idx]
        chunkable_args = args[first_chunkable_idx:]
        min_worker_count = len(chunkable_args[0]) // min_chunk_size

        worker_count = min(min_worker_count, max_worker_count)
        if worker_count <= 1:
            return function(*args, **kwargs)

        chunk_size = (len(chunkable_args[0]) // worker_count) + 1
        chunked_args = list(zip(*(chunked(arg, chunk_size) for arg in chunkable_args)))
        with mp.Pool(worker_count) as pool:
            results = pool.starmap(
                _apply_args_and_kwargs,
                list(
                    zip(
                        repeat(function),
                        repeat(pre_chunk_args),
                        chunked_args,
                        repeat(kwargs),
                    )
                ),
            )
        if isinstance(results[0], tuple):
            return tuple(torch.cat(tensors) for tensors in zip(*results))
        else:
            return torch.cat(results)

    return wrapper


def _apply_func_to_df_chunk(df_chunk, func, args, kwargs):
    """Apply function to a DataFrame chunk using pandas apply."""
    if kwargs.pop("use_progress_apply", False):
        return df_chunk.progress_apply(func, axis=1, args=args, **kwargs)
    else:
        return df_chunk.apply(func, axis=1, args=args, **kwargs)


def parallel_df_apply(
    df,
    func,
    max_workers=10,
    min_chunk_size=1000,
    parallelize=True,
    force_parallel=None,
    *args,
    **kwargs,
):
    """Apply a function to DataFrame rows in parallel.

    The function receives a row of the Dataframe as input.

    Each process receives a subset of the DataFrame and uses pandas apply() on it.
    Preserves the original DataFrame index, including sparse/non-contiguous indices.

    Args:
        df: DataFrame to apply function to.
        func: Function to apply to each row/column.
        max_workers: Maximum number of processes to use.
        min_chunk_size: Minimum chunk size for parallelization.
        parallelize: Whether to use parallel processing.
        use_progress_apply: If True, use tqdm to show progress.
        force_parallel: Provide an integer number of workers to force parallelization.
        *args: Additional positional arguments to pass to func.
        **kwargs: Additional keyword arguments to pass to func.

    Returns:
        Series of results from applying func with original index preserved.
    """
    data_length = len(df)

    max_worker_count = min(mp.cpu_count() // 2, max_workers)
    if (force_parallel is None) and (
        not parallelize or data_length < min_chunk_size or max_worker_count <= 1
    ):
        print("using sequential processing")
        # Fall back to sequential processing.
        if kwargs.pop("use_progress_apply", False):
            return df.progress_apply(func, axis=1, args=args, **kwargs)
        else:
            return df.apply(func, axis=1, args=args, **kwargs)

    force_spawn()

    if force_parallel is not None:
        worker_count = force_parallel
    else:
        # Calculate optimal worker count based on data size.
        min_worker_count = data_length // min_chunk_size
        worker_count = min(min_worker_count, max_worker_count)

    chunk_size = (data_length // worker_count) + 1

    # Create DataFrame chunks preserving index.
    # Chunk rows by position but preserve original index.
    position_chunks = list(chunked(range(len(df)), chunk_size))
    df_chunks = [df.iloc[chunk_positions] for chunk_positions in position_chunks]

    with mp.Pool(worker_count) as pool:
        results = pool.starmap(
            _apply_func_to_df_chunk,
            list(zip(df_chunks, repeat(func), repeat(args), repeat(kwargs))),
        )

    # Concatenate results preserving original index order.
    # pd.concat maintains the index from each chunk, so sparse indices are preserved.
    return pd.concat(results)


def create_optimized_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    collate_fn=None,
    num_workers: int = 2,
) -> DataLoader:
    """Create a DataLoader with optimizations for GPU training.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size for the dataloader
        shuffle: Whether to shuffle the data
        collate_fn: Optional collate function
        num_workers: Number of worker processes for data loading
    Returns:
        DataLoader with GPU optimization settings
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available(),
    )
