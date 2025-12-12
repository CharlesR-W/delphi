import asyncio
from collections.abc import AsyncIterable, Awaitable, Callable
from functools import wraps
from typing import Any

from tqdm.asyncio import tqdm


def process_wrapper(
    function: Callable[..., Awaitable],
    preprocess: Callable | None = None,
    postprocess: Callable | None = None,
) -> Callable[..., Awaitable]:
    """
    Wraps a function with optional preprocessing and postprocessing steps.

    Args:
        function (Callable): The main function to be wrapped.
        preprocess (Callable, optional): A function to preprocess the input.
            Defaults to None.
        postprocess (Callable, optional): A function to postprocess the output.
            Defaults to None.

    Returns:
        Callable: The wrapped function.
    """

    @wraps(function)
    async def wrapped(input: Any):
        fname = getattr(function, '__name__', str(type(function).__name__))
        # print(f"[process_wrapper] Starting {fname}")
        if preprocess is not None:
            # print(f"[process_wrapper] Applying preprocess to {fname}")
            input = preprocess(input)

        # print(f"[process_wrapper] Calling {fname}")
        results = await function(input)
        # print(f"[process_wrapper] {fname} completed with result type: {type(results)}")

        if postprocess is not None:
            print(f"[process_wrapper DEBUG] About to call postprocess for {fname}, result type: {type(results)}")
            # print(f"[process_wrapper] Applying postprocess to {fname}")
            results = postprocess(results)
            print(f"[process_wrapper DEBUG] Postprocess completed for {fname}")

        # print(f"[process_wrapper] {fname} finished")
        return results

    return wrapped


class Pipe:
    """
    Represents a pipe of functions to be executed with the same input.
    """

    def __init__(self, *functions: Callable):
        """
        Initialize the Pipe with a list of functions.

        Args:
            *functions (list[Callable]): Functions to be executed in the pipe.
        """
        self.functions = functions

    async def __call__(self, input: Any) -> list[Any]:
        """
        Execute all functions in the pipe with the given input.

        Args:
            input (Any): The input to be processed by all functions.

        Returns:
            list[Any]: The results of all functions.
        """
        print(f"[Pipe DEBUG] Running {len(self.functions)} functions in parallel")
        tasks = [function(input) for function in self.functions]

        results = await asyncio.gather(*tasks)
        print(f"[Pipe DEBUG] Gathered {len(results)} results from {len(self.functions)} functions")
        for i, result in enumerate(results):
            print(f"[Pipe DEBUG]   Result {i}: type={type(result)}, value={result if not hasattr(result, 'score') else f'ScorerResult(score_len={len(result.score) if result.score else 0})'}")
        return results


class Pipeline:
    """
    Manages the execution of multiple pipes, handling concurrency and progress tracking.
    """

    def __init__(self, loader: AsyncIterable | Callable, *pipes: Pipe | Callable):
        """
        Initialize the Pipeline with a list of pipes.

        Args:
            loader (Callable): The loader to be executed first.
            *pipes (list[Pipe | Callable]): Pipes to be executed in the pipeline.
        """

        self.loader = loader
        self.pipes = pipes

    async def run(self, max_concurrent: int = 10) -> list[Any]:
        """
        Run the pipeline with a maximum number of concurrent tasks.

        Args:
            max_concurrent: Maximum number of concurrent tasks. Defaults to 10.

        Returns:
            list[Any]: The results of all processed items.
        """
        print(f"[Pipeline] Starting pipeline with max_concurrent={max_concurrent}")
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = set()

        progress_bar = tqdm(desc="Processing items")
        number_of_items = 0

        async def process_and_update(item, semaphore):
            result = await self.process_item(item, semaphore)
            progress_bar.update(1)
            return result

        # Process items
        async for item in self.generate_items():
            number_of_items += 1
            task = asyncio.create_task(process_and_update(item, semaphore))
            tasks.add(task)

            if len(tasks) >= max_concurrent:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                results.extend(task.result() for task in done)
                tasks = pending

        if tasks:
            done, _ = await asyncio.wait(tasks)
            results.extend(task.result() for task in done)

        progress_bar.close()
        # print(f"[Pipeline] Completed: processed {number_of_items} items, returning {len(results)} results")
        return results

    async def generate_items(self) -> AsyncIterable[Any]:
        """
        Generates items from the first pipe, which can be an async iterable or callable

        Yields:
            Any: Items generated from the first pipe.

        Raises:
            TypeError: If the first pipe is neither an async iterable nor a callable.
        """
        # print(f"[Pipeline] generate_items: loader type = {type(self.loader)}")
        if isinstance(self.loader, AsyncIterable):
            # print("[Pipeline] generate_items: Using async iterable")
            async for item in self.loader:
                # print("[Pipeline] generate_items: Yielding item from async iterable")
                yield item
        elif callable(self.loader):
            # print("[Pipeline] generate_items: Using callable")
            for item in self.loader():
                # print("[Pipeline] generate_items: Yielding item from callable")
                yield item
                await asyncio.sleep(0)  # Allow other coroutines to run
        else:
            raise TypeError("The first pipe must be an async iterable or a callable")

    async def process_item(self, item: Any, semaphore: asyncio.Semaphore) -> Any:
        """
        Processes a single item through all pipes except the first one.

        Args:
            item (Any): The item to be processed.
            semaphore (asyncio.Semaphore): Semaphore for controlling concurrency.

        Returns:
            Any: The processed item.
        """
        # print("[Pipeline] process_item: Acquiring semaphore")
        async with semaphore:
            # print(
            # f"[Pipeline] process_item: Semaphore acquired, processing through {len(self.pipes)} pipes"
            # )
            result = item
            for i, pipe in enumerate(self.pipes):
                # print(
                # f"[Pipeline] process_item: Processing through pipe {i + 1}/{len(self.pipes)}"
                # )
                if result is not None:
                    result = await pipe(result)
                    # print(
                    # f"[Pipeline] process_item: Completed pipe {i + 1}/{len(self.pipes)}"
                    # )

                else:
                    # print(
                    # f"[Pipeline] process_item: Skipping pipe {i + 1}/{len(self.pipes)} (result is None)"
                    # )
                    pass
        # print("[Pipeline] process_item: Completed processing, releasing semaphore")
        return result
