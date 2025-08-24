"""
Benchmarking setup
"""

from datetime import datetime
import json
from typing import Any, Callable
from collections.abc import Iterator
import triton
from dataclasses import dataclass, field, asdict
import torch
from pydantic import BaseModel
import inspect
import logging

JitType = triton.JITFunction[Any]

logger = logging.getLogger("Bench")
logger.setLevel(logging.INFO)

str_to_dtype = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass
class BenchConfig:
    block_sizes: list[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048])
    dtypes: list[str] = field(
        default_factory=lambda: ["float16", "bfloat16", "float32"]
    )
    num_warps: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    iters: int = field(default_factory=lambda: 10)


default_config = BenchConfig()


class TritonProgram:
    """
    Defines a triton program to be tested with specific settings.
    """

    def __init__(
        self,
        program: JitType,
        bench_runner: Callable[..., Any],
        name: str | None = None,
        bench_runner_args: dict[str, Any] = {},
    ):
        self.bench_runner = bench_runner
        # name of the program. If not provided, the name will be the name of the function.
        self.name = name or program.__name__
        self.program = program
        self.bench_runner_args = bench_runner_args


class ProgramBenchRunnerArgs:
    """
    Creates a list of args for running a program with all the possible combinations of the bench config along with this program's args.
    """

    def __init__(self, name: str, args: dict[str, Any], bench_config: BenchConfig):
        self.name = name
        self.args = args
        self.bench_config = bench_config
        # make sure that all the values in the bench runner args dict are the same length
        keys = list(args.keys())
        values = list(args.values())
        for key, value in zip(keys, values):
            if len(value) != len(values[0]):
                raise ValueError(
                    f"All arguments in the list of bench runner args must be the same length. Mismatch in {name} and arg: {key}. Found lengths: {[len(args) for args in args.values()]}"
                )
            if not isinstance(value, list):
                raise ValueError(
                    f"All arguments in the list of bench runner args must be a list. Mismatch in {name} and arg: {key}. Found type: {type(value)}"
                )

    def __getitem__(self, key: str) -> Any:
        return self.args[key]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        # return an iterator over the args
        # each arg is a key in the args dict. with a list of values
        # we create iterator of length of any of the lists
        iter_length = len(list(self.args.values())[0])
        for i in range(iter_length):
            for block_size in self.bench_config.block_sizes:
                for dtype in self.bench_config.dtypes:
                    for num_warps in self.bench_config.num_warps:
                        args = {key: value[i] for key, value in self.args.items()}
                        args["block_size"] = block_size
                        args["dtype"] = str_to_dtype[dtype]
                        args["num_warps"] = num_warps
                        yield args

    def __len__(self) -> int:
        return (
            len(self.args)
            * len(self.bench_config.block_sizes)
            * len(self.bench_config.dtypes)
            * len(self.bench_config.num)
        )

    def __repr__(self) -> str:
        return f"ProgramBenchRunnerArgs(name={self.name}, args={self.args})"


class Time:
    def __init__(self, name: str, time: float, args: dict[str, Any]):
        self.name = name
        self.time = time
        self.args = args

    def __str__(self) -> str:
        return (
            f"\n\nTime(\n  name={self.name}\n  time={self.time}\n  args={self.args}\n)"
        )

    def __repr__(self) -> str:
        return self.__str__()


class BenchResult:
    def __init__(self, times: list[Time], config: BenchConfig):
        self.times: list[Time] = times
        # pre calculate average time for each program
        self.avg_times: dict[str, float] = {
            name: sum(time.time for time in self.times if time.name == name)
            / len([time for time in self.times if time.name == name])
            for name in set(time.name for time in self.times)
        }
        self.config = config

    def __str__(self) -> str:
        return f"BenchResult(average_times={self.avg_times})"

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(self, key: list[str] | str) -> "BenchResult":
        if isinstance(key, str):
            key = [key]

        filtered_times = [time for time in self.times if time.name in key]

        return BenchResult(filtered_times, self.config)

    def __len__(self) -> int:
        return len(self.times)

    def __add__(self, other: "BenchResult") -> "BenchResult":
        return BenchResult(self.times + other.times, self.config)

    def __repr__(self) -> str:
        return f"BenchResult(times={self.times})"

    def save(self, file_path: str | None = None) -> None:
        """
        Save the bench result to a file.
        """
        if file_path is None:
            file_path = f"bench_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # serialize complex objects
        times_serialized = []
        for t in self.times:
            ser_args = {
                k: (str(v) if isinstance(v, torch.dtype) else v)
                for k, v in t.args.items()
            }
            times_serialized.append(
                {
                    "name": t.name,
                    "time": float(t.time),
                    "args": ser_args,
                }
            )

        with open(file_path, "w") as f:
            json.dump(
                {
                    "config": asdict(self.config),
                    "times": times_serialized,
                    "avg_times": {k: float(v) for k, v in self.avg_times.items()},
                },
                f,
                indent=2,
            )


class Bench:
    def __init__(
        self, programs: list[TritonProgram], config: BenchConfig = default_config
    ):
        self.config = config
        # get the names of the programs using inspect.signature
        self.program_names = [
            program.name or program.program.__name__ for program in programs
        ]
        self.programs = programs
        # ensure that each program's bench runner takes block_size, dtype, and num_warps
        for program in programs:
            signature = inspect.signature(program.bench_runner)
            missing_args = []
            if "block_size" not in signature.parameters:
                missing_args.append("block_size")
            if "dtype" not in signature.parameters:
                missing_args.append("dtype")
            if "num_warps" not in signature.parameters:
                missing_args.append("num_warps")
            if missing_args:
                raise Exception(
                    f"Each program's bench runner must take block_size, dtype, and num_warps as arguments. Program {program.name} has missing args: {missing_args}.\nSignature found: {signature}"
                )

        self.program_bench_runner_args = {
            program.name: ProgramBenchRunnerArgs(
                program.name, program.bench_runner_args, self.config
            )
            for program in programs
        }

    def run_one(self, program: TritonProgram, args: dict[str, Any]) -> BenchResult:
        """
        Run a program with a specific setting.
        """
        times = 0.0
        # run it once with the args to warm up the program
        program.bench_runner(**args)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(self.config.iters):
            program.bench_runner(**args)
            torch.cuda.synchronize()

        end.record()

        end.synchronize()

        avg_time = start.elapsed_time(end) / self.config.iters

        return BenchResult([Time(program.name, avg_time, args)], self.config)

    def run(self, debug: bool = False) -> BenchResult:
        """
        Run all programs with all settings.
        """

        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        runs = BenchResult([], self.config)

        for program in self.programs:
            for args in self.program_bench_runner_args[program.name]:
                logger.debug(f"Running {program.name} with args: {args}")
                runs += self.run_one(program, args)

        return runs


if __name__ == "__main__":
    config = BenchConfig(
        block_sizes=[128, 256],
        dtypes=["float16", "bfloat16", "float32"],
        num_warps=[1, 2, 4, 8],
    )
    Ms = [200, 400]
    Ns = [100, 200]

    from softmax.online_softmax import (
        bench_runner as online_softmax_bench_runner,
        online_softmax,
    )
    from softmax.simple_softmax import (
        bench_runner as simple_softmax_bench_runner,
        simple_softmax,
    )

    bench = Bench(
        config=config,
        programs=[
            # online softmax takes: x_ptr, output_ptr, M, N
            TritonProgram(
                bench_runner=online_softmax_bench_runner,
                program=online_softmax,
                bench_runner_args={"M": Ms, "N": Ns},
            ),
            # simple softmax takes: x_ptr, output_ptr, M, N
            # TritonProgram(
            #     bench_runner=simple_softmax_bench_runner,
            #     program=simple_softmax,
            #     bench_runner_args={"M": Ms, "N": Ns},
            # ),
        ],
    )
    result = bench.run()

    # print(result["online_softmax"].times)
    result.save(file_path="./bench_result.json")
    # print(result["simple_softmax"])

    # print(result["online_softmax"].times)
