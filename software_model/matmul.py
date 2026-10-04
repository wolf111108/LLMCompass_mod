from utils import size
from typing import List, Tuple
from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import Tensor, DataType
from software_model.state_manage import MappingProfiler
from math import ceil, log2, floor
import torch
import time
import statistics
import numpy as np
import pandas as pd
import os
import json
from scalesim.scale_sim import scalesim
import copy
import math

class BatchedMatmul(Operator):
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.input1_shape = None
        self.input2_shape = None
        self.output_shape = None
        self.profiler = None  # add
        self.profiler_scale = 1  # add
        self.profiler_extra_latency_cycles = 0  # add
        self.profiler_extra_dram_write_bytes = 0  # add
        self.profiler_strategy = None  # add

    def __call__(self, input1: Tensor, input2: Tensor) -> Tensor:
        # [b, M, K] * [b, K, N] = [b, M, N]
        assert self.data_type == input1.data_type
        assert self.data_type == input2.data_type
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape 
        assert size(self.input1_shape[:-2]) == size(self.input2_shape[:-2])
        self.bs = size(self.input1_shape[:-2])
        self.M = self.input1_shape[-2]
        self.K = self.input1_shape[-1]
        assert self.input2_shape[-2] == self.K
        self.N = self.input2_shape[-1]
        self.flop_count = 2 * self.bs * self.M * self.K * self.N  # add
        self.output_shape = self.input1_shape[:-2] + [self.M, self.N]
        output = Tensor(self.output_shape, self.data_type)
        return output

    def roofline_model(self, pcb_module: Device):
        matmul = Matmul(self.data_type)
        _ = matmul(Tensor([self.M, self.K], self.data_type), Tensor([self.K, self.N], self.data_type))
        matmul_latency = matmul.roofline_model(pcb_module)
        self.roofline_latency = matmul_latency * self.bs
        return self.roofline_latency

    # def compile_and_simulate(self, pcb_module: Device, compile_mode: str):
    #     matmul = Matmul(self.data_type)
    #     _ = matmul(Tensor([self.M, self.K]), Tensor([self.K, self.N]))
    #     matmul_latency = (
    #         matmul.compile_and_simulate(pcb_module, compile_mode)
    #         # - pcb_module.io_module.latency * 2
    #     )
    #     self.latency = matmul_latency * self.bs  # + pcb_module.io_module.latency * 2
    #     return self.latency

    def compile_and_simulate(self, pcb_module: Device, compile_mode: str, mapping_save_path: str = None, layer_name=None, sparsity_ratio=None):
        # print(f"  BatchedMatmul {layer_name}: bs={self.bs} M={self.M} N={self.N} K={self.K}", flush=True)
        matmul_serialized = Matmul(self.data_type)  # add
        _ = matmul_serialized(Tensor([self.M, self.K], self.data_type), Tensor([self.K, self.N], self.data_type))  # add
        # print(f"  BatchedMatmul {layer_name}: running serialized DSE...", flush=True)
        matmul_latency1 = (
            matmul_serialized.compile_and_simulate(pcb_module, compile_mode, mapping_save_path, (layer_name or "matmul")+"serialized", sparsity_ratio=sparsity_ratio, att=True) * self.bs  # add
        )
        # print(f"  BatchedMatmul {layer_name}: serialized latency={matmul_latency1*1e3:.4f}ms", flush=True)

        # Shared-KV GQA is represented as one physical KV-group Matmul with
        # M=q_heads_per_kv_head.  Do not evaluate the old repeated-KV
        # "parallelized" alternative, which would concatenate distinct K/V
        # operands and defeat the reuse being modeled.
        if compile_mode == "heuristic-CIM-GQA-decode":
            self.profiler = matmul_serialized.profiler
            self.profiler_scale = self.bs
            self.profiler_extra_latency_cycles = 0
            self.profiler_extra_dram_write_bytes = 0
            self.profiler_strategy = "shared-kv-serialized"
            self.latency = matmul_latency1
            return self.latency

        # CIM executes independent batch operands serially. Do not construct
        # the unused K-concatenation alternative, which changes the GEMM.
        if compile_mode in (
            "heuristic-CIM",
            "heuristic-CIM-decode",
            "heuristic-CIM-activation-major",
            "heuristic-CIM-weight-major",
        ):
            self.profiler = matmul_serialized.profiler
            self.profiler_scale = self.bs
            self.profiler_extra_latency_cycles = 0
            self.profiler_extra_dram_write_bytes = 0
            self.profiler_strategy = "serialized"
            self.latency = matmul_latency1
            return self.latency

        matmul_parallelized = Matmul(self.data_type)  # add
        _ = matmul_parallelized(  # add
            Tensor([self.M, self.K * self.bs], self.data_type), Tensor([self.K * self.bs, self.N], self.data_type)
        )
        # print(f"  BatchedMatmul {layer_name}: running parallelized DSE (K={self.K * self.bs})...", flush=True)
        parallelized_extra_latency = (  # add
            (self.bs - 1)  # add
            * self.M  # add
            * self.N  # add
            * self.data_type.word_size  # add
            / pcb_module.io_module.bandwidth  # add
        )  # add
        matmul_latency2 = (
            matmul_parallelized.compile_and_simulate(pcb_module, compile_mode, mapping_save_path, layer_name+"parallelized", sparsity_ratio=sparsity_ratio, att=True)  # add
            + parallelized_extra_latency  # add
        )
        # print(f"  BatchedMatmul {layer_name}: parallelized latency={matmul_latency2*1e3:.4f}ms", flush=True)
        if True:#matmul_latency1 <= matmul_latency2:  # add
            self.profiler = matmul_serialized.profiler  # add
            self.profiler_scale = self.bs  # add
            self.profiler_extra_latency_cycles = 0  # add
            self.profiler_extra_dram_write_bytes = 0  # add
            self.profiler_strategy = "serialized"  # add
            self.latency = matmul_latency1  # add
        # else:  # add
        #     self.profiler = matmul_parallelized.profiler  # add
        #     self.profiler_scale = 1  # add
        #     self.profiler_extra_latency_cycles = parallelized_extra_latency * pcb_module.compute_module.clock_freq  # add
        #     self.profiler_extra_dram_write_bytes = (self.bs - 1) * self.M * self.N * self.data_type.word_size  # add
        #     self.profiler_strategy = "parallelized"  # add
        #     self.latency = matmul_latency2  # add
        return self.latency

    def run_on_gpu(
        self,
    ):
        input1 = torch.randn(self.bs, self.M, self.K, dtype=torch.float16).cuda()
        input2 = torch.randn(self.bs, self.K, self.N, dtype=torch.float16).cuda()
        latencies = []
        # warmup
        for _ in range(3):
            _ = torch.bmm(input1, input2)
            torch.cuda.synchronize()
        for _ in range(self.iterations):
            start = time.time()
            output = torch.bmm(input1, input2)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)

        self.latency_on_gpu = (
            statistics.median(latencies)
            # - self.gpu_kernel_launch_overhead()
            # - 4e-5
            # min(latencies) - 8e-6
        )  # GPU launch kernel overhead and PyTorch overhead
        return self.latency_on_gpu

    @staticmethod
    def gpu_kernel_launch_overhead():
        latencies = []
        for _ in range(50):
            a = torch.randn(1, 1, 1, device="cuda")
            b = torch.randn(1, 1, 1, device="cuda")
            torch.cuda.synchronize()
            start = time.time()
            c = torch.bmm(a, b)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
        avg_overhead = statistics.median(latencies)
        # print('GPU kernel launch overhead: ', avg_overhead*1e3, 'ms')
        # print(latencies)
        return avg_overhead


class Matmul(Operator):
    # Default sparsity ratio for CIM computation (0 = no sparsity, 1 = all zeros)
    DEFAULT_CIM_SPARSITY_RATIO = 0.8

    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.input1_shape = None
        self.input2_shape = None
        self.output_shape = None
        self.look_up_table = None
        self.best_mapping = None
        self.profiler = MappingProfiler({"M": None, "N": None, "K": None})
        self.sparsity_ratio = self.DEFAULT_CIM_SPARSITY_RATIO

    def __call__(self, input1: Tensor, input2: Tensor) -> Tensor:
        # [bs, M, K] * [K, N] = [bs, M, N]
        assert self.data_type == input1.data_type
        assert self.data_type == input2.data_type
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        self.M = size(self.input1_shape[:-1])
        self.K = self.input1_shape[-1]
        assert self.input2_shape[-2] == self.K
        self.N = self.input2_shape[-1]
        if len(self.input1_shape) == 2:
            self.output_shape = [self.M, self.N]
        else:
            self.output_shape = self.input1_shape[:-1] + [self.N]
        output = Tensor(self.output_shape, self.data_type)
        self.computational_graph = self.ComputationalGraph(
            self.M, self.N, self.K, self.data_type
        )
        self.flop_count = 2 * self.M * self.K * self.N
        self.io_count = self.M * self.K + self.K * self.N + self.M * self.N
        self.profiler.layer_shape = {"M": self.M, "N": self.N, "K": self.K}
        # print(f'{self.M}, {self.N}, {self.K}')
        return output

    def roofline_model(self, pcb_module: Device):
        self.roofline_latency = max(
            self.flop_count / pcb_module.compute_module.total_systolic_array_flops,
            self.io_count
            / min(
                pcb_module.io_module.bandwidth,
                pcb_module.compute_module.l2_bandwidth_per_cycle
                * pcb_module.compute_module.clock_freq,
            ),
        )
        return self.roofline_latency

    def print_latency(self):
        print(
            f"{self.computational_graph.M}, {self.computational_graph.N}, {self.computational_graph.K}, {self.best_latency*1e3:.4f}ms, {self.latency_on_gpu*1e3:.4f}ms, {self.best_latency/self.latency_on_gpu*100:.2f}%",
            flush=True,
        )

    @staticmethod
    def generate_tile_loops(loop_M: int, loop_N: int, loop_K: int, loop_order: str):
        assert loop_order in ["mkn", "mnk", "nkm", "nmk", "knm", "kmn"]
        if loop_order == "mnk":
            for m in range(loop_M):
                for n in range(loop_N):
                    for k in range(loop_K):
                        yield m, n, k
        elif loop_order == "mkn":
            for m in range(loop_M):
                for k in range(loop_K):
                    for n in range(loop_N):
                        yield m, n, k
        elif loop_order == "nmk":
            for n in range(loop_N):
                for m in range(loop_M):
                    for k in range(loop_K):
                        yield m, n, k
        elif loop_order == "nkm":
            for n in range(loop_N):
                for k in range(loop_K):
                    for m in range(loop_M):
                        yield m, n, k
        elif loop_order == "knm":
            for k in range(loop_K):
                for n in range(loop_N):
                    for m in range(loop_M):
                        yield m, n, k
        elif loop_order == "kmn":
            for k in range(loop_K):
                for m in range(loop_M):
                    for n in range(loop_N):
                        yield m, n, k

    class ComputationalGraph:
        def __init__(self, M: int, N: int, K: int, data_type: DataType):
            self.M = M
            self.N = N
            self.K = K
            self.data_type = data_type

        def display(self):
            print("-" * 10 + " Computational Graph " + "-" * 10)
            print(
                f"M: {self.M}, N: {self.N}, K: {self.K}, word_size(B): {self.data_type.word_size}"
            )

    class Mapping:
        def __init__(
            self,
            l2_tile_M: int,
            l2_tile_N: int,
            l2_tile_K: int,
            is_l2_double_buffering: bool,
            l1_tile_M: int,
            l1_tile_N: int,
            l1_tile_K: int,
            l2_loop_order: str,
            l1_loop_order: str,
            l0_M_tiling_factor: int,
            l0_N_tiling_factor: int,
            l0_K_tiling_factor: int,
            dataflow: str = "os",
        ):
            self.l2_tile_M = l2_tile_M
            self.l2_tile_N = l2_tile_N
            self.l2_tile_K = l2_tile_K
            self.is_l2_double_buffering = is_l2_double_buffering
            self.l1_tile_M = l1_tile_M
            self.l1_tile_N = l1_tile_N
            self.l1_tile_K = l1_tile_K
            self.l2_loop_order = l2_loop_order
            self.l1_loop_order = l1_loop_order
            self.l0_M_tiling_factor = l0_M_tiling_factor
            self.l0_N_tiling_factor = l0_N_tiling_factor
            self.l0_K_tiling_factor = l0_K_tiling_factor
            self.dataflow = dataflow

        def dump_to_file(self, filename, layer_name, M, N, K, latency):
            record = {
                "layer_name": layer_name,
                "M": M, "N": N, "K": K,
                "latency_ms": round(latency * 1e3, 4),
                "l2_tile_M": self.l2_tile_M,
                "l2_tile_N": self.l2_tile_N,
                "l2_tile_K": self.l2_tile_K,
                "is_l2_double_buffering": self.is_l2_double_buffering,
                "l2_loop_order": self.l2_loop_order,
                "l1_tile_M": self.l1_tile_M,
                "l1_tile_N": self.l1_tile_N,
                "l1_tile_K": self.l1_tile_K,
                "l1_loop_order": self.l1_loop_order,
                "l0_M_tiling_factor": self.l0_M_tiling_factor,
                "l0_N_tiling_factor": self.l0_N_tiling_factor,
                "l0_K_tiling_factor": self.l0_K_tiling_factor,
            }
            with open(filename, "a") as f:
                f.write(json.dumps(record) + "\n")

        def display(self):
            print(f'{"-"*10} Mapping {"-"*10}')
            print(
                f"l2_tile_M: {self.l2_tile_M}, l2_tile_N: {self.l2_tile_N}, l2_tile_K: {self.l2_tile_K}, is_l2_double_buffering: {self.is_l2_double_buffering}, l2_loop_order: {self.l2_loop_order}"
            )
            print(
                f"l1_tile_M: {self.l1_tile_M}, l1_tile_N: {self.l1_tile_N}, l1_tile_K: {self.l1_tile_K}, l1_loop_order: {self.l1_loop_order}"
            )
            print(
                f"l0_M_tiling_factor: {self.l0_M_tiling_factor}, l0_N_tiling_factor: {self.l0_N_tiling_factor}, l0_K_tiling_factor: {self.l0_K_tiling_factor}"
            )

    @staticmethod
    def find_permutations(n):
        permutations = set()

        for i in range(1, n + 1):
            if n % i == 0:
                for j in range(1, n + 1):
                    if (n // i) % j == 0:
                        k = n // (i * j)
                        permutations.add((i, j, k))

        return list(permutations)

    @staticmethod
    def find_kmn_factors(core_count, total_K, total_M, total_N, tile_K, tile_M, tile_N):
        """Find K, M, N parallelization factors across cores with K > M > N priority.

        Returns (K_factor, M_factor, N_factor) such that K_factor * M_factor * N_factor <= core_count.
        Prioritizes maximizing K_factor first, then M_factor, then N_factor.
        """
        max_K_tiles = ceil(total_K / tile_K) if tile_K > 0 else 1
        max_M_tiles = ceil(total_M / tile_M) if tile_M > 0 else 1
        max_N_tiles = ceil(total_N / tile_N) if tile_N > 0 else 1

        best = (1, 1, 1)
        for K_factor in range(min(core_count, max_K_tiles), 0, -1):
            remaining_after_K = core_count // K_factor
            if remaining_after_K == 0:
                continue
            for M_factor in range(min(remaining_after_K, max_M_tiles), 0, -1):
                remaining_after_M = remaining_after_K // M_factor
                if remaining_after_M == 0:
                    continue
                N_factor = min(remaining_after_M, max_N_tiles)
                if K_factor * M_factor * N_factor <= core_count:
                    # Prefer higher K, then higher M
                    if (K_factor, M_factor, N_factor) > best:
                        best = (K_factor, M_factor, N_factor)
                    break  # Take the highest M for this K
            if best[0] == K_factor and best[1] == min(remaining_after_K, max_M_tiles):
                break  # Found a good enough K
        return best

    def compile_and_simulate(
        self,
        pcb_module: Device,
        compile_mode: str = "exhaustive",
        mapping_save_path: str = None,  # 新增参数
        layer_name=None,
        sparsity_ratio=None,
        att: bool = False
    ):
        min_cycle_count = 2**63 - 1
        best_mapping = None
        if layer_name is None:
            self.profiler.layer_name = "Matmul"
        else:
            self.profiler.layer_name = layer_name
        M = self.computational_graph.M
        N = self.computational_graph.N
        K = self.computational_graph.K
        if (M == 1 or N == 1) and (
            compile_mode == "heuristic-GPU"
            or compile_mode == "heuristic-our-throughput"
        ):
            self.profiler.start_new_mapping(None)
            working_set_size = M * K + N * K + M * N
            total_io_count = working_set_size * self.data_type.word_size
            io_latency = total_io_count / pcb_module.io_module.bandwidth
            io_latency_cycles = io_latency * pcb_module.compute_module.clock_freq  # add
            self.profiler.record_dram_bytes((M * K + K * N) * self.data_type.word_size, M * N * self.data_type.word_size)
            self.profiler.record_dram_latency(io_latency_cycles)  # add
            self.profiler.record_l2_to_l1_latency(0)
            total_flop_count = 2 * M * N * K
            compute_latency = (
                total_flop_count
                / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                / pcb_module.compute_module.core_count
                / pcb_module.compute_module.clock_freq
            )
            compute_latency_cycles = compute_latency * pcb_module.compute_module.clock_freq  # add
            self.profiler.record_compute_latency(compute_latency_cycles)  # add
            self.latency = max(
                compute_latency, io_latency
            )  # + pcb_module.io_module.latency * 2
            #self.profiler.record_total_latency(self.latency / pcb_module.compute_module.clock_freq)
            self.profiler.record_l2_l1_bytes(0, 0)  #add
            self.profiler.record_l2_l1_weight_bytes(0, 0)  #add
            self.profiler.record_l2_l1_activation_bytes(0, 0)  #add
            self.profiler.record_total_latency(self.latency * pcb_module.compute_module.clock_freq)  #add
            self.evaluate_current = self.profiler.evaluate_current  #add
            self.profiler.evaluate_current()  # add
            return self.latency
        if compile_mode == "exhaustive":
            for l2_tile_M_log2 in range(5, ceil(log2(self.computational_graph.M)) + 1):
                l2_tile_M = 2**l2_tile_M_log2
                for l2_tile_N_log2 in range(
                    5, ceil(log2(self.computational_graph.N)) + 1
                ):
                    l2_tile_N = 2**l2_tile_N_log2
                    for l2_tile_K_log2 in range(
                        5, ceil(log2(self.computational_graph.K)) + 1
                    ):
                        l2_tile_K = 2**l2_tile_K_log2
                        working_set_size = (
                            l2_tile_N * l2_tile_K
                            + l2_tile_M * l2_tile_K
                            + l2_tile_M * l2_tile_N
                        )
                        if (
                            working_set_size
                            > pcb_module.compute_module.l2_size
                            // self.data_type.word_size
                        ):
                            continue
                        is_l2_double_buffering = True
                        for l1_tile_M_log2 in range(5, l2_tile_M_log2 + 1):
                            l1_tile_M = 2**l1_tile_M_log2
                            for l1_tile_N_log2 in range(5, l2_tile_N_log2 + 1):
                                l1_tile_N = 2**l1_tile_N_log2
                                for l1_tile_K_log2 in range(5, l2_tile_K_log2 + 1):
                                    l1_tile_K = 2**l1_tile_K_log2
                                    if (
                                        l1_tile_M * l1_tile_N
                                        + l1_tile_N * l1_tile_K
                                        + l1_tile_M * l1_tile_K
                                        > pcb_module.compute_module.core.SRAM_size
                                        // self.data_type.word_size
                                        // 2
                                    ):
                                        continue
                                    for l2_loop_order in [
                                        "mkn",
                                        "mnk",
                                        "nkm",
                                        "nmk",
                                        "knm",
                                        "kmn",
                                    ]:
                                        for l1_loop_order in [
                                            "mkn",
                                            "mnk",
                                            "nkm",
                                            "nmk",
                                            "knm",
                                            "kmn",
                                        ]:
                                            for (
                                                l0_M_tiling_factor,
                                                l0_N_tiling_factor,
                                                l0_K_tiling_factor,
                                            ) in self.find_permutations(
                                                pcb_module.compute_module.core.systolic_array_count
                                            ):
                                                mapping = self.Mapping(
                                                    l2_tile_M,
                                                    l2_tile_N,
                                                    l2_tile_K,
                                                    is_l2_double_buffering,
                                                    l1_tile_M,
                                                    l1_tile_N,
                                                    l1_tile_K,
                                                    l2_loop_order,
                                                    l1_loop_order,
                                                    l0_M_tiling_factor,
                                                    l0_N_tiling_factor,
                                                    l0_K_tiling_factor,
                                                )
                                                self.profiler.start_new_mapping(mapping)  #add
                                                cycle_count = self.simulate(
                                                    self.computational_graph,
                                                    mapping,
                                                    pcb_module,
                                                )
                                                self.profiler.record_total_latency(cycle_count)  #add
                                                self.profiler.evaluate_current()  #add
                                                if cycle_count < min_cycle_count:
                                                    min_cycle_count = cycle_count
                                                    best_mapping = mapping
        elif compile_mode == "heuristic-our-throughput":
            i = 0
            for l2_tile_M in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
                for l2_tile_N in [
                    l2_tile_M // 4,
                    l2_tile_M // 2,
                    l2_tile_M,
                    l2_tile_M * 2,
                    l2_tile_M * 4,
                    l2_tile_M * 8,
                    l2_tile_M * 16,
                    l2_tile_M * 32,
                    
                ]:
                    l2_tile_K_max = (
                        pcb_module.compute_module.l2_size
                        // self.data_type.word_size
                        // 2
                        - l2_tile_M * l2_tile_N
                    ) // (l2_tile_M + l2_tile_N)
                    if l2_tile_K_max < 1:
                        continue
                    l2_tile_K = min(l2_tile_K_max, K)
                    l2_tile_K = floor(log2(l2_tile_K))
                    l2_tile_K = 2**l2_tile_K
                    working_set_size = (
                        l2_tile_N * l2_tile_K
                        + l2_tile_M * l2_tile_K
                        + l2_tile_M * l2_tile_N
                    )
                    if (
                        working_set_size
                        > pcb_module.compute_module.l2_size // self.data_type.word_size
                    ):
                        continue
                    is_l2_double_buffering = True

                    for l1_tile_M in [32, 64, 128, 256]:
                        l1_tile_M = min(l1_tile_M, l2_tile_M, l2_tile_N)
                        # if l1_tile_M > min(l2_tile_M, l2_tile_N):
                        #     continue
                        l1_tile_N = l1_tile_M
                        l1_tile_K_max = (
                            pcb_module.compute_module.core.SRAM_size
                            // self.data_type.word_size
                            // 2
                            - l1_tile_M * l1_tile_N
                        ) // (l1_tile_M + l1_tile_N)
                        if l1_tile_K_max < 1:
                            continue
                        l1_tile_K = min(l1_tile_K_max, l2_tile_K)
                        l1_tile_K = floor(log2(l1_tile_K))
                        l1_tile_K = 2**l1_tile_K

                        if (
                            l1_tile_M * l1_tile_N
                            + l1_tile_N * l1_tile_K
                            + l1_tile_M * l1_tile_K
                            > pcb_module.compute_module.core.SRAM_size
                            // self.data_type.word_size
                            // 2
                        ):
                            continue
                        l2_loop_order = "knm"
                        l1_loop_order = "knm"
                        for (
                            l0_M_tiling_factor,
                            l0_N_tiling_factor,
                            l0_K_tiling_factor,
                        ) in [(2, 2, 1)]:
                            # self.find_permutations(
                            #     pcb_module.compute_module.core.systolic_array_count
                            # ):
                            i += 1
                            # start = time.time()
                            mapping = self.Mapping(
                                l2_tile_M,
                                l2_tile_N,
                                l2_tile_K,
                                is_l2_double_buffering,
                                l1_tile_M,
                                l1_tile_N,
                                l1_tile_K,
                                l2_loop_order,
                                l1_loop_order,
                                l0_M_tiling_factor,
                                l0_N_tiling_factor,
                                l0_K_tiling_factor,
                            )
                            self.profiler.start_new_mapping(mapping)  #add
                            cycle_count = self.simulate(
                                self.computational_graph,
                                mapping,
                                pcb_module,
                            )
                            self.profiler.record_total_latency(cycle_count)  #add
                            self.profiler.evaluate_current()  #add
                            # end = time.time()
                            # if i % 1000 == 0:
                            #     print(f"{i} simulation time: {end-start}")
                            if cycle_count < min_cycle_count:
                                min_cycle_count = cycle_count
                                best_mapping = mapping
        elif compile_mode == "heuristic-GPU":
            i = 0
            for l2_tile_M in [64, 128, 256, 512, 1024, 2048]:
                for l2_tile_N in [l2_tile_M // 2, l2_tile_M, l2_tile_M * 2]:
                    if K <= 12288:
                        l2_K_tiling_factor_list = [1, 2, 4, 8]
                    else:
                        l2_K_tiling_factor_list = [
                            K // 1024,
                            K // 2048,
                            K // 4096,
                            K // 8192,
                        ]
                    for l2_K_tiling_factor in l2_K_tiling_factor_list:
                        l2_tile_K = ceil(
                            self.computational_graph.K / l2_K_tiling_factor
                        )
                        l2_tile_K = 2 ** floor(log2(l2_tile_K))
                        working_set_size = (
                            l2_tile_N * l2_tile_K
                            + l2_tile_M * l2_tile_K
                            + l2_tile_M * l2_tile_N
                        )
                        if (
                            working_set_size
                            > pcb_module.compute_module.l2_size
                            // self.data_type.word_size
                        ):
                            continue
                        is_l2_double_buffering = True

                        for l1_tile_M in [32, 64, 128, 256]:
                            if l1_tile_M > min(l2_tile_M, l2_tile_N):
                                continue
                            l1_tile_N = l1_tile_M
                            for l1_K_tiling_factor in [1, 2, 4, 8, 16, 32]:
                                l1_tile_K = ceil(l2_tile_K / l1_K_tiling_factor)
                                if (
                                    l1_tile_M * l1_tile_N
                                    + l1_tile_N * l1_tile_K
                                    + l1_tile_M * l1_tile_K
                                    > pcb_module.compute_module.core.SRAM_size
                                    // self.data_type.word_size
                                    // 2
                                ):
                                    continue
                                l2_loop_order = "knm"
                                l1_loop_order = "knm"
                                for (
                                    l0_M_tiling_factor,
                                    l0_N_tiling_factor,
                                    l0_K_tiling_factor,
                                ) in self.find_permutations(
                                    pcb_module.compute_module.core.systolic_array_count
                                ):
                                    i += 1
                                    start = time.time()
                                    mapping = self.Mapping(
                                        l2_tile_M,
                                        l2_tile_N,
                                        l2_tile_K,
                                        is_l2_double_buffering,
                                        l1_tile_M,
                                        l1_tile_N,
                                        l1_tile_K,
                                        l2_loop_order,
                                        l1_loop_order,
                                        l0_M_tiling_factor,
                                        l0_N_tiling_factor,
                                        l0_K_tiling_factor,
                                    )
                                    #self.profiler.start_new_mapping(mapping)
                                    self.profiler.start_new_mapping(mapping)  #add
                                    cycle_count = self.simulate(
                                        self.computational_graph,
                                        mapping,
                                        pcb_module,
                                    )
                                    self.profiler.record_total_latency(cycle_count)  #add
                                    self.profiler.evaluate_current()  #add
                                    end = time.time()
                                    #self.profiler.record_total_latency(cycle_count / pcb_module.compute_module.clock_freq)
                                    #self.profiler.record_total_latency(cycle_count)  #add
                                    #self.profiler.evaluate_current()
                                    # if i % 1000 == 0:
                                    #     print(f"{i} simulation time: {end-start}")
                                    if cycle_count < min_cycle_count:
                                        min_cycle_count = cycle_count
                                        best_mapping = mapping
            # print("total dse times:", i)
        elif compile_mode == "heuristic-TPU":
            l2_tile_M = self.computational_graph.M
            l2_tile_N = self.computational_graph.N
            l2_tile_K = self.computational_graph.K

            is_l2_double_buffering = True
            for l1_tile_M in [l2_tile_M, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
                if l1_tile_M > l2_tile_M * 2:
                    continue
                for l1_tile_N in [
                    l1_tile_M // 2,
                    l1_tile_M,
                    l1_tile_M * 2,
                    l1_tile_M * 8,
                    l1_tile_M * 16,
                    l1_tile_M * 64,
                    l1_tile_M * 128,
                    l1_tile_M * 256,
                ]:
                    if l1_tile_N > l2_tile_N:
                        continue
                    if l1_tile_N <= 0:
                        continue
                    l1_tile_K_max = (
                        pcb_module.compute_module.core.SRAM_size
                        // self.data_type.word_size
                        // 2
                        - l1_tile_M * l1_tile_N
                    ) // (l1_tile_M + l1_tile_N)
                    if l1_tile_K_max < 1:
                        continue
                    l1_tile_K = min(l1_tile_K_max, l2_tile_K)
                    l1_tile_K = floor(log2(l1_tile_K))
                    l1_tile_K = 2**l1_tile_K

                    l2_loop_order = "knm"
                    l1_loop_order = "knm"
                    for (
                        l0_M_tiling_factor,
                        l0_N_tiling_factor,
                        l0_K_tiling_factor,
                    ) in [(1, 2, 1)]:
                        mapping = self.Mapping(
                            l2_tile_M,
                            l2_tile_N,
                            l2_tile_K,
                            is_l2_double_buffering,
                            l1_tile_M,
                            l1_tile_N,
                            l1_tile_K,
                            l2_loop_order,
                            l1_loop_order,
                            l0_M_tiling_factor,
                            l0_N_tiling_factor,
                            l0_K_tiling_factor,
                        )
                        # mapping.display()
                        # start=time.time()
                        self.profiler.start_new_mapping(mapping)  #add
                        cycle_count = self.simulate(
                            self.computational_graph,
                            mapping,
                            pcb_module,
                        )
                        self.profiler.record_total_latency(cycle_count)  #add
                        self.profiler.evaluate_current()  #add
                        # end=time.time()
                        # print(f'simulation time: {end-start}')
                        if cycle_count < min_cycle_count:
                            min_cycle_count = cycle_count
                            best_mapping = mapping
        elif compile_mode == "heuristic-TPU-new":
            l2_tile_M = self.computational_graph.M
            l2_tile_N = self.computational_graph.N
            l2_tile_K = self.computational_graph.K

            is_l2_double_buffering = True
            for l1_tile_M in [l2_tile_M, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
                if l1_tile_M > l2_tile_M * 2:
                    continue
                for l1_tile_N in [
                    l1_tile_M // 2,
                    l1_tile_M,
                    l1_tile_M * 2,
                    l1_tile_M * 8,
                    l1_tile_M * 16,
                    l1_tile_M * 64,
                    l1_tile_M * 128,
                    l1_tile_M * 256,
                ]:
                    if l1_tile_N > l2_tile_N:
                        continue
                    if l1_tile_N <= 0:
                        continue
                    l1_tile_K_max = (
                        pcb_module.compute_module.core.SRAM_size
                        // self.data_type.word_size
                        // 2
                        - l1_tile_M * l1_tile_N
                    ) // (l1_tile_M + l1_tile_N)
                    if l1_tile_K_max < 1:
                        continue
                    l1_tile_K = min(l1_tile_K_max, l2_tile_K)
                    l1_tile_K = floor(log2(l1_tile_K))
                    l1_tile_K = 2**l1_tile_K

                    l2_loop_order = "knm"
                    l1_loop_order = "knm"
                    for (
                        l0_M_tiling_factor,
                        l0_N_tiling_factor,
                        l0_K_tiling_factor,
                    ) in [(1, 1, 1)]:
                        mapping = self.Mapping(
                            l2_tile_M,
                            l2_tile_N,
                            l2_tile_K,
                            is_l2_double_buffering,
                            l1_tile_M,
                            l1_tile_N,
                            l1_tile_K,
                            l2_loop_order,
                            l1_loop_order,
                            l0_M_tiling_factor,
                            l0_N_tiling_factor,
                            l0_K_tiling_factor,
                        )
                        # mapping.display()
                        # start=time.time()
                        self.profiler.start_new_mapping(mapping)  #add
                        cycle_count = self.simulate(
                            self.computational_graph,
                            mapping,
                            pcb_module,
                        )
                        self.profiler.record_total_latency(cycle_count)  #add
                        self.profiler.evaluate_current()  #add
                        # end=time.time()
                        # print(f'simulation time: {end-start}')
                        if cycle_count < min_cycle_count:
                            min_cycle_count = cycle_count
                            best_mapping = mapping
        # elif compile_mode == "heuristic-CIM":
        #     # CIM-specific compilation: no L1, weights stored in CIM macro
        #     # L2 only stores activation tiles
        #     # Mapping priority: K -> M -> N (to minimize CIM weight writes)
        #     # Optimization criterion: minimize weight_write_cycles (not latency)
        #     import sys
        #     cim_macro = pcb_module.compute_module.core.cim_macro
        #     core_count = pcb_module.compute_module.core_count
        #     weight_buffer_bytes = cim_macro.weight_buffer_size
        #     l2_size = pcb_module.compute_module.l2_size
        #     l2_bw = pcb_module.compute_module.l2_bandwidth_per_cycle
        #     input_ws = cim_macro.input_word_size
        #     output_ws = cim_macro.output_word_size

        #     # Include 1 in tile sizes to handle decode (M=1) case
        #     cim_Mtile_sizes = [1, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        #     cim_tile_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        #     # print(f"    [CIM DSE] Starting: M={M}, N={N}, K={K}, "
        #     #       f"core_count={core_count}, layer={layer_name}", flush=True)

        #     # Track best mapping by weight_write_cycles (primary) and latency (secondary)
        #     min_weight_write_cycles = 2**63 - 1
        #     best_weight_mapping = None
        #     best_weight_cycle_count = 2**63 - 1  # latency of the best weight mapping

        #     i = 0
        #     cim_start_time = time.time()
        #     for tile_K in cim_tile_sizes:
        #         # print(f"Testing tile_K={tile_K}, total_K={K}", flush=True)
        #         if tile_K > K:
        #             tile_K = K
        #         # Weight buffer constraint: tile_K * tile_N * input_word_size <= weight_buffer_bytes
        #         max_tile_N_from_weight = weight_buffer_bytes // (tile_K * input_ws) if tile_K * input_ws > 0 else K
        #         max_tile_N = min(N, max_tile_N_from_weight)
        #         if max_tile_N < 1:
        #             continue
        #         for tile_N in cim_tile_sizes:
        #             # print(f"Testing tile_N={tile_N}, total_N={N}", flush=True)
        #             if tile_N > max_tile_N:
        #                 continue
        #             # Check weight buffer constraint
        #             if tile_K * tile_N * input_ws > weight_buffer_bytes:
        #                 continue
        #             # L2 activation constraint: tile_M * tile_K + tile_M * tile_N <= l2_size / word_size
        #             # (only activation, no weight in L2)
        #             max_tile_M_from_l2 = l2_size // (self.data_type.word_size * (tile_K + tile_N)) if (tile_K + tile_N) > 0 else M
        #             max_tile_M = min(M, max_tile_M_from_l2)
        #             if max_tile_M < 1:
        #                 continue
        #             for tile_M in cim_Mtile_sizes:
        #                 # print(f"Testing tile_M={tile_M}, total_M={M}", flush=True)
        #                 if tile_M > max_tile_M:
        #                     continue
        #                 # Check L2 activation constraint (no weight, only activation)
        #                 activation_l2 = (tile_M * tile_K + tile_M * tile_N) * self.data_type.word_size
        #                 if activation_l2 > l2_size:
        #                     continue

        #                 # Double buffering if activation fits in half L2
        #                 is_l2_double_buffering = True

        #                 # Use nmk loop order: N outermost, M middle, K innermost
        #                 # This enables weight reuse across M iterations (weight depends on K and N only)
        #                 mapping = self.Mapping(
        #                     tile_M, tile_N, tile_K,
        #                     is_l2_double_buffering,
        #                     tile_M, tile_N, tile_K,  # No L1 tiling: l1 = l2
        #                     "nmk",  # L2 loop order: N first, then M, then K (weight reuse across M)
        #                     "nmk",  # L1 loop order: same (not used in CIM)
        #                     1, 1, 1,  # l0 factors (not used in CIM)
        #                 )
        #                 # i += 1
        #                 # if i % 50 == 0:
        #                 #     elapsed = time.time() - cim_start_time
        #                 #     best_ww = min_weight_write_cycles if min_weight_write_cycles < 2**63 - 1 else float('inf')
        #                 #     print(f"    [CIM DSE] iter={i}, tile_K={tile_K}, tile_N={tile_N}, tile_M={tile_M}, "
        #                 #           f"best_ww_cycles={best_ww}, elapsed={elapsed:.1f}s", flush=True)
        #                 self.profiler.start_new_mapping(mapping)
        #                 # Use caller-provided sparsity_ratio if given, otherwise use self.sparsity_ratio
        #                 effective_sparsity = sparsity_ratio if sparsity_ratio is not None else self.sparsity_ratio
        #                 cycle_count = self.simulate_cim(
        #                     self.computational_graph,
        #                     mapping,
        #                     pcb_module,
        #                     effective_sparsity,
        #                 )
        #                 weight_write_cycles = self.last_cim_weight_write_cycles
        #                 self.profiler.record_total_latency(cycle_count)
        #                 self.profiler.evaluate_current()
        #                 # Select mapping with smallest weight_write_cycles
        #                 # Tie-break by latency (cycle_count)
        #                 if (weight_write_cycles < min_weight_write_cycles or
        #                     (weight_write_cycles == min_weight_write_cycles and cycle_count < best_weight_cycle_count)):
        #                     min_weight_write_cycles = weight_write_cycles
        #                     best_weight_cycle_count = cycle_count
        #                     best_weight_mapping = mapping
        #                     weight_write_bytes = self.last_cim_weight_write_bytes
        #                     # print(f"    [CIM DSE] New best! tile_M={tile_M}, tile_N={tile_N}, tile_K={tile_K}, "
        #                     #       f"weight_write_cycles={weight_write_cycles}, "
        #                     #       f"weight_write_bytes={weight_write_bytes}, "
        #                     #       f"latency_cycles={cycle_count}", flush=True)
        #     cim_elapsed = time.time() - cim_start_time
        #     best_latency_ms = best_weight_cycle_count / pcb_module.compute_module.clock_freq * 1e3 if best_weight_cycle_count < 2**63 - 1 else float('inf')
        #     # print(f"    [CIM DSE] Done: {i} iterations in {cim_elapsed:.1f}s, "
        #     #       f"best_weight_write_cycles={min_weight_write_cycles}, "
        #     #       f"best_weight_write_bytes={self.last_cim_weight_write_bytes}, "
        #     #       f"best_latency={best_latency_ms:.4f}ms", flush=True)

        #     # Use the best weight mapping as the final result
        #     best_mapping = best_weight_mapping
        #     min_cycle_count = best_weight_cycle_count
        elif compile_mode == "heuristic-CIM":
            # ============================================================
            # EffLoc strict prefill mapping
            #
            # Macro 间：
            #   - 相同权重块复制到所有 macro
            #   - 不同 macro 处理不同 token
            #
            # Macro 内：
            #   - K 由 Nbank × array_height 展开
            #   - N 由 array_width 展开
            #
            # 因此：
            #   K_factor = 1
            #   M_factor = core_count
            #   N_factor = 1
            #
            # 权重循环：
            #   N -> K -> M
            #   同一权重块处理完所有 token 后才更新
            # ============================================================

            cim_macro = pcb_module.compute_module.core.cim_macro
            core_count = pcb_module.compute_module.core_count
            l2_size = pcb_module.compute_module.l2_size
            output_ws = cim_macro.output_word_size

            # Use the externally configured prefill activation width by default.
            # This lets the weight-major nkm path share the same activation-size
            # sweep interface as activation-major prefill.
            _act_word_size = float(
                getattr(
                    cim_macro,
                    "prefill_activation_element_size",
                    8 / 8,
                )
            )

            if M <= 1:
                raise ValueError(
                    "当前 heuristic-CIM 实现的是 EffLoc prefill "
                    "weight-replication mapping，要求 M > 1。"
                    "Decode 的 M=1 应使用单独的 decode mapping。"
                )

            # ============================================================
            # 物理 CIM tile，由 macro 结构固定，不做 DSE
            # ============================================================

            physical_tile_K = (
                cim_macro.Nbank
                * cim_macro.array_height
            )

            physical_tile_N = (
                cim_macro.array_width
            )

            tile_K = min(K, physical_tile_K)
            tile_N = min(N, physical_tile_N)

            # ============================================================
            # M tile 只受 global buffer/L2 容量限制
            #
            # global buffer 保存：
            #   activation tile: tile_M × tile_K
            #   partial sum:     tile_M × tile_N
            #
            # 预留双缓冲空间。
            # ============================================================

            activation_bytes_per_token = (
                tile_K * _act_word_size
            )
            if activation_bytes_per_token <= 0:
                raise RuntimeError(
                    f"Invalid activation bytes per token="
                    f"{activation_bytes_per_token}"
                )

            # In nkm order, all M-tile partial sums for the current N strip
            # must stay resident across the K loop.  Reserve that full psum
            # strip before choosing the number of activation rows to buffer.
            psum_strip_bytes = M * tile_N * output_ws
            if psum_strip_bytes >= l2_size:
                raise ValueError(
                    "Global Buffer cannot hold the resident N-strip partial "
                    f"sums: required={psum_strip_bytes}B, l2_size={l2_size}B"
                )

            activation_budget_bytes = l2_size - psum_strip_bytes
            max_tile_M_single = max(
                1,
                activation_budget_bytes
                // activation_bytes_per_token,
            )
            max_tile_M_double = max(
                1,
                activation_budget_bytes
                // (2 * activation_bytes_per_token),
            )

            # Prefer a tile size that permits activation double buffering,
            # but fall back to the largest single-buffered tile if needed.
            max_tile_M_from_l2 = (
                max_tile_M_double
                if max_tile_M_double > 1
                else max_tile_M_single
            )

            tile_M = min(
                M,
                max_tile_M_from_l2,
            )

            # 完整 M tile 尽可能是 macro 数量的整数倍，
            # 避免非尾部 tile 出现 token parallelism 尾差。
            if tile_M >= core_count:
                tile_M = (
                    tile_M // core_count
                ) * core_count

            tile_M = max(1, tile_M)

            working_set_bytes = (
                psum_strip_bytes
                + tile_M * activation_bytes_per_token
            )

            is_l2_double_buffering = (
                psum_strip_bytes
                + 2 * tile_M * activation_bytes_per_token
                <= l2_size
            )

            # ============================================================
            # 必须使用 nkm：
            #
            # for n:
            #   for k:
            #       load/multicast W[k, n]
            #       for m:
            #           process tokens with the same weight tile
            # ============================================================

            mapping = self.Mapping(
                tile_M,
                tile_N,
                tile_K,
                is_l2_double_buffering,

                # CIM 没有传统 L1，保持兼容字段
                tile_M,
                tile_N,
                tile_K,

                "nkm",
                "nkm",

                # 不使用 systolic-array L0 factors
                1,
                1,
                1,
            )

            self.profiler.start_new_mapping(mapping)

            # ============================================================
            # 这里传入的是 speedup，而不是 0~1 的 sparsity ratio
            #
            # 例如：
            #   up_proj   = 1.715
            #   down_proj = 4.937
            # ============================================================

            prefill_effective_speedup = (
                1.0
                if sparsity_ratio is None
                else float(sparsity_ratio)
            )

            if prefill_effective_speedup <= 0:
                raise ValueError(
                    "prefill_effective_speedup must be positive, "
                    f"got {prefill_effective_speedup}"
                )

            cycle_count = self.simulate_cim(
                self.computational_graph,
                mapping,
                pcb_module,
                prefill_effective_speedup=(
                    prefill_effective_speedup
                ),
            )

            self.profiler.record_total_latency(
                cycle_count
            )
            self.profiler.evaluate_current()

            # 严格物理映射只有一个 mapping，不再进行 DSE
            best_mapping = mapping
            min_cycle_count = cycle_count

        elif compile_mode == "heuristic-CIM-activation-major":
            # ============================================================
            # Activation-major prefill mapping.
            #
            # Outer schedule:
            #   for each M/token chunk:
            #       1. load A[m_chunk, :] from HBM to Global Buffer;
            #       2. traverse every (N,K) weight tile;
            #       3. each weight tile is multicast and replicated to all
            #          macros, while macros process different tokens.
            #
            # Tile schedule inside an M chunk:
            #   N -> K
            #
            # This is the opposite reuse direction from `nkm`: it streams all
            # weights for a resident activation chunk instead of streaming all
            # activations for a resident weight tile.
            # ============================================================
            cim_macro = pcb_module.compute_module.core.cim_macro
            core_count = pcb_module.compute_module.core_count
            l2_size = pcb_module.compute_module.l2_size
            output_ws = cim_macro.output_word_size
            act_ws = float(
                getattr(
                    cim_macro,
                    "prefill_activation_element_size",
                    8 / 8,
                )
            )

            if M <= 1:
                raise ValueError(
                    "activation-major prefill requires M > 1; "
                    "decode should use heuristic-CIM-decode."
                )

            physical_tile_K = cim_macro.Nbank * cim_macro.array_height
            physical_tile_N = cim_macro.array_width

            # K-major macro layout: give the reduction/feature dimension
            # priority, keep one physical N strip, and use any remaining
            # macros for token (M) parallelism.  This differs from the
            # original activation-major mapping, which used all macros for M
            # and traversed K in physical 1024-element chunks.
            K_factor = min(
                core_count,
                ceil(K / physical_tile_K),
            )
            N_factor = 1
            M_factor = max(
                1,
                core_count // K_factor,
            )

            tile_K = min(
                K,
                physical_tile_K * K_factor,
            )
            tile_N = min(N, physical_tile_N)

            # The resident activation chunk contains all K elements for each
            # selected token.  The current N-strip psum also remains resident.
            bytes_per_token = (
                K * act_ws
                + tile_N * output_ws
            )
            if bytes_per_token <= 0:
                raise RuntimeError(
                    f"Invalid activation-major bytes per token={bytes_per_token}"
                )

            max_tile_M = max(1, l2_size // bytes_per_token)
            tile_M = min(M, max_tile_M)
            if tile_M >= core_count:
                tile_M = (tile_M // core_count) * core_count
            tile_M = max(1, tile_M)

            working_set_bytes = tile_M * bytes_per_token
            if working_set_bytes > l2_size:
                raise ValueError(
                    "Global Buffer cannot hold an activation-major chunk: "
                    f"required={working_set_bytes}B, l2_size={l2_size}B"
                )

            # A future-chunk prefetch buffer is not modeled initially.  Prefer
            # the largest M chunk, which minimizes repeated weight streaming.
            is_l2_double_buffering = False

            mapping = self.Mapping(
                tile_M,
                tile_N,
                tile_K,
                is_l2_double_buffering,
                tile_M,
                tile_N,
                tile_K,
                "mnk",
                "mnk",
                min(M_factor, tile_M),
                N_factor,
                K_factor,
            )
            self.profiler.start_new_mapping(mapping)

            prefill_effective_speedup = (
                1.0
                if sparsity_ratio is None
                else float(sparsity_ratio)
            )
            if prefill_effective_speedup <= 0:
                raise ValueError(
                    "prefill_effective_speedup must be positive, "
                    f"got {prefill_effective_speedup}"
                )

            cycle_count = self.simulate_cim_activation_major(
                self.computational_graph,
                mapping,
                pcb_module,
                prefill_effective_speedup,
            )
            self.profiler.record_total_latency(cycle_count)
            self.profiler.evaluate_current()

            best_mapping = mapping
            min_cycle_count = cycle_count

        elif compile_mode == "heuristic-CIM-weight-major":
            # ============================================================
            # K-major weight-major prefill.
            #
            # This keeps the macro-factor priority of activation-major
            # K-major mapping (K first, N fixed to one strip, then M), but
            # changes the execution order to nkm:
            #
            #   for n:
            #     for k:
            #       load/multicast W[k,n]
            #       for m:
            #         stream activation through the resident weight tile
            # ============================================================
            cim_macro = pcb_module.compute_module.core.cim_macro
            core_count = pcb_module.compute_module.core_count
            l2_size = pcb_module.compute_module.l2_size
            output_ws = cim_macro.output_word_size
            act_ws = float(
                getattr(
                    cim_macro,
                    "prefill_activation_element_size",
                    8 / 8,
                )
            )

            if M <= 1:
                raise ValueError(
                    "weight-major prefill requires M > 1; "
                    "decode should use heuristic-CIM-decode."
                )

            physical_tile_K = (
                cim_macro.Nbank * cim_macro.array_height
            )
            physical_tile_N = cim_macro.array_width

            K_factor = min(
                core_count,
                ceil(K / physical_tile_K),
            )
            N_factor = 1
            M_factor = max(
                1,
                core_count // K_factor,
            )

            tile_K = min(
                K,
                physical_tile_K * K_factor,
            )
            tile_N = min(N, physical_tile_N)

            # Preserve the activation-major tile geometry: the mapper sizes an
            # M chunk as if the full K extent of that chunk can be resident.
            # In weight-major nkm order, only the currently needed K slice is
            # supplied to macros; partial sums that cannot remain resident
            # across K super-tiles are spilled to HBM by the simulator.
            bytes_per_token = (
                K * act_ws
                + tile_N * output_ws
            )
            if bytes_per_token <= 0:
                raise RuntimeError(
                    f"Invalid weight-major bytes per token={bytes_per_token}"
                )

            max_tile_M = max(1, l2_size // bytes_per_token)
            tile_M = min(M, max_tile_M)
            if tile_M >= core_count:
                tile_M = (tile_M // core_count) * core_count
            tile_M = max(1, tile_M)

            # Match activation-major's conservative no-prefetch assumption.
            is_l2_double_buffering = False

            mapping = self.Mapping(
                tile_M,
                tile_N,
                tile_K,
                is_l2_double_buffering,
                tile_M,
                tile_N,
                tile_K,
                "nkm",
                "nkm",
                min(M_factor, tile_M),
                N_factor,
                K_factor,
            )
            self.profiler.start_new_mapping(mapping)

            prefill_effective_speedup = (
                1.0
                if sparsity_ratio is None
                else float(sparsity_ratio)
            )
            if prefill_effective_speedup <= 0:
                raise ValueError(
                    "prefill_effective_speedup must be positive, "
                    f"got {prefill_effective_speedup}"
                )

            cycle_count = self.simulate_cim_weight_major(
                self.computational_graph,
                mapping,
                pcb_module,
                prefill_effective_speedup,
            )
            self.profiler.record_total_latency(cycle_count)
            self.profiler.evaluate_current()

            best_mapping = mapping
            min_cycle_count = cycle_count

        elif compile_mode in (
            "heuristic-CIM-decode",
            "heuristic-CIM-GQA-decode",
        ):
            # ============================================================
            # EffLoc strict decode mapping
            #
            # GEMM:
            #   [1, K] @ [K, N] -> [1, N]
            #
            # Macro 间：
            #   - 同一个 activation K slice 广播给所有活跃 macro；
            #   - 不同 macro 保存不同 N 方向的 weight tile；
            #
            # 因此：
            #   K_factor = 1
            #   M_factor = 1
            #   N_factor = min(core_count, ceil(N / array_width))
            #
            # 单轮最多覆盖：
            #   core_count * array_width 个输出通道。
            #
            # 循环顺序：
            #   K -> N
            #
            # 对每个 K tile：
            #   1. activation K slice 从 HBM 加载到 Global Buffer；
            #   2. 对不同 N super-tile 更新各 Macro 中的不同权重；
            #   3. activation 广播到所有活跃 Macro；
            #   4. partial sum 在 Global Buffer 中累加。
            # ============================================================

            cim_macro = (
                pcb_module.compute_module.core.cim_macro
            )

            core_count = (
                pcb_module.compute_module.core_count
            )

            l2_size = (
                pcb_module.compute_module.l2_size
            )

            if compile_mode == "heuristic-CIM-decode" and M != 1:
                raise ValueError(
                    "heuristic-CIM-decode 只支持单 token decode，"
                    f"要求 M == 1，当前 M={M}。"
                )
            if compile_mode == "heuristic-CIM-GQA-decode" and M < 1:
                raise ValueError(
                    "heuristic-CIM-GQA-decode 要求 M >= 1，"
                    f"当前 M={M}。"
                )

            # ============================================================
            # 计算格式和存储格式分开定义
            #
            # activation_storage_bits:
            #   用于 HBM/Global Buffer traffic。
            #
            # activation_serial_bits:
            #   dense compute baseline 中的串行 bit-plane 数。
            #
            # decode_effective_speedup:
            #   在 dense serial bits 基础上，由 bit skipping 带来的速度提升。
            #
            # 不要同时使用：
            #   activation_serial_bits = 2.69
            #   decode_effective_speedup > 1
            #
            # 如果 speedup 已经包含稀疏跳过，serial bits 应使用 dense 值。
            # ============================================================
            if att:
                activation_storage_bits = float(
                    getattr(
                        cim_macro,
                        "decode_activation_storage_bits_attn",
                        9.2,
                    )
                )
            else:
                activation_storage_bits = float(
                    getattr(
                        cim_macro,
                        "decode_activation_storage_bits",
                        6.4,
                    )
                )


            # FP8 SMMM dense baseline = 4。
            # BF16 S+M 口径改为 8；
            # 论文 mantissa-only BF16 口径改为 7。
            activation_serial_bits = float(
                getattr(
                    cim_macro,
                    "decode_dense_serial_bits",
                    4.0,
                )
            )

            act_ws = activation_storage_bits / 8.0

            psum_ws = float(
                getattr(
                    cim_macro,
                    "psum_word_size",
                    cim_macro.output_word_size,
                )
            )

            # ============================================================
            # 物理 Macro tile
            # ============================================================

            physical_tile_K = (
                cim_macro.Nbank
                * cim_macro.array_height
            )

            physical_tile_N_per_macro = (
                cim_macro.array_width
            )

            Kf, Mf, Nf, Kr, Mr, Nr = compute_optimal_macro_layout_decode(
                pcb_module.compute_module.core_count, K, N, M,
                h=pcb_module.compute_module.core.cim_macro.array_height,
                w=pcb_module.compute_module.core.cim_macro.array_width,
                Nadder=pcb_module.compute_module.core.cim_macro.Nbank,
            )

            # Conventional decode has one token/Q row.  Shared-KV GQA packs
            # the Q heads sharing one KV head into the M dimension, so one
            # K/V tile update is reused by all M rows.
            tile_M = M  # All rows reuse the resident weight tile, including GQA.

            tile_K = min(
                K,
                physical_tile_K * Kf,
            )

            tile_N = min(
                N,
                physical_tile_N_per_macro * Nf,
            )

            # ============================================================
            # Global Buffer 容量
            #
            # 使用 K -> N 顺序时：
            #   - 当前 activation K tile 需要常驻；
            #   - 所有 N 输出通道的 partial sum 需要跨 K tile 保留。
            #
            # 双缓冲只对 activation K tile 做双缓冲。
            # ============================================================

            activation_tile_bytes = (
                tile_M
                * tile_K
                * act_ws
            )

            resident_psum_bytes = (
                M
                * N
                * psum_ws
            )

            single_buffer_required = (
                resident_psum_bytes
                + activation_tile_bytes
            )

            double_buffer_required = (
                resident_psum_bytes
                + 2 * activation_tile_bytes
            )

            if single_buffer_required > l2_size:
                raise ValueError(
                    "Global Buffer 无法容纳 decode partial sums "
                    "和一个 activation K tile："
                    f"required={single_buffer_required} B, "
                    f"l2_size={l2_size} B"
                )

            is_l2_double_buffering = (
                double_buffer_required <= l2_size
            )

            # ============================================================
            # Mapping
            #
            # l2_tile_N 是所有活跃 Macro 合起来覆盖的 N super-tile，
            # 不是单个 Macro 的 48 列。
            # ============================================================

            mapping = self.Mapping(
                tile_M,
                tile_N,
                tile_K,
                is_l2_double_buffering,

                # CIM 没有传统 L1，保留兼容字段
                tile_M,
                tile_N,
                tile_K,

                "knm",
                "knm",

                # 不使用 systolic-array L0 factors
                1,
                1,
                1,
            )

            self.profiler.start_new_mapping(mapping)

            # 保留现有参数接口，但这里实际表示 decode speedup。
            decode_effective_speedup = (
                1.0
                if sparsity_ratio is None
                else float(sparsity_ratio)
            )

            if decode_effective_speedup <= 0:
                raise ValueError(
                    "decode_effective_speedup must be positive, "
                    f"got {decode_effective_speedup}"
                )

            cycle_count = self.simulate_cim_decode(
                self.computational_graph,
                mapping,
                pcb_module,
                decode_effective_speedup=(
                    decode_effective_speedup
                ),
                activation_storage_bits=(
                    activation_storage_bits
                ),
                activation_serial_bits=activation_serial_bits,
                attn = att
            )

            self.profiler.record_total_latency(
                cycle_count
            )

            self.profiler.evaluate_current()

            best_mapping = mapping
            min_cycle_count = cycle_count

        else:
            raise ValueError(f"compile_mode {compile_mode} not supported")
        self.best_mapping = best_mapping
        # if self.best_mapping is not None:
        #     self.best_mapping.display()
        self.best_cycle_count = min_cycle_count
        self.best_latency = min_cycle_count / pcb_module.compute_module.clock_freq
        self.latency = self.best_latency
        # self.best_mapping.display()
        if self.best_mapping is not None and mapping_save_path is not None:
            self.best_mapping.dump_to_file(
                filename=mapping_save_path,
                layer_name=layer_name,
                M=self.computational_graph.M,
                N=self.computational_graph.N,
                K=self.computational_graph.K,
                latency=self.best_latency,
            )
        
        return self.latency

    def simulate(
        self,
        computational_graph: ComputationalGraph,
        mapping: Mapping,
        pcb_module: Device,
    ) -> int:
        if self.look_up_table is None:
            self.look_up_table = pd.read_csv(
                f"./systolic_array_model/look_up_table_{pcb_module.compute_module.core.systolic_array.array_height}_{pcb_module.compute_module.core.systolic_array.array_width}.csv",
                header=None,
                names=[
                    "M",
                    "N",
                    "K",
                    "ArrayHeight",
                    "ArrayWidth",
                    "Dataflow",
                    "cycle_count",
                    "util_rate",
                ],
            )
            self.look_up_table.drop_duplicates(
                inplace=True,
                subset=["M", "N", "K", "ArrayHeight", "ArrayWidth", "Dataflow"],
            )
            # self.look_up_table.reset_index(drop=True, inplace=True)
            # self.look_up_table.to_csv(
            #     f"./systolic_array_model/look_up_table_{pcb_module.compute_module.core.systolic_array.array_height}_{pcb_module.compute_module.core.systolic_array.array_width}.csv",
            #     header=False,
            #     index=False,
            # )
            self.look_up_table.set_index(
                ["M", "N", "K", "ArrayHeight", "ArrayWidth", "Dataflow"],
                inplace=True,
            )
        # print(self.look_up_table)
        # print(self.look_up_table.loc[(32, 16, 256, 16, 16, 'os'), "cycle_count"
        #                              ].item())
        # print('sdfsdfsdfsd')
        # exit()
        M = computational_graph.M
        N = computational_graph.N
        K = computational_graph.K
        data_type = computational_graph.data_type

        l2_tile_M = mapping.l2_tile_M
        l2_tile_N = mapping.l2_tile_N
        l2_tile_K = mapping.l2_tile_K

        assert (
            l2_tile_M * l2_tile_N + l2_tile_N * l2_tile_K + l2_tile_M * l2_tile_K
            <= pcb_module.compute_module.l2_size // self.data_type.word_size
        )

        M_l2_t = M // l2_tile_M
        N_l2_t = N // l2_tile_N
        K_l2_t = K // l2_tile_K
        M_remain = M % l2_tile_M
        N_remain = N % l2_tile_N
        K_remain = K % l2_tile_K

        l2_tiles = np.empty(
            [ceil(M / l2_tile_M), ceil(N / l2_tile_N), ceil(K / l2_tile_K)],
            dtype=self.L2TileSimulator,
        )
        # print('-'*20)
        # print(l2_tiles.shape)
        if M_l2_t * N_l2_t * K_l2_t != 0:
            l2_tiles[:M_l2_t, :N_l2_t, :K_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                l2_tile_N,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain != 0:
            l2_tiles[-1, :N_l2_t, :K_l2_t] = self.L2TileSimulator(
                M_remain,
                l2_tile_N,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if N_remain != 0:
            l2_tiles[:M_l2_t, -1, :K_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                N_remain,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if K_remain != 0:
            l2_tiles[:M_l2_t, :N_l2_t, -1] = self.L2TileSimulator(
                l2_tile_M,
                l2_tile_N,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * N_remain != 0:
            l2_tiles[-1, -1, :K_l2_t] = self.L2TileSimulator(
                M_remain,
                N_remain,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * K_remain != 0:
            l2_tiles[-1, :N_l2_t, -1] = self.L2TileSimulator(
                M_remain,
                l2_tile_N,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if N_remain * K_remain != 0:
            l2_tiles[:M_l2_t, -1, -1] = self.L2TileSimulator(
                l2_tile_M,
                N_remain,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * N_remain * K_remain != 0:
            l2_tiles[-1, -1, -1] = self.L2TileSimulator(
                M_remain,
                N_remain,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )

        total_cycle_count = 0
        total_cycle_count += (
            l2_tiles[0, 0, 0].M_K_io_cycle_count + l2_tiles[0, 0, 0].K_N_io_cycle_count
        )
        previous_m = 0
        previous_n = 0
        previous_k = 0

        total_dram_read_latency = 0  #add
        total_dram_write_latency = 0  #add
        total_dram_read_bytes = 0  #add
        total_dram_write_bytes = 0  #add
        total_compute_latency = 0  #add
        profiler_dram_read_cycle_total = 0  #add
        profiler_dram_write_cycle_total = 0  #add
        profiler_dram_read_bytes_total = 0  #add
        profiler_dram_write_bytes_total = 0  #add
        profiler_compute_cycle_total = 0  #add

        profiler_first_l2_tile = l2_tiles[0, 0, 0]  #add
        profiler_first_read_cycles = profiler_first_l2_tile.M_K_io_cycle_count + profiler_first_l2_tile.K_N_io_cycle_count  #add
        profiler_dram_read_cycle_total += profiler_first_read_cycles  #add
        profiler_dram_read_bytes_total += (profiler_first_l2_tile.M * profiler_first_l2_tile.K + profiler_first_l2_tile.K * profiler_first_l2_tile.N) * data_type.word_size  #add



        for m, n, k in self.generate_tile_loops(
            ceil(M / l2_tile_M),
            ceil(N / l2_tile_N),
            ceil(K / l2_tile_K),
            mapping.l2_loop_order,
        ):
            if m == 0 and n == 0 and k == 0:
                continue

            l2_tile = l2_tiles[m, n, k]
            previous_l2_tile = l2_tiles[previous_m, previous_n, previous_k]

            # current tile read latency
            if m == previous_m and k == previous_k:
                current_tile_read_cycle_count = l2_tile.K_N_io_cycle_count
            elif n == previous_n and k == previous_k:
                current_tile_read_cycle_count = l2_tile.M_K_io_cycle_count
            else:
                current_tile_read_cycle_count = (
                    l2_tile.M_K_io_cycle_count + l2_tile.K_N_io_cycle_count
                )
            if k > 0 and not (m == previous_m and n == previous_n):
                current_tile_read_cycle_count += l2_tile.M_N_io_cycle_count
            profiler_current_tile_read_bytes = 0  #add
            if m == previous_m and k == previous_k:  #add
                profiler_current_tile_read_bytes += l2_tile.K * l2_tile.N * data_type.word_size  #add
            elif n == previous_n and k == previous_k:  #add
                profiler_current_tile_read_bytes += l2_tile.M * l2_tile.K * data_type.word_size  #add
            else:  #add
                profiler_current_tile_read_bytes += (l2_tile.M * l2_tile.K + l2_tile.K * l2_tile.N) * data_type.word_size  #add
            if k > 0 and not (m == previous_m and n == previous_n):  #add
                profiler_current_tile_read_bytes += l2_tile.M * l2_tile.N * data_type.word_size  #add
            profiler_dram_read_cycle_total += current_tile_read_cycle_count  #add
            profiler_dram_read_bytes_total += profiler_current_tile_read_bytes   #add
            # previous tile compute latency
            previous_tile_compute_cycle_count = previous_l2_tile.compute_cycle_count
            if k > 0:
                previous_tile_compute_cycle_count += (
                    previous_l2_tile.K_reduction_cycle_count
                )
            # previous tile write latency
            if m == previous_m and n == previous_n:
                previous_tile_write_cycle_count = 0
            else:
                previous_tile_write_cycle_count = previous_l2_tile.M_N_io_cycle_count
            profiler_previous_tile_write_bytes = 0  #add
            if not (m == previous_m and n == previous_n):  #add
                profiler_previous_tile_write_bytes = previous_l2_tile.M * previous_l2_tile.N * data_type.word_size  #add
            profiler_dram_write_cycle_total += previous_tile_write_cycle_count  #add
            profiler_dram_write_bytes_total += profiler_previous_tile_write_bytes  #add
            profiler_compute_cycle_total += previous_tile_compute_cycle_count  #add

            # read current tile, compute previous tile, write previous tile
            if mapping.is_l2_double_buffering:  # pipelined
                total_cycle_count += (
                    max(
                        current_tile_read_cycle_count, previous_tile_compute_cycle_count
                    )
                    + previous_tile_write_cycle_count
                )
                total_dram_read_latency += max(current_tile_read_cycle_count, previous_tile_compute_cycle_count)
                total_dram_write_latency += previous_tile_write_cycle_count 
                total_compute_latency = None
                total_dram_read_bytes += total_dram_read_latency * pcb_module.io_module.bandwidth / pcb_module.compute_module.clock_freq
                total_dram_write_bytes += total_dram_write_latency * pcb_module.io_module.bandwidth / pcb_module.compute_module.clock_freq
            else:  # non-pipelined
                total_cycle_count += (
                    current_tile_read_cycle_count
                    + previous_tile_compute_cycle_count
                    + previous_tile_write_cycle_count
                )
                total_dram_read_latency += current_tile_read_cycle_count
                total_compute_latency += previous_tile_compute_cycle_count
                total_dram_write_latency += previous_tile_write_cycle_count 
                total_dram_read_bytes += total_dram_read_latency * pcb_module.io_module.bandwidth / pcb_module.compute_module.clock_freq
                total_dram_write_bytes += total_dram_write_latency * pcb_module.io_module.bandwidth / pcb_module.compute_module.clock_freq

            previous_m = m
            previous_n = n
            previous_k = k

        # compute and write last tile
        total_cycle_count += (
            l2_tiles[-1, -1, -1].M_N_io_cycle_count
            + l2_tiles[-1, -1, -1].compute_cycle_count
        )
        profiler_last_l2_tile = l2_tiles[-1, -1, -1]  #add
        profiler_dram_write_cycle_total += profiler_last_l2_tile.M_N_io_cycle_count  #add
        profiler_dram_write_bytes_total += profiler_last_l2_tile.M * profiler_last_l2_tile.N * data_type.word_size  #add
        profiler_compute_cycle_total += profiler_last_l2_tile.compute_cycle_count  #add

        if previous_k > 0:
            total_cycle_count += ceil(l2_tiles[-1, -1, -1].K_reduction_cycle_count)
            profiler_compute_cycle_total += ceil(l2_tiles[-1, -1, -1].K_reduction_cycle_count)  #add
        profiler_l2_to_l1_read_bytes = 0  #add
        profiler_weight_l2_to_l1_read_bytes = 0  #add
        profiler_activation_l2_to_l1_read_bytes = 0  #add
        profiler_activation_l1_to_l2_write_bytes = 0  #add
        profiler_l1_to_l2_write_bytes = 0  #add
        profiler_l2_to_l1_read_cycles = 0  #add
        profiler_l1_to_l2_write_cycles = 0  #add
        profiler_l1_to_core_read_bytes = 0  #add
        profiler_l1_to_core_write_bytes = 0  #add
        profiler_core_compute_cycles = 0  #add
        profiler_core_compute_work_cycles = 0  #add
        for profiler_l2_tile in l2_tiles.flat:  #add
            profiler_l2_to_l1_read_bytes += getattr(profiler_l2_tile, "l2_to_l1_read_bytes", 0)  #add
            profiler_weight_l2_to_l1_read_bytes += getattr(profiler_l2_tile, "weight_l2_to_l1_read_bytes", 0)  #add
            profiler_activation_l2_to_l1_read_bytes += getattr(profiler_l2_tile, "activation_l2_to_l1_read_bytes", 0)  #add
            profiler_l1_to_l2_write_bytes += getattr(profiler_l2_tile, "l1_to_l2_write_bytes", 0)  #add
            profiler_activation_l1_to_l2_write_bytes += getattr(profiler_l2_tile, "l1_to_l2_write_bytes", 0)  #add
            profiler_l2_to_l1_read_cycles += getattr(profiler_l2_tile, "l2_to_l1_read_cycle_count", 0)  #add
            profiler_l1_to_l2_write_cycles += getattr(profiler_l2_tile, "l1_to_l2_write_cycle_count", 0)  #add
            profiler_l1_to_core_read_bytes += getattr(profiler_l2_tile, "l1_to_core_read_bytes", 0)  #add
            profiler_l1_to_core_write_bytes += getattr(profiler_l2_tile, "l1_to_core_write_bytes", 0)  #add
            profiler_core_compute_cycles += getattr(profiler_l2_tile, "core_compute_cycle_count", 0)  #add
            profiler_core_compute_work_cycles += getattr(profiler_l2_tile, "core_compute_work_cycle_count", 0)  #add
        self.profiler.record_dram_latency(profiler_dram_read_cycle_total + profiler_dram_write_cycle_total)  #add
        self.profiler.record_dram_bytes(int(profiler_dram_read_bytes_total), int(profiler_dram_write_bytes_total))  #add
        self.profiler.record_l2_to_l1_latency(profiler_l2_to_l1_read_cycles + profiler_l1_to_l2_write_cycles)  #add
        self.profiler.record_l2_l1_bytes(int(profiler_l2_to_l1_read_bytes), int(profiler_l1_to_l2_write_bytes))  #add
        self.profiler.record_l2_l1_weight_bytes(int(profiler_weight_l2_to_l1_read_bytes), 0)  #add
        self.profiler.record_l2_l1_activation_bytes(int(profiler_activation_l2_to_l1_read_bytes), int(profiler_activation_l1_to_l2_write_bytes))  #add
        self.profiler.record_compute_latency(profiler_core_compute_cycles + profiler_compute_cycle_total)  #add
        self.profiler.current_record["other_stats"]["dram_read_cycles"] = profiler_dram_read_cycle_total  #add
        self.profiler.current_record["other_stats"]["dram_write_cycles"] = profiler_dram_write_cycle_total  #add
        self.profiler.current_record["other_stats"]["l2_to_l1_read_cycles"] = profiler_l2_to_l1_read_cycles  #add
        self.profiler.current_record["other_stats"]["l1_to_l2_write_cycles"] = profiler_l1_to_l2_write_cycles  #add
        self.profiler.current_record["other_stats"]["l1_to_core_read_bytes_model_estimate"] = int(profiler_l1_to_core_read_bytes)  #add
        self.profiler.current_record["other_stats"]["l1_to_core_write_bytes_model_estimate"] = int(profiler_l1_to_core_write_bytes)  #add
        self.profiler.current_record["other_stats"]["core_compute_cycles_from_l2_tiles"] = profiler_core_compute_cycles  #add
        self.profiler.current_record["other_stats"]["core_compute_work_cycles_sum_over_cores"] = profiler_core_compute_work_cycles  #add
        self.profiler.current_record["other_stats"]["l1_to_core_latency_cycles_model_note"] = "not explicitly modeled in original LLMCompass Matmul; recorded bytes only"  #add
        return total_cycle_count  #add
        self.profiler.record_dram_latency((total_dram_read_latency + total_dram_write_latency) / pcb_module.compute_module.clock_freq)
        self.profiler.record_dram_bytes(total_dram_read_bytes + total_dram_write_bytes)
        self.profiler.record_compute_latency(total_compute_latency)
        return total_cycle_count #+ ceil(
        # pcb_module.io_module.latency * 2 * pcb_module.compute_module.clock_freq
        # )

    # def simulate_cim(
    #     self,
    #     computational_graph: ComputationalGraph,
    #     mapping: Mapping,
    #     pcb_module: Device,
    #     sparsity_ratio: float = 0.8,
    # ) -> int:
    #     """CIM-specific simulation: no L1, weights stored in CIM macro.

    #     Architecture:
    #     - No L1 cache (SRAM_size = 0)
    #     - L2 stores only activation tiles
    #     - CIM macro stores weight tiles (size = array_height * array_width * Nbank KB)
    #     - Compute throughput = max_throughput * (1 - sparsity_ratio)
    #     - Weight write latency = weight_tile_bytes / array_width
    #     - Core mapping priority: K -> M -> N
    #     """
    #     M = computational_graph.M
    #     N = computational_graph.N
    #     K = computational_graph.K
    #     data_type = computational_graph.data_type

    #     cim_macro = pcb_module.compute_module.core.cim_macro
    #     core_count = pcb_module.compute_module.core_count
    #     clock_freq = pcb_module.compute_module.clock_freq
    #     l2_size = pcb_module.compute_module.l2_size
    #     l2_bw = pcb_module.compute_module.l2_bandwidth_per_cycle
    #     input_ws = cim_macro.input_word_size
    #     output_ws = cim_macro.output_word_size
    #     array_width = cim_macro.array_width

    #     # Weight write bandwidth: use wu_io_module if available, otherwise fallback to array_width
    #     if hasattr(pcb_module, 'wu_io_module') and pcb_module.wu_io_module is not None:
    #         weight_write_bw_per_cycle = pcb_module.wu_io_module.bandwidth / clock_freq
    #     else:
    #         weight_write_bw_per_cycle = array_width

    #     tile_M = mapping.l2_tile_M
    #     tile_N = mapping.l2_tile_N
    #     tile_K = mapping.l2_tile_K

    #     # Effective throughput considering sparsity
    #     effective_throughput = cim_macro.max_throughput_per_cycle * sparsity_ratio
    #     # effective_throughput = cim_macro.max_throughput_per_cycle
    #     if effective_throughput <= 0:
    #         effective_throughput = cim_macro.max_throughput_per_cycle  # fallback

    #     num_K_tiles = ceil(K / tile_K)
    #     num_M_tiles = ceil(M / tile_M)
    #     num_N_tiles = ceil(N / tile_N)

    #     # Determine K-M-N parallelization factors across cores
    #     K_factor, M_factor, N_factor = self.find_kmn_factors(
    #         core_count, K, M, N, tile_K, tile_M, tile_N
    #     )

    #     cores_used = M_factor

    #     # Per-core tile dimensions
    #     per_core_K = ceil(tile_K / max(K_factor, 1))
    #     per_core_M = ceil(tile_M / max(M_factor, 1))
    #     per_core_N = ceil(tile_N / max(N_factor, 1))

    #     total_cycle_count = 0
    #     profiler_dram_read_bytes_total = 0
    #     profiler_dram_write_bytes_total = 0
    #     profiler_dram_read_cycle_total = 0
    #     profiler_dram_write_cycle_total = 0
    #     profiler_compute_cycle_total = 0
    #     profiler_weight_write_cycles_total = 0
    #     profiler_weight_write_bytes_total = 0

    #     # Iterate over tiles in nmk order: N outermost, M middle, K innermost
    #     # Weight [tile_K, tile_N] is reused across M iterations, only written when (n_idx, k_idx) changes
    #     prev_nk = None
    #     for m_idx, n_idx, k_idx in self.generate_tile_loops(
    #         num_M_tiles, num_N_tiles, num_K_tiles, "nmk"
    #     ):
    #         # Actual tile dimensions (handle remainder)
    #         cur_tile_M = min(tile_M, M - m_idx * tile_M)
    #         cur_tile_N = min(tile_N, N - n_idx * tile_N)
    #         cur_tile_K = min(tile_K, K - k_idx * tile_K)

    #         # Per-core compute cycles
    #         per_core_M_actual = min(per_core_M, cur_tile_M)
    #         per_core_N_actual = min(per_core_N, cur_tile_N)
    #         per_core_K_actual = min(per_core_K, cur_tile_K)

    #         total_ops = 2 * cur_tile_M * cur_tile_N * cur_tile_K * 4
    #         # Total compute cycles: all cores work in parallel
    #         # Each core handles a portion, total throughput = core_count_used * effective_throughput
    #         cores_used = K_factor * M_factor * N_factor
    #         total_compute_cycles = ceil(total_ops / (cores_used * effective_throughput))

    #         # Weight write: only when (n_idx, k_idx) changes (weight reuse across M iterations)
    #         current_nk = (n_idx, k_idx)
    #         if current_nk != prev_nk:
    #             weight_tile_bytes = cur_tile_K * cur_tile_N * input_ws * 0.25 #bitnet
    #             weight_write_cycles = ceil(weight_tile_bytes / weight_write_bw_per_cycle)
    #             profiler_weight_write_bytes_total += weight_tile_bytes
    #             prev_nk = current_nk
    #         else:
    #             weight_write_cycles = 0

    #         # Activation read from DRAM: only M*K (input activation)
    #         activation_read_bytes = cur_tile_M * cur_tile_K * data_type.word_size
    #         activation_read_cycles = ceil(activation_read_bytes / l2_bw)

    #         # Activation write to DRAM: M*N (output activation)
    #         activation_write_bytes = cur_tile_M * cur_tile_N * output_ws
    #         activation_write_cycles_dram = ceil(
    #             activation_write_bytes
    #             / (pcb_module.io_module.bandwidth / pcb_module.compute_module.clock_freq)
    #         )

    #         # K reduction: if K_factor > 1, partial sums need to be reduced
    #         K_reduction_cycles = 0
    #         if K_factor > 1:
    #             K_reduction_cycles = ceil(
    #                 cur_tile_M * cur_tile_N * output_ws / l2_bw
    #             ) * (K_factor - 1)

    #         # Profiling
    #         profiler_dram_read_bytes_total += activation_read_bytes
    #         profiler_dram_write_bytes_total += activation_write_bytes
    #         profiler_compute_cycle_total += total_compute_cycles
    #         profiler_weight_write_cycles_total += weight_write_cycles

    #         # Pipeline: max(compute, activation_read) + weight_write + K_reduction
    #         tile_cycles = (
    #             max(total_compute_cycles, activation_read_cycles)
    #             + weight_write_cycles
    #             + K_reduction_cycles
    #         )
    #         profiler_dram_read_cycle_total += activation_read_cycles
    #         profiler_dram_write_cycle_total += activation_write_cycles_dram

    #         total_cycle_count += tile_cycles

    #     # Add final write
    #     total_cycle_count += profiler_dram_write_cycle_total

    #     # Record profiler stats
    #     self.profiler.record_dram_bytes(
    #         int(profiler_dram_read_bytes_total),
    #         int(profiler_dram_write_bytes_total)
    #     )
    #     self.profiler.record_dram_latency(
    #         profiler_dram_read_cycle_total + profiler_dram_write_cycle_total
    #     )
    #     self.profiler.record_l2_to_l1_latency(0)  # No L1 in CIM
    #     self.profiler.record_l2_l1_bytes(0, 0)
    #     self.profiler.record_l2_l1_weight_bytes(0, 0)
    #     self.profiler.record_l2_l1_activation_bytes(0, 0)
    #     self.profiler.record_compute_latency(profiler_compute_cycle_total)
    #     self.profiler.current_record["other_stats"]["sparsity_ratio"] = sparsity_ratio
    #     self.profiler.current_record["other_stats"]["effective_throughput"] = effective_throughput
    #     self.profiler.current_record["other_stats"]["K_factor"] = K_factor
    #     self.profiler.current_record["other_stats"]["M_factor"] = M_factor
    #     self.profiler.current_record["other_stats"]["N_factor"] = N_factor
    #     self.profiler.current_record["other_stats"]["weight_write_cycles"] = profiler_weight_write_cycles_total
    #     self.profiler.current_record["other_stats"]["weight_write_bytes"] = int(profiler_weight_write_bytes_total)
    #     self.profiler.current_record["other_stats"]["cim_arch"] = "CIM_macro"

    #     # Store weight_write_cycles and bytes as instance attribute for external access
    #     self.last_cim_weight_write_cycles = profiler_weight_write_cycles_total
    #     self.last_cim_weight_write_bytes = profiler_weight_write_bytes_total

    #     return total_cycle_count

    def simulate_cim_weight_major(
        self,
        computational_graph: ComputationalGraph,
        mapping: Mapping,
        pcb_module: Device,
        prefill_effective_speedup: float = 1.0,
    ) -> int:
        """
        K-major weight-major prefill.

        Execution order:

            for n:
              for k:
                load and multicast W[k,n]
                for m:
                  stream A[m,k] through the resident weight tile

        The macro-factor priority matches K-major activation-major prefill:
        K_factor is maximized, N_factor is one physical strip, and M_factor
        uses the remaining macro groups.
        """
        M = computational_graph.M
        N = computational_graph.N
        K = computational_graph.K

        cim_macro = pcb_module.compute_module.core.cim_macro
        core_count = pcb_module.compute_module.core_count
        clock_freq = pcb_module.compute_module.clock_freq
        l2_size = pcb_module.compute_module.l2_size
        l2_bw = pcb_module.compute_module.l2_bandwidth_per_cycle
        output_ws = cim_macro.output_word_size
        psum_ws = int(
            getattr(cim_macro, "psum_word_size", output_ws)
        )
        act_ws = float(
            getattr(
                cim_macro,
                "prefill_activation_element_size",
                8 / 8,
            )
        )

        tile_M = mapping.l2_tile_M
        tile_N = mapping.l2_tile_N
        tile_K = mapping.l2_tile_K
        physical_macro_K = (
            cim_macro.Nbank * cim_macro.array_height
        )
        physical_macro_N = cim_macro.array_width

        K_factor = max(1, ceil(tile_K / physical_macro_K))
        expected_tile_K = min(
            K,
            physical_macro_K
            * min(core_count, ceil(K / physical_macro_K)),
        )
        if tile_K != expected_tile_K:
            raise ValueError(
                "weight-major prefill requires the maximum K-major "
                f"super-tile: tile_K={tile_K}, expected={expected_tile_K}"
            )
        if tile_N != min(N, physical_macro_N):
            raise ValueError(
                "weight-major prefill requires the physical N strip"
            )
        if mapping.l2_loop_order != "nkm":
            raise ValueError(
                "weight-major prefill requires nkm, got "
                f"{mapping.l2_loop_order!r}"
            )

        num_M_tiles = ceil(M / tile_M)
        num_N_tiles = ceil(N / tile_N)
        num_K_tiles = ceil(K / tile_K)
        M_factor = max(
            1,
            min(core_count // K_factor, tile_M),
        )

        weight_bytes_per_element = float(
            getattr(
                cim_macro,
                "weight_storage_bytes_per_element",
                1,
            )
        )
        if (
            hasattr(pcb_module, "wu_io_module")
            and pcb_module.wu_io_module is not None
        ):
            weight_write_bw_per_cycle = (
                pcb_module.wu_io_module.bandwidth / clock_freq
            )
        else:
            weight_write_bw_per_cycle = max(
                1.0,
                float(cim_macro.array_width),
            )

        io_bw_per_cycle = (
            pcb_module.io_module.bandwidth / clock_freq
        )
        effective_throughput_per_macro = (
            cim_macro.max_throughput_per_cycle
            * prefill_effective_speedup
        )
        if (
            io_bw_per_cycle <= 0
            or l2_bw <= 0
            or effective_throughput_per_macro <= 0
        ):
            raise ValueError(
                "Invalid bandwidth or throughput in weight-major prefill"
            )

        hbm_output_write_bytes = M * N * output_ws
        hbm_output_write_cycles = ceil(
            hbm_output_write_bytes / io_bw_per_cycle
        )

        total_cycle_count = 0
        compute_cycles = 0
        useful_ops = 0.0
        weight_cycles = 0
        weight_hbm_bytes = 0.0
        local_weight_bytes = 0.0
        weight_update_count = 0
        hbm_activation_bytes = 0.0
        hbm_activation_cycles = 0
        hbm_psum_read_bytes = 0.0
        hbm_psum_read_cycles = 0
        hbm_psum_write_bytes = 0.0
        hbm_psum_write_cycles = 0
        gb_activation_read_bytes = 0
        gb_psum_read_bytes = 0
        gb_psum_write_bytes = 0
        gb_read_cycles = 0
        gb_write_cycles = 0

        cache_hits = 0
        cache_misses = 0
        cache_evictions = 0
        activation_tile_cache = {}
        max_activation_tile_bytes = 0

        # The mapper preserves the activation-major M-chunk geometry.  In nkm
        # order, however, a new activation chunk is streamed for every (n,k)
        # tile.  If more than one K super-tile exists, the simulator spills
        # intermediate psums to HBM rather than requiring all M psums for the
        # current N strip to remain simultaneously resident.
        resident_psum_bytes = tile_M * tile_N * psum_ws
        pipeline_activation_reserve = (
            (2 if mapping.is_l2_double_buffering else 1)
            * tile_M
            * K
            * act_ws
        )
        activation_cache_capacity = max(
            0,
            l2_size
            - resident_psum_bytes
            - pipeline_activation_reserve,
        )

        # If the full N-strip psum cannot remain resident together with the
        # activation working set, intermediate K super-tiles spill through HBM.
        # With an effectively unlimited Global Buffer this evaluates to False,
        # so psums stay on-chip and no HBM spill/reload traffic is charged.
        psum_spill_enabled = (
            M * tile_N * psum_ws
            + pipeline_activation_reserve
            > l2_size
        )

        # N -> K -> M.  A weight tile is resident for all M tiles.
        for n_idx in range(num_N_tiles):
            cur_tile_N = min(
                tile_N,
                N - n_idx * tile_N,
            )

            for k_idx in range(num_K_tiles):
                cur_tile_K = min(
                    tile_K,
                    K - k_idx * tile_K,
                )

                compressed_weight_tile_bytes = (
                    cur_tile_K
                    * cur_tile_N
                    * weight_bytes_per_element
                )
                local_replicated_weight_tile_bytes = (
                    compressed_weight_tile_bytes * M_factor
                )
                weight_update_cycles = ceil(
                    compressed_weight_tile_bytes
                    / weight_write_bw_per_cycle
                )

                weight_hbm_bytes += compressed_weight_tile_bytes
                local_weight_bytes += local_replicated_weight_tile_bytes
                weight_cycles += weight_update_cycles
                weight_update_count += 1

                # The same weight tile now streams every activation tile.
                for m_idx in range(num_M_tiles):
                    cur_tile_M = min(
                        tile_M,
                        M - m_idx * tile_M,
                    )

                    activation_tile_bytes = (
                        cur_tile_M
                        * cur_tile_K
                        * act_ws
                    )
                    max_activation_tile_bytes = max(
                        max_activation_tile_bytes,
                        activation_tile_bytes,
                    )

                    act_key = (m_idx, k_idx)
                    if act_key in activation_tile_cache:
                        activation_tile_cache[act_key] = (
                            activation_tile_cache.pop(act_key)
                        )
                        cache_hits += 1
                        activation_load_cycles = 0
                    else:
                        cache_misses += 1
                        hbm_activation_bytes += activation_tile_bytes
                        activation_load_cycles = ceil(
                            activation_tile_bytes
                            / io_bw_per_cycle
                        )
                        hbm_activation_cycles += activation_load_cycles

                        if (
                            activation_tile_bytes
                            > activation_cache_capacity
                        ):
                            cache_evictions += len(
                                activation_tile_cache
                            )
                            activation_tile_cache.clear()
                        else:
                            activation_tile_cache[act_key] = (
                                activation_tile_bytes
                            )
                            while (
                                sum(activation_tile_cache.values())
                                > activation_cache_capacity
                            ):
                                evicted = next(
                                    iter(activation_tile_cache)
                                )
                                activation_tile_cache.pop(evicted)
                                cache_evictions += 1

                    tokens_per_slowest_macro = ceil(
                        cur_tile_M / M_factor
                    )
                    physical_compute_K = (
                        ceil(
                            (cur_tile_K / K_factor)
                            / cim_macro.Nbank
                        )
                        * cim_macro.Nbank
                    )
                    per_macro_physical_ops = (
                        2
                        * tokens_per_slowest_macro
                        * physical_macro_N
                        * physical_compute_K
                    )
                    tile_compute_cycles = ceil(
                        per_macro_physical_ops
                        / effective_throughput_per_macro
                    )
                    compute_cycles += tile_compute_cycles
                    useful_ops += (
                        2
                        * cur_tile_M
                        * cur_tile_N
                        * cur_tile_K
                        * 4
                    )

                    current_gb_activation_read = (
                        cur_tile_M
                        * cur_tile_K
                        * act_ws
                    )
                    psum_tile_bytes = (
                        cur_tile_M
                        * cur_tile_N
                        * psum_ws
                    )
                    current_gb_psum_read = (
                        psum_tile_bytes
                        if k_idx > 0
                        else 0
                    )
                    current_gb_psum_write = psum_tile_bytes

                    # Intermediate K super-tiles spill/reload partial sums in
                    # HBM.  The final K super-tile is eventually covered by the
                    # one-time final output write.
                    current_hbm_psum_read_bytes = (
                        psum_tile_bytes
                        if psum_spill_enabled and k_idx > 0
                        else 0
                    )
                    current_hbm_psum_write_bytes = (
                        psum_tile_bytes
                        if (
                            psum_spill_enabled
                            and k_idx < num_K_tiles - 1
                        )
                        else 0
                    )
                    current_hbm_psum_read_cycles = ceil(
                        current_hbm_psum_read_bytes
                        / io_bw_per_cycle
                    )
                    current_hbm_psum_write_cycles = ceil(
                        current_hbm_psum_write_bytes
                        / io_bw_per_cycle
                    )

                    hbm_psum_read_bytes += current_hbm_psum_read_bytes
                    hbm_psum_read_cycles += current_hbm_psum_read_cycles
                    hbm_psum_write_bytes += current_hbm_psum_write_bytes
                    hbm_psum_write_cycles += current_hbm_psum_write_cycles

                    current_gb_read_cycles = ceil(
                        (
                            current_gb_activation_read
                            + current_gb_psum_read
                        )
                        / l2_bw
                    )
                    current_gb_write_cycles = ceil(
                        current_gb_psum_write / l2_bw
                    )

                    gb_activation_read_bytes += (
                        current_gb_activation_read
                    )
                    gb_psum_read_bytes += current_gb_psum_read
                    gb_psum_write_bytes += current_gb_psum_write
                    gb_read_cycles += current_gb_read_cycles
                    gb_write_cycles += current_gb_write_cycles

                    if mapping.is_l2_double_buffering:
                        compute_and_read_cycles = max(
                            tile_compute_cycles,
                            current_gb_read_cycles,
                        )
                    else:
                        compute_and_read_cycles = (
                            tile_compute_cycles
                            + current_gb_read_cycles
                        )

                    total_cycle_count += (
                        activation_load_cycles
                        + current_hbm_psum_read_cycles
                        + compute_and_read_cycles
                        + current_gb_write_cycles
                        + current_hbm_psum_write_cycles
                    )

                # Charge the weight update once after all M tiles have used
                # this resident (N,K) weight tile.
                total_cycle_count += weight_update_cycles

        total_cycle_count += hbm_output_write_cycles

        dram_read_bytes = int(
            hbm_activation_bytes
            + weight_hbm_bytes
            + hbm_psum_read_bytes
        )
        dram_write_bytes = int(
            hbm_output_write_bytes
            + hbm_psum_write_bytes
        )
        dram_cycles = (
            hbm_activation_cycles
            + weight_cycles
            + hbm_psum_read_cycles
            + hbm_psum_write_cycles
            + hbm_output_write_cycles
        )
        gb_read_bytes = (
            gb_activation_read_bytes + gb_psum_read_bytes
        )
        gb_write_bytes = gb_psum_write_bytes

        self.profiler.record_dram_bytes(
            dram_read_bytes,
            dram_write_bytes,
        )
        self.profiler.record_dram_latency(dram_cycles)
        self.profiler.record_l2_to_l1_latency(
            gb_read_cycles + gb_write_cycles
        )
        self.profiler.record_l2_l1_bytes(
            int(gb_read_bytes),
            int(gb_write_bytes),
        )
        self.profiler.record_l2_l1_weight_bytes(0, 0)
        self.profiler.record_l2_l1_activation_bytes(
            int(gb_read_bytes),
            int(gb_write_bytes),
        )
        self.profiler.record_compute_latency(compute_cycles)

        stats = self.profiler.current_record["other_stats"]
        stats.update(
            {
                "mapping_mode": (
                    "effloc_prefill_kmajor_weight_replication"
                ),
                "prefill_effective_speedup": (
                    prefill_effective_speedup
                ),
                "effective_throughput_per_macro": (
                    effective_throughput_per_macro
                ),
                "K_factor": K_factor,
                "M_factor": M_factor,
                "N_factor": 1,
                "token_parallel_macros": M_factor,
                "tile_M": tile_M,
                "tile_N": tile_N,
                "tile_K": tile_K,
                "num_M_tiles": num_M_tiles,
                "num_N_tiles": num_N_tiles,
                "num_K_tiles": num_K_tiles,
                "weight_update_count": weight_update_count,
                "weight_write_cycles": weight_cycles,
                "weight_write_bytes": int(weight_hbm_bytes),
                "local_replicated_weight_write_bytes": int(
                    local_weight_bytes
                ),
                "hbm_activation_read_bytes": int(
                    hbm_activation_bytes
                ),
                "hbm_psum_read_bytes": int(hbm_psum_read_bytes),
                "hbm_psum_write_bytes": int(hbm_psum_write_bytes),
                "hbm_psum_read_cycles": hbm_psum_read_cycles,
                "hbm_psum_write_cycles": hbm_psum_write_cycles,
                "activation_tile_bytes": int(
                    max_activation_tile_bytes
                ),
                "activation_cache_capacity_bytes": int(
                    activation_cache_capacity
                ),
                "activation_cache_hits": cache_hits,
                "activation_cache_misses": cache_misses,
                "activation_cache_evictions": cache_evictions,
                "hbm_output_write_bytes": int(
                    hbm_output_write_bytes
                ),
                "global_buffer_activation_read_bytes": int(
                    gb_activation_read_bytes
                ),
                "global_buffer_psum_read_bytes": int(
                    gb_psum_read_bytes
                ),
                "global_buffer_psum_write_bytes": int(
                    gb_psum_write_bytes
                ),
                "useful_ops": float(useful_ops),
                "cim_arch": "EffLoc",
            }
        )

        return int(total_cycle_count)

    def simulate_cim_activation_major(
        self,
        computational_graph: ComputationalGraph,
        mapping: Mapping,
        pcb_module: Device,
        prefill_effective_speedup: float = 1.0,
    ) -> int:
        """
        Activation-major prefill:

            for m_chunk:
                load A[m_chunk, all K]
                for all N tiles:
                    for all K tiles:
                        replicate W[k,n] in all macros
                        compute the M-token chunk

        This mode makes activation resident first and streams all weight tiles
        for each activation chunk.
        """
        M = computational_graph.M
        N = computational_graph.N
        K = computational_graph.K

        cim_macro = pcb_module.compute_module.core.cim_macro
        core_count = pcb_module.compute_module.core_count
        clock_freq = pcb_module.compute_module.clock_freq
        l2_size = pcb_module.compute_module.l2_size
        l2_bw = pcb_module.compute_module.l2_bandwidth_per_cycle
        output_ws = cim_macro.output_word_size
        psum_ws = int(getattr(cim_macro, "psum_word_size", output_ws))
        act_ws = float(
            getattr(
                cim_macro,
                "prefill_activation_element_size",
                8 / 8,
            )
        )

        tile_M = mapping.l2_tile_M
        tile_N = mapping.l2_tile_N
        tile_K = mapping.l2_tile_K
        physical_macro_K = cim_macro.Nbank * cim_macro.array_height
        physical_macro_N = cim_macro.array_width

        # A K-major super-tile may span multiple physical macro-K groups.
        K_factor = max(1, ceil(tile_K / physical_macro_K))
        expected_tile_K = min(
            K,
            physical_macro_K
            * min(core_count, ceil(K / physical_macro_K)),
        )
        if tile_K != expected_tile_K:
            raise ValueError(
                "activation-major prefill requires the maximum K-major "
                f"super-tile: tile_K={tile_K}, expected={expected_tile_K}"
            )
        if tile_N != min(N, physical_macro_N):
            raise ValueError("activation-major prefill requires the physical N tile")
        if mapping.l2_loop_order != "mnk":
            raise ValueError(
                f"activation-major prefill requires mnk, got {mapping.l2_loop_order!r}"
            )

        num_M_tiles = ceil(M / tile_M)
        num_N_tiles = ceil(N / tile_N)
        num_K_tiles = ceil(K / tile_K)
        M_factor = max(
            1,
            min(core_count // K_factor, tile_M),
        )

        weight_bytes_per_element = float(
            getattr(cim_macro, "weight_storage_bytes_per_element", 1)
        )
        if hasattr(pcb_module, "wu_io_module") and pcb_module.wu_io_module is not None:
            weight_write_bw_per_cycle = (
                pcb_module.wu_io_module.bandwidth / clock_freq
            )
        else:
            weight_write_bw_per_cycle = max(1.0, float(cim_macro.array_width))

        io_bw_per_cycle = pcb_module.io_module.bandwidth / clock_freq
        effective_throughput_per_macro = (
            cim_macro.max_throughput_per_cycle
            * prefill_effective_speedup
        )
        if io_bw_per_cycle <= 0 or l2_bw <= 0 or effective_throughput_per_macro <= 0:
            raise ValueError("Invalid CIM bandwidth or throughput in activation-major prefill")

        hbm_output_write_bytes = M * N * output_ws
        hbm_output_write_cycles = ceil(hbm_output_write_bytes / io_bw_per_cycle)
        total_cycle_count = 0

        compute_cycles = 0
        useful_ops = 0.0
        weight_cycles = 0
        weight_hbm_bytes = 0.0
        local_weight_bytes = 0.0
        weight_update_count = 0
        hbm_activation_bytes = 0.0
        hbm_activation_cycles = 0
        gb_activation_read_bytes = 0
        gb_psum_read_bytes = 0
        gb_psum_write_bytes = 0
        gb_read_cycles = 0
        gb_write_cycles = 0
        cache_hits = 0
        cache_misses = 0
        max_activation_chunk_bytes = 0

        # M is outermost: bring in an activation chunk before any weights.
        for m_idx in range(num_M_tiles):
            cur_tile_M = min(tile_M, M - m_idx * tile_M)
            activation_chunk_bytes = cur_tile_M * K * act_ws
            resident_psum_bytes = cur_tile_M * tile_N * psum_ws
            if activation_chunk_bytes + resident_psum_bytes > l2_size:
                raise ValueError(
                    "Global Buffer cannot hold activation-major chunk: "
                    f"required={activation_chunk_bytes + resident_psum_bytes}B, "
                    f"l2_size={l2_size}B"
                )

            activation_load_cycles = ceil(activation_chunk_bytes / io_bw_per_cycle)
            hbm_activation_bytes += activation_chunk_bytes
            hbm_activation_cycles += activation_load_cycles
            total_cycle_count += activation_load_cycles
            max_activation_chunk_bytes = max(
                max_activation_chunk_bytes,
                activation_chunk_bytes,
            )

            # The (m,k) tiles are compulsory misses when the chunk arrives and
            # hit from GB for every subsequent N tile.
            cache_misses += num_K_tiles
            cache_hits += (num_N_tiles - 1) * num_K_tiles

            # N -> K keeps the psum for the current N strip local while the
            # reduction dimension K is traversed.
            for n_idx in range(num_N_tiles):
                cur_tile_N = min(tile_N, N - n_idx * tile_N)
                for k_idx in range(num_K_tiles):
                    cur_tile_K = min(tile_K, K - k_idx * tile_K)

                    compressed_weight_tile_bytes = (
                        cur_tile_K
                        * cur_tile_N
                        * weight_bytes_per_element
                    )
                    local_replicated_weight_tile_bytes = (
                        compressed_weight_tile_bytes * M_factor
                    )
                    weight_update_cycles = ceil(
                        compressed_weight_tile_bytes / weight_write_bw_per_cycle
                    )

                    weight_hbm_bytes += compressed_weight_tile_bytes
                    local_weight_bytes += local_replicated_weight_tile_bytes
                    weight_cycles += weight_update_cycles
                    weight_update_count += 1

                    tokens_per_slowest_macro = ceil(cur_tile_M / M_factor)
                    physical_compute_K = (
                        ceil(
                            (cur_tile_K / K_factor)
                            / cim_macro.Nbank
                        )
                        * cim_macro.Nbank
                    )
                    per_macro_physical_ops = (
                        2
                        * tokens_per_slowest_macro
                        * physical_macro_N
                        * physical_compute_K
                    )
                    tile_compute_cycles = ceil(
                        per_macro_physical_ops / effective_throughput_per_macro
                    )
                    compute_cycles += tile_compute_cycles
                    useful_ops += (
                        2 * cur_tile_M * cur_tile_N * cur_tile_K * 4
                    )

                    current_gb_activation_read_bytes = (
                        cur_tile_M * cur_tile_K * act_ws
                    )
                    psum_tile_bytes = cur_tile_M * cur_tile_N * psum_ws
                    current_gb_psum_read_bytes = (
                        psum_tile_bytes if k_idx > 0 else 0
                    )
                    current_gb_psum_write_bytes = psum_tile_bytes
                    current_gb_read_cycles = ceil(
                        (
                            current_gb_activation_read_bytes
                            + current_gb_psum_read_bytes
                        )
                        / l2_bw
                    )
                    current_gb_write_cycles = ceil(
                        current_gb_psum_write_bytes / l2_bw
                    )

                    gb_activation_read_bytes += current_gb_activation_read_bytes
                    gb_psum_read_bytes += current_gb_psum_read_bytes
                    gb_psum_write_bytes += current_gb_psum_write_bytes
                    gb_read_cycles += current_gb_read_cycles
                    gb_write_cycles += current_gb_write_cycles

                    if mapping.is_l2_double_buffering:
                        compute_and_read_cycles = max(
                            tile_compute_cycles,
                            current_gb_read_cycles,
                        )
                    else:
                        compute_and_read_cycles = (
                            tile_compute_cycles + current_gb_read_cycles
                        )

                    total_cycle_count += (
                        weight_update_cycles
                        + compute_and_read_cycles
                        + current_gb_write_cycles
                    )

        total_cycle_count += hbm_output_write_cycles

        dram_read_bytes = int(hbm_activation_bytes + weight_hbm_bytes)
        dram_write_bytes = int(hbm_output_write_bytes)
        dram_cycles = (
            hbm_activation_cycles
            + weight_cycles
            + hbm_output_write_cycles
        )
        gb_read_bytes = gb_activation_read_bytes + gb_psum_read_bytes
        gb_write_bytes = gb_psum_write_bytes

        self.profiler.record_dram_bytes(dram_read_bytes, dram_write_bytes)
        self.profiler.record_dram_latency(dram_cycles)
        self.profiler.record_l2_to_l1_latency(gb_read_cycles + gb_write_cycles)
        self.profiler.record_l2_l1_bytes(
            int(gb_read_bytes),
            int(gb_write_bytes),
        )
        self.profiler.record_l2_l1_weight_bytes(0, 0)
        self.profiler.record_l2_l1_activation_bytes(
            int(gb_read_bytes),
            int(gb_write_bytes),
        )
        self.profiler.record_compute_latency(compute_cycles)

        stats = self.profiler.current_record["other_stats"]
        stats.update(
            {
                "mapping_mode": "effloc_prefill_activation_major_weight_replication",
                "prefill_effective_speedup": prefill_effective_speedup,
                "effective_throughput_per_macro": effective_throughput_per_macro,
                "K_factor": K_factor,
                "M_factor": M_factor,
                "N_factor": 1,
                "token_parallel_macros": M_factor,
                "tile_M": tile_M,
                "tile_N": tile_N,
                "tile_K": tile_K,
                "num_M_tiles": num_M_tiles,
                "num_N_tiles": num_N_tiles,
                "num_K_tiles": num_K_tiles,
                "weight_update_count": weight_update_count,
                "weight_write_cycles": weight_cycles,
                "weight_write_bytes": int(weight_hbm_bytes),
                "local_replicated_weight_write_bytes": int(local_weight_bytes),
                "hbm_activation_read_bytes": int(hbm_activation_bytes),
                "activation_chunk_bytes": int(max_activation_chunk_bytes),
                "activation_cache_capacity_bytes": int(
                    max_activation_chunk_bytes
                ),
                "activation_cache_hits": cache_hits,
                "activation_cache_misses": cache_misses,
                "activation_cache_evictions": 0,
                "hbm_output_write_bytes": int(hbm_output_write_bytes),
                "global_buffer_activation_read_bytes": int(
                    gb_activation_read_bytes
                ),
                "global_buffer_psum_read_bytes": int(gb_psum_read_bytes),
                "global_buffer_psum_write_bytes": int(gb_psum_write_bytes),
                "useful_ops": float(useful_ops),
                "cim_arch": "EffLoc",
            }
        )

        return int(total_cycle_count)

    def simulate_cim(
        self,
        computational_graph: ComputationalGraph,
        mapping: Mapping,
        pcb_module: Device,
        prefill_effective_speedup: float = 1.0,
    ) -> int:
        """
        EffLoc prefill simulation with strict cross-macro
        weight replication.

        Macro mapping:
            K_factor = 1
            M_factor = min(core_count, current_M_tile)
            N_factor = 1

        所有 macro 保存相同权重块，各自处理不同 token。

        K 和 N 不跨 macro 分片：
            - K tile 串行更新并累加 partial sum；
            - N tile 串行更新权重；
            - 每个 (N_tile, K_tile) 权重块只加载一次；
            - 该权重块处理完所有 M tile 后才更新。

        prefill_effective_speedup 应来自 Mapping_stat，
        其中已经包含：
            - FP8 SMMM effective-bit skipping；
            - exponent-range overhead；
            - cross-bank barrier；
            - inter-macro asynchronous token scheduling。
        """

        M = computational_graph.M
        N = computational_graph.N
        K = computational_graph.K
        data_type = computational_graph.data_type

        cim_macro = (
            pcb_module.compute_module.core.cim_macro
        )

        core_count = (
            pcb_module.compute_module.core_count
        )

        l2_size = (
            pcb_module.compute_module.l2_size
        )

        clock_freq = (
            pcb_module.compute_module.clock_freq
        )

        l2_bw = (
            pcb_module.compute_module
            .l2_bandwidth_per_cycle
        )

        output_ws = cim_macro.output_word_size

        # Stored prefill activation width.  This must match the width used
        # by heuristic-CIM when selecting tile_M and deciding double buffering.
        act_ws = float(
            getattr(
                cim_macro,
                "prefill_activation_element_size",
                8 / 8,
            )
        )

        # Partial sum 的实际存储宽度。
        # 若硬件是 INT32 accumulator，可在 CIMMacro 中设置为 4。
        psum_ws = int(
            getattr(
                cim_macro,
                "psum_word_size",
                output_ws,
            )
        )

        if M <= 1:
            raise ValueError(
                "该 simulate_cim 只支持 EffLoc prefill，"
                "要求 M > 1。"
            )

        if prefill_effective_speedup <= 0:
            raise ValueError(
                "prefill_effective_speedup must be positive, "
                f"got {prefill_effective_speedup}"
            )

        if l2_bw <= 0:
            raise ValueError(
                f"Invalid L2/global-buffer bandwidth: {l2_bw}"
            )

        tile_M = mapping.l2_tile_M
        tile_N = mapping.l2_tile_N
        tile_K = mapping.l2_tile_K

        # ============================================================
        # 检查是否严格使用物理 macro tile
        # ============================================================

        physical_tile_K = (
            cim_macro.Nbank
            * cim_macro.array_height
        )

        physical_tile_N = (
            cim_macro.array_width
        )

        expected_tile_K = min(
            K,
            physical_tile_K,
        )

        expected_tile_N = min(
            N,
            physical_tile_N,
        )

        if tile_K != expected_tile_K:
            raise ValueError(
                "EffLoc prefill 必须使用物理 K tile："
                f"tile_K={tile_K}, "
                f"expected={expected_tile_K}"
            )

        if tile_N != expected_tile_N:
            raise ValueError(
                "EffLoc prefill 必须使用物理 N tile："
                f"tile_N={tile_N}, "
                f"expected={expected_tile_N}"
            )

        if mapping.l2_loop_order != "nkm":
            raise ValueError(
                "EffLoc prefill 权重复用要求 loop order='nkm'，"
                f"当前为 {mapping.l2_loop_order!r}"
            )

        num_M_tiles = ceil(M / tile_M)
        num_N_tiles = ceil(N / tile_N)
        num_K_tiles = ceil(K / tile_K)

        # ============================================================
        # BitNet ternary weight：2 bit / weight
        #
        # 建议在 CIMMacro 中显式设置：
        #   weight_storage_bytes_per_element = 0.25
        # ============================================================

        weight_bytes_per_element = float(
            getattr(
                cim_macro,
                "weight_storage_bytes_per_element",
                1,
            )
        )

        # ============================================================
        # Weight-update bandwidth
        # ============================================================

        if (
            hasattr(pcb_module, "wu_io_module")
            and pcb_module.wu_io_module is not None
        ):
            weight_write_bw_per_cycle = (
                pcb_module.wu_io_module.bandwidth
                / clock_freq
            )
        else:
            # fallback：每周期写一行的全部物理列
            weight_write_bw_per_cycle = max(
                1.0,
                float(cim_macro.array_width),
            )

        io_bw_per_cycle = (
            pcb_module.io_module.bandwidth
            / clock_freq
        )

        if io_bw_per_cycle <= 0:
            raise ValueError(
                f"Invalid external IO bandwidth: "
                f"{io_bw_per_cycle}"
            )

        # ============================================================
        # 单个 macro 的 effective throughput
        # ============================================================

        base_throughput_per_macro = (
            cim_macro.max_throughput_per_cycle
        )

        effective_throughput_per_macro = (
            base_throughput_per_macro
            * prefill_effective_speedup
        )

        # ============================================================
        # Off-chip activation traffic
        #
        # Activation 按 (M_tile, K_tile) 分批从 HBM 加载到 GB。
        # 同一 (m,k) activation tile 被所有 N tile 复用，
        # 只在第一次遇到时从 HBM 加载。
        #
        # 最终 output 整层只写回一次。
        #
        # N tile 引起的 activation 重复访问属于片上
        # global-buffer -> macro 流量，不是 DRAM 流量。
        # ============================================================

        hbm_output_write_bytes = (
            M
            * N
            * output_ws
            * num_K_tiles
        )

        hbm_output_write_cycles = ceil(
            hbm_output_write_bytes
            / io_bw_per_cycle
        )

        total_cycle_count = 0

        # ============================================================
        # Profiler counters
        # ============================================================

        profiler_compute_cycles = 0

        profiler_weight_fetch_cycles = 0
        profiler_weight_hbm_bytes = 0.0
        profiler_local_weight_write_bytes = 0.0
        profiler_weight_update_count = 0

        profiler_gb_activation_read_bytes = 0
        profiler_gb_psum_read_bytes = 0
        profiler_gb_psum_write_bytes = 0

        profiler_gb_read_cycles = 0
        profiler_gb_write_cycles = 0

        profiler_hbm_activation_bytes = 0.0
        profiler_hbm_activation_cycles = 0

        prev_nk = None

        # ==================================================================
        # Finite Global-Buffer activation cache
        #
        # The old `loaded_act_tiles` set modeled an infinitely large GB: once
        # an (m,k) activation tile was read, it was assumed available for all
        # later N tiles.  Here we reserve the resident N-strip partial sums
        # and the activation pipeline buffers first.  Only the remaining GB
        # capacity may retain older activation tiles for cross-N reuse.
        # Replacement is LRU.
        # ==================================================================
        psum_strip_bytes = M * tile_N * psum_ws
        pipeline_activation_reserve_bytes = (
            (2 if mapping.is_l2_double_buffering else 1)
            * tile_M
            * tile_K
            * act_ws
        )
        activation_cache_capacity_bytes = max(
            0,
            l2_size
            - psum_strip_bytes
            - pipeline_activation_reserve_bytes,
        )
        activation_tile_cache = {}
        activation_cache_hits = 0
        activation_cache_misses = 0
        activation_cache_evictions = 0

        # ============================================================
        # N -> K -> M
        #
        # 每个 (n,k) 权重块：
        #   1. 从 off-chip 读取一次；
        #   2. multicast 并复制到全部 macro；
        #   3. 所有 macro 异步处理不同 token；
        #   4. 全部 token 处理完成后更新下一权重块。
        # ============================================================

        for m_idx, n_idx, k_idx in self.generate_tile_loops(
            num_M_tiles,
            num_N_tiles,
            num_K_tiles,
            "nkm",
        ):
            cur_tile_M = min(
                tile_M,
                M - m_idx * tile_M,
            )

            cur_tile_N = min(
                tile_N,
                N - n_idx * tile_N,
            )

            cur_tile_K = min(
                tile_K,
                K - k_idx * tile_K,
            )

            # ========================================================
            # HBM -> GB: activation tile load
            #
            # 同一 (m,k) activation tile 被所有 N tile 复用，
            # 只在第一次遇到时从 HBM 加载到 GB。
            # ========================================================

            act_key = (m_idx, k_idx)
            activation_tile_bytes = (
                cur_tile_M
                * cur_tile_K
                * act_ws
            )
            if act_key in activation_tile_cache:
                # LRU touch.
                activation_tile_cache[act_key] = (
                    activation_tile_cache.pop(act_key)
                )
                activation_cache_hits += 1
                hbm_act_tile_bytes = 0
                hbm_act_load_cycles = ceil(
                    hbm_act_tile_bytes
                    / io_bw_per_cycle
                )
            else:
                activation_cache_misses += 1
                hbm_act_tile_bytes = activation_tile_bytes
                hbm_act_load_cycles = ceil(
                    activation_tile_bytes
                    / io_bw_per_cycle
                )

                # If this tile cannot fit in the residual cache capacity, all
                # older tiles must be evicted.  Otherwise insert it and evict
                # LRU entries until the residual capacity is satisfied.
                if activation_tile_bytes > activation_cache_capacity_bytes:
                    activation_cache_evictions += len(activation_tile_cache)
                    activation_tile_cache.clear()
                else:
                    activation_tile_cache[act_key] = activation_tile_bytes
                    while sum(activation_tile_cache.values()) > activation_cache_capacity_bytes:
                        evicted_key = next(iter(activation_tile_cache))
                        activation_tile_cache.pop(evicted_key)
                        activation_cache_evictions += 1

                profiler_hbm_activation_bytes += activation_tile_bytes
                profiler_hbm_activation_cycles += hbm_act_load_cycles

            # ========================================================
            # Weight update / multicast
            # ========================================================

            current_nk = (
                n_idx,
                k_idx,
            )

            if current_nk != prev_nk:
                compressed_weight_tile_bytes = (
                    cur_tile_K
                    * cur_tile_N
                    * weight_bytes_per_element
                )

                # HBM/NoC 只发送一份；
                # 每个 macro 内部各写入一份本地副本。
                local_replicated_weight_bytes = (
                    compressed_weight_tile_bytes
                    * core_count
                )

                weight_update_cycles = ceil(
                    compressed_weight_tile_bytes
                    / weight_write_bw_per_cycle
                )

                profiler_weight_hbm_bytes += (
                    compressed_weight_tile_bytes
                )

                profiler_local_weight_write_bytes += (
                    local_replicated_weight_bytes
                )

                profiler_weight_fetch_cycles += (
                    weight_update_cycles
                )

                profiler_weight_update_count += 1

                prev_nk = current_nk

            else:
                weight_update_cycles = 0

            # ========================================================
            # Strict prefill macro mapping
            # ========================================================

            K_factor = 1
            N_factor = 1

            M_factor = min(
                core_count,
                cur_tile_M,
            )

            active_macros = M_factor

            # 4 表示 FP8 E4M3 的 SMMM 四个 bit-plane
            # ============================================================
            # Useful operations
            #
            # 只用于 FLOP/BOP 统计：
            #   最后一个 N/K tile 只统计真实有效元素。
            # ============================================================

            useful_ops = (
                2
                * cur_tile_M
                * cur_tile_N
                * cur_tile_K
                * 4
            )

            # ============================================================
            # Physical compute dimensions
            #
            # 固定物理阵列：
            #   - N 方向始终占用一个完整的 array_width=48 tile；
            #   - K 方向需要均匀映射到 Nadder=16 个 bank，
            #     因此向上补齐到 Nadder 的整数倍。
            #
            # 尾部无效列/维度不会产生 useful ops，
            # 但仍然占用物理执行周期。
            # ============================================================

            physical_compute_N = tile_N

            physical_compute_K = (
                ceil(
                    cur_tile_K / cim_macro.Nbank
                )
                * cim_macro.Nbank
            )

            # ============================================================
            # Strict token replication scheduling
            #
            # 每个 macro 处理整数个 token。
            # 当前 M=2048、macro=16 时：
            #   2048 / 16 = 128 token/macro。
            # ============================================================

            active_macros = min(
                core_count,
                cur_tile_M,
            )

            tokens_per_slowest_macro = ceil(
                cur_tile_M / active_macros
            )

            # 一个最慢 macro 的物理 dense-equivalent 工作量
            per_macro_physical_ops = (
                2
                * tokens_per_slowest_macro
                * physical_compute_N
                * physical_compute_K
                * 1
            )

            total_compute_cycles = ceil(
                per_macro_physical_ops
                / effective_throughput_per_macro
            )

            profiler_compute_cycles += (
                total_compute_cycles
            )

            # ========================================================
            # Global-buffer -> macro activation traffic
            #
            # 同一 activation K tile 会随不同 N weight tile 重复使用，
            # 但重复流量都发生在片上 global buffer。
            # ========================================================

            gb_activation_read_bytes = (
                cur_tile_M
                * cur_tile_K
                * act_ws
            )

            # ========================================================
            # K tile 串行累加
            #
            # k_idx == 0：
            #   不需要读取旧 partial sum；
            #
            # k_idx > 0：
            #   从 global buffer 读取旧 partial sum；
            #
            # 每个 K tile 都写回更新后的 partial sum。
            # 最后一个 K tile 写出的就是当前 output tile。
            # ========================================================

            psum_tile_bytes = (
                cur_tile_M
                * cur_tile_N
                * psum_ws
            )

            gb_psum_read_bytes = (
                psum_tile_bytes
                if k_idx > 0
                else 0
            )

            gb_psum_write_bytes = (
                psum_tile_bytes
            )

            gb_read_cycles = ceil(
                (
                    gb_activation_read_bytes
                    + gb_psum_read_bytes
                )
                / l2_bw
            )

            gb_write_cycles = ceil(
                gb_psum_write_bytes
                / l2_bw
            )

            profiler_gb_activation_read_bytes += (
                gb_activation_read_bytes
            )

            profiler_gb_psum_read_bytes += (
                gb_psum_read_bytes
            )

            profiler_gb_psum_write_bytes += (
                gb_psum_write_bytes
            )

            profiler_gb_read_cycles += (
                gb_read_cycles
            )

            profiler_gb_write_cycles += (
                gb_write_cycles
            )

            # ========================================================
            # Pipeline
            #
            # Weight update：
            #   上一权重块的 token FIFO 清空后才执行，不能与当前
            #   权重块的计算重叠。
            #
            # Activation/psum read：
            #   双缓冲时可以与 compute 重叠。
            #
            # Psum write：
            #   当前计算完成后写回。
            # ========================================================

            if mapping.is_l2_double_buffering:
                compute_and_read_cycles = max(
                    total_compute_cycles,
                    gb_read_cycles,
                )
            else:
                compute_and_read_cycles = (
                    total_compute_cycles
                    + gb_read_cycles
                )

            tile_cycles = (
                hbm_act_load_cycles
                + weight_update_cycles
                + compute_and_read_cycles
                + gb_write_cycles
            )

            total_cycle_count += tile_cycles

        # 最终 output 从 global buffer 写回 off-chip
        total_cycle_count += (
            hbm_output_write_cycles
        )

        # ============================================================
        # Profiler：off-chip traffic
        # ============================================================

        profiler_dram_read_bytes = int(
            profiler_hbm_activation_bytes
            + profiler_weight_hbm_bytes
        )

        profiler_dram_write_bytes = int(
            hbm_output_write_bytes
        )

        profiler_dram_cycles = (
            profiler_hbm_activation_cycles
            + profiler_weight_fetch_cycles
            + hbm_output_write_cycles
        )

        # ============================================================
        # Profiler：global-buffer <-> CIM traffic
        # ============================================================

        profiler_gb_read_bytes = (
            profiler_gb_activation_read_bytes
            + profiler_gb_psum_read_bytes
        )

        profiler_gb_write_bytes = (
            profiler_gb_psum_write_bytes
        )

        self.profiler.record_dram_bytes(
            profiler_dram_read_bytes,
            profiler_dram_write_bytes,
        )

        self.profiler.record_dram_latency(
            profiler_dram_cycles
        )

        # 复用原来的 L2/L1 profiler 字段记录
        # global-buffer <-> CIM 流量。
        self.profiler.record_l2_to_l1_latency(
            profiler_gb_read_cycles
            + profiler_gb_write_cycles
        )

        self.profiler.record_l2_l1_bytes(
            int(profiler_gb_read_bytes),
            int(profiler_gb_write_bytes),
        )

        self.profiler.record_l2_l1_weight_bytes(
            0,
            0,
        )

        self.profiler.record_l2_l1_activation_bytes(
            int(profiler_gb_read_bytes),
            int(profiler_gb_write_bytes),
        )

        self.profiler.record_compute_latency(
            profiler_compute_cycles
        )

        # ============================================================
        # Extra profiler information
        # ============================================================

        stats = self.profiler.current_record[
            "other_stats"
        ]

        stats.update(
            {
                "mapping_mode": (
                    "effloc_prefill_weight_replication"
                ),
                "prefill_effective_speedup": (
                    prefill_effective_speedup
                ),
                "effective_throughput_per_macro": (
                    effective_throughput_per_macro
                ),

                "K_factor": 1,
                "M_factor": min(core_count, M),
                "N_factor": 1,
                "token_parallel_macros": min(
                    core_count,
                    M,
                ),

                "tile_M": tile_M,
                "tile_N": tile_N,
                "tile_K": tile_K,

                "num_M_tiles": num_M_tiles,
                "num_N_tiles": num_N_tiles,
                "num_K_tiles": num_K_tiles,

                "weight_update_count": (
                    profiler_weight_update_count
                ),
                "weight_write_cycles": (
                    profiler_weight_fetch_cycles
                ),
                "weight_write_bytes": int(
                    profiler_weight_hbm_bytes
                ),
                "local_replicated_weight_write_bytes": int(
                    profiler_local_weight_write_bytes
                ),

                "hbm_activation_read_bytes": int(
                    profiler_hbm_activation_bytes
                ),
                "activation_cache_capacity_bytes": int(
                    activation_cache_capacity_bytes
                ),
                "activation_cache_hits": (
                    activation_cache_hits
                ),
                "activation_cache_misses": (
                    activation_cache_misses
                ),
                "activation_cache_evictions": (
                    activation_cache_evictions
                ),
                "hbm_output_write_bytes": int(
                    hbm_output_write_bytes
                ),

                "global_buffer_activation_read_bytes": int(
                    profiler_gb_activation_read_bytes
                ),
                "global_buffer_psum_read_bytes": int(
                    profiler_gb_psum_read_bytes
                ),
                "global_buffer_psum_write_bytes": int(
                    profiler_gb_psum_write_bytes
                ),

                "cim_arch": "EffLoc",
            }
        )

        # ============================================================
        # External access
        # ============================================================

        self.last_cim_weight_write_cycles = (
            profiler_weight_fetch_cycles
        )

        # off-chip 只传一份 compressed weights
        self.last_cim_weight_write_bytes = int(
            profiler_weight_hbm_bytes
        )

        # 本地所有 macro 的权重副本写入总量
        self.last_cim_local_weight_write_bytes = int(
            profiler_local_weight_write_bytes
        )

        self.last_cim_weight_update_count = (
            profiler_weight_update_count
        )

        return int(total_cycle_count)

    def simulate_cim_decode(
        self,
        computational_graph: ComputationalGraph,
        mapping: Mapping,
        pcb_module: Device,
        decode_effective_speedup: float = 1.0,
        activation_storage_bits: float = 7.4,
        activation_serial_bits: float = 4.0,
        attn: bool = False,
    ) -> int:
        """
        EffLoc 单 token decode 模拟。

        GEMM:
            [1, K] @ [K, N] -> [1, N]

        Macro mapping:
            K_factor * N_factor <= core_count; M rows reuse each weight tile.
            K shards receive disjoint activation slices, multicast along N.
            Cross-shard partial sums accumulate through the Global Buffer.

        数据映射:
            - Global Buffer 中只保存一份 activation；
            - activation K slice 通过 NoC 广播给所有活跃 Macro；
            - 每个 Macro 保存不同的 N 方向权重块；
            - 每个 Macro 覆盖 array_width 个输出通道；
            - K tile 串行执行并在 Global Buffer 中累加 partial sum。

        参数:
            decode_effective_speedup:
                相对于 activation_serial_bits 对应 dense baseline
                的实际计算速度提升。

            activation_storage_bits:
                activation 在 HBM/Global Buffer 中的平均存储位宽。

            activation_serial_bits:
                dense compute baseline 的串行 bit-plane 数。
                FP8 SMMM 为 4；
                BF16 S+M 为 8；
                BF16 mantissa-only 为 7。

        注意:
            如果 decode_effective_speedup 已经包含 bit skipping，
            activation_serial_bits 必须使用 dense 位宽，不能再使用
            平均 effective bit 数，否则会重复计算稀疏收益。
        """

        M = computational_graph.M
        N = computational_graph.N
        K = computational_graph.K

        Kf, Mf, Nf, Kr, Mr, Nr = compute_optimal_macro_layout_decode(
            pcb_module.compute_module.core_count, K, N, M,
            h=pcb_module.compute_module.core.cim_macro.array_height,
            w=pcb_module.compute_module.core.cim_macro.array_width,
            Nadder=pcb_module.compute_module.core.cim_macro.Nbank,
        )

        data_type = computational_graph.data_type

        cim_macro = (
            pcb_module.compute_module.core.cim_macro
        )

        core_count = (
            pcb_module.compute_module.core_count
        )

        clock_freq = (
            pcb_module.compute_module.clock_freq
        )

        l2_bw = (
            pcb_module.compute_module
            .l2_bandwidth_per_cycle
        )

        if M < 1:
            raise ValueError(
                "simulate_cim_decode 要求 M >= 1，"
                f"当前 M={M}。"
            )

        if decode_effective_speedup <= 0:
            raise ValueError(
                "decode_effective_speedup must be positive, "
                f"got {decode_effective_speedup}"
            )

        if activation_storage_bits <= 0:
            raise ValueError(
                "activation_storage_bits must be positive, "
                f"got {activation_storage_bits}"
            )

        if activation_serial_bits <= 0:
            raise ValueError(
                "activation_serial_bits must be positive, "
                f"got {activation_serial_bits}"
            )

        if l2_bw <= 0:
            raise ValueError(
                f"Invalid Global Buffer bandwidth: {l2_bw}"
            )

        act_ws = (
            activation_storage_bits / 8.0
        )

        psum_ws = float(
            getattr(
                cim_macro,
                "psum_word_size",
                cim_macro.output_word_size,
            )
        )

        # 最终输出位宽和 partial sum 位宽分开。
        final_output_ws = float(
            getattr(
                data_type,
                "word_size",
                cim_macro.output_word_size,
            )
        )

        tile_M = mapping.l2_tile_M
        tile_N = mapping.l2_tile_N
        tile_K = mapping.l2_tile_K

        physical_macro_K = (
            cim_macro.Nbank
            * cim_macro.array_height
        )

        physical_macro_N = (
            cim_macro.array_width
        )

        # expected_tile_M = 1

        # expected_tile_K = min(
        #     K,
        #     physical_macro_K,
        # )

        # expected_tile_N = min(
        #     N,
        #     core_count * physical_macro_N,
        # )

        # if tile_M != expected_tile_M:
        #     raise ValueError(
        #         "Decode tile_M 必须为 1："
        #         f"tile_M={tile_M}"
        #     )

        # if tile_K != expected_tile_K:
        #     raise ValueError(
        #         "Decode 必须使用物理 K tile："
        #         f"tile_K={tile_K}, "
        #         f"expected={expected_tile_K}"
        #     )

        # if tile_N != expected_tile_N:
        #     raise ValueError(
        #         "Decode 的 N tile 必须是所有 Macro "
        #         "同时覆盖的 N super-tile："
        #         f"tile_N={tile_N}, "
        #         f"expected={expected_tile_N}"
        #     )

        if mapping.l2_loop_order != "knm":
            raise ValueError(
                "Decode activation reuse 要求 "
                "loop order='knm'，"
                f"当前为 {mapping.l2_loop_order!r}"
            )

        num_K_tiles = ceil(K / tile_K)

        num_N_groups = ceil(N / tile_N)

        # ============================================================
        # Weight storage
        #
        # Decode 中不同 Macro 存不同权重，不存在同一权重的 16 份复制。
        # ============================================================
        if attn:
            offchip_weight_bytes_per_element = float(
                getattr(
                    cim_macro,
                    "weight_storage_bytes_per_element",
                    1.0,
                )
            )
        else:
            offchip_weight_bytes_per_element = float(
                getattr(
                    cim_macro,
                    "weight_storage_bytes_per_element",
                    1,
                )
            )
        local_weight_bytes_per_element = float(
            getattr(
                cim_macro,
                "local_weight_storage_bytes_per_element",
                offchip_weight_bytes_per_element,
            )
        )

        # ============================================================
        # Bandwidth
        # ============================================================

        if (
            hasattr(pcb_module, "wu_io_module")
            and pcb_module.wu_io_module is not None
        ):
            # 假定这是整个系统的 aggregate weight-update bandwidth。
            weight_write_bw_per_cycle = (
                pcb_module.wu_io_module.bandwidth
                / clock_freq
            )
        else:
            weight_write_bw_per_cycle = max(
                1.0,
                float(
                    core_count
                    * physical_macro_N
                ),
            )

        io_bw_per_cycle = (
            pcb_module.io_module.bandwidth
            / clock_freq
        )

        if weight_write_bw_per_cycle <= 0:
            raise ValueError(
                "Invalid weight-update bandwidth: "
                f"{weight_write_bw_per_cycle}"
            )

        if io_bw_per_cycle <= 0:
            raise ValueError(
                "Invalid external IO bandwidth: "
                f"{io_bw_per_cycle}"
            )

        # ============================================================
        # Compute throughput
        # ============================================================

        base_throughput_per_macro = (
            cim_macro.max_throughput_per_cycle
        )

        effective_throughput_per_macro = (
            base_throughput_per_macro
            * decode_effective_speedup
            # Existing throughput is calibrated to a four-plane dense baseline.
            * (4.0 / activation_serial_bits)
        )

        if effective_throughput_per_macro <= 0:
            raise ValueError(
                "Invalid effective throughput: "
                f"{effective_throughput_per_macro}"
            )

        # ============================================================
        # 最终 output 只从 Global Buffer 写回 HBM 一次
        # ============================================================

        hbm_output_write_bytes = (
            M
            * N
            * final_output_ws
        )

        hbm_output_write_cycles = ceil(
            hbm_output_write_bytes
            / io_bw_per_cycle
        )

        total_cycle_count = 0

        # ============================================================
        # Profiler counters
        # ============================================================

        profiler_compute_cycles = 0
        profiler_useful_ops = 0.0

        profiler_weight_update_cycles = 0
        profiler_weight_hbm_bytes = 0.0
        profiler_local_weight_write_bytes = 0.0
        profiler_weight_update_count = 0

        profiler_hbm_activation_bytes = 0.0
        profiler_hbm_activation_cycles = 0

        profiler_gb_activation_source_bytes = 0.0
        profiler_gb_activation_delivered_bytes = 0.0

        profiler_gb_psum_read_bytes = 0.0
        profiler_gb_psum_write_bytes = 0.0

        profiler_gb_read_cycles = 0
        profiler_gb_write_cycles = 0

        max_active_macros = 0

        # ============================================================
        # K -> N
        #
        # 对固定 K tile：
        #   activation slice 从 HBM 加载一次；
        #   随后依次广播给每个 N Macro group。
        # ============================================================

        for k_idx in range(num_K_tiles):
            cur_tile_K = min(
                tile_K,
                K - k_idx * tile_K,
            )

            # --------------------------------------------------------
            # HBM -> Global Buffer
            #
            # 每个 K slice 只加载一次，不随 N group 重复加载。
            # --------------------------------------------------------

            hbm_act_tile_bytes = (
                M
                * cur_tile_K
                * act_ws
            )

            hbm_act_load_cycles = ceil(
                hbm_act_tile_bytes
                / io_bw_per_cycle
            )

            profiler_hbm_activation_bytes += (
                hbm_act_tile_bytes
            )

            profiler_hbm_activation_cycles += (
                hbm_act_load_cycles
            )

            # HBM activation load 可在第一个 N group 前发生。
            first_n_group = True

            for n_group_idx in range(num_N_groups):
                cur_tile_N = min(
                    tile_N,
                    N - n_group_idx * tile_N,
                )

                # ====================================================
                # 当前 N super-tile 需要多少个 Macro
                # ====================================================

                K_factor = ceil(cur_tile_K / physical_macro_K)
                N_factor = ceil(cur_tile_N / physical_macro_N)
                active_macros = K_factor * N_factor
                if active_macros > core_count:
                    raise ValueError("Decode mapping exceeds physical macro count")
                max_active_macros = max(max_active_macros, active_macros)

                # ====================================================
                # Weight update
                #
                # 每个 Macro 获得不同的 N slice。
                #
                # 因此：
                #   - off-chip 读取的是一份完整 N super-tile；
                #   - 本地各 Macro 写入不同权重；
                #   - 不乘 16 倍 replication factor。
                # ====================================================

                compressed_weight_group_bytes = (
                    cur_tile_K
                    * cur_tile_N
                    * offchip_weight_bytes_per_element
                )

                local_weight_group_bytes = (
                    cur_tile_K
                    * cur_tile_N
                    * local_weight_bytes_per_element
                )

                weight_update_cycles = ceil(
                    compressed_weight_group_bytes
                    / weight_write_bw_per_cycle
                )

                profiler_weight_hbm_bytes += (
                    compressed_weight_group_bytes
                )

                profiler_local_weight_write_bytes += (
                    local_weight_group_bytes
                )

                profiler_weight_update_cycles += (
                    weight_update_cycles
                )

                profiler_weight_update_count += 1

                # ====================================================
                # Physical compute dimensions
                #
                # 每个活跃 Macro 的执行时间由：
                #   - 一个完整物理 N tile；
                #   - 均匀映射到 Nbank 的 K tile；
                # 决定。
                #
                # 最后一个 Macro 的无效 N 列只降低利用率，
                # 不减少 bit-serial traversal 周期。
                # ====================================================

                physical_compute_N = (
                    physical_macro_N
                )

                physical_compute_K = (
                    ceil(
                        min(cur_tile_K, physical_macro_K)
                        / cim_macro.Nbank
                    )
                    * cim_macro.Nbank
                )

                # 一个 Macro 处理当前 M 维的所有 activation rows。
                # M=1 是普通 decode；M=q/kv 是 shared-KV GQA decode。
                # K/V weight tile 只更新一次，由这些 M 行复用。
                per_macro_col_physical_ops = (
                    2
                    * tile_M
                    * physical_compute_N
                    * physical_compute_K
                )

                total_compute_cycles = ceil(
                    per_macro_col_physical_ops
                    / (effective_throughput_per_macro)
                )

                profiler_compute_cycles += (
                    total_compute_cycles
                )

                useful_ops = (
                    2
                    * M
                    * cur_tile_N
                    * cur_tile_K
                    * 1
                )

                profiler_useful_ops += (
                    useful_ops
                )

                # ====================================================
                # Global Buffer -> Macro activation broadcast
                #
                # source traffic:
                #   Global Buffer 只读一份 activation。
                #
                # delivered traffic:
                #   每个活跃 Macro 都接收一份。
                #
                # 延迟按 source multicast traffic 计算。
                # ====================================================

                gb_activation_source_bytes = (
                    M
                    * cur_tile_K
                    * act_ws
                )

                gb_activation_delivered_bytes = (
                    gb_activation_source_bytes
                    * N_factor
                )

                profiler_gb_activation_source_bytes += (
                    gb_activation_source_bytes
                )

                profiler_gb_activation_delivered_bytes += (
                    gb_activation_delivered_bytes
                )

                # ====================================================
                # K partial-sum accumulation
                # ====================================================

                psum_tile_bytes = (
                    M
                    * cur_tile_N
                    * psum_ws
                )

                gb_psum_read_bytes = (
                    psum_tile_bytes * (K_factor - 1 + int(k_idx > 0))
                )

                # Each K shard writes a partial sum; subsequent shards read
                # the accumulator. Reduction is bandwidth-bound in this model.
                gb_psum_write_bytes = psum_tile_bytes * K_factor

                # 假设 multicast NoC 从 Global Buffer 只读取一份
                # activation，因此不乘 active_macros。
                gb_read_cycles = ceil(
                    (
                        gb_activation_source_bytes
                        + (psum_tile_bytes if k_idx > 0 else 0)
                    )
                    / l2_bw
                )

                # Intra-round K reduction depends on this round's compute;
                # its accumulator reads cannot overlap that compute.
                reduction_read_cycles = ceil(psum_tile_bytes * (K_factor - 1) / l2_bw)
                gb_write_cycles = ceil(
                    gb_psum_write_bytes
                    / l2_bw
                )

                profiler_gb_psum_read_bytes += (
                    gb_psum_read_bytes
                )

                profiler_gb_psum_write_bytes += (
                    gb_psum_write_bytes
                )

                profiler_gb_read_cycles += (
                    gb_read_cycles + reduction_read_cycles
                )

                profiler_gb_write_cycles += (
                    gb_write_cycles
                )

                # ====================================================
                # Pipeline
                #
                # 保守假设：
                #   weight update 不能和本轮 compute 重叠；
                #
                # 双缓冲：
                #   Global Buffer read 可以和 compute 重叠；
                #
                # psum write：
                #   在计算后完成。
                # ====================================================

                if mapping.is_l2_double_buffering:
                    compute_and_read_cycles = max(
                        total_compute_cycles,
                        gb_read_cycles,
                    )
                else:
                    compute_and_read_cycles = (
                        total_compute_cycles
                        + gb_read_cycles
                    )

                # 每个 K tile 的 activation 只从 HBM 加载一次。
                current_hbm_act_cycles = (
                    hbm_act_load_cycles
                    if first_n_group
                    else 0
                )

                tile_cycles = (
                    current_hbm_act_cycles
                    + weight_update_cycles
                    + compute_and_read_cycles
                    + gb_write_cycles
                    + reduction_read_cycles
                )

                total_cycle_count += (
                    tile_cycles
                )

                first_n_group = False

        # 最终输出从 Global Buffer 写回 HBM 一次。
        total_cycle_count += (
            hbm_output_write_cycles
        )

        # ============================================================
        # Profiler：off-chip traffic
        # ============================================================

        profiler_dram_read_bytes = int(
            ceil(
                profiler_hbm_activation_bytes
                + profiler_weight_hbm_bytes
            )
        )

        profiler_dram_write_bytes = int(
            ceil(hbm_output_write_bytes)
        )

        profiler_dram_cycles = (
            profiler_hbm_activation_cycles
            + profiler_weight_update_cycles
            + hbm_output_write_cycles
        )

        self.profiler.record_dram_bytes(
            profiler_dram_read_bytes,
            profiler_dram_write_bytes,
        )

        self.profiler.record_dram_latency(
            profiler_dram_cycles
        )

        # ============================================================
        # Profiler：Global Buffer <-> CIM
        # ============================================================

        profiler_gb_source_read_bytes = (
            profiler_gb_activation_source_bytes
            + profiler_gb_psum_read_bytes
        )

        profiler_gb_write_bytes = (
            profiler_gb_psum_write_bytes
        )

        self.profiler.record_l2_to_l1_latency(
            profiler_gb_read_cycles
            + profiler_gb_write_cycles
        )

        self.profiler.record_l2_l1_bytes(
            int(ceil(profiler_gb_source_read_bytes)),
            int(ceil(profiler_gb_write_bytes)),
        )

        self.profiler.record_l2_l1_weight_bytes(
            0,
            0,
        )

        # 此处字段包含 activation broadcast source 和 psum。
        self.profiler.record_l2_l1_activation_bytes(
            int(ceil(profiler_gb_source_read_bytes)),
            int(ceil(profiler_gb_write_bytes)),
        )

        self.profiler.record_compute_latency(
            profiler_compute_cycles
        )

        # ============================================================
        # Extra profiler information
        # ============================================================

        stats = self.profiler.current_record[
            "other_stats"
        ]

        stats.update(
            {
                "mapping_mode": (
                    "effloc_decode_activation_broadcast"
                ),

                "decode_effective_speedup": (
                    decode_effective_speedup
                ),

                "activation_storage_bits": (
                    activation_storage_bits
                ),

                "activation_serial_bits": (
                    activation_serial_bits
                ),

                "effective_throughput_per_macro": (
                    effective_throughput_per_macro
                ),

                "K_factor": Kf,
                "M_factor": Mf,
                "N_factor": Nf,

                "max_active_macros": (
                    max_active_macros
                ),

                "activation_broadcast_fanout": Nf,
                "reduction_model": "global_buffer_bandwidth_bound",

                "weight_replication_factor": 1,

                "tile_M": tile_M,
                "tile_N": tile_N,
                "tile_K": tile_K,

                "num_K_tiles": (
                    num_K_tiles
                ),

                "num_N_groups": (
                    num_N_groups
                ),

                "weight_update_count": (
                    profiler_weight_update_count
                ),

                "weight_write_cycles": (
                    profiler_weight_update_cycles
                ),

                "weight_write_bytes": int(
                    ceil(profiler_weight_hbm_bytes)
                ),

                "local_weight_write_bytes": int(
                    ceil(
                        profiler_local_weight_write_bytes
                    )
                ),

                "hbm_activation_read_bytes": int(
                    ceil(
                        profiler_hbm_activation_bytes
                    )
                ),

                "hbm_output_write_bytes": int(
                    ceil(
                        hbm_output_write_bytes
                    )
                ),

                "global_buffer_activation_source_bytes": int(
                    ceil(
                        profiler_gb_activation_source_bytes
                    )
                ),

                "global_buffer_activation_delivered_bytes": int(
                    ceil(
                        profiler_gb_activation_delivered_bytes
                    )
                ),

                "global_buffer_psum_read_bytes": int(
                    ceil(
                        profiler_gb_psum_read_bytes
                    )
                ),

                "global_buffer_psum_write_bytes": int(
                    ceil(
                        profiler_gb_psum_write_bytes
                    )
                ),


                "useful_ops": float(
                    profiler_useful_ops
                ),

                "cim_arch": "EffLoc",
            }
        )

        # ============================================================
        # External access
        # ============================================================

        self.last_cim_weight_write_cycles = (
            profiler_weight_update_cycles
        )

        self.last_cim_weight_write_bytes = int(
            ceil(profiler_weight_hbm_bytes)
        )

        # Decode 中是不同权重分片，不是同一权重复制。
        self.last_cim_local_weight_write_bytes = int(
            ceil(
                profiler_local_weight_write_bytes
            )
        )

        self.last_cim_weight_update_count = (
            profiler_weight_update_count
        )

        return int(total_cycle_count)

    class L2TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: "Matmul.Mapping",
            pcb_module: Device,
            look_up_table: pd.DataFrame,
        ):
            # print(f'L2 tile: {M} {N} {K}')
            self.M = M
            self.N = N
            self.K = K
            self.l2_to_l1_read_bytes = 0  #add
            self.l1_to_l2_write_bytes = 0  #add
            self.l2_to_l1_read_cycle_count = 0  #add
            self.l1_to_l2_write_cycle_count = 0  #add
            self.core_compute_cycle_count = 0  #add
            self.core_compute_work_cycle_count = 0  #add
            self.l1_to_core_read_bytes = 0  #add
            self.l1_to_core_write_bytes = 0  #add
            self.K_reduction_cycle_count = ceil(
                M * N / pcb_module.compute_module.total_vector_flops_per_cycle
            ) + 2 * ceil(
                M
                * N
                * data_type.word_size
                / pcb_module.compute_module.l2_bandwidth_per_cycle
            )
            self.K_reduction_io_count = 2 * M * N * data_type.word_size
            self.M_K_io_cycle_count = self.simulate_l2_tile_io_cycle_count(
                M, K, data_type, pcb_module
            )
            self.K_N_io_cycle_count = self.simulate_l2_tile_io_cycle_count(
                K, N, data_type, pcb_module
            )
            self.M_N_io_cycle_count = self.simulate_l2_tile_io_cycle_count(
                M, N, data_type, pcb_module
            )
            self.compute_cycle_count = self.simulate_l2_tile_compute_cycle_count(
                M, N, K, data_type, mapping, pcb_module, look_up_table
            )

        def simulate_l2_tile_io_cycle_count(
            self, M: int, N: int, data_type: DataType, chiplet_module: Device
        ):
            return ceil(
                M
                * N
                * data_type.word_size
                / (
                    chiplet_module.io_module.bandwidth
                    / chiplet_module.compute_module.clock_freq
                )
            )

        def simulate_l2_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: "Matmul.Mapping",
            chiplet_module: Device,
            look_up_table: pd.DataFrame,
        ) -> int:
            l1_tile_M = mapping.l1_tile_M
            l1_tile_N = mapping.l1_tile_N
            l1_tile_K = mapping.l1_tile_K

            M_l1_t = M // l1_tile_M
            N_l1_t = N // l1_tile_N
            K_l1_t = K // l1_tile_K
            M_remain = M % l1_tile_M
            N_remain = N % l1_tile_N
            K_remain = K % l1_tile_K

            l1_tiles = np.empty(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N), ceil(K / l1_tile_K)],
                dtype=Matmul.L1TileSimulator,
            )
            if M_l1_t * N_l1_t * K_l1_t != 0:
                l1_tiles[:M_l1_t, :N_l1_t, :K_l1_t] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    l1_tile_N,
                    l1_tile_K,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if M_remain != 0:
                l1_tiles[-1, :N_l1_t, :K_l1_t] = Matmul.L1TileSimulator(
                    M_remain,
                    l1_tile_N,
                    l1_tile_K,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if N_remain != 0:
                l1_tiles[:M_l1_t, -1, :K_l1_t] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    N_remain,
                    l1_tile_K,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if K_remain != 0:
                l1_tiles[:M_l1_t, :N_l1_t, -1] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    l1_tile_N,
                    K_remain,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if M_remain * N_remain != 0:
                l1_tiles[-1, -1, :K_l1_t] = Matmul.L1TileSimulator(
                    M_remain,
                    N_remain,
                    l1_tile_K,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if M_remain * K_remain != 0:
                l1_tiles[-1, :N_l1_t, -1] = Matmul.L1TileSimulator(
                    M_remain,
                    l1_tile_N,
                    K_remain,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if N_remain * K_remain != 0:
                l1_tiles[:M_l1_t, -1, -1] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    N_remain,
                    K_remain,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if M_remain * N_remain * K_remain != 0:
                l1_tiles[-1, -1, -1] = Matmul.L1TileSimulator(
                    M_remain,
                    N_remain,
                    K_remain,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )

            M_K_tile_size = np.zeros(
                [ceil(M / l1_tile_M), ceil(K / l1_tile_K)], dtype=int
            )
            M_K_tile_size[:M_l1_t, :K_l1_t] = l1_tile_M * l1_tile_K
            if M_remain > 0:
                M_K_tile_size[-1, :K_l1_t] = M_remain * l1_tile_K
            if K_remain > 0:
                M_K_tile_size[:M_l1_t, -1] = l1_tile_M * K_remain
            if M_remain > 0 and K_remain > 0:
                M_K_tile_size[-1, -1] = M_remain * K_remain

            K_N_tile_size = np.zeros(
                [ceil(K / l1_tile_K), ceil(N / l1_tile_N)], dtype=int
            )
            K_N_tile_size[:K_l1_t, :N_l1_t] = l1_tile_K * l1_tile_N
            if K_remain > 0:
                K_N_tile_size[-1, :N_l1_t] = K_remain * l1_tile_N
            if N_remain > 0:
                K_N_tile_size[:K_l1_t, -1] = l1_tile_K * N_remain
            if K_remain > 0 and N_remain > 0:
                K_N_tile_size[-1, -1] = K_remain * N_remain

            M_N_tile_size = np.zeros(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=int
            )
            M_N_tile_size[:M_l1_t, :N_l1_t] = l1_tile_M * l1_tile_N
            if M_remain > 0:
                M_N_tile_size[-1, :N_l1_t] = M_remain * l1_tile_N
            if N_remain > 0:
                M_N_tile_size[:M_l1_t, -1] = l1_tile_M * N_remain
            if M_remain > 0 and N_remain > 0:
                M_N_tile_size[-1, -1] = M_remain * N_remain

            total_cycle_count = 0
            total_l2_to_l1_read_bytes = 0  #add
            total_l2_to_l1_weight_read_bytes = 0  #add
            total_l2_to_l1_activation_read_bytes = 0  #add
            total_l1_to_l2_write_bytes = 0  #add
            total_l1_to_l2_activation_write_bytes = 0  #add
            total_l2_to_l1_read_cycles = 0  #add
            total_l1_to_l2_write_cycles = 0  #add
            total_core_compute_cycles = 0  #add
            total_core_compute_work_cycles = 0  #add
            total_l1_to_core_read_bytes = 0  #add
            total_l1_to_core_write_bytes = 0  #add
            previous_batch_Read_M_K = np.zeros(
                [ceil(M / l1_tile_M), ceil(K / l1_tile_K)], dtype=bool
            )
            previous_batch_Read_K_N = np.zeros(
                [ceil(K / l1_tile_K), ceil(N / l1_tile_N)], dtype=bool
            )
            previous_batch_Read_M_N = np.zeros(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
            )
            previous_batch_Write_M_N = np.zeros(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
            )
            previous_batch_compute_cycle_count = 0
            active_l1_tile_list = []
            for m, n, k in Matmul.generate_tile_loops(
                ceil(M / l1_tile_M),
                ceil(N / l1_tile_N),
                ceil(K / l1_tile_K),
                mapping.l1_loop_order,
            ):
                active_l1_tile_list.append((m, n, k, l1_tiles[m, n, k]))
                if (
                    m == ceil(M / l1_tile_M) - 1
                    and n == ceil(N / l1_tile_N) - 1
                    and k == ceil(K / l1_tile_K) - 1
                ):
                    pass
                elif (
                    len(active_l1_tile_list) < chiplet_module.compute_module.core_count
                ):
                    continue

                assert (
                    len(active_l1_tile_list) <= chiplet_module.compute_module.core_count
                )
                current_batch_Read_M_K = np.zeros(
                    [ceil(M / l1_tile_M), ceil(K / l1_tile_K)], dtype=bool
                )
                current_batch_Read_K_N = np.zeros(
                    [ceil(K / l1_tile_K), ceil(N / l1_tile_N)], dtype=bool
                )
                current_batch_Read_M_N = np.zeros(
                    [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
                )
                current_batch_Write_M_N = np.zeros(
                    [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
                )

                current_batch_compute_cycle_count = 0
                for i in range(len(active_l1_tile_list)):
                    temp_m, temp_n, temp_k, temp_l1_tile = active_l1_tile_list[i]
                    current_batch_Read_M_K[temp_m, temp_k] = 1
                    current_batch_Read_K_N[temp_k, temp_n] = 1
                    current_batch_Read_M_N[temp_m, temp_n] = temp_k > 0
                    current_batch_Write_M_N[temp_m, temp_n] = 1
                    temp_l1_tile_compute_cycle_count = temp_l1_tile.compute_cycle_count
                    if temp_k > 0:
                        temp_l1_tile_compute_cycle_count += ceil(
                            temp_l1_tile.M
                            * temp_l1_tile.N
                            / chiplet_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                        )
                    current_batch_compute_cycle_count = max(
                        current_batch_compute_cycle_count,
                        temp_l1_tile_compute_cycle_count,
                    )
                    total_core_compute_work_cycles += temp_l1_tile_compute_cycle_count  #add
                    total_l1_to_core_read_bytes += (  #add
                        temp_l1_tile.M * temp_l1_tile.K * chiplet_module.compute_module.core.systolic_array.input_word_size  #add
                        + temp_l1_tile.K * temp_l1_tile.N * chiplet_module.compute_module.core.systolic_array.input_word_size  #add
                    )  #add
                    if temp_k > 0:  #add
                        total_l1_to_core_read_bytes += temp_l1_tile.M * temp_l1_tile.N * chiplet_module.compute_module.core.systolic_array.output_word_size  #add
                    total_l1_to_core_write_bytes += temp_l1_tile.M * temp_l1_tile.N * chiplet_module.compute_module.core.systolic_array.output_word_size  #add

                # if one output tile in this batch shares input/output with another output tile in the previous batch, assign them to the same core to avoid data movement
                # note that of the three input matrix mk, kn, mn, at most one of them can be the same if we change m,n,k
                current_batch_M_K_read_count = np.sum(
                    (current_batch_Read_M_K * (~previous_batch_Read_M_K))
                    * M_K_tile_size
                )
                current_batch_K_N_read_count = np.sum(
                    (current_batch_Read_K_N * (~previous_batch_Read_K_N))
                    * K_N_tile_size
                )
                current_batch_M_N_read_count = np.sum(
                    (
                        current_batch_Read_M_N
                        * (~(previous_batch_Read_M_N + previous_batch_Write_M_N))
                    )
                    * M_N_tile_size
                )
                previous_batch_M_N_write_count = np.sum(
                    (previous_batch_Write_M_N * (~current_batch_Read_M_N))
                    * M_N_tile_size
                )

                # read current batch while compute and write previous batch
                current_batch_read_count = (
                    current_batch_M_K_read_count
                    + current_batch_K_N_read_count
                    + current_batch_M_N_read_count
                )
                current_batch_read_cycle_count = ceil(
                    current_batch_read_count
                    * chiplet_module.compute_module.core.systolic_array.input_word_size
                    / chiplet_module.compute_module.l2_bandwidth_per_cycle
                )
                prvious_batch_write_cycle_count = ceil(
                    previous_batch_M_N_write_count
                    * chiplet_module.compute_module.core.systolic_array.output_word_size
                    / chiplet_module.compute_module.l2_bandwidth_per_cycle
                )
                current_batch_read_bytes_profile = (  #add
                    current_batch_M_K_read_count * chiplet_module.compute_module.core.systolic_array.input_word_size  #add
                    + current_batch_K_N_read_count * chiplet_module.compute_module.core.systolic_array.input_word_size  #add
                    + current_batch_M_N_read_count * chiplet_module.compute_module.core.systolic_array.output_word_size  #add
                )  #add
                current_batch_weight_read_bytes_profile = (
                    current_batch_K_N_read_count * chiplet_module.compute_module.core.systolic_array.input_word_size
                )  #add
                current_batch_activation_read_bytes_profile = (
                    current_batch_M_K_read_count * chiplet_module.compute_module.core.systolic_array.input_word_size
                    + current_batch_M_N_read_count * chiplet_module.compute_module.core.systolic_array.output_word_size
                )  #add
                previous_batch_write_bytes_profile = previous_batch_M_N_write_count * chiplet_module.compute_module.core.systolic_array.output_word_size  #add
                total_l2_to_l1_read_bytes += int(current_batch_read_bytes_profile)  #add
                total_l2_to_l1_weight_read_bytes += int(current_batch_weight_read_bytes_profile)  #add
                total_l2_to_l1_activation_read_bytes += int(current_batch_activation_read_bytes_profile)  #add
                total_l1_to_l2_write_bytes += int(previous_batch_write_bytes_profile)  #add
                total_l1_to_l2_activation_write_bytes += int(previous_batch_write_bytes_profile)  #add
                total_l2_to_l1_read_cycles += current_batch_read_cycle_count  #add
                total_l1_to_l2_write_cycles += prvious_batch_write_cycle_count  #add
                total_core_compute_cycles += previous_batch_compute_cycle_count  #add

                total_cycle_count += (
                    max(
                        current_batch_read_cycle_count,
                        previous_batch_compute_cycle_count,
                    )
                    + prvious_batch_write_cycle_count
                )

                previous_batch_compute_cycle_count = current_batch_compute_cycle_count
                previous_batch_Read_M_K = copy.deepcopy(current_batch_Read_M_K)
                previous_batch_Read_K_N = copy.deepcopy(current_batch_Read_K_N)
                previous_batch_Read_M_N = copy.deepcopy(current_batch_Read_M_N)
                previous_batch_Write_M_N = copy.deepcopy(current_batch_Write_M_N)

                active_l1_tile_list = []

            # last batch's compute and write
            total_cycle_count += previous_batch_compute_cycle_count + ceil(
                np.sum(previous_batch_Write_M_N * M_N_tile_size)
                * data_type.word_size
                / chiplet_module.compute_module.l2_bandwidth_per_cycle
            )
            last_batch_write_bytes_profile = int(np.sum(previous_batch_Write_M_N * M_N_tile_size) * data_type.word_size)  #add
            last_batch_write_cycles_profile = ceil(last_batch_write_bytes_profile / chiplet_module.compute_module.l2_bandwidth_per_cycle)  #add
            total_l1_to_l2_write_bytes += last_batch_write_bytes_profile  #add
            total_l1_to_l2_write_cycles += last_batch_write_cycles_profile  #add
            total_core_compute_cycles += previous_batch_compute_cycle_count  #add
            self.l2_to_l1_read_bytes = total_l2_to_l1_read_bytes  #add
            self.weight_l2_to_l1_read_bytes = total_l2_to_l1_weight_read_bytes  #add
            self.activation_l2_to_l1_read_bytes = total_l2_to_l1_activation_read_bytes  #add
            self.l1_to_l2_write_bytes = total_l1_to_l2_write_bytes  #add
            self.l1_to_l2_activation_write_bytes = total_l1_to_l2_activation_write_bytes  #add
            self.l2_to_l1_read_cycle_count = total_l2_to_l1_read_cycles  #add
            self.l1_to_l2_write_cycle_count = total_l1_to_l2_write_cycles  #add
            self.core_compute_cycle_count = total_core_compute_cycles  #add
            self.core_compute_work_cycle_count = total_core_compute_work_cycles  #add
            self.l1_to_core_read_bytes = total_l1_to_core_read_bytes  #add
            self.l1_to_core_write_bytes = total_l1_to_core_write_bytes  #add

            return total_cycle_count

    class L1TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: "Matmul.Mapping",
            chiplet_module: Device,
            look_up_table: pd.DataFrame,
        ):
            # print(f'L1 tile: {M} {N} {K}')
            self.M = M
            self.N = N
            self.K = K
            self.compute_cycle_count = self.simulate_l1_tile_compute_cycle_count(
                M, N, K, data_type, mapping, chiplet_module, look_up_table
            )

        def simulate_l1_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: "Matmul.Mapping",
            chiplet_module: Device,
            look_up_table: pd.DataFrame,
        ):
            assert (
                M * K + K * N + M * N
                <= chiplet_module.compute_module.core.SRAM_size
                // data_type.word_size
                // 2
            )

            M_tiling_factor = mapping.l0_M_tiling_factor
            N_tiling_factor = mapping.l0_N_tiling_factor
            K_tiling_factor = mapping.l0_K_tiling_factor
            assert (
                M_tiling_factor * K_tiling_factor * N_tiling_factor
                <= chiplet_module.compute_module.core.systolic_array_count
            )

            compute_cycle_count = ceil(
                Matmul.simulate_systolic_array_cycle_count(
                    look_up_table,
                    ceil(M / M_tiling_factor),
                    ceil(N / N_tiling_factor),
                    ceil(K / K_tiling_factor),
                    chiplet_module.compute_module.core.systolic_array.array_height,
                    chiplet_module.compute_module.core.systolic_array.array_width,
                    chiplet_module.compute_module.core.systolic_array.mac_per_cycle,
                    mapping.dataflow,
                )
                + (K_tiling_factor - 1)
                * M
                * N
                / chiplet_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            )

            return compute_cycle_count

    @staticmethod
    def simulate_systolic_array_cycle_count(
        look_up_table: pd.DataFrame,
        M,
        N,
        K,
        array_height,
        array_width,
        mac_per_clock,
        dataflow="os",
    ):
        # print(f'start: {M} {N} {K} {array_height} {array_width} {mac_per_clock} {dataflow}')
        assert M * N * K * array_height * array_width * mac_per_clock != 0
        if M >= array_height and N >= array_width:
            if (
                M * N * K / array_height / array_width / max(array_height, array_width)
                >= 128
            ):
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / 0.99
                )
            elif (
                M * N * K / array_height / array_width / max(array_height, array_width)
                >= 64
            ):
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / 0.98
                )
        elif M >= array_height and N < array_width:
            if K * M / array_height / max(array_height, array_width) >= 64:
                util_rate = N / array_width / 0.98
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / util_rate
                )
        elif M < array_height and N >= array_width:
            if K * N / array_width / max(array_height, array_width) >= 64:
                util_rate = M / array_height / 0.98
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / util_rate
                )
        else:
            assert M < array_height and N < array_width
            if K / max(array_height, array_width) >= 64:
                util_rate = M / array_height * N / array_width / 0.98
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / util_rate
                )
        # print('start look up table')
        try:
            cycle_count = look_up_table.loc[
                (M, N, K, array_height, array_width, dataflow), "cycle_count"
            ].item()
        except KeyError:
            try:
                cycle_count = look_up_table.loc[
                    (N, M, K, array_height, array_width, dataflow), "cycle_count"
                ].item()
            except KeyError:
                # print('not found in look up table')
                config = f"./systolic_array_model/temp/systolic_array_{os.getpid()}.cfg"
                with open(config, "w") as f:
                    f.writelines("[general]\n")
                    f.writelines("run_name = systolic_array\n\n")
                    f.writelines("[architecture_presets]\n")
                    f.writelines("ArrayHeight:    " + str(array_height) + "\n")
                    f.writelines("ArrayWidth:     " + str(array_width) + "\n")
                    f.writelines("IfmapSramSzkB:    " + str(1024) + "\n")
                    f.writelines("FilterSramSzkB:   " + str(1024) + "\n")
                    f.writelines("OfmapSramSzkB:    " + str(1024) + "\n")
                    f.writelines("IfmapOffset:    0\n")
                    f.writelines("FilterOffset:   10000000\n")
                    f.writelines("OfmapOffset:    20000000\n")
                    f.writelines("Dataflow : " + dataflow + "\n")
                    f.writelines("Bandwidth : " + "100" + "\n")
                    f.writelines("MemoryBanks: 1\n\n")
                    f.writelines("[run_presets]\n")
                    f.writelines("InterfaceBandwidth: CALC\n")

                topology = f"./systolic_array_model/temp/matmul_{os.getpid()}.csv"
                with open(topology, "w") as f:
                    f.writelines("Layer, M, N, K\n")
                    f.writelines(f"matmul1, {M}, {N}, {K},\n")

                logpath = f"./systolic_array_model/temp/"
                s = scalesim(
                    save_disk_space=True,
                    verbose=False,
                    config=config,
                    topology=topology,
                    input_type_gemm=True,
                )
                s.run_scale(top_path=logpath)

                cycle_count = s.runner.single_layer_sim_object_list[0].total_cycles
                util_rate = s.runner.single_layer_sim_object_list[0].overall_util
                with open(
                    f"./systolic_array_model/look_up_table_{array_height}_{array_width}.csv",
                    "a",
                ) as f:
                    f.writelines(
                        f"{M},{N},{K},{array_height},{array_width},{dataflow},{cycle_count},{util_rate:.3f}\n"
                    )
                look_up_table.loc[(M, N, K, array_height, array_width, dataflow), :] = [
                    cycle_count,
                    util_rate,
                ]
                if len(look_up_table) % 10 == 0:
                    look_up_table.sort_index(inplace=True)
        # if (
        #     dataflow == "os"
        # ):  # scalesim assumes collecting output is not on critical path in os
        #     cycle_count += min(array_height, array_width, M, N)
        # if True:
        #     print(f"{M}x{N}x{K}x{array_height}x{array_width}x{dataflow}: {cycle_count}")
        # new_table = look_up_table[~look_up_table.index.duplicated(keep='first')]
        # if look_up_table.shape[0]-new_table.shape[0]>=1:
        #     print(look_up_table)
        #     print(look_up_table.duplicated(keep=False))
        #     exit()
        # print(f'end: {M} {N} {K} {array_height} {array_width} {mac_per_clock} {dataflow}')
        # assert isinstance(cycle_count, float), f"cycle_count: {cycle_count}"
        return ceil(cycle_count / mac_per_clock)

    def run_on_gpu(
        self,
    ):
        # import subprocess
        # subprocess.run(['nvidia-smi', '-q', '–d', 'CLOCK'])
        input1 = torch.randn(
            self.computational_graph.M,
            self.computational_graph.K,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        input2 = torch.randn(
            self.computational_graph.K,
            self.computational_graph.N,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        latencies = []
        input1_dummy = torch.ones(4096, 4096).cuda()
        input2_dummy = torch.ones(4096, 4096).cuda()
        # warmup
        for _ in range(3):
            torch.matmul(input1_dummy, input2_dummy)
            torch.cuda.synchronize()
            time.sleep(1)
        for _ in range(self.iterations):
            # x = torch.matmul(input1_dummy, input2_dummy)  # flush the cache
            # torch.cuda.synchronize()
            start = time.time()
            output = torch.matmul(input1, input2)
            torch.cuda.synchronize()
            end = time.time()
            assert list(output.shape) == [
                self.computational_graph.M,
                self.computational_graph.N,
            ]
            latencies.append(end - start)
            # time.sleep(1)

        self.latency_on_gpu = (
            statistics.median(latencies)
            # min(latencies)
            # - self.gpu_kernel_launch_overhead()
            # - 4e-5
            # min(latencies) - 8e-6
        )  # GPU launch kernel overhead and PyTorch overhead
        return self.latency_on_gpu

    @staticmethod
    def gpu_kernel_launch_overhead():
        size = 1
        latencies = []
        for _ in range(50):
            a = torch.randn(size, size, device="cuda")
            b = torch.randn(size, size, device="cuda")
            torch.cuda.synchronize()
            start = time.time()
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
        avg_overhead = statistics.median(latencies)
        print("GPU kernel launch overhead: ", avg_overhead * 1e3, "ms")
        print(latencies)
        return avg_overhead


def compute_optimal_macro_layout_decode(
    Nmacro: int,
    in_features: int,
    out_features: int,
    seq_length: int,
    h: int = 64,
    w: int = 48,
    Nadder: int = 16,
):
    if Nmacro <= 0:
        raise ValueError(
            f"Nmacro must be positive, got {Nmacro}"
        )

    if in_features <= 0:
        raise ValueError(
            f"in_features must be positive, got {in_features}"
        )

    if out_features <= 0:
        raise ValueError(
            f"out_features must be positive, got {out_features}"
        )

    if seq_length <= 0:
        raise ValueError(
            f"seq_length must be positive, got {seq_length}"
        )

    if h <= 0 or w <= 0 or Nadder <= 0:
        raise ValueError(
            f"h, w and Nadder must be positive, "
            f"got h={h}, w={w}, Nadder={Nadder}"
        )

    # ------------------------------------------------------------
    # 一个 Macro 在一个 K round 中可以覆盖的输入维度
    #
    # 16 banks × 64 dimensions = 1024 dimensions
    # ------------------------------------------------------------
    k_capacity_per_macro_round = Nadder * h

    # 完整 GEMM 在三个方向上至少需要多少个基础 tile
    total_K_tiles = math.ceil(
        in_features / k_capacity_per_macro_round
    )

    total_M_tiles = seq_length

    total_N_tiles = math.ceil(
        out_features / w
    )

    max_K_factor = min(
        Nmacro,
        total_K_tiles,
    )

    # max_M_factor = min(
    #     Nmacro,
    #     total_M_tiles,
    # )

    max_N_factor = min(
        Nmacro,
        total_N_tiles,
    )

    # # N_factor 固定为 1
    # N_factor = 1

    M_factor = 1

    best_layout = None
    best_score = None

    for K_factor in range(1, max_K_factor + 1):
        for N_factor in range(1, max_N_factor + 1):

                macros_used = (
                    K_factor
                    * M_factor
                    * N_factor
                )

                if macros_used > Nmacro:
                    continue

                # ====================================================
                # 三个方向仍需顺序执行的轮数
                # ====================================================

                # K_factor 个 Macro group 同时处理不同 K slice
                K_rounds = math.ceil(
                    in_features
                    / (
                        K_factor
                        * k_capacity_per_macro_round
                    )
                )

                # M_factor 个 Macro 处理不同 token
                M_rounds = math.ceil(
                    seq_length / M_factor
                )

                # N_factor=1 固定, 输出通道不并行
                N_rounds = math.ceil(
                    out_features / (w * N_factor)
                )

                # 粗粒度总串行轮数
                total_serial_rounds = (
                    K_rounds
                    * M_rounds
                    * N_rounds
                )

                # ====================================================
                # 计算三个维度上的 padding/utilization
                # ====================================================

                K_capacity = (
                    K_factor
                    * K_rounds
                    * k_capacity_per_macro_round
                )

                M_capacity = (
                    M_factor
                    * M_rounds
                )

                N_capacity = (
                    N_factor
                    * N_rounds
                    * w
                )

                K_utilization = (
                    in_features / K_capacity
                )

                M_utilization = (
                    seq_length / M_capacity
                )

                N_utilization = (
                    out_features / N_capacity
                )

                macro_utilization = (
                    macros_used / Nmacro
                )

                overall_utilization = (
                    K_utilization
                    * M_utilization
                    * N_utilization
                    * macro_utilization
                )

                # K 并行时会产生 K_factor 份 partial sum。
                # 这里先用一个无量纲的简单 penalty 做次级比较。
                # 真正计算 latency 时仍应使用精确 reduction cycles。
                K_reduction_penalty = (
                    0
                    if K_factor == 1
                    else (
                        (K_factor - 1)
                        * M_rounds
                        * N_rounds
                    )
                )

                # ====================================================
                # 评分
                #
                # Python tuple 按顺序比较：
                #   1. 总串行轮数越小越好
                #   2. K reduction 越少越好
                #   3. 综合利用率越高越好
                #   4. 同条件优先 K > M > N
                # ====================================================
                score = (
                    # total_serial_rounds,
                    # K_reduction_penalty,
                    -overall_utilization,
                    -K_factor,
                    -M_factor,
                    -N_factor,
                )

                if best_score is None or score < best_score:
                    best_score = score

                    best_layout = {
                        "K_factor": K_factor,
                        "M_factor": M_factor,
                        "N_factor": N_factor,
                        "K_rounds": K_rounds,
                        "M_rounds": M_rounds,
                        "N_rounds": N_rounds,
                        "macros_used": macros_used,
                        "num_groups": (
                            K_factor * N_factor
                        ),
                        "macros_per_group": M_factor,
                        "K_utilization": K_utilization,
                        "M_utilization": M_utilization,
                        "N_utilization": N_utilization,
                        "macro_utilization": macro_utilization,
                        "overall_utilization": overall_utilization,
                        "total_serial_rounds": total_serial_rounds,
                        "K_reduction_penalty": K_reduction_penalty,
                    }

    if best_layout is None:
        raise RuntimeError(
            "Unable to find a valid macro layout"
        )

    return (
        best_layout["K_factor"],
        best_layout["M_factor"],
        best_layout["N_factor"],
        best_layout["K_rounds"],
        best_layout["M_rounds"],
        best_layout["N_rounds"],
    )
