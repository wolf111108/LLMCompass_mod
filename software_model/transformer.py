from software_model.operators import (
    Operator,
    Reshape,
    Concat,
    Transpose,
)
from software_model.matmul import Matmul, BatchedMatmul
from software_model.softmax import Softmax
from software_model.layernorm import LayerNorm
from software_model.gelu import GeLU

from software_model.utils import Tensor, DataType
from software_model.communication_primitives import AllReduceMultiPCB
from math import ceil
from typing import List
from hardware_model.system import System


class TransformerBlockInitComputationTP(Operator):
    def __init__(self, d_model, n_heads, device_count, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.n_heads = n_heads
        self.device_count = device_count
        # parameters per device
        d = d_model
        self.Wq = Tensor([d, d // device_count], data_type)
        self.Wk = Tensor([d, d // device_count], data_type)
        self.Wv = Tensor([d, d // device_count], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)
        self.W1 = Tensor([d, 4 * d // device_count], data_type)
        self.W2 = Tensor([4 * d // device_count, d], data_type)
        # operators per device
        # # multi-head attention
        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)
        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)
        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)
        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type)
        self.layer_norm0 = LayerNorm(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)
        # # feed-forward network
        self.H_matmul1 = Matmul(data_type)
        self.H_gelu = GeLU(data_type)
        self.H_matmul2 = Matmul(data_type)
        self.layer_norm1 = LayerNorm(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)

    def __call__(self, X: Tensor) -> Tensor:
        # b: batch size
        # s: sequence length
        # d: hidden dimension
        # d_h: dimension per head
        b, s, d = X.shape
        assert d == self.d_model
        h = self.n_heads
        dev_cnt = self.device_count
        d_h = d // h

        # multi-head attention
        Q = self.Q_proj(X, self.Wq)  # [b, s, d / dev_cnt]
        assert Q.shape == [b, s, d // dev_cnt]
        K = self.K_proj(X, self.Wk)  # [b, s, d / dev_cnt]
        V = self.V_proj(X, self.Wv)  # [b, s, d / dev_cnt]
        Q = self.Q_reshape(Q, [b, s, h // dev_cnt, d_h])
        K = self.K_reshape(K, [b, s, h // dev_cnt, d_h])
        V = self.V_reshape(V, [b, s, h // dev_cnt, d_h])
        Q_T = self.Q_transpose(Q, [0, 2, 1, 3])  # [b, h / dev_cnt, s, d_h]
        assert Q_T.shape == [b, h // dev_cnt, s, d_h]
        K_T = self.K_transpose(K, [0, 2, 3, 1])  # [b, h / dev_cnt, d_h, s]
        assert K_T.shape == [b, h // dev_cnt, d_h, s]
        V_T = self.V_transpose(V, [0, 2, 1, 3])  # [b, h / dev_cnt, s, d_h]
        assert V_T.shape == [b, h // dev_cnt, s, d_h]
        A = self.Q_mul_K(Q_T, K_T)  # [b, h / dev_cnt, s, s]
        assert A.shape == [b, h // dev_cnt, s, s]
        A_prob = self.A_softmax(A)
        H = self.A_mul_V(A_prob, V_T)  #  [b, h / dev_cnt, s, d_h]
        assert H.shape == [b, h // dev_cnt, s, d_h]
        H = self.H_transpose(H, [0, 2, 1, 3])  #  [b, s, h / dev_cnt, d_h]
        assert H.shape == [b, s, h // dev_cnt, d_h]
        H = self.H_reshape(H, [b, s, d // dev_cnt])
        assert H.shape == [b, s, d // dev_cnt]
        H0 = self.H_matmul0(H, self.W0)  #  [b, s, d]
        assert H0.shape == [b, s, d]
        H0 = self.layer_norm0(H0)
        assert H0.shape == [b, s, d]
        if dev_cnt > 1:
            H0 = self.allreduce_mha(H0)

        # feed-forward network
        H1 = self.H_matmul1(H0, self.W1)  # [b, s, 4 * d / dev_cnt]
        assert H1.shape == [b, s, 4 * d // dev_cnt]
        H1 = self.H_gelu(H1)
        H2 = self.H_matmul2(H1, self.W2)  #  [b, s, d]
        assert H2.shape == [b, s, d]
        H2 = self.layer_norm1(H2)
        if dev_cnt > 1:
            H2 = self.allreduce_ffn(H2)

        assert H2.shape == [b, s, d]
        return H2

    def roofline_model(self, system: System):
        device = system.device
        interconnect = system.interconnect

        qkv_latency = 3 * (
            self.Q_proj.roofline_model(device) + device.compute_module.overhead.matmul
        )
        q_mul_k_latency = (
            self.Q_mul_K.roofline_model(device) + device.compute_module.overhead.matmul
        )
        a_mul_v_latency = (
            self.A_mul_V.roofline_model(device) + device.compute_module.overhead.matmul
        )
        h_matmul0_latency = (
            self.H_matmul0.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h1_matmul1_latency = (
            self.H_matmul1.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h2_matmul2_latency = (
            self.H_matmul2.roofline_model(device)
            + device.compute_module.overhead.matmul
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.roofline_model(device)
            + device.compute_module.overhead.softmax
        )
        layernorm_latency = (
            self.layer_norm0.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.roofline_model(device) + device.compute_module.overhead.gelu
        )

        # allreduce
        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_total_latency = 0
            allreduce_total_latency = 0

        # others

        # print
        print("Roofline breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        )
        self.roofline_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {gelu_latency}, {allreduce_latency}, {allreduce_latency}"
        print("total:")
        print(
            f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        )
        self.roofline_latency = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        return self.roofline_latency

    def compile_and_simulate(self, system: System, compile_mode: str, mapping_save_path: str = None):
        import sys
        device = system.device
        interconnect = system.interconnect

        # matmul
        print(f"[Init TP] simulating qkv (M={self.Q_proj.M}, N={self.Q_proj.N}, K={self.Q_proj.K})", flush=True)
        qkv_latency = 3 * (
            self.Q_proj.compile_and_simulate(device, compile_mode, mapping_save_path, "Q_proj")
            + device.compute_module.overhead.matmul
        )
        print(f"[Init TP] simulating q_mul_k done, qkv_latency={qkv_latency*1e3:.4f}ms", flush=True)
        q_mul_k_latency = (
            self.Q_mul_K.compile_and_simulate(device, compile_mode, mapping_save_path, "Q_mul_K")
            + device.compute_module.overhead.matmul
        )
        print(f"[Init TP] simulating a_mul_v done, q_mul_k_latency={q_mul_k_latency*1e3:.4f}ms", flush=True)
        a_mul_v_latency = (
            self.A_mul_V.compile_and_simulate(device, compile_mode, mapping_save_path, "A_mul_V")
            + device.compute_module.overhead.matmul
        )
        print(f"[Init TP] simulating h_matmul0 done, a_mul_v_latency={a_mul_v_latency*1e3:.4f}ms", flush=True)
        h_matmul0_latency = (
            self.H_matmul0.compile_and_simulate(device, compile_mode, mapping_save_path, "H_matmul0")
            + device.compute_module.overhead.matmul
        )
        print(f"[Init TP] simulating h1_matmul1 done, h_matmul0_latency={h_matmul0_latency*1e3:.4f}ms", flush=True)
        h1_matmul1_latency = (
            self.H_matmul1.compile_and_simulate(device, compile_mode, mapping_save_path, "H_matmul1")
            + device.compute_module.overhead.matmul
        )
        print(f"[Init TP] simulating h2_matmul2 done, h1_matmul1_latency={h1_matmul1_latency*1e3:.4f}ms", flush=True)
        h2_matmul2_latency = (
            self.H_matmul2.compile_and_simulate(device, compile_mode, mapping_save_path, "H_matmul2")
            + device.compute_module.overhead.matmul
        )
        print(f"[Init TP] finish matmul simulation, h2_matmul2_latency={h2_matmul2_latency*1e3:.4f}ms", flush=True)

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.softmax
        )
        layernorm_latency = (
            self.layer_norm0.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.layernorm
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.gelu
        )

        # allreduce
        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        # others

        # print
        # print("breakdown:")
        # print(
        #     f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        # )
        # print("total:")
        # print(
        #     f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        # )
        self.latency = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        self.simluate_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {gelu_latency}, {allreduce_latency}, {allreduce_latency}"
        return self.latency

    def run_on_gpu(self):
        # matmul
        qkv_latency = (
            self.Q_proj.run_on_gpu()  # - self.Q_proj.gpu_kernel_launch_overhead()
        ) * 3
        q_mul_k_latency = (
            self.Q_mul_K.run_on_gpu()  # - self.Q_mul_K.gpu_kernel_launch_overhead()
        )
        a_mul_v_latency = (
            self.A_mul_V.run_on_gpu()  # - self.A_mul_V.gpu_kernel_launch_overhead()
        )
        h_matmul0_latency = (
            self.H_matmul0.run_on_gpu()  # - self.H_matmul0.gpu_kernel_launch_overhead()
        )
        h1_matmul1_latency = (
            self.H_matmul1.run_on_gpu()  # - self.H_matmul1.gpu_kernel_launch_overhead()
        )
        h2_matmul2_latency = (
            self.H_matmul2.run_on_gpu()  # - self.H_matmul2.gpu_kernel_launch_overhead()
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.run_on_gpu()  # - self.A_softmax.gpu_kernel_launch_overhead()
        )
        layernorm_latency = (
            self.layer_norm0.run_on_gpu()
            - self.layer_norm0.gpu_kernel_launch_overhead()
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.run_on_gpu()  # - self.H_gelu.gpu_kernel_launch_overhead()
        )

        # allreduce
        allreduce_total_latency = 0

        # others

        # print
        print("breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n"
        )
        print("total:")
        print(
            f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        )
        self.latency_on_gpu = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        return self.latency_on_gpu


class TransformerBlockAutoRegressionTP(Operator):
    def __init__(self, d_model, n_heads, device_count, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.n_heads = n_heads
        self.device_count = device_count
        # parameters per device
        d = d_model
        self.Wq = Tensor([d, d // device_count], data_type)
        self.Wk = Tensor([d, d // device_count], data_type)
        self.Wv = Tensor([d, d // device_count], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)
        self.W1 = Tensor([d, 4 * d // device_count], data_type)
        self.W2 = Tensor([4 * d // device_count, d], data_type)
        # operators per device
        # # multi-head attention
        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)
        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)
        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)
        self.K_concat = Concat(data_type)
        self.V_concat = Concat(data_type)
        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type)
        self.layer_norm0 = LayerNorm(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)
        # # feed-forward network
        self.H_matmul1 = Matmul(data_type)
        self.H_gelu = GeLU(data_type)
        self.H_matmul2 = Matmul(data_type)
        self.layer_norm1 = LayerNorm(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)

    def __call__(self, x: Tensor, seq_len: int) -> Tensor:
        # b: batch size
        # s: sequence length
        # d: hidden dimension
        # d_h: dimension per head
        b, _, d = x.shape
        assert d == self.d_model
        s = seq_len
        h = self.n_heads
        dev_cnt = self.device_count
        d_h = d // h

        # KV cache
        K_cache = Tensor([b, h // dev_cnt, d_h, s], self.data_type)
        V_cache = Tensor([b, h // dev_cnt, s, d_h], self.data_type)

        # multi-head attention
        q = self.Q_proj(x, self.Wq)  # [b, 1, d / dev_cnt]
        assert q.shape == [b, 1, d // dev_cnt]
        k = self.K_proj(x, self.Wk)  # [b, 1, d / dev_cnt]
        v = self.V_proj(x, self.Wv)  # [b, 1, d / dev_cnt]
        q = self.Q_reshape(q, [b, 1, h // dev_cnt, d_h])
        k = self.K_reshape(k, [b, 1, h // dev_cnt, d_h])
        v = self.V_reshape(v, [b, 1, h // dev_cnt, d_h])
        q_T = self.Q_transpose(q, [0, 2, 1, 3])  # [b, h / dev_cnt, 1, d_h]
        assert q_T.shape == [b, h // dev_cnt, 1, d_h]
        k_T = self.K_transpose(k, [0, 2, 3, 1])  # [b, h / dev_cnt, d_h, 1]
        assert k_T.shape == [b, h // dev_cnt, d_h, 1]
        v_T = self.V_transpose(v, [0, 2, 1, 3])  # [b, h / dev_cnt, 1, d_h]
        assert v_T.shape == [b, h // dev_cnt, 1, d_h]
        K_T = self.K_concat(K_cache, k_T, 3)  # [b, h / dev_cnt, d_h, s+1]
        assert K_T.shape == [b, h // dev_cnt, d_h, s + 1]
        V_T = self.V_concat(V_cache, v_T, 2)  # [b, h / dev_cnt, s+1, d_h]
        assert V_T.shape == [b, h // dev_cnt, s + 1, d_h]
        a = self.Q_mul_K(q_T, K_T)  # [b, h / dev_cnt, 1, s+1]
        assert a.shape == [b, h // dev_cnt, 1, s + 1]
        a_prob = self.A_softmax(a)
        h0 = self.A_mul_V(a_prob, V_T)  #  [b, h / dev_cnt, 1, d_h]
        assert h0.shape == [b, h // dev_cnt, 1, d_h]
        h0 = self.H_transpose(h0, [0, 2, 1, 3])  #  [b, 1, h / dev_cnt, d_h]
        assert h0.shape == [b, 1, h // dev_cnt, d_h]
        h0 = self.H_reshape(h0, [b, 1, d // dev_cnt])
        assert h0.shape == [b, 1, d // dev_cnt]
        h0 = self.H_matmul0(h0, self.W0)  #  [b, 1, d]
        assert h0.shape == [b, 1, d]
        h0 = self.layer_norm0(h0)
        assert h0.shape == [b, 1, d]
        if dev_cnt > 1:
            h0 = self.allreduce_mha(h0)

        # feed-forward network
        h1 = self.H_matmul1(h0, self.W1)  # [b, 1, 4 * d / dev_cnt]
        assert h1.shape == [b, 1, 4 * d // dev_cnt]
        h1 = self.H_gelu(h1)
        h2 = self.H_matmul2(h1, self.W2)  #  [b, 1, d]
        assert h2.shape == [b, 1, d]
        h2 = self.layer_norm1(h2)
        if dev_cnt > 1:
            h2 = self.allreduce_ffn(h2)

        assert h2.shape == [b, 1, d]
        self.memory_requirement = (
            self.Wq.size * self.Wq.data_type.word_size
            + self.Wk.size * self.Wk.data_type.word_size
            + self.Wv.size * self.Wv.data_type.word_size
            + self.W0.size * self.W0.data_type.word_size
            + self.W1.size * self.W1.data_type.word_size
            + self.W2.size * self.W2.data_type.word_size
            + K_cache.size * K_cache.data_type.word_size
            + V_cache.size * V_cache.data_type.word_size
        )
        return h2

    def roofline_model(self, system: System):
        device = system.device
        interconnect = system.interconnect

        qkv_latency = 3 * (
            self.Q_proj.roofline_model(device) + device.compute_module.overhead.matmul
        )
        q_mul_k_latency = (
            self.Q_mul_K.roofline_model(device) + device.compute_module.overhead.matmul
        )
        a_mul_v_latency = (
            self.A_mul_V.roofline_model(device) + device.compute_module.overhead.matmul
        )
        h_matmul0_latency = (
            self.H_matmul0.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h1_matmul1_latency = (
            self.H_matmul1.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h2_matmul2_latency = (
            self.H_matmul2.roofline_model(device)
            + device.compute_module.overhead.matmul
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.roofline_model(device)
            + device.compute_module.overhead.softmax
        )
        layernorm_latency = (
            self.layer_norm0.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.roofline_model(device) + device.compute_module.overhead.gelu
        )

        # allreduce
        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        # others

        # print
        print("Roofline breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        )
        print("total:")
        print(
            f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        )
        self.roofline_latency = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        # print(f'memory requirement: {self.memory_requirement/1e9*96}GB')
        self.roofline_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {gelu_latency}, {allreduce_latency}, {allreduce_latency}"
        return self.roofline_latency

    def compile_and_simulate(self, system: System, compile_mode: str, mapping_save_path: str = None):
        pcb = system.device
        interconnect = system.interconnect

        # matmul
        # print("simulating qkv")
        qkv_latency = 3 * (
            self.Q_proj.compile_and_simulate(pcb, compile_mode, mapping_save_path, "Q_proj")
            + pcb.compute_module.overhead.matmul
        )
        # print("simulating q_mul_k")
        q_mul_k_latency = (
            self.Q_mul_K.compile_and_simulate(pcb, compile_mode, mapping_save_path, "Q_mul_K")
            + pcb.compute_module.overhead.matmul
        )
        # print("simulating a_mul_v")
        a_mul_v_latency = (
            self.A_mul_V.compile_and_simulate(pcb, compile_mode, mapping_save_path, "A_mul_V")
            + pcb.compute_module.overhead.matmul
        )
        # print("simulating h_matmul0")
        h_matmul0_latency = (
            self.H_matmul0.compile_and_simulate(pcb, compile_mode, mapping_save_path, "H_matmul0")
            + pcb.compute_module.overhead.matmul
        )
        # print("simulating h1_matmul1")
        h1_matmul1_latency = (
            self.H_matmul1.compile_and_simulate(pcb, compile_mode, mapping_save_path, "H_matmul1")
            + pcb.compute_module.overhead.matmul
        )
        # print("simulating h2_matmul2")
        h2_matmul2_latency = (
            self.H_matmul2.compile_and_simulate(pcb, compile_mode, mapping_save_path, "H_matmul2")
            + pcb.compute_module.overhead.matmul
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.softmax
        )
        layernorm_latency = (
            self.layer_norm0.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.layernorm
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.compile_and_simulate(pcb, compile_mode)
            + pcb.compute_module.overhead.gelu
        )

        # allreduce
        if self.device_count > 1:
            allreduce_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_total_latency = allreduce_latency * 2
        else:
            allreduce_latency = 0
            allreduce_total_latency = 0

        # others

        # print
        # print("breakdown:")
        # print(
        #     f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n{allreduce_latency}\n{allreduce_latency}\n"
        # )
        # print("total:")
        # print(
        #     f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        # )
        self.latency = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        self.simluate_log = f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, {h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, {softmax_latency}, {layernorm_latency}, {layernorm_latency}, {gelu_latency}, {allreduce_latency}, {allreduce_latency}"
        return self.latency

    def run_on_gpu(self):
        # matmul
        qkv_latency = (
            self.Q_proj.run_on_gpu()  # - self.Q_proj.gpu_kernel_launch_overhead()
        ) * 3
        q_mul_k_latency = (
            self.Q_mul_K.run_on_gpu()  # - self.Q_mul_K.gpu_kernel_launch_overhead()
        )
        a_mul_v_latency = (
            self.A_mul_V.run_on_gpu()  # - self.A_mul_V.gpu_kernel_launch_overhead()
        )
        h_matmul0_latency = (
            self.H_matmul0.run_on_gpu()  # - self.H_matmul0.gpu_kernel_launch_overhead()
        )
        h1_matmul1_latency = (
            self.H_matmul1.run_on_gpu()  # - self.H_matmul1.gpu_kernel_launch_overhead()
        )
        h2_matmul2_latency = (
            self.H_matmul2.run_on_gpu()  # - self.H_matmul2.gpu_kernel_launch_overhead()
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.run_on_gpu()  # - self.A_softmax.gpu_kernel_launch_overhead()
        )
        layernorm_latency = (
            self.layer_norm0.run_on_gpu()
            - self.layer_norm0.gpu_kernel_launch_overhead()
        )

        normlization_total_latency = softmax_latency + layernorm_latency * 2

        # gelu
        gelu_latency = (
            self.H_gelu.run_on_gpu()  # - self.H_gelu.gpu_kernel_launch_overhead()
        )
        # gelu_latency = max(gelu_latency, 1e-7)

        # allreduce
        allreduce_total_latency = 0

        # others

        # print
        print("breakdown:")
        print(
            f"{qkv_latency}\n{q_mul_k_latency}\n{a_mul_v_latency}\n{h_matmul0_latency}\n{h1_matmul1_latency}\n{h2_matmul2_latency}\n{softmax_latency}\n{layernorm_latency}\n{layernorm_latency}\n{gelu_latency}\n"
        )
        print("total:")
        print(
            f"{matmul_total_latency}\n{normlization_total_latency}\n{gelu_latency}\n{allreduce_total_latency}\n"
        )
        self.latency_on_gpu = (
            matmul_total_latency
            + normlization_total_latency
            + gelu_latency
            + allreduce_total_latency
        )
        return self.latency_on_gpu


class LLMInitComputationTP:
    def __init__(
        self,
        d_model,
        n_heads,
        n_layers,
        device_count,
    ) -> None:
        pass

class TransformerBlockOPTInitComputationTP(Operator):
    """
    OPT prefill / init computation block with tensor parallelism.

    OPT主要改动：
    1. FFN hidden dim 从固定 4*d 改成 config.ffn_dim
    2. LayerNorm 默认改成 pre-LN:
       LN -> Attention -> residual
       LN -> FFN -> residual
    3. activation 从 GPT-3 常用的 GeLU 改成 OPT 默认的 ReLU
       但如果 LLMCompass 里没有 ReLU operator，则自动退回 GeLU 近似
    4. all-reduce 放在 row-parallel projection 之后，再进入 residual / post-LN
    """

    def __init__(
        self,
        d_model,
        n_heads,
        ffn_dim,
        device_count,
        data_type: DataType,
        do_layer_norm_before=True,
        activation_function="relu",
    ):
        super().__init__(0, 0, 0, 0, data_type)

        self.d_model = d_model
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.device_count = device_count
        self.do_layer_norm_before = do_layer_norm_before
        self.activation_function = activation_function

        d = d_model
        f = ffn_dim
        dev_cnt = device_count

        assert d % dev_cnt == 0
        assert f % dev_cnt == 0
        assert n_heads % dev_cnt == 0
        assert d % n_heads == 0

        # ============================================================
        # parameters per device
        # ============================================================

        # OPT self-attention projections.
        # Tensor parallelism:
        # q/k/v are column-parallel: [d, d/dev_cnt]
        self.Wq = Tensor([d, d // dev_cnt], data_type)
        self.Wk = Tensor([d, d // dev_cnt], data_type)
        self.Wv = Tensor([d, d // dev_cnt], data_type)

        # output projection is row-parallel:
        # local input is d/dev_cnt, output is full d, then all-reduce
        self.W0 = Tensor([d // dev_cnt, d], data_type)

        # ===================== OPT CHANGE ==========================
        # GPT-3原代码:
        #   self.W1 = Tensor([d, 4 * d // device_count], data_type)
        #   self.W2 = Tensor([4 * d // device_count, d], data_type)
        #
        # OPT不能写死 4*d，而应该使用 config.ffn_dim。
        # fc1: hidden_size -> ffn_dim
        # fc2: ffn_dim -> hidden_size
        # ============================================================
        self.W1 = Tensor([d, f // dev_cnt], data_type)
        self.W2 = Tensor([f // dev_cnt, d], data_type)

        # ============================================================
        # operators per device
        # ============================================================

        # ===================== OPT CHANGE ==========================
        # OPT大多数模型是 pre-LN:
        #   self_attn_layer_norm before attention
        #   final_layer_norm before FFN
        #
        # 因此这里把两个LayerNorm明确命名为：
        #   layer_norm_attn: attention前/后使用
        #   layer_norm_ffn : FFN前/后使用
        # ============================================================
        self.layer_norm_attn = LayerNorm(data_type)
        self.layer_norm_ffn = LayerNorm(data_type)

        # multi-head attention
        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)

        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)

        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)

        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)

        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)

        self.H_matmul0 = Matmul(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)

        # feed-forward network
        self.H_matmul1 = Matmul(data_type)

        # ===================== OPT CHANGE ==========================
        # OPT默认 activation_function 通常是 "relu"。
        # 但是LLMCompass原始代码里一般只有 GeLU operator。
        #
        # 如果你已经在算子库里实现了 ReLU，这里会自动使用 ReLU；
        # 如果没有 ReLU，就退回 GeLU 近似，保证代码先能跑。
        # 更严谨的OPT建模建议你后续补一个 ReLU operator。
        # ============================================================
        if activation_function == "relu" and "ReLU" in globals():
            self.H_act = ReLU(data_type)
            self.activation_overhead_name = "relu"
        else:
            self.H_act = GeLU(data_type)
            self.activation_overhead_name = "gelu"

        self.H_matmul2 = Matmul(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)

    def _activation_overhead(self, device):
        """
        LLMCompass原来的Overhead里通常只有 gelu。
        如果你后续给 Overhead 加了 relu 字段，这里会自动使用 relu；
        否则使用 gelu overhead 作为近似。
        """
        return getattr(
            device.compute_module.overhead,
            self.activation_overhead_name,
            device.compute_module.overhead.gelu,
        )

    def __call__(self, X: Tensor) -> Tensor:
        # b: batch size
        # s: sequence length
        # d: hidden dimension
        # d_h: dimension per head
        b, s, d = X.shape

        assert d == self.d_model

        h = self.n_heads
        dev_cnt = self.device_count
        d_h = d // h
        f = self.ffn_dim

        # ============================================================
        # OPT MHA block
        #
        # GPT-3原代码大致是：
        #   Attention(X)
        #   output projection
        #   LayerNorm
        #
        # OPT大多数模型是：
        #   residual = X
        #   X_attn = LayerNorm(X)
        #   H0 = Attention(X_attn)
        #   H0 = output projection
        #   H0 = all-reduce
        #   Y = residual + H0
        #
        # 注意：
        # LLMCompass原始TransformerBlock里没有显式建 residual Add。
        # 为了和原框架风格一致，这里也不显式统计 Add。
        # 如果你想更精确，需要额外实现 elementwise Add operator。
        # ============================================================

        if self.do_layer_norm_before:
            X_attn = self.layer_norm_attn(X)
            assert X_attn.shape == [b, s, d]
        else:
            X_attn = X

        Q = self.Q_proj(X_attn, self.Wq)  # [b, s, d / dev_cnt]
        assert Q.shape == [b, s, d // dev_cnt]

        K = self.K_proj(X_attn, self.Wk)  # [b, s, d / dev_cnt]
        assert K.shape == [b, s, d // dev_cnt]

        V = self.V_proj(X_attn, self.Wv)  # [b, s, d / dev_cnt]
        assert V.shape == [b, s, d // dev_cnt]

        Q = self.Q_reshape(Q, [b, s, h // dev_cnt, d_h])
        K = self.K_reshape(K, [b, s, h // dev_cnt, d_h])
        V = self.V_reshape(V, [b, s, h // dev_cnt, d_h])

        Q_T = self.Q_transpose(Q, [0, 2, 1, 3])  # [b, h/dev_cnt, s, d_h]
        assert Q_T.shape == [b, h // dev_cnt, s, d_h]

        K_T = self.K_transpose(K, [0, 2, 3, 1])  # [b, h/dev_cnt, d_h, s]
        assert K_T.shape == [b, h // dev_cnt, d_h, s]

        V_T = self.V_transpose(V, [0, 2, 1, 3])  # [b, h/dev_cnt, s, d_h]
        assert V_T.shape == [b, h // dev_cnt, s, d_h]

        A = self.Q_mul_K(Q_T, K_T)  # [b, h/dev_cnt, s, s]
        assert A.shape == [b, h // dev_cnt, s, s]

        A_prob = self.A_softmax(A)

        H = self.A_mul_V(A_prob, V_T)  # [b, h/dev_cnt, s, d_h]
        assert H.shape == [b, h // dev_cnt, s, d_h]

        H = self.H_transpose(H, [0, 2, 1, 3])  # [b, s, h/dev_cnt, d_h]
        assert H.shape == [b, s, h // dev_cnt, d_h]

        H = self.H_reshape(H, [b, s, d // dev_cnt])
        assert H.shape == [b, s, d // dev_cnt]

        H0 = self.H_matmul0(H, self.W0)  # [b, s, d]
        assert H0.shape == [b, s, d]

        # ===================== OPT CHANGE ==========================
        # row-parallel output projection之后应该先 all-reduce。
        # 原代码是 H0 = layer_norm0(H0) 之后再 all-reduce。
        # 对 tensor parallel 的 row-parallel projection 来说，
        # all-reduce 应该发生在完整输出进入下一步之前。
        # ============================================================
        if dev_cnt > 1:
            H0 = self.allreduce_mha(H0)
            assert H0.shape == [b, s, d]

        # residual add: Y = X + H0
        # LLMCompass原始代码没有Add operator，这里只保留shape流。
        Y = H0
        assert Y.shape == [b, s, d]

        # OPT-350m这种post-LN结构会走这里
        if not self.do_layer_norm_before:
            Y = self.layer_norm_attn(Y)
            assert Y.shape == [b, s, d]

        # ============================================================
        # OPT FFN block
        #
        # GPT-3原代码：
        #   H1 = H_matmul1(H0, W1)       # d -> 4d/dev
        #   H1 = GeLU(H1)
        #   H2 = H_matmul2(H1, W2)       # 4d/dev -> d
        #   H2 = LayerNorm(H2)
        #
        # OPT大多数模型：
        #   residual = Y
        #   Y_ffn = LayerNorm(Y)
        #   H1 = fc1(Y_ffn)              # d -> ffn_dim/dev
        #   H1 = ReLU(H1)
        #   H2 = fc2(H1)                 # ffn_dim/dev -> d
        #   H2 = all-reduce
        #   Z = residual + H2
        # ============================================================

        if self.do_layer_norm_before:
            Y_ffn = self.layer_norm_ffn(Y)
            assert Y_ffn.shape == [b, s, d]
        else:
            Y_ffn = Y

        H1 = self.H_matmul1(Y_ffn, self.W1)  # [b, s, ffn_dim/dev_cnt]
        assert H1.shape == [b, s, f // dev_cnt]

        H1 = self.H_act(H1)
        assert H1.shape == [b, s, f // dev_cnt]

        H2 = self.H_matmul2(H1, self.W2)  # [b, s, d]
        assert H2.shape == [b, s, d]

        # row-parallel fc2之后也需要 all-reduce
        if dev_cnt > 1:
            H2 = self.allreduce_ffn(H2)
            assert H2.shape == [b, s, d]

        # residual add: Z = Y + H2
        # 同样，这里不显式统计Add，只保留shape流。
        Z = H2
        assert Z.shape == [b, s, d]

        if not self.do_layer_norm_before:
            Z = self.layer_norm_ffn(Z)
            assert Z.shape == [b, s, d]

        return Z

    def roofline_model(self, system: System):
        device = system.device
        interconnect = system.interconnect

        # matmul
        qkv_latency = 3 * (
            self.Q_proj.roofline_model(device) + device.compute_module.overhead.matmul
        )
        q_mul_k_latency = (
            self.Q_mul_K.roofline_model(device) + device.compute_module.overhead.matmul
        )
        a_mul_v_latency = (
            self.A_mul_V.roofline_model(device) + device.compute_module.overhead.matmul
        )
        h_matmul0_latency = (
            self.H_matmul0.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h1_matmul1_latency = (
            self.H_matmul1.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h2_matmul2_latency = (
            self.H_matmul2.roofline_model(device)
            + device.compute_module.overhead.matmul
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.roofline_model(device)
            + device.compute_module.overhead.softmax
        )

        # ===================== OPT CHANGE ==========================
        # 原代码只调用 layer_norm0，然后 *2。
        # 这里明确区分 attention LN 和 FFN LN。
        # 它们shape通常一样，所以数值可能相同，但语义更准确。
        # ============================================================
        layernorm_attn_latency = (
            self.layer_norm_attn.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )
        layernorm_ffn_latency = (
            self.layer_norm_ffn.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )

        normalization_total_latency = (
            softmax_latency + layernorm_attn_latency + layernorm_ffn_latency
        )

        # activation
        activation_latency = (
            self.H_act.roofline_model(device) + self._activation_overhead(device)
        )

        # allreduce
        if self.device_count > 1:
            allreduce_mha_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_ffn_latency = self.allreduce_ffn.simulate(interconnect)
            allreduce_total_latency = allreduce_mha_latency + allreduce_ffn_latency
        else:
            allreduce_mha_latency = 0
            allreduce_ffn_latency = 0
            allreduce_total_latency = 0

        print("Roofline breakdown:")
        print(
            f"{qkv_latency}\n"
            f"{q_mul_k_latency}\n"
            f"{a_mul_v_latency}\n"
            f"{h_matmul0_latency}\n"
            f"{h1_matmul1_latency}\n"
            f"{h2_matmul2_latency}\n"
            f"{softmax_latency}\n"
            f"{layernorm_attn_latency}\n"
            f"{layernorm_ffn_latency}\n"
            f"{activation_latency}\n"
            f"{allreduce_mha_latency}\n"
            f"{allreduce_ffn_latency}\n"
        )

        self.roofline_log = (
            f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, "
            f"{h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, "
            f"{softmax_latency}, {layernorm_attn_latency}, {layernorm_ffn_latency}, "
            f"{activation_latency}, {allreduce_mha_latency}, {allreduce_ffn_latency}"
        )

        print("total:")
        print(
            f"{matmul_total_latency}\n"
            f"{normalization_total_latency}\n"
            f"{activation_latency}\n"
            f"{allreduce_total_latency}\n"
        )

        self.roofline_latency = (
            matmul_total_latency
            + normalization_total_latency
            + activation_latency
            + allreduce_total_latency
        )

        return self.roofline_latency

    def compile_and_simulate(
        self,
        system: System,
        compile_mode: str,
        mapping_save_path: str = None,
    ):
        device = system.device
        interconnect = system.interconnect

        # matmul
        print("simulating qkv")
        qkv_latency = 3 * (
            self.Q_proj.compile_and_simulate(
                device, compile_mode, mapping_save_path, "Q_proj", sparsity_ratio=1.7623148071785417
            )
            + device.compute_module.overhead.matmul
        )

        # print("simulating q_mul_k")
        # q_mul_k_latency = (
        #     self.Q_mul_K.compile_and_simulate(
        #         device, compile_mode, mapping_save_path, "Q_mul_K"
        #     )
        #     + device.compute_module.overhead.matmul
        # )
        q_mul_k_latency = 0

        # print("simulating a_mul_v")
        # a_mul_v_latency = (
        #     self.A_mul_V.compile_and_simulate(
        #         device, compile_mode, mapping_save_path, "A_mul_V"
        #     )
        #     + device.compute_module.overhead.matmul
        # )
        a_mul_v_latency = 0

        print("simulating h_matmul0")
        h_matmul0_latency = (
            self.H_matmul0.compile_and_simulate(
                device, compile_mode, mapping_save_path, "H_matmul0", sparsity_ratio= 1.6495933957066935
            )
            + device.compute_module.overhead.matmul
        )

        print("simulating h1_matmul1")
        h1_matmul1_latency = (
            self.H_matmul1.compile_and_simulate(
                device, compile_mode, mapping_save_path, "H_matmul1", sparsity_ratio=1.764290135580415
            )
            + device.compute_module.overhead.matmul
        )

        print("simulating h2_matmul2")
        h2_matmul2_latency = (
            self.H_matmul2.compile_and_simulate(
                device, compile_mode, mapping_save_path, "H_matmul2", sparsity_ratio=17.646223634697243
            )
            + device.compute_module.overhead.matmul
        )

        print("finish matmul simulation")

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.softmax
        )

        layernorm_attn_latency = (
            self.layer_norm_attn.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.layernorm
        )

        layernorm_ffn_latency = (
            self.layer_norm_ffn.compile_and_simulate(device, compile_mode)
            + device.compute_module.overhead.layernorm
        )

        normalization_total_latency = (
            softmax_latency + layernorm_attn_latency + layernorm_ffn_latency
        )

        # activation
        activation_latency = (
            self.H_act.compile_and_simulate(device, compile_mode)
            + self._activation_overhead(device)
        )

        # allreduce
        if self.device_count > 1:
            allreduce_mha_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_ffn_latency = self.allreduce_ffn.simulate(interconnect)
            allreduce_total_latency = allreduce_mha_latency + allreduce_ffn_latency
        else:
            allreduce_mha_latency = 0
            allreduce_ffn_latency = 0
            allreduce_total_latency = 0

        self.latency = (
            matmul_total_latency
            + normalization_total_latency
            + activation_latency
            + allreduce_total_latency
        )

        self.simluate_log = (
            f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, "
            f"{h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, "
            f"{softmax_latency}, {layernorm_attn_latency}, {layernorm_ffn_latency}, "
            f"{activation_latency}, {allreduce_mha_latency}, {allreduce_ffn_latency}"
        )

        return self.latency

    def run_on_gpu(self):
        # matmul
        qkv_latency = self.Q_proj.run_on_gpu() * 3
        q_mul_k_latency = self.Q_mul_K.run_on_gpu()
        a_mul_v_latency = self.A_mul_V.run_on_gpu()
        h_matmul0_latency = self.H_matmul0.run_on_gpu()
        h1_matmul1_latency = self.H_matmul1.run_on_gpu()
        h2_matmul2_latency = self.H_matmul2.run_on_gpu()

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = self.A_softmax.run_on_gpu()

        layernorm_attn_latency = (
            self.layer_norm_attn.run_on_gpu()
            - self.layer_norm_attn.gpu_kernel_launch_overhead()
        )

        layernorm_ffn_latency = (
            self.layer_norm_ffn.run_on_gpu()
            - self.layer_norm_ffn.gpu_kernel_launch_overhead()
        )

        normalization_total_latency = (
            softmax_latency + layernorm_attn_latency + layernorm_ffn_latency
        )

        # activation
        activation_latency = self.H_act.run_on_gpu()

        # allreduce
        allreduce_total_latency = 0

        print("breakdown:")
        print(
            f"{qkv_latency}\n"
            f"{q_mul_k_latency}\n"
            f"{a_mul_v_latency}\n"
            f"{h_matmul0_latency}\n"
            f"{h1_matmul1_latency}\n"
            f"{h2_matmul2_latency}\n"
            f"{softmax_latency}\n"
            f"{layernorm_attn_latency}\n"
            f"{layernorm_ffn_latency}\n"
            f"{activation_latency}\n"
        )

        print("total:")
        print(
            f"{matmul_total_latency}\n"
            f"{normalization_total_latency}\n"
            f"{activation_latency}\n"
            f"{allreduce_total_latency}\n"
        )

        self.latency_on_gpu = (
            matmul_total_latency
            + normalization_total_latency
            + activation_latency
            + allreduce_total_latency
        )

        return self.latency_on_gpu
    
class TransformerBlockOPTAutoRegressionTP(Operator):
    """
    OPT autoregressive decoding block with tensor parallelism.

    对应 decode / autoregression 阶段：
        输入 x: [b, 1, d]
        KV cache length: seq_len = s
        输出: [b, 1, d]

    相比原 TransformerBlockAutoRegressionTP 的主要修改：
    1. 增加 ffn_dim，不再写死 4*d
    2. 默认使用 OPT pre-LN:
          LN -> Self-Attention -> residual
          LN -> FFN -> residual
    3. activation 从固定 GeLU 改成 OPT 的 activation_function，通常是 ReLU
    4. all-reduce 放在 row-parallel projection 之后
    5. KV cache / concat 逻辑保持和原 decode 版本一致
    """

    def __init__(
        self,
        d_model,
        n_heads,
        ffn_dim,
        device_count,
        data_type: DataType,
        do_layer_norm_before=True,
        activation_function="relu",
    ):
        super().__init__(0, 0, 0, 0, data_type)

        self.d_model = d_model
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.device_count = device_count
        self.do_layer_norm_before = do_layer_norm_before
        self.activation_function = activation_function

        d = d_model
        f = ffn_dim
        dev_cnt = device_count

        assert d % dev_cnt == 0
        assert f % dev_cnt == 0
        assert n_heads % dev_cnt == 0
        assert d % n_heads == 0

        # ============================================================
        # parameters per device
        # ============================================================

        # q/k/v column-parallel projection
        self.Wq = Tensor([d, d // dev_cnt], data_type)
        self.Wk = Tensor([d, d // dev_cnt], data_type)
        self.Wv = Tensor([d, d // dev_cnt], data_type)

        # output projection is row-parallel
        self.W0 = Tensor([d // dev_cnt, d], data_type)

        # ===================== OPT CHANGE ==========================
        # 原 GPT-style decode 写法:
        #     self.W1 = Tensor([d, 4 * d // device_count], data_type)
        #     self.W2 = Tensor([4 * d // device_count, d], data_type)
        #
        # OPT 应该使用 config.ffn_dim:
        #     fc1: d_model -> ffn_dim
        #     fc2: ffn_dim -> d_model
        # ============================================================
        self.W1 = Tensor([d, f // dev_cnt], data_type)
        self.W2 = Tensor([f // dev_cnt, d], data_type)

        # ============================================================
        # operators per device
        # ============================================================

        # ===================== OPT CHANGE ==========================
        # OPT 大多数模型使用 pre-LN。
        # 注意这里不再叫 layer_norm0 / layer_norm1，
        # 而是明确区分 attention LN 和 FFN LN。
        # ============================================================
        self.layer_norm_attn = LayerNorm(data_type)
        self.layer_norm_ffn = LayerNorm(data_type)

        # multi-head attention
        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)

        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)

        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)

        # decode 阶段需要把新 token 的 k/v 拼接到 KV cache
        self.K_concat = Concat(data_type)
        self.V_concat = Concat(data_type)

        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)

        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)

        self.H_matmul0 = Matmul(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)

        # FFN
        self.H_matmul1 = Matmul(data_type)

        # ===================== OPT CHANGE ==========================
        # OPT activation_function 通常是 "relu"。
        # 如果你已经实现了 ReLU operator，则使用 ReLU；
        # 如果没有，就临时退回 GeLU，保证代码可以先跑。
        # ============================================================
        if activation_function == "relu" and "ReLU" in globals():
            self.H_act = ReLU(data_type)
            self.activation_overhead_name = "relu"
        else:
            self.H_act = GeLU(data_type)
            self.activation_overhead_name = "gelu"

        self.H_matmul2 = Matmul(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)

    def _activation_overhead(self, device):
        """
        原 LLMCompass 的 Overhead 里一般只有 gelu。
        如果你后续加了 relu overhead，这里会自动使用；
        否则用 gelu overhead 作为 elementwise activation 的近似。
        """
        return getattr(
            device.compute_module.overhead,
            self.activation_overhead_name,
            device.compute_module.overhead.gelu,
        )

    def __call__(self, x: Tensor, seq_len: int) -> Tensor:
        # b: batch size
        # decode 阶段当前 token 长度固定为 1
        # s: KV cache 已有 sequence length
        # d: hidden dimension
        # d_h: dimension per head

        b, one, d = x.shape
        assert one == 1
        assert d == self.d_model

        s = seq_len
        h = self.n_heads
        dev_cnt = self.device_count
        d_h = d // h
        f = self.ffn_dim

        # ============================================================
        # KV cache
        #
        # K_cache: [b, h/dev_cnt, d_h, s]
        # V_cache: [b, h/dev_cnt, s, d_h]
        #
        # 新 token 的 k/v 会 concat 后变成长度 s+1。
        # ============================================================
        K_cache = Tensor([b, h // dev_cnt, d_h, s], self.data_type)
        V_cache = Tensor([b, h // dev_cnt, s, d_h], self.data_type)

        # ============================================================
        # OPT MHA decode block
        #
        # 原 GPT-style decode:
        #     q/k/v = proj(x)
        #     attention
        #     h0 = out_proj(...)
        #     h0 = layer_norm0(h0)
        #     allreduce
        #
        # OPT pre-LN decode:
        #     residual = x
        #     x_attn = layer_norm_attn(x)
        #     q/k/v = proj(x_attn)
        #     attention with KV cache
        #     h0 = out_proj(...)
        #     allreduce
        #     y = residual + h0
        #
        # 注意：这里仍然没有显式统计 residual Add，
        # 因为原 LLMCompass 代码也没有 Add operator。
        # ============================================================

        if self.do_layer_norm_before:
            x_attn = self.layer_norm_attn(x)
            assert x_attn.shape == [b, 1, d]
        else:
            x_attn = x

        q = self.Q_proj(x_attn, self.Wq)  # [b, 1, d/dev_cnt]
        assert q.shape == [b, 1, d // dev_cnt]

        k = self.K_proj(x_attn, self.Wk)  # [b, 1, d/dev_cnt]
        assert k.shape == [b, 1, d // dev_cnt]

        v = self.V_proj(x_attn, self.Wv)  # [b, 1, d/dev_cnt]
        assert v.shape == [b, 1, d // dev_cnt]

        q = self.Q_reshape(q, [b, 1, h // dev_cnt, d_h])
        k = self.K_reshape(k, [b, 1, h // dev_cnt, d_h])
        v = self.V_reshape(v, [b, 1, h // dev_cnt, d_h])

        q_T = self.Q_transpose(q, [0, 2, 1, 3])
        assert q_T.shape == [b, h // dev_cnt, 1, d_h]

        k_T = self.K_transpose(k, [0, 2, 3, 1])
        assert k_T.shape == [b, h // dev_cnt, d_h, 1]

        v_T = self.V_transpose(v, [0, 2, 1, 3])
        assert v_T.shape == [b, h // dev_cnt, 1, d_h]

        K_T = self.K_concat(K_cache, k_T, 3)
        assert K_T.shape == [b, h // dev_cnt, d_h, s + 1]

        V_T = self.V_concat(V_cache, v_T, 2)
        assert V_T.shape == [b, h // dev_cnt, s + 1, d_h]

        a = self.Q_mul_K(q_T, K_T)
        assert a.shape == [b, h // dev_cnt, 1, s + 1]

        a_prob = self.A_softmax(a)
        assert a_prob.shape == [b, h // dev_cnt, 1, s + 1]

        h0 = self.A_mul_V(a_prob, V_T)
        assert h0.shape == [b, h // dev_cnt, 1, d_h]

        h0 = self.H_transpose(h0, [0, 2, 1, 3])
        assert h0.shape == [b, 1, h // dev_cnt, d_h]

        h0 = self.H_reshape(h0, [b, 1, d // dev_cnt])
        assert h0.shape == [b, 1, d // dev_cnt]

        h0 = self.H_matmul0(h0, self.W0)
        assert h0.shape == [b, 1, d]

        # ===================== OPT CHANGE ==========================
        # row-parallel output projection 后先 all-reduce。
        # 原 decode 代码是 layer_norm0 后再 all-reduce。
        # 这里调整为更合理的 tensor-parallel 顺序。
        # ============================================================
        if dev_cnt > 1:
            h0 = self.allreduce_mha(h0)
            assert h0.shape == [b, 1, d]

        # residual add: y = x + h0
        # 不显式建 Add operator，只保留 shape 流。
        y = h0
        assert y.shape == [b, 1, d]

        if not self.do_layer_norm_before:
            y = self.layer_norm_attn(y)
            assert y.shape == [b, 1, d]

        # ============================================================
        # OPT FFN decode block
        #
        # 原 GPT-style decode:
        #     h1 = H_matmul1(h0, W1)      # d -> 4d/dev
        #     h1 = GeLU(h1)
        #     h2 = H_matmul2(h1, W2)      # 4d/dev -> d
        #     h2 = layer_norm1(h2)
        #     allreduce
        #
        # OPT pre-LN:
        #     residual = y
        #     y_ffn = layer_norm_ffn(y)
        #     h1 = fc1(y_ffn)             # d -> ffn_dim/dev
        #     h1 = activation(h1)
        #     h2 = fc2(h1)                # ffn_dim/dev -> d
        #     allreduce
        #     z = residual + h2
        # ============================================================

        if self.do_layer_norm_before:
            y_ffn = self.layer_norm_ffn(y)
            assert y_ffn.shape == [b, 1, d]
        else:
            y_ffn = y

        h1 = self.H_matmul1(y_ffn, self.W1)
        assert h1.shape == [b, 1, f // dev_cnt]

        h1 = self.H_act(h1)
        assert h1.shape == [b, 1, f // dev_cnt]

        h2 = self.H_matmul2(h1, self.W2)
        assert h2.shape == [b, 1, d]

        if dev_cnt > 1:
            h2 = self.allreduce_ffn(h2)
            assert h2.shape == [b, 1, d]

        # residual add: z = y + h2
        z = h2
        assert z.shape == [b, 1, d]

        if not self.do_layer_norm_before:
            z = self.layer_norm_ffn(z)
            assert z.shape == [b, 1, d]

        # memory requirement:
        # 权重 + KV cache。
        # 注意 W1/W2 已经从 4*d 改成 ffn_dim。
        self.memory_requirement = (
            self.Wq.size * self.Wq.data_type.word_size
            + self.Wk.size * self.Wk.data_type.word_size
            + self.Wv.size * self.Wv.data_type.word_size
            + self.W0.size * self.W0.data_type.word_size
            + self.W1.size * self.W1.data_type.word_size
            + self.W2.size * self.W2.data_type.word_size
            + K_cache.size * K_cache.data_type.word_size
            + V_cache.size * V_cache.data_type.word_size
        )

        return z

    def roofline_model(self, system: System):
        device = system.device
        interconnect = system.interconnect

        qkv_latency = 3 * (
            self.Q_proj.roofline_model(device)
            + device.compute_module.overhead.matmul
        )

        q_mul_k_latency = (
            self.Q_mul_K.roofline_model(device)
            + device.compute_module.overhead.matmul
        )

        a_mul_v_latency = (
            self.A_mul_V.roofline_model(device)
            + device.compute_module.overhead.matmul
        )

        h_matmul0_latency = (
            self.H_matmul0.roofline_model(device)
            + device.compute_module.overhead.matmul
        )

        h1_matmul1_latency = (
            self.H_matmul1.roofline_model(device)
            + device.compute_module.overhead.matmul
        )

        h2_matmul2_latency = (
            self.H_matmul2.roofline_model(device)
            + device.compute_module.overhead.matmul
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = (
            self.A_softmax.roofline_model(device)
            + device.compute_module.overhead.softmax
        )

        # ===================== OPT CHANGE ==========================
        # 原代码只用 layer_norm0，然后乘 2。
        # 这里拆成 attention LN 和 FFN LN。
        # shape 都是 [b, 1, d]，数值可能一样，但语义更清楚。
        # ============================================================
        layernorm_attn_latency = (
            self.layer_norm_attn.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )

        layernorm_ffn_latency = (
            self.layer_norm_ffn.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )

        normalization_total_latency = (
            softmax_latency
            + layernorm_attn_latency
            + layernorm_ffn_latency
        )

        # activation
        activation_latency = (
            self.H_act.roofline_model(device)
            + self._activation_overhead(device)
        )

        # allreduce
        if self.device_count > 1:
            allreduce_mha_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_ffn_latency = self.allreduce_ffn.simulate(interconnect)
            allreduce_total_latency = allreduce_mha_latency + allreduce_ffn_latency
        else:
            allreduce_mha_latency = 0
            allreduce_ffn_latency = 0
            allreduce_total_latency = 0

        print("Roofline breakdown:")
        print(
            f"{qkv_latency}\n"
            f"{q_mul_k_latency}\n"
            f"{a_mul_v_latency}\n"
            f"{h_matmul0_latency}\n"
            f"{h1_matmul1_latency}\n"
            f"{h2_matmul2_latency}\n"
            f"{softmax_latency}\n"
            f"{layernorm_attn_latency}\n"
            f"{layernorm_ffn_latency}\n"
            f"{activation_latency}\n"
            f"{allreduce_mha_latency}\n"
            f"{allreduce_ffn_latency}\n"
        )

        print("total:")
        print(
            f"{matmul_total_latency}\n"
            f"{normalization_total_latency}\n"
            f"{activation_latency}\n"
            f"{allreduce_total_latency}\n"
        )

        self.roofline_latency = (
            matmul_total_latency
            + normalization_total_latency
            + activation_latency
            + allreduce_total_latency
        )

        self.roofline_log = (
            f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, "
            f"{h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, "
            f"{softmax_latency}, {layernorm_attn_latency}, {layernorm_ffn_latency}, "
            f"{activation_latency}, {allreduce_mha_latency}, {allreduce_ffn_latency}"
        )

        return self.roofline_latency

    def compile_and_simulate(
        self,
        system: System,
        compile_mode: str,
        mapping_save_path: str = None,
    ):
        pcb = system.device
        interconnect = system.interconnect

        # matmul
        qkv_latency = 3 * (
            self.Q_proj.compile_and_simulate(
                pcb, compile_mode, mapping_save_path, "Q_proj", sparsity_ratio=1.610625655365
            )
            + pcb.compute_module.overhead.matmul
        )

        q_mul_k_latency =(
            self.Q_mul_K.compile_and_simulate(
                pcb, compile_mode, mapping_save_path, "Q_mul_K", sparsity_ratio= 9.545313309166
            )
            + pcb.compute_module.overhead.matmul
        )

        a_mul_v_latency =(
            self.A_mul_V.compile_and_simulate(
                pcb, compile_mode, mapping_save_path, "A_mul_V", sparsity_ratio= 1.769075727037
            )
            + pcb.compute_module.overhead.matmul
        )

        h_matmul0_latency = (
            self.H_matmul0.compile_and_simulate(
                pcb, compile_mode, mapping_save_path, "H_matmul0", sparsity_ratio=1.618545837724
            )
            + pcb.compute_module.overhead.matmul
        )

        h1_matmul1_latency = (
            self.H_matmul1.compile_and_simulate(
                pcb, compile_mode, mapping_save_path, "H_matmul1", sparsity_ratio=1.586230636833
            )
            + pcb.compute_module.overhead.matmul
        )

        h2_matmul2_latency = (
            self.H_matmul2.compile_and_simulate(
                pcb, compile_mode, mapping_save_path, "H_matmul2", sparsity_ratio=8.332730560579
            )
            + pcb.compute_module.overhead.matmul
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = 0
        # (
        #     self.A_softmax.compile_and_simulate(pcb, compile_mode)
        #     + pcb.compute_module.overhead.softmax
        # )

        layernorm_attn_latency = 0
        # (
        #     self.layer_norm_attn.compile_and_simulate(pcb, compile_mode)
        #     + pcb.compute_module.overhead.layernorm
        # )

        layernorm_ffn_latency = 0
        # (
        #     self.layer_norm_ffn.compile_and_simulate(pcb, compile_mode)
        #     + pcb.compute_module.overhead.layernorm
        # )

        normalization_total_latency = 0
        # (
        #     softmax_latency
        #     + layernorm_attn_latency
        #     + layernorm_ffn_latency
        # )

        # activation
        activation_latency = 0
        # (
        #     self.H_act.compile_and_simulate(pcb, compile_mode)
        #     + self._activation_overhead(pcb)
        # )

        # allreduce
        if self.device_count > 1:
            allreduce_mha_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_ffn_latency = self.allreduce_ffn.simulate(interconnect)
            allreduce_total_latency = allreduce_mha_latency + allreduce_ffn_latency
        else:
            allreduce_mha_latency = 0
            allreduce_ffn_latency = 0
            allreduce_total_latency = 0

        self.latency = (
            matmul_total_latency
            + normalization_total_latency
            + activation_latency
            + allreduce_total_latency
        )

        self.simluate_log = (
            f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, "
            f"{h_matmul0_latency}, {h1_matmul1_latency}, {h2_matmul2_latency}, "
            f"{softmax_latency}, {layernorm_attn_latency}, {layernorm_ffn_latency}, "
            f"{activation_latency}, {allreduce_mha_latency}, {allreduce_ffn_latency}"
        )

        return self.latency

    def run_on_gpu(self):
        # matmul
        qkv_latency = self.Q_proj.run_on_gpu() * 3
        q_mul_k_latency = self.Q_mul_K.run_on_gpu()
        a_mul_v_latency = self.A_mul_V.run_on_gpu()
        h_matmul0_latency = self.H_matmul0.run_on_gpu()
        h1_matmul1_latency = self.H_matmul1.run_on_gpu()
        h2_matmul2_latency = self.H_matmul2.run_on_gpu()

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + h1_matmul1_latency
            + h2_matmul2_latency
        )

        # normalization
        softmax_latency = self.A_softmax.run_on_gpu()

        layernorm_attn_latency = (
            self.layer_norm_attn.run_on_gpu()
            - self.layer_norm_attn.gpu_kernel_launch_overhead()
        )

        layernorm_ffn_latency = (
            self.layer_norm_ffn.run_on_gpu()
            - self.layer_norm_ffn.gpu_kernel_launch_overhead()
        )

        normalization_total_latency = (
            softmax_latency
            + layernorm_attn_latency
            + layernorm_ffn_latency
        )

        # activation
        activation_latency = self.H_act.run_on_gpu()

        # allreduce
        allreduce_total_latency = 0

        print("breakdown:")
        print(
            f"{qkv_latency}\n"
            f"{q_mul_k_latency}\n"
            f"{a_mul_v_latency}\n"
            f"{h_matmul0_latency}\n"
            f"{h1_matmul1_latency}\n"
            f"{h2_matmul2_latency}\n"
            f"{softmax_latency}\n"
            f"{layernorm_attn_latency}\n"
            f"{layernorm_ffn_latency}\n"
            f"{activation_latency}\n"
        )

        print("total:")
        print(
            f"{matmul_total_latency}\n"
            f"{normalization_total_latency}\n"
            f"{activation_latency}\n"
            f"{allreduce_total_latency}\n"
        )

        self.latency_on_gpu = (
            matmul_total_latency
            + normalization_total_latency
            + activation_latency
            + allreduce_total_latency
        )

        return self.latency_on_gpu

class TransformerBlockQwen25InitComputationTP(Operator):
    """
    Qwen2.5 prefill / init computation block with tensor parallelism.

    Qwen2.5-1.5B 主要结构：
    1. hidden_size = 1536
    2. num_attention_heads = 12
    3. num_key_value_heads = 2, 即 GQA
    4. intermediate_size = 8960
    5. RMSNorm + Attention + residual
    6. RMSNorm + SwiGLU MLP + residual

    注意：
    - LLMCompass 原始代码通常没有 RMSNorm / SiLU / elementwise Mul / RoPE 算子。
    - 这里用 LayerNorm 近似 RMSNorm。
    - 如果你已经实现了 SiLU，则自动使用 SiLU；否则用 GeLU 近似 SiLU。
    - SwiGLU 的 gate * up 这个 elementwise mul 暂时不单独统计，只保留 shape 流。
    - RoPE 和 QKV bias 也不单独建模。
    """

    def __init__(
        self,
        d_model,
        n_heads,
        n_kv_heads,
        ffn_dim,
        device_count,
        data_type: DataType,
        kv_partition_mode="replicate",
    ):
        super().__init__(0, 0, 0, 0, data_type)

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.ffn_dim = ffn_dim
        self.device_count = device_count
        self.kv_partition_mode = kv_partition_mode

        d = d_model
        h = n_heads
        h_kv = n_kv_heads
        f = ffn_dim
        dev_cnt = device_count
        d_h = d // h

        assert d % h == 0
        assert h % dev_cnt == 0
        assert f % dev_cnt == 0
        assert d % dev_cnt == 0

        self.q_heads_local = h // dev_cnt
        self.q_dim_local = self.q_heads_local * d_h

        # Qwen2.5-1.5B: n_kv_heads=2。
        # 如果 device_count=4，则 2 不能整除 4。
        # 所以默认 replicate：每个 device 都保留完整 KV heads，
        # 这样可以避免 assert 失败。
        if kv_partition_mode == "shard":
            assert h_kv % dev_cnt == 0, (
                "num_key_value_heads must be divisible by device_count "
                "when kv_partition_mode='shard'. "
                "For Qwen2.5-1.5B with device_count=4, use 'replicate'."
            )
            self.kv_heads_local = h_kv // dev_cnt
        elif kv_partition_mode == "replicate":
            self.kv_heads_local = h_kv
        else:
            raise ValueError("kv_partition_mode must be 'replicate' or 'shard'.")

        self.kv_dim_local = self.kv_heads_local * d_h

        # ============================================================
        # Parameters per device
        # ============================================================

        # Q projection: hidden_size -> local query heads
        self.Wq = Tensor([d, self.q_dim_local], data_type)

        # K/V projection: hidden_size -> local KV heads
        # GQA 下 K/V 维度远小于 Q。
        self.Wk = Tensor([d, self.kv_dim_local], data_type)
        self.Wv = Tensor([d, self.kv_dim_local], data_type)

        # o_proj: local query heads -> hidden_size
        self.W0 = Tensor([self.q_dim_local, d], data_type)

        # SwiGLU MLP:
        # gate_proj: hidden_size -> intermediate_size
        # up_proj  : hidden_size -> intermediate_size
        # down_proj: intermediate_size -> hidden_size
        self.W_gate = Tensor([d, f // dev_cnt], data_type)
        self.W_up = Tensor([d, f // dev_cnt], data_type)
        self.W_down = Tensor([f // dev_cnt, d], data_type)

        # ============================================================
        # Operators per device
        # ============================================================

        NormOp = globals().get("RMSNorm", LayerNorm)
        self.rms_norm_attn = NormOp(data_type)
        self.rms_norm_ffn = NormOp(data_type)

        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)

        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)

        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)

        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)

        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)

        self.H_matmul0 = Matmul(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)

        self.Gate_proj = Matmul(data_type)
        self.Up_proj = Matmul(data_type)

        if "SiLU" in globals():
            self.H_act = SiLU(data_type)
            self.activation_overhead_name = "silu"
        else:
            self.H_act = GeLU(data_type)
            self.activation_overhead_name = "gelu"

        self.Down_proj = Matmul(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)

    def _activation_overhead(self, device):
        return getattr(
            device.compute_module.overhead,
            self.activation_overhead_name,
            device.compute_module.overhead.gelu,
        )

    def __call__(self, X: Tensor) -> Tensor:
        b, s, d = X.shape
        assert d == self.d_model

        h = self.n_heads
        dev_cnt = self.device_count
        d_h = d // h
        f = self.ffn_dim

        # ============================================================
        # Attention block: RMSNorm -> GQA Attention -> o_proj
        # ============================================================

        X_attn = self.rms_norm_attn(X)
        assert X_attn.shape == [b, s, d]

        Q = self.Q_proj(X_attn, self.Wq)
        assert Q.shape == [b, s, self.q_dim_local]

        K = self.K_proj(X_attn, self.Wk)
        assert K.shape == [b, s, self.kv_dim_local]

        V = self.V_proj(X_attn, self.Wv)
        assert V.shape == [b, s, self.kv_dim_local]

        Q = self.Q_reshape(Q, [b, s, self.q_heads_local, d_h])
        assert Q.shape == [b, s, self.q_heads_local, d_h]

        K = self.K_reshape(K, [b, s, self.kv_heads_local, d_h])
        assert K.shape == [b, s, self.kv_heads_local, d_h]

        V = self.V_reshape(V, [b, s, self.kv_heads_local, d_h])
        assert V.shape == [b, s, self.kv_heads_local, d_h]

        Q_T = self.Q_transpose(Q, [0, 2, 1, 3])
        assert Q_T.shape == [b, self.q_heads_local, s, d_h]

        K_T_small = self.K_transpose(K, [0, 2, 3, 1])
        assert K_T_small.shape == [b, self.kv_heads_local, d_h, s]

        V_T_small = self.V_transpose(V, [0, 2, 1, 3])
        assert V_T_small.shape == [b, self.kv_heads_local, s, d_h]

        # GQA repeat_kv：
        # 这里不单独统计 repeat 的开销，只把参与 attention 的 shape 展开到 q_heads_local。
        K_T = Tensor([b, self.q_heads_local, d_h, s], self.data_type)
        V_T = Tensor([b, self.q_heads_local, s, d_h], self.data_type)

        A = self.Q_mul_K(Q_T, K_T)
        assert A.shape == [b, self.q_heads_local, s, s]

        A_prob = self.A_softmax(A)
        assert A_prob.shape == [b, self.q_heads_local, s, s]

        H = self.A_mul_V(A_prob, V_T)
        assert H.shape == [b, self.q_heads_local, s, d_h]

        H = self.H_transpose(H, [0, 2, 1, 3])
        assert H.shape == [b, s, self.q_heads_local, d_h]

        H = self.H_reshape(H, [b, s, self.q_dim_local])
        assert H.shape == [b, s, self.q_dim_local]

        H0 = self.H_matmul0(H, self.W0)
        assert H0.shape == [b, s, d]

        if dev_cnt > 1:
            H0 = self.allreduce_mha(H0)
            assert H0.shape == [b, s, d]

        # residual add: X + H0
        # LLMCompass 原始代码没有显式 Add operator，这里只保留 shape。
        Y = H0
        assert Y.shape == [b, s, d]

        # ============================================================
        # MLP block: RMSNorm -> gate/up -> SiLU(gate) * up -> down
        # ============================================================

        Y_ffn = self.rms_norm_ffn(Y)
        assert Y_ffn.shape == [b, s, d]

        Gate = self.Gate_proj(Y_ffn, self.W_gate)
        assert Gate.shape == [b, s, f // dev_cnt]

        Up = self.Up_proj(Y_ffn, self.W_up)
        assert Up.shape == [b, s, f // dev_cnt]

        Gate_act = self.H_act(Gate)
        assert Gate_act.shape == [b, s, f // dev_cnt]

        # SwiGLU: Gate_act * Up
        # 暂时不统计 elementwise mul，shape 使用 Gate_act。
        H1 = Gate_act
        assert H1.shape == [b, s, f // dev_cnt]

        H2 = self.Down_proj(H1, self.W_down)
        assert H2.shape == [b, s, d]

        if dev_cnt > 1:
            H2 = self.allreduce_ffn(H2)
            assert H2.shape == [b, s, d]

        # residual add: Y + H2
        Z = H2
        assert Z.shape == [b, s, d]

        return Z

    def roofline_model(self, system: System):
        device = system.device
        interconnect = system.interconnect

        q_latency = (
            self.Q_proj.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        k_latency = (
            self.K_proj.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        v_latency = (
            self.V_proj.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        qkv_latency = q_latency + k_latency + v_latency

        q_mul_k_latency = (
            self.Q_mul_K.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        a_mul_v_latency = (
            self.A_mul_V.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h_matmul0_latency = (
            self.H_matmul0.roofline_model(device)
            + device.compute_module.overhead.matmul
        )

        gate_latency = (
            self.Gate_proj.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        up_latency = (
            self.Up_proj.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        gate_up_latency = gate_latency + up_latency

        down_latency = (
            self.Down_proj.roofline_model(device)
            + device.compute_module.overhead.matmul
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + gate_up_latency
            + down_latency
        )

        softmax_latency = (
            self.A_softmax.roofline_model(device)
            + device.compute_module.overhead.softmax
        )

        norm_attn_latency = (
            self.rms_norm_attn.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )
        norm_ffn_latency = (
            self.rms_norm_ffn.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )

        normalization_total_latency = (
            softmax_latency + norm_attn_latency + norm_ffn_latency
        )

        activation_latency = (
            self.H_act.roofline_model(device)
            + self._activation_overhead(device)
        )

        if self.device_count > 1:
            allreduce_mha_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_ffn_latency = self.allreduce_ffn.simulate(interconnect)
            allreduce_total_latency = allreduce_mha_latency + allreduce_ffn_latency
        else:
            allreduce_mha_latency = 0
            allreduce_ffn_latency = 0
            allreduce_total_latency = 0

        print("Roofline breakdown:")
        print(
            f"{qkv_latency}\n"
            f"{q_mul_k_latency}\n"
            f"{a_mul_v_latency}\n"
            f"{h_matmul0_latency}\n"
            f"{gate_up_latency}\n"
            f"{down_latency}\n"
            f"{softmax_latency}\n"
            f"{norm_attn_latency}\n"
            f"{norm_ffn_latency}\n"
            f"{activation_latency}\n"
            f"{allreduce_mha_latency}\n"
            f"{allreduce_ffn_latency}\n"
        )

        print("total:")
        print(
            f"{matmul_total_latency}\n"
            f"{normalization_total_latency}\n"
            f"{activation_latency}\n"
            f"{allreduce_total_latency}\n"
        )

        self.roofline_log = (
            f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, "
            f"{h_matmul0_latency}, {gate_up_latency}, {down_latency}, "
            f"{softmax_latency}, {norm_attn_latency}, {norm_ffn_latency}, "
            f"{activation_latency}, {allreduce_mha_latency}, {allreduce_ffn_latency}"
        )

        self.roofline_latency = (
            matmul_total_latency
            + normalization_total_latency
            + activation_latency
            + allreduce_total_latency
        )

        return self.roofline_latency

    def compile_and_simulate(
        self,
        system: System,
        compile_mode: str,
        mapping_save_path: str = None,
        include_attention: bool = False,
    ):
        device = system.device
        interconnect = system.interconnect

        elementwise_compile_mode = (
            "heuristic-CIM"
            if compile_mode in (
                "heuristic-CIM-activation-major",
                "heuristic-CIM-weight-major",
            )
            else compile_mode
        )

        print("simulating q_proj")
        q_latency = (
            self.Q_proj.compile_and_simulate(
                device, compile_mode, mapping_save_path, "Q_proj", sparsity_ratio=3.519874201545
            )
            + device.compute_module.overhead.matmul
        )

        print("simulating k_proj")
        k_latency = (
            self.K_proj.compile_and_simulate(
                device, compile_mode, mapping_save_path, "K_proj", sparsity_ratio=3.519874201545
            )
            + device.compute_module.overhead.matmul
        )

        print("simulating v_proj")
        v_latency = (
            self.V_proj.compile_and_simulate(
                device, compile_mode, mapping_save_path, "V_proj", sparsity_ratio=3.519874201545
            )
            + device.compute_module.overhead.matmul
        )

        qkv_latency = q_latency + k_latency + v_latency

        if include_attention:
            print("simulating q_mul_k")
            q_mul_k_latency = (
                self.Q_mul_K.compile_and_simulate(
                    device,
                    compile_mode,
                    mapping_save_path,
                    "Q_mul_K",
                    sparsity_ratio=13.240733719052,
                )
                + device.compute_module.overhead.matmul
            )

            print("simulating a_mul_v")
            a_mul_v_latency = (
                self.A_mul_V.compile_and_simulate(
                    device,
                    compile_mode,
                    mapping_save_path,
                    "A_mul_V",
                    sparsity_ratio=3.072269148579,
                )
                + device.compute_module.overhead.matmul
            )
        else:
            q_mul_k_latency = 0
            a_mul_v_latency = 0

        print("simulating o_proj")
        h_matmul0_latency = (
            self.H_matmul0.compile_and_simulate(
                device, compile_mode, mapping_save_path, "H_matmul0", sparsity_ratio=3.017747346852
            )
            + device.compute_module.overhead.matmul
        )

        print("simulating gate_proj")
        gate_latency = (
            self.Gate_proj.compile_and_simulate(
                device, compile_mode, mapping_save_path, "Gate_proj", sparsity_ratio=3.074047363576
            )
            + device.compute_module.overhead.matmul
        )

        print("simulating up_proj")
        up_latency = (
            self.Up_proj.compile_and_simulate(
                device, compile_mode, mapping_save_path, "Up_proj", sparsity_ratio=3.074047363576
            )
            + device.compute_module.overhead.matmul
        )

        gate_up_latency = gate_latency + up_latency

        print("simulating down_proj")
        down_latency = (
            self.Down_proj.compile_and_simulate(
                device, compile_mode, mapping_save_path, "Down_proj", sparsity_ratio=4.473686427507
            )
            + device.compute_module.overhead.matmul
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + gate_up_latency
            + down_latency
        )

        softmax_latency = (
            self.A_softmax.compile_and_simulate(device, elementwise_compile_mode)
            + device.compute_module.overhead.softmax
        )

        norm_attn_latency = (
            self.rms_norm_attn.compile_and_simulate(device, elementwise_compile_mode)
            + device.compute_module.overhead.layernorm
        )

        norm_ffn_latency = (
            self.rms_norm_ffn.compile_and_simulate(device, elementwise_compile_mode)
            + device.compute_module.overhead.layernorm
        )

        normalization_total_latency = (
            softmax_latency + norm_attn_latency + norm_ffn_latency
        )

        activation_latency = (
            self.H_act.compile_and_simulate(device, elementwise_compile_mode)
            + self._activation_overhead(device)
        )

        if self.device_count > 1:
            allreduce_mha_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_ffn_latency = self.allreduce_ffn.simulate(interconnect)
            allreduce_total_latency = allreduce_mha_latency + allreduce_ffn_latency
        else:
            allreduce_mha_latency = 0
            allreduce_ffn_latency = 0
            allreduce_total_latency = 0

        self.latency = (
            matmul_total_latency
            + normalization_total_latency
            + activation_latency
            + allreduce_total_latency
        )

        self.simluate_log = (
            f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, "
            f"{h_matmul0_latency}, {gate_up_latency}, {down_latency}, "
            f"{softmax_latency}, {norm_attn_latency}, {norm_ffn_latency}, "
            f"{activation_latency}, {allreduce_mha_latency}, {allreduce_ffn_latency}"
        )

        return self.latency

    def run_on_gpu(self):
        q_latency = self.Q_proj.run_on_gpu()
        k_latency = self.K_proj.run_on_gpu()
        v_latency = self.V_proj.run_on_gpu()
        qkv_latency = q_latency + k_latency + v_latency

        q_mul_k_latency = self.Q_mul_K.run_on_gpu()
        a_mul_v_latency = self.A_mul_V.run_on_gpu()
        h_matmul0_latency = self.H_matmul0.run_on_gpu()

        gate_latency = self.Gate_proj.run_on_gpu()
        up_latency = self.Up_proj.run_on_gpu()
        gate_up_latency = gate_latency + up_latency

        down_latency = self.Down_proj.run_on_gpu()

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + gate_up_latency
            + down_latency
        )

        softmax_latency = self.A_softmax.run_on_gpu()

        norm_attn_latency = (
            self.rms_norm_attn.run_on_gpu()
            - self.rms_norm_attn.gpu_kernel_launch_overhead()
        )

        norm_ffn_latency = (
            self.rms_norm_ffn.run_on_gpu()
            - self.rms_norm_ffn.gpu_kernel_launch_overhead()
        )

        normalization_total_latency = (
            softmax_latency + norm_attn_latency + norm_ffn_latency
        )

        activation_latency = self.H_act.run_on_gpu()

        allreduce_total_latency = 0

        print("breakdown:")
        print(
            f"{qkv_latency}\n"
            f"{q_mul_k_latency}\n"
            f"{a_mul_v_latency}\n"
            f"{h_matmul0_latency}\n"
            f"{gate_up_latency}\n"
            f"{down_latency}\n"
            f"{softmax_latency}\n"
            f"{norm_attn_latency}\n"
            f"{norm_ffn_latency}\n"
            f"{activation_latency}\n"
        )

        print("total:")
        print(
            f"{matmul_total_latency}\n"
            f"{normalization_total_latency}\n"
            f"{activation_latency}\n"
            f"{allreduce_total_latency}\n"
        )

        self.latency_on_gpu = (
            matmul_total_latency
            + normalization_total_latency
            + activation_latency
            + allreduce_total_latency
        )

        return self.latency_on_gpu


class TransformerBlockQwen25AutoRegressionTP(Operator):
    """
    Qwen2.5 decode / autoregressive block with tensor parallelism.

    输入:
        x: [b, 1, d]
        seq_len: KV cache 已有长度 s

    输出:
        z: [b, 1, d]
    """

    def __init__(
        self,
        d_model,
        n_heads,
        n_kv_heads,
        ffn_dim,
        device_count,
        data_type: DataType,
        kv_partition_mode="replicate",
        shared_kv_gqa=False,
    ):
        super().__init__(0, 0, 0, 0, data_type)

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.ffn_dim = ffn_dim
        self.device_count = device_count
        self.kv_partition_mode = kv_partition_mode
        self.shared_kv_gqa = shared_kv_gqa

        d = d_model
        h = n_heads
        h_kv = n_kv_heads
        f = ffn_dim
        dev_cnt = device_count
        d_h = d // h

        assert d % h == 0
        assert h % dev_cnt == 0
        assert f % dev_cnt == 0
        assert d % dev_cnt == 0

        self.q_heads_local = h // dev_cnt
        self.q_dim_local = self.q_heads_local * d_h

        if kv_partition_mode == "shard":
            assert h_kv % dev_cnt == 0, (
                "num_key_value_heads must be divisible by device_count "
                "when kv_partition_mode='shard'. "
                "For Qwen2.5-1.5B with device_count=4, use 'replicate'."
            )
            self.kv_heads_local = h_kv // dev_cnt
        elif kv_partition_mode == "replicate":
            self.kv_heads_local = h_kv
        else:
            raise ValueError("kv_partition_mode must be 'replicate' or 'shard'.")

        self.kv_dim_local = self.kv_heads_local * d_h

        if self.shared_kv_gqa:
            assert (
                self.q_heads_local % self.kv_heads_local == 0
            ), "shared-KV GQA requires local Q heads divisible by local KV heads"
            self.q_heads_per_kv_head = (
                self.q_heads_local // self.kv_heads_local
            )

        self.Wq = Tensor([d, self.q_dim_local], data_type)
        self.Wk = Tensor([d, self.kv_dim_local], data_type)
        self.Wv = Tensor([d, self.kv_dim_local], data_type)
        self.W0 = Tensor([self.q_dim_local, d], data_type)

        self.W_gate = Tensor([d, f // dev_cnt], data_type)
        self.W_up = Tensor([d, f // dev_cnt], data_type)
        self.W_down = Tensor([f // dev_cnt, d], data_type)

        NormOp = globals().get("RMSNorm", LayerNorm)
        self.rms_norm_attn = NormOp(data_type)
        self.rms_norm_ffn = NormOp(data_type)

        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)

        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)

        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)

        self.K_concat = Concat(data_type)
        self.V_concat = Concat(data_type)

        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)

        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)

        self.H_matmul0 = Matmul(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)

        self.Gate_proj = Matmul(data_type)
        self.Up_proj = Matmul(data_type)

        if "SiLU" in globals():
            self.H_act = SiLU(data_type)
            self.activation_overhead_name = "silu"
        else:
            self.H_act = GeLU(data_type)
            self.activation_overhead_name = "gelu"

        self.Down_proj = Matmul(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)

    def _activation_overhead(self, device):
        return getattr(
            device.compute_module.overhead,
            self.activation_overhead_name,
            device.compute_module.overhead.gelu,
        )

    def __call__(self, x: Tensor, seq_len: int) -> Tensor:
        b, one, d = x.shape
        assert one == 1
        assert d == self.d_model

        s = seq_len
        h = self.n_heads
        dev_cnt = self.device_count
        d_h = d // h
        f = self.ffn_dim

        # KV cache 用真实 GQA KV heads，而不是完整 Q heads。
        K_cache_small = Tensor([b, self.kv_heads_local, d_h, s], self.data_type)
        V_cache_small = Tensor([b, self.kv_heads_local, s, d_h], self.data_type)

        x_attn = self.rms_norm_attn(x)
        assert x_attn.shape == [b, 1, d]

        q = self.Q_proj(x_attn, self.Wq)
        assert q.shape == [b, 1, self.q_dim_local]

        k = self.K_proj(x_attn, self.Wk)
        assert k.shape == [b, 1, self.kv_dim_local]

        v = self.V_proj(x_attn, self.Wv)
        assert v.shape == [b, 1, self.kv_dim_local]

        if self.shared_kv_gqa:
            q_grouped = self.Q_reshape(
                q,
                [b, self.kv_heads_local, self.q_heads_per_kv_head, d_h],
            )
            assert q_grouped.shape == [
                b,
                self.kv_heads_local,
                self.q_heads_per_kv_head,
                d_h,
            ]
        else:
            q = self.Q_reshape(q, [b, 1, self.q_heads_local, d_h])
            assert q.shape == [b, 1, self.q_heads_local, d_h]

        k = self.K_reshape(k, [b, 1, self.kv_heads_local, d_h])
        assert k.shape == [b, 1, self.kv_heads_local, d_h]

        v = self.V_reshape(v, [b, 1, self.kv_heads_local, d_h])
        assert v.shape == [b, 1, self.kv_heads_local, d_h]

        q_T = None
        if not self.shared_kv_gqa:
            q_T = self.Q_transpose(q, [0, 2, 1, 3])
            assert q_T.shape == [b, self.q_heads_local, 1, d_h]

        k_T_small = self.K_transpose(k, [0, 2, 3, 1])
        assert k_T_small.shape == [b, self.kv_heads_local, d_h, 1]

        v_T_small = self.V_transpose(v, [0, 2, 1, 3])
        assert v_T_small.shape == [b, self.kv_heads_local, 1, d_h]

        K_cat_small = self.K_concat(K_cache_small, k_T_small, 3)
        assert K_cat_small.shape == [b, self.kv_heads_local, d_h, s + 1]

        V_cat_small = self.V_concat(V_cache_small, v_T_small, 2)
        assert V_cat_small.shape == [b, self.kv_heads_local, s + 1, d_h]

        if self.shared_kv_gqa:
            a = self.Q_mul_K(q_grouped, K_cat_small)
            assert a.shape == [
                b,
                self.kv_heads_local,
                self.q_heads_per_kv_head,
                s + 1,
            ]
        else:
            K_T = Tensor(
                [b, self.q_heads_local, d_h, s + 1],
                self.data_type,
            )
            V_T = Tensor(
                [b, self.q_heads_local, s + 1, d_h],
                self.data_type,
            )
            a = self.Q_mul_K(q_T, K_T)
            assert a.shape == [b, self.q_heads_local, 1, s + 1]

        a_prob = self.A_softmax(a)
        if self.shared_kv_gqa:
            assert a_prob.shape == [
                b,
                self.kv_heads_local,
                self.q_heads_per_kv_head,
                s + 1,
            ]
        else:
            assert a_prob.shape == [b, self.q_heads_local, 1, s + 1]

        if self.shared_kv_gqa:
            h0 = self.A_mul_V(a_prob, V_cat_small)
            assert h0.shape == [
                b,
                self.kv_heads_local,
                self.q_heads_per_kv_head,
                d_h,
            ]

            h0 = self.H_reshape(h0, [b, 1, self.q_dim_local])
            assert h0.shape == [b, 1, self.q_dim_local]
        else:
            h0 = self.A_mul_V(a_prob, V_T)
            assert h0.shape == [b, self.q_heads_local, 1, d_h]

            h0 = self.H_transpose(h0, [0, 2, 1, 3])
            assert h0.shape == [b, 1, self.q_heads_local, d_h]

            h0 = self.H_reshape(h0, [b, 1, self.q_dim_local])
            assert h0.shape == [b, 1, self.q_dim_local]

        h0 = self.H_matmul0(h0, self.W0)
        assert h0.shape == [b, 1, d]

        if dev_cnt > 1:
            h0 = self.allreduce_mha(h0)
            assert h0.shape == [b, 1, d]

        y = h0
        assert y.shape == [b, 1, d]

        y_ffn = self.rms_norm_ffn(y)
        assert y_ffn.shape == [b, 1, d]

        gate = self.Gate_proj(y_ffn, self.W_gate)
        assert gate.shape == [b, 1, f // dev_cnt]

        up = self.Up_proj(y_ffn, self.W_up)
        assert up.shape == [b, 1, f // dev_cnt]

        gate_act = self.H_act(gate)
        assert gate_act.shape == [b, 1, f // dev_cnt]

        h1 = gate_act
        assert h1.shape == [b, 1, f // dev_cnt]

        h2 = self.Down_proj(h1, self.W_down)
        assert h2.shape == [b, 1, d]

        if dev_cnt > 1:
            h2 = self.allreduce_ffn(h2)
            assert h2.shape == [b, 1, d]

        z = h2
        assert z.shape == [b, 1, d]

        self.memory_requirement = (
            self.Wq.size * self.Wq.data_type.word_size
            + self.Wk.size * self.Wk.data_type.word_size
            + self.Wv.size * self.Wv.data_type.word_size
            + self.W0.size * self.W0.data_type.word_size
            + self.W_gate.size * self.W_gate.data_type.word_size
            + self.W_up.size * self.W_up.data_type.word_size
            + self.W_down.size * self.W_down.data_type.word_size
            + K_cache_small.size * K_cache_small.data_type.word_size
            + V_cache_small.size * V_cache_small.data_type.word_size
        )

        return z

    def roofline_model(self, system: System):
        device = system.device
        interconnect = system.interconnect

        q_latency = (
            self.Q_proj.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        k_latency = (
            self.K_proj.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        v_latency = (
            self.V_proj.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        qkv_latency = q_latency + k_latency + v_latency

        q_mul_k_latency = (
            self.Q_mul_K.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        a_mul_v_latency = (
            self.A_mul_V.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        h_matmul0_latency = (
            self.H_matmul0.roofline_model(device)
            + device.compute_module.overhead.matmul
        )

        gate_latency = (
            self.Gate_proj.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        up_latency = (
            self.Up_proj.roofline_model(device)
            + device.compute_module.overhead.matmul
        )
        gate_up_latency = gate_latency + up_latency

        down_latency = (
            self.Down_proj.roofline_model(device)
            + device.compute_module.overhead.matmul
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + gate_up_latency
            + down_latency
        )

        softmax_latency = (
            self.A_softmax.roofline_model(device)
            + device.compute_module.overhead.softmax
        )

        norm_attn_latency = (
            self.rms_norm_attn.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )
        norm_ffn_latency = (
            self.rms_norm_ffn.roofline_model(device)
            + device.compute_module.overhead.layernorm
        )

        normalization_total_latency = (
            softmax_latency + norm_attn_latency + norm_ffn_latency
        )

        activation_latency = (
            self.H_act.roofline_model(device)
            + self._activation_overhead(device)
        )

        if self.device_count > 1:
            allreduce_mha_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_ffn_latency = self.allreduce_ffn.simulate(interconnect)
            allreduce_total_latency = allreduce_mha_latency + allreduce_ffn_latency
        else:
            allreduce_mha_latency = 0
            allreduce_ffn_latency = 0
            allreduce_total_latency = 0

        print("Roofline breakdown:")
        print(
            f"{qkv_latency}\n"
            f"{q_mul_k_latency}\n"
            f"{a_mul_v_latency}\n"
            f"{h_matmul0_latency}\n"
            f"{gate_up_latency}\n"
            f"{down_latency}\n"
            f"{softmax_latency}\n"
            f"{norm_attn_latency}\n"
            f"{norm_ffn_latency}\n"
            f"{activation_latency}\n"
            f"{allreduce_mha_latency}\n"
            f"{allreduce_ffn_latency}\n"
        )

        print("total:")
        print(
            f"{matmul_total_latency}\n"
            f"{normalization_total_latency}\n"
            f"{activation_latency}\n"
            f"{allreduce_total_latency}\n"
        )

        self.roofline_log = (
            f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, "
            f"{h_matmul0_latency}, {gate_up_latency}, {down_latency}, "
            f"{softmax_latency}, {norm_attn_latency}, {norm_ffn_latency}, "
            f"{activation_latency}, {allreduce_mha_latency}, {allreduce_ffn_latency}"
        )

        self.roofline_latency = (
            matmul_total_latency
            + normalization_total_latency
            + activation_latency
            + allreduce_total_latency
        )

        return self.roofline_latency

    def compile_and_simulate(
        self,
        system: System,
        compile_mode: str,
        mapping_save_path: str = None,
    ):
        device = system.device
        interconnect = system.interconnect

        attention_compile_mode = (
            "heuristic-CIM-GQA-decode"
            if (
                self.shared_kv_gqa
                and compile_mode == "heuristic-CIM-decode"
            )
            else compile_mode
        )

        q_latency = (
            self.Q_proj.compile_and_simulate(
                device, compile_mode, mapping_save_path, "Q_proj", sparsity_ratio=10.052872029157, att = False
            )
            + device.compute_module.overhead.matmul
        )
        k_latency = (
            self.K_proj.compile_and_simulate(
                device, compile_mode, mapping_save_path, "K_proj", sparsity_ratio=10.052872029157, att = False
            )
            + device.compute_module.overhead.matmul
        )
        v_latency = (
            self.V_proj.compile_and_simulate(
                device, compile_mode, mapping_save_path, "V_proj", sparsity_ratio=10.052872029157, att = False
            )
            + device.compute_module.overhead.matmul
        )
        qkv_latency = q_latency + k_latency + v_latency

        q_mul_k_latency = (
            self.Q_mul_K.compile_and_simulate(
                device,
                attention_compile_mode,
                mapping_save_path,
                "Q_mul_K",
                sparsity_ratio=9.285158421345,
            )
            + device.compute_module.overhead.matmul
        )
        a_mul_v_latency = (
            self.A_mul_V.compile_and_simulate(
                device,
                attention_compile_mode,
                mapping_save_path,
                "A_mul_V",
                sparsity_ratio=2.118058224728,
            )
            + device.compute_module.overhead.matmul
        )



        h_matmul0_latency = (
            self.H_matmul0.compile_and_simulate(
                device, compile_mode, mapping_save_path, "H_matmul0", sparsity_ratio= 4.117520909286, att = False
            )
            + device.compute_module.overhead.matmul
        )

        gate_latency = (
            self.Gate_proj.compile_and_simulate(
                device, compile_mode, mapping_save_path, "Gate_proj", sparsity_ratio=4.694950482944, att = False
            )
            + device.compute_module.overhead.matmul
        )
        up_latency = (
            self.Up_proj.compile_and_simulate(
                device, compile_mode, mapping_save_path, "Up_proj", sparsity_ratio=4.694950482944, att = False
            )
            + device.compute_module.overhead.matmul
        )
        gate_up_latency = gate_latency + up_latency

        down_latency = (
            self.Down_proj.compile_and_simulate(
                device, compile_mode, mapping_save_path, "Down_proj", sparsity_ratio=11.418861512319, att = False
            )
            + device.compute_module.overhead.matmul
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + gate_up_latency
            + down_latency
        )

        softmax_latency = 0#(
        #     self.A_softmax.compile_and_simulate(device, compile_mode)
        #     + device.compute_module.overhead.softmax
        # )

        norm_attn_latency = 0#(
        #     self.rms_norm_attn.compile_and_simulate(device, compile_mode)
        #     + device.compute_module.overhead.layernorm
        # )

        norm_ffn_latency = 0#(
        #     self.rms_norm_ffn.compile_and_simulate(device, compile_mode)
        #     + device.compute_module.overhead.layernorm
        # )

        normalization_total_latency = 0#(
        #     softmax_latency + norm_attn_latency + norm_ffn_latency
        # )

        activation_latency = 0#(
        #     self.H_act.compile_and_simulate(device, compile_mode)
        #     + self._activation_overhead(device)
        # )

        if self.device_count > 1:
            allreduce_mha_latency = self.allreduce_mha.simulate(interconnect)
            allreduce_ffn_latency = self.allreduce_ffn.simulate(interconnect)
            allreduce_total_latency = allreduce_mha_latency + allreduce_ffn_latency
        else:
            allreduce_mha_latency = 0
            allreduce_ffn_latency = 0
            allreduce_total_latency = 0

        self.latency = (
            matmul_total_latency
            + normalization_total_latency
            + activation_latency
            + allreduce_total_latency
        )

        self.simluate_log = (
            f"{qkv_latency}, {q_mul_k_latency}, {a_mul_v_latency}, "
            f"{h_matmul0_latency}, {gate_up_latency}, {down_latency}, "
            f"{softmax_latency}, {norm_attn_latency}, {norm_ffn_latency}, "
            f"{activation_latency}, {allreduce_mha_latency}, {allreduce_ffn_latency}"
        )

        return self.latency

    def run_on_gpu(self):
        q_latency = self.Q_proj.run_on_gpu()
        k_latency = self.K_proj.run_on_gpu()
        v_latency = self.V_proj.run_on_gpu()
        qkv_latency = q_latency + k_latency + v_latency

        q_mul_k_latency = self.Q_mul_K.run_on_gpu()
        a_mul_v_latency = self.A_mul_V.run_on_gpu()
        h_matmul0_latency = self.H_matmul0.run_on_gpu()

        gate_latency = self.Gate_proj.run_on_gpu()
        up_latency = self.Up_proj.run_on_gpu()
        gate_up_latency = gate_latency + up_latency

        down_latency = self.Down_proj.run_on_gpu()

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + h_matmul0_latency
            + gate_up_latency
            + down_latency
        )

        softmax_latency = self.A_softmax.run_on_gpu()

        norm_attn_latency = (
            self.rms_norm_attn.run_on_gpu()
            - self.rms_norm_attn.gpu_kernel_launch_overhead()
        )
        norm_ffn_latency = (
            self.rms_norm_ffn.run_on_gpu()
            - self.rms_norm_ffn.gpu_kernel_launch_overhead()
        )

        normalization_total_latency = (
            softmax_latency + norm_attn_latency + norm_ffn_latency
        )

        activation_latency = self.H_act.run_on_gpu()

        allreduce_total_latency = 0

        print("breakdown:")
        print(
            f"{qkv_latency}\n"
            f"{q_mul_k_latency}\n"
            f"{a_mul_v_latency}\n"
            f"{h_matmul0_latency}\n"
            f"{gate_up_latency}\n"
            f"{down_latency}\n"
            f"{softmax_latency}\n"
            f"{norm_attn_latency}\n"
            f"{norm_ffn_latency}\n"
            f"{activation_latency}\n"
        )

        print("total:")
        print(
            f"{matmul_total_latency}\n"
            f"{normalization_total_latency}\n"
            f"{activation_latency}\n"
            f"{allreduce_total_latency}\n"
        )

        self.latency_on_gpu = (
            matmul_total_latency
            + normalization_total_latency
            + activation_latency
            + allreduce_total_latency
        )

        return self.latency_on_gpu


# class TransformerBlockQwen25_1_5BInitComputationTP(
#     TransformerBlockQwen25InitComputationTP
# ):
#     def __init__(
#         self,
#         device_count,
#         data_type: DataType,
#         kv_partition_mode="replicate",
#     ):
#         super().__init__(
#             d_model=1536,
#             n_heads=12,
#             n_kv_heads=2,
#             ffn_dim=8960,
#             device_count=device_count,
#             data_type=data_type,
#             kv_partition_mode=kv_partition_mode,
#         )


# class TransformerBlockQwen25_1_5BAutoRegressionTP(
#     TransformerBlockQwen25AutoRegressionTP
# ):
#     def __init__(
#         self,
#         device_count,
#         data_type: DataType,
#         kv_partition_mode="replicate",
#     ):
#         super().__init__(
#             d_model=1536,
#             n_heads=12,
#             n_kv_heads=2,
#             ffn_dim=8960,
#             device_count=device_count,
#             data_type=data_type,
#             kv_partition_mode=kv_partition_mode,
#         )
class TransformerBlockBitNetInitComputationTP(Operator):
    """
    BitNet-b1.58 prefill / init computation block。

    当前版本针对：
        device_count = 1
        只统计 Linear 层的计算与访存开销

    BitNet-b1.58-2B-4T 结构：
        hidden_size          = 2560
        num_attention_heads  = 20
        num_key_value_heads  = 5
        head_dim             = 128
        intermediate_size    = 6912
        num_hidden_layers    = 30

    每层结构：
        RMSNorm
        -> q_proj / k_proj / v_proj
        -> GQA attention
        -> inner_attn_norm
        -> o_proj
        -> residual
        -> RMSNorm
        -> ReLU2(gate_proj) * up_proj
        -> ffn_inner_norm
        -> down_proj
        -> residual

    注意：
    1. RoPE、attention scale、causal mask 不单独统计。
    2. repeat_kv 不单独统计。
    3. residual add 不单独统计。
    4. gate * up 不单独统计。
    5. compile_and_simulate 只统计 Linear。
    6. sparsity_ratio 参数在你当前 Matmul 中实际表示 throughput speedup。
    """

    def __init__(
        self,
        d_model,
        n_heads,
        n_kv_heads,
        ffn_dim,
        device_count,
        data_type: DataType,
        weight_data_type: DataType = None,
        linear_sparsity_speedup=None,
    ):
        super().__init__(0, 0, 0, 0, data_type)

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.ffn_dim = ffn_dim
        self.device_count = device_count

        # 当前工程已经明确 dev_cnt=1。
        # BitNet 的 KV heads=5，多卡 TP 需要额外设计 KV head mapping。
        if device_count != 1:
            raise NotImplementedError(
                "This BitNet prefill block currently supports device_count=1 only. "
                "Multi-device GQA requires explicit KV-head-to-device mapping."
            )

        d = d_model
        h = n_heads
        h_kv = n_kv_heads
        f = ffn_dim
        dev_cnt = device_count

        assert d % h == 0
        assert h % h_kv == 0
        assert h % dev_cnt == 0
        assert d % dev_cnt == 0
        assert f % dev_cnt == 0

        self.head_dim = d // h
        self.q_heads_local = h
        self.kv_heads_local = h_kv

        self.q_dim_local = self.q_heads_local * self.head_dim
        self.kv_dim_local = self.kv_heads_local * self.head_dim

        assert self.q_dim_local == d
        assert self.kv_dim_local == h_kv * self.head_dim

        # ------------------------------------------------------------
        # Linear sparsity throughput speedup
        #
        # 这里的数值应当是：
        #     dense_latency / sparse_latency
        #
        # 而不是零比特比例或 utilization ratio。
        # ------------------------------------------------------------
        default_speedup = {
            "q_proj": 1.0,
            "k_proj": 1.0,
            "v_proj": 1.0,
            "o_proj": 1.0,
            "gate_proj": 1.0,
            "up_proj": 1.0,
            "down_proj": 1.0,
        }

        if linear_sparsity_speedup is not None:
            default_speedup.update(linear_sparsity_speedup)

        self.linear_sparsity_speedup = default_speedup

        # 如果框架支持单独的 2-bit weight DataType，可以传进来。
        # 否则继续使用 data_type，并在 CIM simulator 中单独处理
        # BitNet 的 2-bit weight storage / bandwidth。
        w_dtype = (
            weight_data_type
            if weight_data_type is not None
            else data_type
        )

        # ============================================================
        # Parameters
        # ============================================================

        # BitNet 使用 GQA：
        # Q: d -> d
        # K: d -> n_kv_heads * head_dim
        # V: d -> n_kv_heads * head_dim
        self.Wq = Tensor(
            [d, self.q_dim_local],
            w_dtype,
        )
        self.Wk = Tensor(
            [d, self.kv_dim_local],
            w_dtype,
        )
        self.Wv = Tensor(
            [d, self.kv_dim_local],
            w_dtype,
        )

        # Attention 输出经过 inner_attn_norm 后送入 o_proj。
        self.W0 = Tensor(
            [self.q_dim_local, d],
            w_dtype,
        )

        # BitNet MLP：
        # ReLU2(gate_proj(x)) * up_proj(x)
        # -> inner RMSNorm
        # -> down_proj
        self.W_gate = Tensor(
            [d, f],
            w_dtype,
        )
        self.W_up = Tensor(
            [d, f],
            w_dtype,
        )
        self.W_down = Tensor(
            [f, d],
            w_dtype,
        )

        # ============================================================
        # Normalization operators
        # ============================================================

        # BitNet 使用 subln/RMSNorm。
        # 如果工程中没有 RMSNorm，则暂时用 LayerNorm 近似。
        NormOp = globals().get("RMSNorm", LayerNorm)

        # Decoder layer 输入处
        self.input_layernorm = NormOp(data_type)

        # Attention output 在 o_proj 前
        self.inner_attn_norm = NormOp(data_type)

        # Attention residual 后、MLP 前
        self.post_attention_layernorm = NormOp(data_type)

        # gate * up 后、down_proj 前
        # 这个 norm 的最后一维是 ffn_dim。
        self.ffn_inner_norm = NormOp(data_type)

        # ============================================================
        # Attention operators
        # ============================================================

        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)

        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)

        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)

        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)

        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)

        self.H_matmul0 = Matmul(data_type)

        # ============================================================
        # MLP operators
        # ============================================================

        self.Gate_proj = Matmul(data_type)
        self.Up_proj = Matmul(data_type)

        # BitNet 使用 ReLU²。
        #
        # 如果工程中已经定义 ReLU2，则直接使用。
        # 如果只有 ReLU，则用 ReLU 近似。
        # 如果都没有，则使用 GeLU 保证 shape 建模可运行。
        #
        # 因为当前只统计 Linear，所以这里不会影响 Linear latency。
        if "ReLU2" in globals():
            self.H_act = ReLU2(data_type)
        elif "ReLU" in globals():
            self.H_act = ReLU(data_type)
        else:
            self.H_act = GeLU(data_type)

        self.Down_proj = Matmul(data_type)

    def _speedup(self, layer_name: str) -> float:
        speedup = float(
            self.linear_sparsity_speedup.get(
                layer_name,
                1.0,
            )
        )

        if speedup <= 0:
            raise ValueError(
                f"Invalid sparsity throughput speedup for "
                f"{layer_name}: {speedup}"
            )

        return speedup

    def __call__(self, X: Tensor) -> Tensor:
        """
        构建 BitNet prefill 计算图和各 Linear 的 M/N/K shape。

        当前只要求 shape 正确，不显式统计：
            residual add
            RoPE
            repeat_kv
            attention scale/mask
            gate * up
        """
        b, s, d = X.shape

        assert d == self.d_model

        h = self.n_heads
        h_kv = self.n_kv_heads
        d_h = self.head_dim
        f = self.ffn_dim

        # ============================================================
        # Attention
        # ============================================================

        residual = X

        X_attn = self.input_layernorm(X)
        assert X_attn.shape == [b, s, d]

        # ------------------------------------------------------------
        # Q/K/V projections
        # ------------------------------------------------------------

        Q = self.Q_proj(
            X_attn,
            self.Wq,
        )
        assert Q.shape == [
            b,
            s,
            d,
        ]

        K = self.K_proj(
            X_attn,
            self.Wk,
        )
        assert K.shape == [
            b,
            s,
            h_kv * d_h,
        ]

        V = self.V_proj(
            X_attn,
            self.Wv,
        )
        assert V.shape == [
            b,
            s,
            h_kv * d_h,
        ]

        # ------------------------------------------------------------
        # Reshape heads
        # ------------------------------------------------------------

        Q = self.Q_reshape(
            Q,
            [b, s, h, d_h],
        )
        assert Q.shape == [
            b,
            s,
            h,
            d_h,
        ]

        K = self.K_reshape(
            K,
            [b, s, h_kv, d_h],
        )
        assert K.shape == [
            b,
            s,
            h_kv,
            d_h,
        ]

        V = self.V_reshape(
            V,
            [b, s, h_kv, d_h],
        )
        assert V.shape == [
            b,
            s,
            h_kv,
            d_h,
        ]

        # ------------------------------------------------------------
        # Transpose
        # ------------------------------------------------------------

        Q_T = self.Q_transpose(
            Q,
            [0, 2, 1, 3],
        )
        assert Q_T.shape == [
            b,
            h,
            s,
            d_h,
        ]

        K_T_small = self.K_transpose(
            K,
            [0, 2, 3, 1],
        )
        assert K_T_small.shape == [
            b,
            h_kv,
            d_h,
            s,
        ]

        V_T_small = self.V_transpose(
            V,
            [0, 2, 1, 3],
        )
        assert V_T_small.shape == [
            b,
            h_kv,
            s,
            d_h,
        ]

        # ------------------------------------------------------------
        # GQA repeat_kv
        #
        # BitNet:
        #     20 Q heads
        #      5 KV heads
        # repeat factor = 4
        #
        # 这里只创建参与 attention 的逻辑 shape，
        # 不单独统计 repeat/broadcast。
        # ------------------------------------------------------------

        num_key_value_groups = h // h_kv
        assert num_key_value_groups == 4

        K_T = Tensor(
            [b, h, d_h, s],
            self.data_type,
        )
        V_T = Tensor(
            [b, h, s, d_h],
            self.data_type,
        )

        # ------------------------------------------------------------
        # Attention MatMul
        # ------------------------------------------------------------

        A = self.Q_mul_K(
            Q_T,
            K_T,
        )
        assert A.shape == [
            b,
            h,
            s,
            s,
        ]

        A_prob = self.A_softmax(A)
        assert A_prob.shape == [
            b,
            h,
            s,
            s,
        ]

        H = self.A_mul_V(
            A_prob,
            V_T,
        )
        assert H.shape == [
            b,
            h,
            s,
            d_h,
        ]

        H = self.H_transpose(
            H,
            [0, 2, 1, 3],
        )
        assert H.shape == [
            b,
            s,
            h,
            d_h,
        ]

        H = self.H_reshape(
            H,
            [b, s, d],
        )
        assert H.shape == [
            b,
            s,
            d,
        ]

        # BitNet 特有：
        # attention output 在 o_proj 前有 inner_attn_norm。
        H = self.inner_attn_norm(H)
        assert H.shape == [
            b,
            s,
            d,
        ]

        H0 = self.H_matmul0(
            H,
            self.W0,
        )
        assert H0.shape == [
            b,
            s,
            d,
        ]

        # residual + attention output
        # 当前不显式统计 Add。
        Y = H0
        assert Y.shape == residual.shape

        # ============================================================
        # MLP
        # ============================================================

        residual_ffn = Y

        Y_ffn = self.post_attention_layernorm(Y)
        assert Y_ffn.shape == [
            b,
            s,
            d,
        ]

        Gate = self.Gate_proj(
            Y_ffn,
            self.W_gate,
        )
        assert Gate.shape == [
            b,
            s,
            f,
        ]

        Up = self.Up_proj(
            Y_ffn,
            self.W_up,
        )
        assert Up.shape == [
            b,
            s,
            f,
        ]

        Gate_act = self.H_act(Gate)
        assert Gate_act.shape == [
            b,
            s,
            f,
        ]

        # BitNet:
        # H1 = ReLU2(Gate) * Up
        #
        # 当前框架没有 elementwise Mul operator，
        # Linear 已经分别执行 Gate_proj 和 Up_proj。
        H1 = Gate_act
        assert H1.shape == Up.shape

        # BitNet 特有：
        # gate*up 后、down_proj 前还有一次 RMSNorm。
        H1 = self.ffn_inner_norm(H1)
        assert H1.shape == [
            b,
            s,
            f,
        ]

        H2 = self.Down_proj(
            H1,
            self.W_down,
        )
        assert H2.shape == [
            b,
            s,
            d,
        ]

        # residual + MLP output
        Z = H2
        assert Z.shape == residual_ffn.shape

        return Z

    def roofline_model(self, system: System):
        """
        只统计七个 Linear 层。

        不包含：
            QK
            AV
            Softmax
            RMSNorm
            ReLU2
            elementwise Mul
            residual
        """
        device = system.device
        matmul_overhead = device.compute_module.overhead.matmul

        q_latency = (
            self.Q_proj.roofline_model(device)
            + matmul_overhead
        )

        k_latency = (
            self.K_proj.roofline_model(device)
            + matmul_overhead
        )

        v_latency = (
            self.V_proj.roofline_model(device)
            + matmul_overhead
        )

        o_latency = (
            self.H_matmul0.roofline_model(device)
            + matmul_overhead
        )

        gate_latency = (
            self.Gate_proj.roofline_model(device)
            + matmul_overhead
        )

        up_latency = (
            self.Up_proj.roofline_model(device)
            + matmul_overhead
        )

        down_latency = (
            self.Down_proj.roofline_model(device)
            + matmul_overhead
        )

        self.roofline_latency = (
            q_latency
            + k_latency
            + v_latency
            + o_latency
            + gate_latency
            + up_latency
            + down_latency
        )

        self.roofline_log = (
            f"{q_latency}, "
            f"{k_latency}, "
            f"{v_latency}, "
            f"{o_latency}, "
            f"{gate_latency}, "
            f"{up_latency}, "
            f"{down_latency}"
        )

        print("BitNet prefill Linear roofline breakdown:")
        print(f"q_proj    : {q_latency}")
        print(f"k_proj    : {k_latency}")
        print(f"v_proj    : {v_latency}")
        print(f"o_proj    : {o_latency}")
        print(f"gate_proj : {gate_latency}")
        print(f"up_proj   : {up_latency}")
        print(f"down_proj : {down_latency}")
        print(f"total     : {self.roofline_latency}")

        return self.roofline_latency

    def compile_and_simulate(
        self,
        system: System,
        compile_mode: str,
        mapping_save_path: str = None,
    ):
        """
        只模拟七个 Linear 层。

        注意：
        当前 Matmul.compile_and_simulate() 中的 sparsity_ratio
        实际上传入的是吞吐加速倍数：
            dense_latency / sparse_latency
        """
        device = system.device
        matmul_overhead = device.compute_module.overhead.matmul

        print("simulating BitNet q_proj")
        q_latency = (
            self.Q_proj.compile_and_simulate(
                device,
                compile_mode,
                mapping_save_path,
                "Q_proj",
                sparsity_ratio=1.785560543296901,
            )
            + matmul_overhead
        )

        print("simulating BitNet k_proj")
        k_latency = (
            self.K_proj.compile_and_simulate(
                device,
                compile_mode,
                mapping_save_path,
                "K_proj",
                sparsity_ratio=1.785560543296901,
            )
            + matmul_overhead
        )

        print("simulating BitNet v_proj")
        v_latency = (
            self.V_proj.compile_and_simulate(
                device,
                compile_mode,
                mapping_save_path,
                "V_proj",
                sparsity_ratio=1.785560543296901,
            )
            + matmul_overhead
        )

        print("simulating BitNet o_proj")
        o_latency = (
            self.H_matmul0.compile_and_simulate(
                device,
                compile_mode,
                mapping_save_path,
                "H_matmul0",
                sparsity_ratio=2.5834651293703605,
            )
            + matmul_overhead
        )

        print("simulating BitNet gate_proj")
        gate_latency = (
            self.Gate_proj.compile_and_simulate(
                device,
                compile_mode,
                mapping_save_path,
                "Gate_proj",
                sparsity_ratio=1.7857765686582259,
            )
            + matmul_overhead
        )

        print("simulating BitNet up_proj")
        up_latency = (
            self.Up_proj.compile_and_simulate(
                device,
                compile_mode,
                mapping_save_path,
                "Up_proj",
                sparsity_ratio=1.7857765686582259,
            )
            + matmul_overhead
        )

        print("simulating BitNet down_proj")
        down_latency = (
            self.Down_proj.compile_and_simulate(
                device,
                compile_mode,
                mapping_save_path,
                "Down_proj",
                sparsity_ratio=2.911975487783497,
            )
            + matmul_overhead
        )

        self.latency = (
            q_latency
            + k_latency
            + v_latency
            + o_latency
            + gate_latency
            + up_latency
            + down_latency
        )

        self.simluate_log = (
            f"{q_latency}, "
            f"{k_latency}, "
            f"{v_latency}, "
            f"{o_latency}, "
            f"{gate_latency}, "
            f"{up_latency}, "
            f"{down_latency}"
        )

        print("BitNet prefill Linear simulation breakdown:")
        print(f"q_proj    : {q_latency * 1e3:.6f} ms")
        print(f"k_proj    : {k_latency * 1e3:.6f} ms")
        print(f"v_proj    : {v_latency * 1e3:.6f} ms")
        print(f"o_proj    : {o_latency * 1e3:.6f} ms")
        print(f"gate_proj : {gate_latency * 1e3:.6f} ms")
        print(f"up_proj   : {up_latency * 1e3:.6f} ms")
        print(f"down_proj : {down_latency * 1e3:.6f} ms")
        print(f"total     : {self.latency * 1e3:.6f} ms")

        return self.latency

    def run_on_gpu(self):
        """
        GPU 路径同样只统计七个 Linear。
        """
        q_latency = self.Q_proj.run_on_gpu()
        k_latency = self.K_proj.run_on_gpu()
        v_latency = self.V_proj.run_on_gpu()
        o_latency = self.H_matmul0.run_on_gpu()
        gate_latency = self.Gate_proj.run_on_gpu()
        up_latency = self.Up_proj.run_on_gpu()
        down_latency = self.Down_proj.run_on_gpu()

        self.latency_on_gpu = (
            q_latency
            + k_latency
            + v_latency
            + o_latency
            + gate_latency
            + up_latency
            + down_latency
        )

        print("BitNet prefill GPU Linear breakdown:")
        print(f"q_proj    : {q_latency}")
        print(f"k_proj    : {k_latency}")
        print(f"v_proj    : {v_latency}")
        print(f"o_proj    : {o_latency}")
        print(f"gate_proj : {gate_latency}")
        print(f"up_proj   : {up_latency}")
        print(f"down_proj : {down_latency}")
        print(f"total     : {self.latency_on_gpu}")

        return self.latency_on_gpu


class TransformerBlockBitNetAutoRegressionTP(Operator):
    """
    BitNet-b1.58 decode / autoregressive block。

    当前版本针对：
        device_count = 1
        只统计 Linear 层的计算与访存开销

    BitNet-b1.58-2B-4T 结构：
        hidden_size          = 2560
        num_attention_heads  = 20
        num_key_value_heads  = 5
        head_dim             = 128
        intermediate_size    = 6912
        num_hidden_layers    = 30

    每层结构：
        RMSNorm
        -> q_proj / k_proj / v_proj
        -> GQA attention (KV cache)
        -> inner_attn_norm
        -> o_proj
        -> residual
        -> RMSNorm
        -> ReLU2(gate_proj) * up_proj
        -> ffn_inner_norm
        -> down_proj
        -> residual

    注意：
    1. RoPE、attention scale、causal mask 不单独统计。
    2. repeat_kv 不单独统计。
    3. residual add 不单独统计。
    4. gate * up 不单独统计。
    5. compile_and_simulate 只统计 Linear。
    6. sparsity_ratio 参数在当前 Matmul 中实际表示 throughput speedup。
    """

    def __init__(
        self,
        d_model,
        n_heads,
        n_kv_heads,
        ffn_dim,
        device_count,
        data_type: DataType,
        weight_data_type: DataType = None,
        linear_sparsity_speedup=None,
    ):
        super().__init__(0, 0, 0, 0, data_type)

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.ffn_dim = ffn_dim
        self.device_count = device_count

        # 当前工程已经明确 dev_cnt=1。
        # BitNet 的 KV heads=5，多卡 TP 需要额外设计 KV head mapping。
        if device_count != 1:
            raise NotImplementedError(
                "This BitNet decode block currently supports device_count=1 only. "
                "Multi-device GQA requires explicit KV-head-to-device mapping."
            )

        d = d_model
        h = n_heads
        h_kv = n_kv_heads
        f = ffn_dim
        dev_cnt = device_count

        assert d % h == 0
        assert h % h_kv == 0
        assert h % dev_cnt == 0
        assert d % dev_cnt == 0
        assert f % dev_cnt == 0

        self.head_dim = d // h
        self.q_heads_local = h
        self.kv_heads_local = h_kv

        self.q_dim_local = self.q_heads_local * self.head_dim
        self.kv_dim_local = self.kv_heads_local * self.head_dim

        assert self.q_dim_local == d
        assert self.kv_dim_local == h_kv * self.head_dim

        # ------------------------------------------------------------
        # Linear sparsity throughput speedup
        # ------------------------------------------------------------
        default_speedup = {
            "q_proj": 1.0,
            "k_proj": 1.0,
            "v_proj": 1.0,
            "o_proj": 1.0,
            "gate_proj": 1.0,
            "up_proj": 1.0,
            "down_proj": 1.0,
        }

        if linear_sparsity_speedup is not None:
            default_speedup.update(linear_sparsity_speedup)

        self.linear_sparsity_speedup = default_speedup

        w_dtype = (
            weight_data_type
            if weight_data_type is not None
            else data_type
        )

        # ============================================================
        # Parameters
        # ============================================================

        # Q: d -> d
        self.Wq = Tensor([d, self.q_dim_local], w_dtype)
        # K: d -> n_kv_heads * head_dim
        self.Wk = Tensor([d, self.kv_dim_local], w_dtype)
        # V: d -> n_kv_heads * head_dim
        self.Wv = Tensor([d, self.kv_dim_local], w_dtype)
        # o_proj: d -> d
        self.W0 = Tensor([self.q_dim_local, d], w_dtype)

        # SwiGLU-like MLP:
        # ReLU2(gate_proj(x)) * up_proj(x)
        # -> ffn_inner_norm
        # -> down_proj
        self.W_gate = Tensor([d, f], w_dtype)
        self.W_up = Tensor([d, f], w_dtype)
        self.W_down = Tensor([f, d], w_dtype)

        # ============================================================
        # Normalization operators
        # ============================================================

        NormOp = globals().get("RMSNorm", LayerNorm)

        self.input_layernorm = NormOp(data_type)
        self.inner_attn_norm = NormOp(data_type)
        self.post_attention_layernorm = NormOp(data_type)
        self.ffn_inner_norm = NormOp(data_type)

        # ============================================================
        # Attention operators
        # ============================================================

        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)

        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)

        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)

        # decode 阶段需要把新 token 的 k/v 拼接到 KV cache
        self.K_concat = Concat(data_type)
        self.V_concat = Concat(data_type)

        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)

        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)

        self.H_matmul0 = Matmul(data_type)

        # ============================================================
        # MLP operators
        # ============================================================

        self.Gate_proj = Matmul(data_type)
        self.Up_proj = Matmul(data_type)

        if "ReLU2" in globals():
            self.H_act = ReLU2(data_type)
        elif "ReLU" in globals():
            self.H_act = ReLU(data_type)
        else:
            self.H_act = GeLU(data_type)

        self.Down_proj = Matmul(data_type)

    def _speedup(self, layer_name: str) -> float:
        speedup = float(
            self.linear_sparsity_speedup.get(
                layer_name,
                1.0,
            )
        )

        if speedup <= 0:
            raise ValueError(
                f"Invalid sparsity throughput speedup for "
                f"{layer_name}: {speedup}"
            )

        return speedup

    def __call__(self, x: Tensor, seq_len: int) -> Tensor:
        """
        构建 BitNet decode 计算图和各 Linear 的 M/N/K shape。

        输入:
            x: [b, 1, d]  (当前 token)
            seq_len: KV cache 已有长度 s

        输出:
            z: [b, 1, d]

        不显式统计：
            residual add
            RoPE
            repeat_kv
            attention scale/mask
            gate * up
        """
        b, one, d = x.shape
        assert one == 1
        assert d == self.d_model

        s = seq_len
        h = self.n_heads
        h_kv = self.n_kv_heads
        d_h = self.head_dim
        f = self.ffn_dim

        # ============================================================
        # KV cache (GQA: 使用 kv_heads_local 而不是 q_heads_local)
        # ============================================================

        K_cache_small = Tensor([b, self.kv_heads_local, d_h, s], self.data_type)
        V_cache_small = Tensor([b, self.kv_heads_local, s, d_h], self.data_type)

        # ============================================================
        # Attention block
        # ============================================================

        x_attn = self.input_layernorm(x)
        assert x_attn.shape == [b, 1, d]

        # Q/K/V projections
        q = self.Q_proj(x_attn, self.Wq)
        assert q.shape == [b, 1, self.q_dim_local]

        k = self.K_proj(x_attn, self.Wk)
        assert k.shape == [b, 1, self.kv_dim_local]

        v = self.V_proj(x_attn, self.Wv)
        assert v.shape == [b, 1, self.kv_dim_local]

        # Reshape heads
        q = self.Q_reshape(q, [b, 1, self.q_heads_local, d_h])
        assert q.shape == [b, 1, self.q_heads_local, d_h]

        k = self.K_reshape(k, [b, 1, self.kv_heads_local, d_h])
        assert k.shape == [b, 1, self.kv_heads_local, d_h]

        v = self.V_reshape(v, [b, 1, self.kv_heads_local, d_h])
        assert v.shape == [b, 1, self.kv_heads_local, d_h]

        # Transpose
        q_T = self.Q_transpose(q, [0, 2, 1, 3])
        assert q_T.shape == [b, self.q_heads_local, 1, d_h]

        k_T_small = self.K_transpose(k, [0, 2, 3, 1])
        assert k_T_small.shape == [b, self.kv_heads_local, d_h, 1]

        v_T_small = self.V_transpose(v, [0, 2, 1, 3])
        assert v_T_small.shape == [b, self.kv_heads_local, 1, d_h]

        # Concat new KV to cache
        K_cat_small = self.K_concat(K_cache_small, k_T_small, 3)
        assert K_cat_small.shape == [b, self.kv_heads_local, d_h, s + 1]

        V_cat_small = self.V_concat(V_cache_small, v_T_small, 2)
        assert V_cat_small.shape == [b, self.kv_heads_local, s + 1, d_h]

        # GQA repeat_kv：不单独统计 repeat 开销。
        # BitNet: 20 Q heads, 5 KV heads, repeat factor = 4
        K_T = Tensor([b, self.q_heads_local, d_h, s + 1], self.data_type)
        V_T = Tensor([b, self.q_heads_local, s + 1, d_h], self.data_type)

        # Attention MatMul
        a = self.Q_mul_K(q_T, K_T)
        assert a.shape == [b, self.q_heads_local, 1, s + 1]

        a_prob = self.A_softmax(a)
        assert a_prob.shape == [b, self.q_heads_local, 1, s + 1]

        h0 = self.A_mul_V(a_prob, V_T)
        assert h0.shape == [b, self.q_heads_local, 1, d_h]

        h0 = self.H_transpose(h0, [0, 2, 1, 3])
        assert h0.shape == [b, 1, self.q_heads_local, d_h]

        h0 = self.H_reshape(h0, [b, 1, self.q_dim_local])
        assert h0.shape == [b, 1, self.q_dim_local]

        # BitNet 特有：attention output 在 o_proj 前有 inner_attn_norm。
        h0 = self.inner_attn_norm(h0)
        assert h0.shape == [b, 1, self.q_dim_local]

        h0 = self.H_matmul0(h0, self.W0)
        assert h0.shape == [b, 1, d]

        # residual add: y = x + h0
        y = h0
        assert y.shape == [b, 1, d]

        # ============================================================
        # MLP block
        # ============================================================

        y_ffn = self.post_attention_layernorm(y)
        assert y_ffn.shape == [b, 1, d]

        gate = self.Gate_proj(y_ffn, self.W_gate)
        assert gate.shape == [b, 1, f]

        up = self.Up_proj(y_ffn, self.W_up)
        assert up.shape == [b, 1, f]

        gate_act = self.H_act(gate)
        assert gate_act.shape == [b, 1, f]

        # BitNet: H1 = ReLU2(gate) * up
        # 不单独统计 elementwise mul。
        h1 = gate_act
        assert h1.shape == [b, 1, f]

        # BitNet 特有：gate*up 后、down_proj 前还有一次 RMSNorm。
        h1 = self.ffn_inner_norm(h1)
        assert h1.shape == [b, 1, f]

        h2 = self.Down_proj(h1, self.W_down)
        assert h2.shape == [b, 1, d]

        # residual add: z = y + h2
        z = h2
        assert z.shape == [b, 1, d]

        # memory requirement:
        # 权重 + KV cache。
        self.memory_requirement = (
            self.Wq.size * self.Wq.data_type.word_size
            + self.Wk.size * self.Wk.data_type.word_size
            + self.Wv.size * self.Wv.data_type.word_size
            + self.W0.size * self.W0.data_type.word_size
            + self.W_gate.size * self.W_gate.data_type.word_size
            + self.W_up.size * self.W_up.data_type.word_size
            + self.W_down.size * self.W_down.data_type.word_size
            + K_cache_small.size * K_cache_small.data_type.word_size
            + V_cache_small.size * V_cache_small.data_type.word_size
        )

        return z

    def roofline_model(self, system: System):
        """
        只统计七个 Linear 层。
        """
        device = system.device
        matmul_overhead = device.compute_module.overhead.matmul

        q_latency = (
            self.Q_proj.roofline_model(device)
            + matmul_overhead
        )
        k_latency = (
            self.K_proj.roofline_model(device)
            + matmul_overhead
        )
        v_latency = (
            self.V_proj.roofline_model(device)
            + matmul_overhead
        )
        qkv_latency = q_latency + k_latency + v_latency

        q_mul_k_latency = (
            self.Q_mul_K.roofline_model(device)
            + matmul_overhead
        )
        a_mul_v_latency = (
            self.A_mul_V.roofline_model(device)
            + matmul_overhead
        )
        o_latency = (
            self.H_matmul0.roofline_model(device)
            + matmul_overhead
        )

        gate_latency = (
            self.Gate_proj.roofline_model(device)
            + matmul_overhead
        )
        up_latency = (
            self.Up_proj.roofline_model(device)
            + matmul_overhead
        )
        gate_up_latency = gate_latency + up_latency

        down_latency = (
            self.Down_proj.roofline_model(device)
            + matmul_overhead
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + o_latency
            + gate_up_latency
            + down_latency
        )

        softmax_latency = (
            self.A_softmax.roofline_model(device)
            + device.compute_module.overhead.softmax
        )

        norm_latency = (
            self.input_layernorm.roofline_model(device)
            + self.inner_attn_norm.roofline_model(device)
            + self.post_attention_layernorm.roofline_model(device)
            + self.ffn_inner_norm.roofline_model(device)
            + 4 * device.compute_module.overhead.layernorm
        )

        normalization_total_latency = softmax_latency + norm_latency

        activation_latency = (
            self.H_act.roofline_model(device)
            + device.compute_module.overhead.gelu
        )

        self.roofline_latency = (
            matmul_total_latency
            + normalization_total_latency
            + activation_latency
        )

        self.roofline_log = (
            f"{q_latency}, {k_latency}, {v_latency}, "
            f"{q_mul_k_latency}, {a_mul_v_latency}, {o_latency}, "
            f"{gate_latency}, {up_latency}, {down_latency}, "
            f"{softmax_latency}, {norm_latency}, {activation_latency}"
        )

        print("BitNet decode Linear roofline breakdown:")
        print(f"q_proj    : {q_latency}")
        print(f"k_proj    : {k_latency}")
        print(f"v_proj    : {v_latency}")
        print(f"o_proj    : {o_latency}")
        print(f"gate_proj : {gate_latency}")
        print(f"up_proj   : {up_latency}")
        print(f"down_proj : {down_latency}")
        print(f"total     : {self.roofline_latency}")

        return self.roofline_latency

    def compile_and_simulate(
        self,
        system: System,
        compile_mode: str,
        mapping_save_path: str = None,
    ):
        """
        只模拟七个 Linear 层。

        注意：
        当前 Matmul.compile_and_simulate() 中的 sparsity_ratio
        实际上传入的是吞吐加速倍数：
            dense_latency / sparse_latency
        """
        device = system.device
        matmul_overhead = device.compute_module.overhead.matmul

        print("simulating BitNet decode q_proj")
        q_latency = (
            self.Q_proj.compile_and_simulate(
                device,
                compile_mode,
                mapping_save_path,
                "Q_proj",
                sparsity_ratio=1.894113778362,
            )
            + matmul_overhead
        )

        print("simulating BitNet decode k_proj")
        k_latency = (
            self.K_proj.compile_and_simulate(
                device,
                compile_mode,
                mapping_save_path,
                "K_proj",
                sparsity_ratio=1.894113778362,
            )
            + matmul_overhead
        )

        print("simulating BitNet decode v_proj")
        v_latency = (
            self.V_proj.compile_and_simulate(
                device,
                compile_mode,
                mapping_save_path,
                "V_proj",
                sparsity_ratio=1.894113778362,
            )
            + matmul_overhead
        )

        qkv_latency = q_latency + k_latency + v_latency

        print("simulating BitNet decode q_mul_k")
        q_mul_k_latency = (
            self.Q_mul_K.compile_and_simulate(
                device,
                compile_mode,
                mapping_save_path,
                "Q_mul_K",
                sparsity_ratio=7.668752496339,
            )
            + matmul_overhead
        )

        print("simulating BitNet decode a_mul_v")
        a_mul_v_latency = (
            self.A_mul_V.compile_and_simulate(
                device,
                compile_mode,
                mapping_save_path,
                "A_mul_V",
                sparsity_ratio=20.674802584350,
            )
            + matmul_overhead
        )

        print("simulating BitNet decode o_proj")
        o_latency = (
            self.H_matmul0.compile_and_simulate(
                device,
                compile_mode,
                mapping_save_path,
                "H_matmul0",
                sparsity_ratio=1.660420870568,
            )
            + matmul_overhead
        )

        print("simulating BitNet decode gate_proj")
        gate_latency = (
            self.Gate_proj.compile_and_simulate(
                device,
                compile_mode,
                mapping_save_path,
                "Gate_proj",
                sparsity_ratio=1.686841077704,
            )
            + matmul_overhead
        )

        print("simulating BitNet decode up_proj")
        up_latency = (
            self.Up_proj.compile_and_simulate(
                device,
                compile_mode,
                mapping_save_path,
                "Up_proj",
                sparsity_ratio=1.686841077704,
            )
            + matmul_overhead
        )

        gate_up_latency = gate_latency + up_latency

        print("simulating BitNet decode down_proj")
        down_latency = (
            self.Down_proj.compile_and_simulate(
                device,
                compile_mode,
                mapping_save_path,
                "Down_proj",
                sparsity_ratio=2.682813227760,
            )
            + matmul_overhead
        )

        matmul_total_latency = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + o_latency
            + gate_up_latency
            + down_latency
        )

        # normalization / softmax / activation: 当前不统计
        normalization_total_latency = 0
        activation_latency = 0

        self.latency = (
            matmul_total_latency
            + normalization_total_latency
            + activation_latency
        )

        self.simluate_log = (
            f"{q_latency}, {k_latency}, {v_latency}, "
            f"{q_mul_k_latency}, {a_mul_v_latency}, {o_latency}, "
            f"{gate_latency}, {up_latency}, {down_latency}"
        )

        print("BitNet decode Linear simulation breakdown:")
        print(f"q_proj    : {q_latency * 1e3:.6f} ms")
        print(f"k_proj    : {k_latency * 1e3:.6f} ms")
        print(f"v_proj    : {v_latency * 1e3:.6f} ms")
        print(f"q_mul_k   : {q_mul_k_latency * 1e3:.6f} ms")
        print(f"a_mul_v   : {a_mul_v_latency * 1e3:.6f} ms")
        print(f"o_proj    : {o_latency * 1e3:.6f} ms")
        print(f"gate_proj : {gate_latency * 1e3:.6f} ms")
        print(f"up_proj   : {up_latency * 1e3:.6f} ms")
        print(f"down_proj : {down_latency * 1e3:.6f} ms")
        print(f"total     : {self.latency * 1e3:.6f} ms")

        return self.latency

    def run_on_gpu(self):
        """
        GPU 路径同样只统计七个 Linear。
        """
        q_latency = self.Q_proj.run_on_gpu()
        k_latency = self.K_proj.run_on_gpu()
        v_latency = self.V_proj.run_on_gpu()
        qkv_latency = q_latency + k_latency + v_latency

        q_mul_k_latency = self.Q_mul_K.run_on_gpu()
        a_mul_v_latency = self.A_mul_V.run_on_gpu()

        o_latency = self.H_matmul0.run_on_gpu()

        gate_latency = self.Gate_proj.run_on_gpu()
        up_latency = self.Up_proj.run_on_gpu()
        gate_up_latency = gate_latency + up_latency

        down_latency = self.Down_proj.run_on_gpu()

        self.latency_on_gpu = (
            qkv_latency
            + q_mul_k_latency
            + a_mul_v_latency
            + o_latency
            + gate_up_latency
            + down_latency
        )

        print("BitNet decode GPU Linear breakdown:")
        print(f"q_proj    : {q_latency}")
        print(f"k_proj    : {k_latency}")
        print(f"v_proj    : {v_latency}")
        print(f"q_mul_k   : {q_mul_k_latency}")
        print(f"a_mul_v   : {a_mul_v_latency}")
        print(f"o_proj    : {o_latency}")
        print(f"gate_proj : {gate_latency}")
        print(f"up_proj   : {up_latency}")
        print(f"down_proj : {down_latency}")
        print(f"total     : {self.latency_on_gpu}")

        return self.latency_on_gpu
