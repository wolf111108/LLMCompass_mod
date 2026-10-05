"""Figure-10 scope wrappers; preserve the existing GEMM-only Qwen entry points."""
from software_model.transformer import (
    TransformerBlockQwen25InitComputationTP,
    TransformerBlockQwen25AutoRegressionTP,
)


class QwenFigure10Prefill(TransformerBlockQwen25InitComputationTP):
    def compile_and_simulate(self, system, compile_mode="heuristic-CIM"):
        return super().compile_and_simulate(system, compile_mode, include_attention=True)


class QwenFigure10Decode(TransformerBlockQwen25AutoRegressionTP):
    def compile_and_simulate(self, system, compile_mode="heuristic-CIM-decode"):
        base = super().compile_and_simulate(system, compile_mode)
        device = system.device
        overhead = device.compute_module.overhead
        # Matrix mapping modes do not apply to vector operators.
        vector_mode = "heuristic-CIM"
        self.vector_breakdown = {
            "softmax": self.A_softmax.compile_and_simulate(device, vector_mode) + overhead.softmax,
            "norm_attn": self.rms_norm_attn.compile_and_simulate(device, vector_mode) + overhead.layernorm,
            "norm_ffn": self.rms_norm_ffn.compile_and_simulate(device, vector_mode) + overhead.layernorm,
            "activation": self.H_act.compile_and_simulate(device, vector_mode) + self._activation_overhead(device),
        }
        self.latency = base + sum(self.vector_breakdown.values())
        # Keep the original 12-column breakdown contract, now with nonzero vectors.
        columns = [float(x) for x in self.simluate_log.split(",")]
        columns[6:10] = self.vector_breakdown.values()
        self.simluate_log = ",".join(map(str, columns))
        return self.latency
