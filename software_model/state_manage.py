import json
import os
import copy


class MappingProfiler:
    def __init__(self, layer_shape: dict):
        self.layer_shape = layer_shape
        self.current_record = None
        self.best_record = None
        self.min_cycle_count = float("inf")
        self.layer_name = None

    def start_new_mapping(self, mapping):
        self.current_record = {
            "mapping": mapping,

            # DRAM / HBM <-> L2 / global buffer
            "dram_bytes": {"read": 0, "write": 0},
            "dram_latency_cycles": 0,

            # L2 / global buffer <-> L1 / local buffer
            "l2_to_l1_latency_cycles": 0,
            "l2_to_l1_bytes": {"read": 0, "write": 0},

            # L1/local buffer 内部供给 core 后的计算时间
            # 注意：这里不是 L1->core 传输 latency，而是 core compute latency
            "compute_latency_cycles": 0,

            "other_stats": {},
            "total_latency": float("inf"),
        }

    def record_total_latency(self, cycle_count: float):
        self.current_record["total_latency"] = cycle_count

    def record_dram_bytes(self, read_bytes: int, write_bytes: int):
        self.current_record["dram_bytes"]["read"] = int(read_bytes)
        self.current_record["dram_bytes"]["write"] = int(write_bytes)

    def record_l2_l1_bytes(self, read_bytes: int, write_bytes: int):
        self.current_record["l2_to_l1_bytes"]["read"] = int(read_bytes)
        self.current_record["l2_to_l1_bytes"]["write"] = int(write_bytes)

    def record_dram_latency(self, cycle_count: float):
        self.current_record["dram_latency_cycles"] = cycle_count

    def record_l2_to_l1_latency(self, cycle_count: float):
        self.current_record["l2_to_l1_latency_cycles"] = cycle_count

    def record_compute_latency(self, cycle_count: float):
        self.current_record["compute_latency_cycles"] = cycle_count

    def record_other_stat(self, key: str, value):
        self.current_record["other_stats"][key] = value

    def evaluate_current(self):
        if self.current_record is None:
            return

        total_latency = self.current_record["total_latency"]

        if total_latency < self.min_cycle_count:
            self.min_cycle_count = total_latency
            self.best_record = copy.deepcopy(self.current_record)

    def get_best_record(self):
        return self.best_record

    def dump_best_to_file(self, filepath: str, extra_info: dict = None):
        if self.best_record is None:
            print(f"Warning: No valid mapping found for {self.layer_name}.")
            return

        mapping_obj = self.best_record["mapping"]
        try:
            mapping_details = mapping_obj.__dict__
        except AttributeError:
            mapping_details = str(mapping_obj)

        output_data = {
            "layer_name": self.layer_name,
            "layer_shape": self.layer_shape,
            "best_cycle_count": self.best_record["total_latency"],
            "dram_bytes": self.best_record["dram_bytes"],
            "dram_latency_cycles": self.best_record["dram_latency_cycles"],
            "l2_to_l1_bytes": self.best_record["l2_to_l1_bytes"],
            "l2_to_l1_latency_cycles": self.best_record["l2_to_l1_latency_cycles"],
            "compute_latency_cycles": self.best_record["compute_latency_cycles"],
            "other_stats": self.best_record["other_stats"],
            "mapping_details": mapping_details,
        }

        if extra_info:
            output_data.update(extra_info)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)