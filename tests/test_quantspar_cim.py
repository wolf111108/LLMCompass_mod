"""Golden layout/cycle checks against quantspar ca07eb0 and system integration."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from software_model.quantspar_cim import compute_optimal_macro_layout_prefill, load_speedup_manifest
from software_model.matmul import Matmul, compute_optimal_macro_layout_decode
from software_model.utils import Tensor, data_type_dict
from ae.figure5.ijkl.test_transformer import build_cim_system


class QuantsparCIMTests(unittest.TestCase):
    def run_op(self, m, n, k, phase='prefill', speed=1, system=None):
        system = system or build_cim_system(64,48,16,16)
        op = Matmul(data_type_dict['int8'])
        op(Tensor([m,k],op.data_type),Tensor([k,n],op.data_type))
        mode = 'heuristic-CIM' if phase == 'prefill' else 'heuristic-CIM-GQA-decode'
        with contextlib.redirect_stdout(io.StringIO()):
            op.compile_and_simulate(system.device,mode,layer_name='Q_proj',sparsity_ratio=speed)
        return op.profiler.get_best_record()

    def test_source_golden_layouts(self):
        self.assertEqual(compute_optimal_macro_layout_prefill(16,4096,128,4096),(4,4,1,1,1024,3))
        self.assertEqual(compute_optimal_macro_layout_prefill(16,5120,5120,4096),(1,16,1,5,256,107))
        self.assertEqual(compute_optimal_macro_layout_decode(16,5120,5120,1),(1,1,16,5,1,7))
        self.assertEqual(compute_optimal_macro_layout_decode(16,8192,128,5),(5,1,3,2,5,1))

    def test_small_k_source_compatibility_and_corrected_contract(self):
        # Golden extracted Mapping_stat_dynamic result: all three bits active,
        # K=128 -> h_eff=8, but published dense denominator still uses 64.
        rec=self.run_op(16,4096,128,speed=8)
        self.assertEqual(rec['other_stats']['dense_effective_bit_steps'],16512)
        self.assertEqual(rec['compute_latency_cycles'],2064)
        system=build_cim_system(64,48,16,16)
        system.device.compute_module.core.cim_macro.quantspar_baseline='effective'
        fixed=self.run_op(16,4096,128,speed=1,system=system)
        self.assertEqual(fixed['compute_latency_cycles'],2064)
        self.assertEqual(fixed['other_stats']['baseline_policy'],'effective')

    def test_operator_epoch_and_storage_independent_of_speed(self):
        a=self.run_op(4096,128,4096)
        b=self.run_op(4096,128,4096,speed=2)
        self.assertEqual(a['compute_latency_cycles'],589824)
        self.assertEqual(a['compute_latency_cycles'],2*b['compute_latency_cycles'])
        self.assertEqual(a['dram_bytes'],b['dram_bytes'])
        self.assertEqual(a['l2_to_l1_bytes'],b['l2_to_l1_bytes'])
        st=a['other_stats']
        self.assertEqual((st['K_factor'],st['M_factor'],st['N_factor']),(4,4,1))
        self.assertEqual(st['local_weight_write_bytes'],4*4096*128)
        self.assertEqual(st['useful_ops'],2*4096*128*4096)
        self.assertEqual(a['total_latency']-b['total_latency'],
                         a['compute_latency_cycles']-b['compute_latency_cycles'])

    def test_decode_shared_rows_and_explicit_bit_calibration(self):
        system=build_cim_system(64,48,16,16)
        a=self.run_op(5,128,8192,'decode',system=system)
        st=a['other_stats']
        self.assertEqual((st['K_factor'],st['M_factor'],st['N_factor']),(5,1,3))
        self.assertEqual(st['weight_write_bytes'],8192*128)
        macro=system.device.compute_module.core.cim_macro
        macro.quantspar_decode_dense_bits=7
        macro.quantspar_cycles_per_effective_bit=3
        b=self.run_op(5,128,8192,'decode',system=system)
        self.assertEqual(b['compute_latency_cycles'],7*a['compute_latency_cycles'])

    def test_invalid_calibration_and_manifest_geometry(self):
        for speed in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError): self.run_op(16,48,1024,speed=speed)
        macro=build_cim_system(64,48,16,16).device.compute_module.core.cim_macro
        doc=dict(geometry=dict(height=64,width=48,banks=16,macros=16),baseline='source',
                 dense_bits=dict(prefill=3.8,decode=3),cycles_per_effective_bit=1,
                 source_commit='ca07eb0',workload={'model':'test'},
                 speedups=dict(prefill={'Q_proj':2},decode={'Q_proj':4}))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'stats.json';path.write_text(json.dumps(doc))
            load_speedup_manifest(macro,16,path)
            self.assertEqual(macro.quantspar_prefill_dense_bits,3.8)
            with self.assertRaisesRegex(ValueError,'workload'):
                load_speedup_manifest(macro,16,path,expected_workload={'model':'different'})
            system=build_cim_system(64,48,16,16)
            system.device.compute_module.core.cim_macro=macro
            a=self.run_op(16,48,1024,system=system,speed=99)
            self.assertEqual(a['other_stats']['effective_speedup'],2)
            with self.assertRaisesRegex(ValueError,'missing quantspar speedup'):
                macro.quantspar_speedups['prefill']={}
                self.run_op(16,48,1024,system=system)
            doc['geometry']['macros']=32;path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,'geometry'): load_speedup_manifest(macro,16,path)

if __name__=='__main__': unittest.main()
