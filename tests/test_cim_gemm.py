"""Analytical CIM regressions. Run: python -m unittest discover -s tests -p test_cim_gemm.py"""
import contextlib
import io
import unittest
from math import ceil
from types import SimpleNamespace

from software_model.matmul import Matmul, BatchedMatmul
from software_model.utils import Tensor, data_type_dict
from ae.figure5.ijkl.test_transformer import build_cim_system, collect_matmul_profiler_stats

DTYPE = data_type_dict['int8']


def simulate(m, n, k, *, system=None, mode='heuristic-CIM-decode', speed=1):
    system = system or build_cim_system(64, 48, 16, 16)
    system.device.compute_module.core.cim_macro.cim_backend = "legacy"
    op = Matmul(DTYPE)
    op(Tensor([m, k], DTYPE), Tensor([k, n], DTYPE))
    with contextlib.redirect_stdout(io.StringIO()):
        op.compile_and_simulate(system.device, mode, layer_name='regression', sparsity_ratio=speed)
    return op, op.profiler.get_best_record()


class CIMRegression(unittest.TestCase):
    def test_decode_conserves_work_across_geometry_and_tails(self):
        for h, w, banks, cores in [(64,48,16,16),(32,48,16,16),(64,24,16,16),(64,48,16,4),(17,13,3,7)]:
            system = build_cim_system(h,w,banks,cores)
            for m,n,k in [(1,128,8192),(1,131,8201),(5,137,129)]:
                with self.subTest(geometry=(h,w,banks,cores), shape=(m,n,k)):
                    _, rec=simulate(m,n,k,system=system,mode='heuristic-CIM-GQA-decode')
                    st=rec['other_stats']
                    self.assertEqual(st['weight_write_bytes'],k*n)
                    self.assertEqual(st['local_weight_write_bytes'],k*n)
                    self.assertEqual(st['useful_ops'],2*m*n*k)
                    psum_ws=system.device.compute_module.core.cim_macro.output_word_size
                    self.assertEqual(st['global_buffer_psum_write_bytes'],m*n*psum_ws*ceil(k/(h*banks)))
                    self.assertEqual(st['global_buffer_psum_read_bytes'],m*n*psum_ws*(ceil(k/(h*banks))-1))
                    self.assertLessEqual(st['max_active_macros'],cores)
                    self.assertLessEqual(st['K_factor']*st['N_factor'],cores)
                    self.assertEqual(system.device.compute_module.core.cim_macro.weight_buffer_size,h*w*banks)

    def test_speedup_and_dense_bits_affect_compute(self):
        system=build_cim_system(64,48,16,16)
        _,a=simulate(1,128,8192,system=system)
        _,b=simulate(1,128,8192,system=system,speed=2)
        self.assertAlmostEqual(a['compute_latency_cycles'],2*b['compute_latency_cycles'],delta=2)
        system.device.compute_module.core.cim_macro.decode_dense_serial_bits=8
        _,c=simulate(1,128,8192,system=system)
        self.assertAlmostEqual(c['compute_latency_cycles'],2*a['compute_latency_cycles'],delta=2)

    def test_all_output_psums_must_fit(self):
        system=build_cim_system(64,48,16,16)
        system.device.compute_module.l2_size=4096
        with self.assertRaisesRegex(ValueError,'Global Buffer'):
            simulate(5,1024,128,system=system,mode='heuristic-CIM-GQA-decode')

    def test_gqa_reuses_kv_without_changing_flops(self):
        system=build_cim_system(64,48,16,16)
        for n,k in [(1024,128),(128,1024)]:
            results=[]
            for batch,m,mode in [(40,1,'heuristic-CIM-decode'),(8,5,'heuristic-CIM-GQA-decode')]:
                op=BatchedMatmul(DTYPE)
                op(Tensor([batch,m,k],DTYPE),Tensor([batch,k,n],DTYPE))
                with contextlib.redirect_stdout(io.StringIO()):
                    op.compile_and_simulate(system.device,mode,layer_name='attention',sparsity_ratio=1)
                rec=op.profiler.get_best_record()
                results.append((op.flop_count,rec['other_stats']['weight_write_bytes']*op.profiler_scale))
            self.assertEqual(results[0][0],results[1][0])
            self.assertEqual(results[0][1],5*results[1][1])

    def test_report_single_token_decode_and_scaled_traffic(self):
        system=build_cim_system(64,48,16,16)
        op,rec=simulate(5,128,1024,system=system,mode='heuristic-CIM-GQA-decode')
        op.profiler_scale=8
        op.flop_count*=8
        model=SimpleNamespace(**{name:op for name in ['Q_mul_K','A_mul_V','H_matmul0','Q_proj','K_proj','V_proj','Gate_proj','Up_proj','Down_proj']},A_softmax=None)
        report=collect_matmul_profiler_stats(model,system,48,1,phase='decode')
        summary=report['_full_model_summary']
        expected=rec['total_latency']*8*9*48/system.device.compute_module.clock_freq*1000
        self.assertEqual(summary['mode'],'Decode')
        self.assertAlmostEqual(summary['gemm_tpot_ms'],expected)
        self.assertAlmostEqual(report['Q_mul_K']['activation_read_trips'],1,places=3)
        self.assertEqual(report['Q_mul_K']['hbm_weight_read_bytes'],8*128*1024)
        many=collect_matmul_profiler_stats(model,system,48,10,phase='decode')
        self.assertEqual(many['_full_model_summary']['gemm_tpot_ms'],summary['gemm_tpot_ms'])
        prefill=collect_matmul_profiler_stats(model,system,48,1,phase='prefill')
        self.assertIsNone(prefill['_full_model_summary']['gemm_tpot_ms'])
        model.A_mul_V=None
        partial=collect_matmul_profiler_stats(model,system,48,1,phase='decode')
        self.assertIsNone(partial['_full_model_summary']['gemm_tpot_ms'])
        self.assertEqual(partial['_metadata']['missing_gemm_records'],['A_mul_V'])

    def test_prefill_uses_physical_dimensions_and_replica_traffic(self):
        for mode in ['heuristic-CIM','heuristic-CIM-weight-major','heuristic-CIM-activation-major']:
            system=build_cim_system(32,24,8,4)
            _,rec=simulate(8,48,256,system=system,mode=mode)
            st=rec['other_stats']
            self.assertEqual(st['weight_write_bytes'],48*256)
            self.assertEqual(st['local_replicated_weight_write_bytes'],4*48*256)
            # Two N tiles, two rows/macro, full K, 2 ops/MAC.
            expected=2*2*24*256/system.device.compute_module.core.cim_macro.max_throughput_per_cycle*2
            self.assertAlmostEqual(rec['compute_latency_cycles'],expected,delta=2)

if __name__=='__main__':
    unittest.main()
