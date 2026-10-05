import contextlib
import io
import unittest
from ae.figure10_qwen.test_latency import sample_lengths, sum_interpolated
from software_model.qwen_fig10 import QwenFigure10Prefill, QwenFigure10Decode
from software_model.utils import Tensor, data_type_dict
from hardware_model.compute_module import Overhead
from ae.figure5.ijkl.test_transformer import build_cim_system, collect_matmul_profiler_stats


class QwenFigure10Tests(unittest.TestCase):
    def test_discrete_linear_sum_and_request_boundaries(self):
        samples={8:{'v':3},12:{'v':11},20:{'v':27}}
        self.assertEqual(sum_interpolated(samples,8,0,'v'),0)
        self.assertEqual(sum_interpolated(samples,8,1,'v'),3)
        self.assertEqual(sum_interpolated(samples,9,4,'v'),5+7+9+11)
        self.assertEqual(sum_interpolated(samples,8,13,'v'),sum(range(3,28,2)))
        self.assertEqual(sample_lengths(8,0,64),[])
        self.assertEqual(sample_lengths(8,2,64),[8,9])
        with self.assertRaises(ValueError):
            sum_interpolated(samples,8,14,'v')

    def test_complete_vectors_control_and_gemm_unchanged(self):
        dt=data_type_dict['int8']
        for cls,phase in [(QwenFigure10Prefill,'prefill'),(QwenFigure10Decode,'decode')]:
            totals=[]; gemms=[]
            for control in [0,1e-6]:
                system=build_cim_system(64,48,16,16)
                system.device.compute_module.overhead=Overhead(control,control,control,control)
                kwargs=dict(d_model=5120,n_heads=40,n_kv_heads=8,ffn_dim=13824,device_count=1,data_type=dt)
                if phase=='decode': kwargs['shared_kv_gqa']=True
                model=cls(**kwargs)
                with contextlib.redirect_stdout(io.StringIO()):
                    if phase=='prefill': model(Tensor([1,8,5120],dt))
                    else: model(Tensor([1,1,5120],dt),8)
                    totals.append(model.compile_and_simulate(system))
                    report=collect_matmul_profiler_stats(model,system,phase=phase)
                self.assertTrue(report['_metadata']['complete_block_gemm'])
                cols=list(map(float,model.simluate_log.split(',')))
                self.assertTrue(all(x>0 for x in cols[6:10]))
                self.assertAlmostEqual(sum(cols),totals[-1])
                gemms.append(report['_full_model_summary']['total_latency_cycles'])
            self.assertEqual(gemms[0],gemms[1])
            self.assertAlmostEqual(totals[1]-totals[0],13e-6)
