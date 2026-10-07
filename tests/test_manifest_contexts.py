"""Context reuse must not bypass model/hardware/format validation."""
import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from software_model.quantspar_cim import load_speedup_manifest, compute_cycles, compute_optimal_macro_layout_prefill
from ae.figure10_qwen.test_latency import main, parser, sample_lengths


class ManifestContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "manifest.json"
        self.macro = SimpleNamespace(array_height=64, array_width=48, Nbank=16)
        self.source = dict(d_model=5120, ffn_dim=13824, q_heads=40, kv_heads=8,
            batch_size=1, shared_kv_gqa=True, prefill_lengths=[256],
            decode_cache_lengths=list(range(256,288)))
        self.target = dict(self.source, prefill_lengths=[8192],
            decode_cache_lengths=sample_lengths(8192,1024,1))
        self.doc = dict(geometry=dict(height=64,width=48,banks=16,macros=16),
            source_commit="measured-source", workload=self.source, baseline="effective",
            dense_bits=dict(prefill=3,decode=3), cycles_per_effective_bit=1,
            activation_storage_bits=8,
            transport=dict(linear_weight_storage_bits=4,kv_storage_bits=8,local_linear_weight_storage_bits=8),
            speedups={p:{op:2 for op in ["Q_proj","K_proj","V_proj","Q_mul_K","A_mul_V",
                "H_matmul0","Gate_proj","Up_proj","Down_proj"]} for p in ["prefill","decode"]})
        self.write()

    def write(self):
        self.path.write_text(json.dumps(self.doc))

    def test_strict_rejects_and_describes_context_mismatch(self):
        with self.assertRaisesRegex(ValueError,"prefill_lengths: source=.*requested="):
            load_speedup_manifest(self.macro,16,self.path,self.target)
        self.assertFalse(hasattr(self.macro,"quantspar_speedups"))
        load_speedup_manifest(self.macro,16,self.path,self.source)
        self.assertFalse(self.macro.quantspar_manifest_context_extrapolated)
        self.assertEqual(self.macro.quantspar_manifest_context_policy,"strict")

    def test_reuse_preserves_provenance_and_recomputes_target_layout(self):
        raw=self.path.read_bytes()
        with self.assertWarnsRegex(RuntimeWarning,"long-context sparsity is not measured"):
            load_speedup_manifest(self.macro,16,self.path,self.target,allow_context_extrapolation=True)
        self.assertEqual(self.path.read_bytes(),raw)
        self.assertEqual(self.macro.quantspar_manifest_sha256,hashlib.sha256(raw).hexdigest())
        self.assertEqual(self.macro.quantspar_manifest_workload,self.source)
        self.assertEqual(self.macro.quantspar_manifest_requested_workload,self.target)
        self.assertEqual(self.macro.quantspar_speedups,self.doc['speedups'])
        self.assertEqual(set(self.macro.quantspar_manifest_context_mismatches),
                         {"prefill_lengths","decode_cache_lengths"})
        self.assertTrue(self.macro.quantspar_manifest_context_extrapolated)
        steps=[]
        for length in (256,8192):
            layout=compute_optimal_macro_layout_prefill(16,5120,5120,length)
            steps.append(compute_cycles(layout,height=64,banks=16,k=5120,
                bits=self.macro.quantspar_prefill_dense_bits,
                speedup=self.macro.quantspar_speedups['prefill']['Q_proj'],baseline='effective')[1])
        self.assertEqual(steps[1],32*steps[0])

    def test_extrapolation_keeps_model_geometry_and_format_checks(self):
        for key,value in [('d_model',4096),('kv_heads',4),('batch_size',2),('shared_kv_gqa',False)]:
            with self.subTest(key=key):
                target=dict(self.target,**{key:value})
                with self.assertRaisesRegex(ValueError,key):
                    load_speedup_manifest(self.macro,16,self.path,target,allow_context_extrapolation=True)
        for key,value,pattern in [('geometry',dict(height=64,width=48,banks=16,macros=32),'geometry'),
                                 ('baseline','invalid','phase'),
                                 ('transport',dict(linear_weight_storage_bits=4,kv_storage_bits=0,local_linear_weight_storage_bits=8),'transport')]:
            original=self.doc[key];self.doc[key]=value;self.write()
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,pattern):
                load_speedup_manifest(self.macro,16,self.path,self.target,allow_context_extrapolation=True)
            self.doc[key]=original;self.write()

    def test_missing_or_malformed_source_contexts_are_not_relabelled(self):
        for value in (None,'256',[True],[0],[256,255],[],{'length':256}):
            self.doc['workload']=dict(self.source,prefill_lengths=value);self.write()
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'source prefill_lengths'):
                load_speedup_manifest(self.macro,16,self.path,self.target,allow_context_extrapolation=True)

    def test_cli_requires_manifest_and_counts_decode_calls(self):
        args=parser().parse_args(['--speedups-json','stats.json','--allow-context-extrapolation',
                                 '--input-lengths','8192','--output-lengths','1025'])
        self.assertTrue(args.allow_context_extrapolation)
        lengths=sample_lengths(args.input_lengths[0],args.output_lengths[0]-1,1)
        self.assertEqual((len(lengths),lengths[0],lengths[-1]),(1024,8192,9215))
        with contextlib.redirect_stderr(io.StringIO()) as err,self.assertRaises(SystemExit) as exc:
            main(['--allow-context-extrapolation'])
        self.assertEqual(exc.exception.code,2)
        self.assertIn('requires --speedups-json',err.getvalue())


if __name__ == '__main__':
    unittest.main()
