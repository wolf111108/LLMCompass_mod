# 修改记录

记录每次提交的目的、内容、验证与限制。后续所有提交必须在同一提交中新增或更新本次条目，详细规则见 [AGENTS.md](AGENTS.md)。日期采用 `YYYY-MM-DD`，条目按时间倒序排列；同一天的提交按提交顺序倒序排列。

当前提交用与 commit message 一致的标题标识，可通过 `git log -- CHANGELOG.md` 找到所属提交；自身 SHA 无法在提交前确定，不要求写入。历史补录使用已有 SHA。以下历史记录依据 `CIM_Arch` 的提交内容补录，不代表重新运行了历史实验。


## 2026-10-07 — Allow explicit context extrapolation of measured CIM speedups

- 目的：复用短序列实测统计的phase/operator倍率，为8192＋1024等长请求做离线条件外推，同时避免把短统计重新标成目标长度实测。
- 内容：Figure-10新增`--allow-context-extrapolation`；只放行prefill/decode上下文长度差异，仍校验模型、batch、GQA、硬件、位口径及transport；错误列出不匹配字段；源manifest不改写，报告保留SHA与源workload并记录目标workload和外推假设。更新使用说明与输出token计数示例。
- 验证：新增5项无Torch依赖回归通过，覆盖默认严格拒绝、显式复用及源文件不变、目标布局重算、模型/硬件/格式不匹配仍拒绝、无效上下文拒绝及1025→1024次decode。CLI help、Python3.9语法与补丁检查通过。
- 限制：编辑环境缺Torch，未运行完整Figure-10仿真；未获得用户实际manifest，不报告目标性能。固定短序列倍率仅是外推假设，长上下文稀疏、分片及负载不均衡没有重新测量。保持现有Transformer-stack E2E口径，不补缺失生成算子。

## 2026-10-06 — Import FP8 W4 transport for Asyn-CIM profiles

- 目的：让quantspar新的FP8/W4 manifest明确驱动Linear和Attention不同的搬运位宽。
- 内容：loader校验可选transport并设置packed INT4 Linear、FP8 K/V及本地Linear写入位宽；GEMM按算子选择格式，旧manifest保持原默认。SCALEsim改为systolic路径按需导入，纯CIM不依赖它。
- 验证：真实PyTorch环境下全套14项后端测试通过；新增回归验证W4片外0.5B、Linear本地1B、FP8 K/V1B、无效transport拒绝及旧manifest恢复。真实小Qwen256＋32文件已通过loader和Figure10 CLI联调（32个decode context），未使用Torch/SCALEsim占位；修改文件语法与diff检查通过。
- 限制与旧结果影响：不改变旧manifest、legacy和systolic算式；新文件按实际明确transport计量。仅为原Figure10 Transformer-stack估计，未补完整生成链的LM head/RoPE/residual等成本，未执行CUDA/芯片验证。

## 条目模板

```markdown
### YYYY-MM-DD — 提交标题

- 目的：解释要解决的问题或预期行为。
- 内容：描述最终修改及关键文件；涉及默认行为、接口或统计口径时明确写出。
- 验证：实际执行的检查及结果；未验证时注明原因。
- 限制：剩余问题、估计假设或适用范围；无新增限制时注明。
```

## 提交记录

### 2026-10-05 — Document every commit in repository changelog

- 目的：让每次修改的原因、实际内容与验证结果可追溯，并建立后续提交时同步维护文档的规则。
- 内容：新增根目录 `CHANGELOG.md`，补录当前分支此前 7 次提交；新增 `AGENTS.md`，要求每次提交在同一提交内更新记录；在 `README.md` 添加文档入口。
- 验证：核对历史条目对应的 Git 提交与变更文件；检查文档相对链接和 `git diff --check`。本次仅修改文档，不重复运行模型测试。
- 限制：这是仓库协作约定，不是自动生成修改说明的 Git hook 或强制 CI 门禁；其他提交者也需要遵守。历史实验未重新执行。

### 2026-10-05 — Align default CIM GEMM mapping and compute contract with quantspar（d01c914）

- 目的：解决 LLMCompass 与 quantspar 在 prefill 布局、串行位宽及周期聚合范围上的差异，使上游统计倍率能按相同计算口径用于仿真。
- 内容：新增 `software_model/quantspar_cim.py`；默认 CIM GEMM 改用 quantspar 布局和算子级异步计算周期；加入 `source/effective` 基准选择、位宽与每有效位周期配置；fig10 支持严格 JSON 倍率导入并校验硬件、模型及上下文；保留 legacy 后端；新增接口文档与回归测试。
- 验证：400 组 prefill 布局与 quantspar 原始函数一致；13 项解析测试通过；fig10 默认、JSON 导入及 legacy 回退冒烟运行通过；远端 tree 与本地验证版本一致。
- 限制：验证环境缺少 Torch/SCALEsim，使用导入占位执行解析代码，未做完整推理或硬件测量。访存采用串行带宽估计；`source` 兼容上游小 K 分母缺陷，`effective` 需要重新统计倍率；上游混精 decode 采集仍缺失。未导入匹配的统计文件时，旧硬编码倍率未经重新校准。

### 2026-10-05 — Add Figure-10-style Qwen CIM end-to-end latency evaluation（c7ce110）

- 目的：为当前 Qwen 模型提供类似官方 fig10 的请求延迟估计，纳入 Transformer block 的 GEMM 与向量算子。
- 内容：新增 `ae/figure10_qwen/` 脚本、绘图入口与说明；新增 `software_model/qwen_fig10.py` 包装类，补齐 decode 的向量算子成本；按一次 prefill 与 G−1 次 decode 累加请求延迟，输出 E2E、GEMM E2E 与平均 TPOT；加入请求边界及算子统计测试。
- 验证：当次验证中已有 6 项 CIM 测试及新增 2 项 fig10 测试通过；生成解析验证结果；检查离散 decode 求和及插值边界。
- 限制：这是 Transformer-stack 估计，不是 checkpoint 推理计时；LayerNorm/GeLU 分别近似 RMSNorm/SiLU；不包含 embedding、LM head、采样等。当次验证使用 Torch/SCALEsim 导入占位；该提交留下的验证 CSV 属于后续更改前的 legacy 后端。

### 2026-10-05 — Add shared-KV decode sweep results（1bd47aa）

- 目的：保存 shared-KV decode 在不同上下文长度和带宽配置下的仿真结果，便于核对 TPOT 与数据搬运趋势。
- 内容：加入 sweep 日志、GEMM/仿真 CSV、profiler JSON、指标汇总、manifest、校验和与实验说明。
- 验证：本次补录核对了提交内结果文件类型与实验说明；未重新运行 sweep，具体运行证据以该提交的 README、manifest 和日志为准。
- 限制：结果对应当时的后端及硬件配置；不能直接作为新 quantspar 后端的验证结果。

### 2026-10-04 — Fix CIM geometry accounting and report GEMM-only TPOT（61738e9）

- 目的：修正 CIM 几何参数、尾部 tile 和数据流量计数，并明确 GEMM-only TPOT 的统计范围。
- 内容：修改 `hardware_model/compute_module.py`、`software_model/matmul.py` 及 figure5 profiler 汇总；加入 `GEMM_METRICS.md` 与 `tests/test_cim_gemm.py`，核对配置尺寸、shared-KV 复用、倍率及统计缩放。
- 验证：6 项解析回归测试通过；后续 quantspar 对齐提交中保留这些测试，并显式以 legacy 后端运行。
- 限制：GEMM-only TPOT 不包含向量算子、LM head、采样等；解析验证不等于硬件测量。

### 2026-10-04 — Update CIM model and profiling results（f5e0776）

- 目的：更新 CIM prefill/decode 建模路径及 OPT/Qwen profiler 输出，为后续映射与流量分析提供基础。
- 内容：修改 CIM 计算单元、I/O、Matmul、Transformer 与 figure5 实验入口，并更新 4 份 OPT/Qwen profiler JSON。
- 验证：本次补录核对了提交涉及的文件及结果产物；未重跑该历史版本，不能据此声明其所有建模路径均正确。
- 限制：该版本后续接受了几何计数、GEMM 指标及 quantspar 口径修正，应按具体提交解释旧结果。

### 2026-10-04 — Ignore bulk CIM experiment artifacts（d5424c5）

- 目的：减少批量实验产物进入版本控制带来的仓库体积和审阅负担。
- 内容：在 `.gitignore` 中增加批量 CIM 实验产物的忽略规则。
- 验证：本次补录核对了 `.gitignore` 变更；未执行历史实验。
- 限制：忽略规则不替代关键结果的可复现记录，已跟踪文件也不会因此自动移除。

### 2026-06-23 — Add CIM-based architecture evaluation T5（5702a86）

- 目的：建立该分支的 LLMCompass/CIM 架构评估代码基础。
- 内容：初始提交加入软件算子、硬件与系统模型、实验脚本、配置、成本模型、查表数据及环境说明。
- 验证：本次补录核对了初始提交的文件范围；没有重新执行全部原始实验。
- 限制：初始代码与当前后端存在较大差异，历史结果需要结合后续修改记录阅读。
