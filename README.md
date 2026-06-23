# LLMCompass_mod 项目说明

`LLMCompass_mod` 是一个面向大语言模型推理硬件评估的 Python 项目。项目把 LLM 推理中的 Transformer、矩阵乘、Softmax、LayerNorm、GeLU、AllReduce 等计算过程抽象为软件模型，再结合 GPU/TPU 风格的硬件模型、互连模型、存储模型和芯片面积成本模型，用于估算延迟、吞吐、硬件面积，并支持一定范围的设计空间探索。

当前仓库更像是研究代码和实验复现代码的集合，包含论文图表复现实验、预设硬件配置、映射记录和成本估算脚本。

## 主要能力

- 对 Transformer block 的 prefill 和 autoregressive decoding 阶段建模。
- 对 Matmul、BatchedMatmul、Softmax、LayerNorm、GeLU、AllReduce 等算子进行 roofline 估算或启发式模拟。
- 建模 A100、MI210、TPUv3 等硬件配置，包括计算单元、I/O、内存和互连。
- 使用 JSON 架构模板生成系统模型，并在不同设备数、core 数、systolic array 尺寸、SRAM、global buffer、内存协议等参数上做设计空间探索。
- 使用成本模型估算 compute chiplet、I/O die、SRAM、寄存器文件、存储/互连 PHY 与 controller 等面积。
- 提供 `ae/` 目录下的实验脚本，用于复现 figure5 到 figure12 相关结果。

## 目录结构

```text
.
├── ae/                         # Artifact evaluation / 论文图表复现实验
│   ├── figure5/                # 算子级性能对比：matmul、softmax、layernorm、gelu、allreduce、transformer
│   ├── figure6/                # 成本模型实验
│   ├── figure7-12/             # core size、memory bandwidth、cache、latency、decoding、throughput 等实验
│   ├── expected_results/       # 预期输出 PDF
│   └── map_rcd/                # Transformer 映射记录和硬件配置记录
├── configs/                    # 硬件/系统 JSON 配置模板
├── cost_model/                 # 芯片面积和供应链成本模型
├── design_space_exploration/   # 设计空间探索入口
├── hardware_model/             # 设备、计算单元、I/O、内存、互连、系统模型
├── software_model/             # LLM 算子和 Transformer 软件模型
├── systolic_array_model/       # systolic array 查表数据
├── environment.yml             # Conda 环境描述
└── utils.py                    # 通用工具函数
```

## 环境安装

推荐使用 Conda：

```bash
conda env create -f environment.yml
conda activate llmcompass_ae
```

`environment.yml` 中声明了：

- Python 3.9
- PyTorch
- scalesim
- matplotlib
- seaborn
- scipy

部分模块还会导入 `numpy`、`pandas` 等库。通常 PyTorch/科学计算环境会间接安装 `numpy`，但如果运行脚本时报缺包，可以补装：

```bash
pip install numpy pandas
```

## 常用入口

### 1. 成本模型示例

```bash
python -m cost_model.cost_examples
```

该脚本读取 `configs/prefilling_system.json`，输出 compute area、I/O area 和总面积。

### 2. 设计空间探索

```bash
python -m design_space_exploration.dse
```

该入口会读取 `configs/template.json`，遍历设备数量、互连、core 数、systolic array、vector unit、SRAM、global buffer、内存协议等设计参数，尝试寻找满足 prefill 和 decoding 延迟约束的最低面积设计，并将结果写入：

```text
configs/best_arch_specs.json
```

注意：当前 `dse.py` 中成本模型 import 被注释掉了，但函数内部仍调用 `calc_compute_chiplet_area_mm2` 和 `calc_io_die_area_mm2`。直接运行前需要恢复：

```python
from cost_model.cost_model import calc_compute_chiplet_area_mm2, calc_io_die_area_mm2
```

### 3. 算子级实验

以 Matmul 为例：

```bash
python -m ae.figure5.ab.test_matmul --simgpu
python -m ae.figure5.ab.test_matmul --simgpu --roofline
python -m ae.figure5.ab.test_matmul --simamd
```

Figure 5 整组实验入口：

```bash
cd ae/figure5
bash run_figure5.sh
```

### 4. Transformer 映射记录实验

```bash
python -m ae.map_rcd.map_transformer --init --simgpu
python -m ae.map_rcd.map_transformer --simgpu
```

生成的映射记录会写入 `ae/map_rcd/*.jsonl`，硬件配置会写入 `ae/map_rcd/hw_config_*.json`。

### 5. 其他图表复现实验

各图表目录通常提供独立 shell 入口：

```bash
bash ae/figure6/run_figure6.sh
bash ae/figure7/run_figure7.sh
bash ae/figure8/run_figure8.sh
bash ae/figure9/run_figure9.sh
bash ae/figure10/run_figure10.sh
bash ae/figure11/run_figure11.sh
bash ae/figure12/run_figure12.sh
```

运行前建议先查看对应 `run_figure*.sh`，其中部分脚本会删除当前目录下的 `*.csv` 或 `*.pdf` 输出文件。

## 配置文件说明

- `configs/template.json`：A100-like 默认架构模板，也是 DSE 的基础模板。
- `configs/GA100.json`：GA100/A100 相关配置。
- `configs/mi210.json`、`configs/mi210_template.json`：AMD MI210 相关配置。
- `configs/prefilling_system.json`、`configs/generation_system.json`：prefill / generation 场景配置。
- `configs/latency_design.json`：用于 figure10 等实验的自定义设计配置。

配置通常包含：

- `device_count`：设备数量。
- `interconnect`：互连链路、带宽、延迟、拓扑。
- `device.frequency_Hz`：设备频率。
- `compute_chiplet`：core 数、工艺节点、systolic array、vector unit、寄存器文件、SRAM。
- `io`：global buffer、内存 channel、pin 数和带宽。
- `memory`：总容量。

## 当前代码状态与注意事项

我在当前快照上运行了：

```bash
python -m compileall -q .
```

发现以下语法级问题，说明项目当前不是完全可导入/可运行状态：

- `hardware_model/arch_template.py`：`ArchitectureTemplate.__init__` 函数体不完整，触发 `IndentationError`。
- `software_model/matmul_old.py`：第 1365 行附近存在未完成表达式，触发 `SyntaxError`。
- `software_model/transformer.py`：存在残缺语句 `from software_model.state_manage import`，触发 `SyntaxError`。

此外：

- `design_space_exploration/dse.py` 中成本模型 import 被注释，但后续仍调用相关函数。
- 多个实验脚本会向 `ae/figure*/` 目录写入 CSV/PDF/JSONL 输出。
- GPU 实测路径依赖 CUDA 和可用 GPU，例如 `Matmul.run_on_gpu()` 使用 `torch.cuda`。
- `systolic_array_model/` 中的 CSV 查表文件是矩阵乘启发式模拟的重要输入。

## 建议的后续整理顺序

1. 修复 `software_model/transformer.py` 的残缺 import，否则 Transformer 相关实验无法导入。
2. 判断 `hardware_model/arch_template.py` 和 `software_model/matmul_old.py` 是否仍需保留；如果只是废弃代码，可以移出包导入路径或标注 deprecated。
3. 恢复 `dse.py` 中成本模型函数 import。
4. 补齐 `environment.yml` 中实际使用但未显式声明的依赖，例如 `numpy`、`pandas`。
5. 给核心入口增加最小 smoke test，例如成本模型示例、单个 Matmul roofline、单个配置模板加载。


# Hardware Evaluation步骤描述
1. 打开configs文件，复制

如果which python指向的是错误环境的python，可以执行：
export PATH=/home/zyzhao/.conda/envs/llmcompass_ae/bin:$PATH

