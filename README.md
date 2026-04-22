# LLM from Scratch

从零开始动手学大语言模型（LLM）。本项目包含一系列 Jupyter Notebook，以渐进方式拆解并复现 Transformer 的核心组件，最终训练一个可生成中文科幻小说的字符级/Token 级语言模型。

## 项目结构

```
.
├── llm.ipynb                          # 从零手写 Transformer 核心模块
├── scifi-demo/
│   └── all_code.ipynb                 # 完整训练管线：数据 → 模型 → 训练 → 生成 → 微调
├── Karpathy LLM Course/
│   └── class1.ipynb                   # Andrej Karpathy LLM 课程笔记
├── pytorch_modulelist_and_unpack.md   # PyTorch ModuleList 与解包技巧笔记
├── pyproject.toml                     # 依赖管理
└── README.md                          # 本文件
```

## 环境准备

本项目使用 [uv](https://github.com/astral-sh/uv) 管理依赖，Python >= 3.9。

```bash
# 安装依赖
uv sync

# 或直接使用 pip
pip install torch numpy tiktoken matplotlib pandas requests tqdm
```

## 内容概览

### 1. `llm.ipynb` — 逐层拆解 Transformer

以 sales textbook 为示例数据，手写以下组件：

- **Input Embedding**：使用 `nn.Embedding` 构建词嵌入表
- **Positional Embedding**：正弦/余弦位置编码
- **Multi-Head Self-Attention**：Q/K/V 拆分、Scaled Dot-Product Attention、Causal Mask
- **Residual Connection & LayerNorm**
- **Feed-Forward Network (FFN)**
- **输出层**：预测下一个 Token

适合想理解 Transformer 内部张量流动的初学者。

### 2. `scifi-demo/all_code.ipynb` — 端到端训练管线

基于中文科幻小说语料，完整实现：

| 阶段 | 说明 |
|------|------|
| 数据准备 | 从 HuggingFace 下载 webnovel_cn 数据集；合并本地 `.txt` 科幻小说 |
| 模型定义 | 完整的 Decoder-only Transformer（含 MultiheadAttention、FFN、TransformerBlock） |
| 训练 | AdamW 优化器，支持 train/valid loss 监控（集成 Aim 日志） |
| 生成 | 基于训练好的模型进行自回归文本生成 |
| 微调 | 在科幻小说子集上进一步微调，提升风格一致性 |

关键超参数：
- `context_length = 128`
- `d_model = 512`
- `num_heads = 8`
- `num_blocks = 12`
- `dropout = 0.1`

### 3. `pytorch_modulelist_and_unpack.md`

记录 `nn.ModuleList` 与 Python 解包操作（`*` / `**`）在构建重复 Transformer Block 时的使用技巧。

## 数据集说明

- **sales_textbook**：英文销售教材，用于 `llm.ipynb` 的基础演示
- **webnovel_cn / 本地科幻小说**：中文科幻小说语料，用于 `scifi-demo` 的训练与微调

> 注意：小说数据文件体积较大，未放入本仓库。运行 Notebook 时会自动下载或从本地 `scifi-demo/data/` 读取。

## 参考资源

- [Andrej Karpathy - Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [HuggingFace datasets - webnovel_cn](https://huggingface.co/datasets/zxbsmk/webnovel_cn)

## License

MIT
