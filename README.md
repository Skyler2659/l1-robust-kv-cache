# L1/L2 Leverage Score 用于 KV Cache 淘汰

我们尝试了用**随机线性代数里的 leverage score(几何信号)作为 LLM KV cache
的 token 重要性打分**,用来决定长上下文推理时淘汰哪些 token。出发点来自
attention 的计算方式:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V
$$

query 和 key 的匹配度($QK^\top$ 大)只决定了权重,最终输出是这些权重对
**V 行向量的加权和**,所以一个 token 对输出实际的贡献还取决于它在 V 空间
里的几何结构——而这部分是纯 attention 打分看不出来的。因此我们直接考察
KV 矩阵的几何重要性(leverage score),把它当作一个独立于 attention 的信号源,
在 tight budget 下用来保住 attention 抓不住的证据 token。

## 我们做了什么

1. **实现了 L1 leverage score 估计算法**:概念出自 ℓp 回归行采样
   (Dasgupta et al., SODA 2008),嵌入构造参考 Exp(1) 重加权
   (Woodruff & Zhang, COLT 2013)加 CountSketch(Charikar et al. 2002),
   具体管线(Exp(1) 重加权 → CountSketch → QR → 行 ℓ1 范数)是我们自己
   拼装的近似实现,全部 PyTorch 手写;以及精确的 L2
   leverage score(QR 后行范数平方)。代码在 `src/sketching/`。
2. **实现了 30+ 种淘汰方法并统一到一套框架**:leverage 族(L1/L2、prefill-only、
   按 K 打分、decode 阶段更新)、attention 族(累计 attention / H2O、窗口化)、
   hybrid(attention + leverage)、以及 SnapKV、PyramidKV、Compactor、recency、
   random、norm、聚类、PCA、oracle 等对照组,全部走统一的
   `BaseEviction` 抽象和方法注册表,带预算合法性不变量校验。
3. **搭建了完整的实验平台**:YAML 实验配置、统一 benchmark runner、
   官方口径打分(RULER / LongBench)、overlap/相关系数/案例分析、出图脚本、
   13 个单元测试和 smoke test。
4. **移植到 MLX 4bit 后端**:在 Apple Silicon 上用 MLX 4bit 量化跑通
   Qwen2.5-7B,实现了 prefill 压缩、decode 阶段打分淘汰、按头 attention
   观测,15 种方法 sanity 全过。
5. **跑了两组受控诊断实验**:RULER NIAH(单针检索)和 RULER VT(变量追踪)。

## 工程实现

### 改 KV cache 的两种方式

**HF/Torch 路径**(早期,`l1_llm/pos_shift/`):对 llama / gpt_neox / qwen2 /
falcon 四种架构的 attention forward 打补丁,替换成我们自己的实现。补丁里
把 Q/K/V 投影、reshape、RoPE 之后的结果存进 `shared_q` 全局仓库(每层最后
token 的 query、按头取均值的 key 行),leverage 打分和 attention 累加都从
这里取。同时兼容了不同 transformers 版本的差异(`past_key_value` 与
`past_key_values` 参数名、`layer_idx` 属性 polyfill)。

**MLX 路径**(最终主路径,`src/runners/mlx_runner.py`):直接编辑 MLX-LM 的
cache 对象。cache 里 `keys/values` 是物理存储,`offset` 是物理长度,
`logical_offset` 记录逻辑位置——压缩之后仍知道每个 slot 对应真实 token 的
序号。prefill 结束做 `prefill_compress`,decode 每步先 `evict_for_space`
腾出位置再追加;每层独立打分、独立裁剪。SnapKV / PyramidKV / Compactor
这类按头保留的方法(snapkv 每头保留的 token 集合可以不同),用
`head_valid_mask` 逐头标记有效位置,算 attention 时按掩码取。attention
权重通过替换 `scaled_dot_product_attention` 装 hook 拿到。

### RoPE 和淘汰怎么配合

这是整个系统的核心难点:key 一旦按 RoPE 旋转,删掉中间 token 后位置就是
稀疏的,不能当普通数组用。

- **pos_shift 方案**(HF 路径):用**完整 cache 长度**生成 cos/sin,再按
  **原始 position_ids** 取对应频率——Q 用当前真实位置旋转,K 用完整长度
  重新旋转。这样即使 cache 被淘汰得稀疏,attention 分数依然正确,同时保留
  了 L1 打分需要的旋转后 Q。
- **MLX 方案**:`rope_offset` 直接取 `logical_offset`,压缩不改变旋转位置;
  淘汰只做 `index_select`(按 seq 维取保留行),**从不重排位置**。
- 数值兜底:Q·K^T 用 float32 计算,避免 fp16 溢出;`rotary_emb` 的新老 API
  (传 `position_ids` / 传 `seq_len`)都兼容。

### 位置追踪

每次淘汰后,`BaseEviction` 维护每个 cache slot 到**原始 token 位置**的
映射(position map),`gather_by_dim` 只沿 seq 维收集、不重排。这让打分、
oracle 方法(用真实证据位置做上限)和后处理分析都能知道保留的 token 原本
在第几个位置。MLX 侧对应的是 `logical_offset` + 每层同步。

### 预算语义与打分更新策略

- 普通方法:`budget` 是总 live KV,decode 时先腾位再追加。
- SnapKV / PyramidKV / Compactor:`prompt_prefill` 语义,只压缩 prefill
  的 prompt cache,decode 生成的新 token 自由追加(结果里显式记录
  `cache_budget_scope`)。
- 逐层预算:MLX runner 支持按层分配预算,而不是每层同一个数。
- 打分更新:支持 prefill 只算一次(`*_prefill_only`)、每 N 步重算
  (`update_interval`,摊销 sketch 成本)、以及 decode 阶段持续更新三种策略。

## 实验结果

### RULER NIAH(单针检索)

`results/mlx_qwen25_7b_inst_4bit/ruler/20260611_225910` — 15 方法 × 3 预算
(128/256/512) × 20 样本,共 900 条,全部通过官方口径打分。

| 方法 | 均分(0–100) | 分预算 |
| --- | --- | --- |
| full | 100.0 | 100/100/100 |
| l1_leverage / l2_leverage | 100.0 | 100/100/100 |
| l1/l2_prefill_only | 100.0 | 100/100/100 |
| attention_l1 / attention_l2 | 100.0 | 100/100/100 |
| attention | 60.0 | 0/80/100 |
| snapkv | 43.3 | 5/35/90 |
| pyramidkv | 13.3 | 0/5/35 |
| recency / sink_recent | 1.7 | 0/0/5 |
| compactor / random | 0.0 | 0/0/0 |

说明:leverage 及其 hybrid 在最紧的 128 预算下就 100% 保住针,**attention 到
512 预算才追平**——几何信号能在 tight budget 下守住证据 token,attention
做不到。

### RULER VT(变量追踪)

`results/mlx_qwen25_7b_inst_4bit/ruler/20260615_223853` — 8 方法 × 3 预算
(256/512/1024) × 10 样本,共 240 条。

| 方法 | 均分(0–100) | 分预算 |
| --- | --- | --- |
| full | 98.0 | 98/98/98 |
| l2_prefill_only | 97.3 | 100/94/98 |
| attention_l1 | 94.7 | 86/98/100 |
| attention_l2 | 92.0 | 80/96/100 |
| l1_prefill_only | 91.3 | 92/86/96 |
| attention | 85.3 | 56/100/100 |
| snapkv | 73.3 | 38/88/94 |
| compactor | 38.7 | 2/24/90 |

说明:状态跟踪任务上,几何信号在 tight budget(256)下依然强,attention 在
预算放宽后追平;attention + leverage 的 hybrid 全程稳定,是我们最看好的组合。

## 结论

leverage score 是独立于 attention 的 token 重要性信号:在 tight budget 下
保留证据 token 上明显强于 attention,且与 attention 互补(attention 负责宽
预算下的语义选择,几何信号负责守住关键证据)。这是代码库和两组实验支持的主要发现。

## 代码结构

```text
src/eviction/        30+ 种淘汰方法 + 注册表
src/sketching/       L1/L2 leverage 估计算法
src/runners/         MLX 4bit 推理 + cache 编辑(2500+ 行)
src/benchmarks/      RULER / LongBench / NIAH 适配器
src/evaluation/      官方口径打分
tests/               单元测试(预算不变量、数学恒等式、smoke)
results/             两个完整实验 run 及图、分析
configs/             YAML 实验配置
l1_llm/ h2o_llm/ snapkv/ rocketkv/ streaming_llm/   早期原型(保留存档)
```

## 快速使用

```bash
pip install -r requirements.txt
PYTHONPATH=. .venv/bin/python -m pytest tests/test_p0.py -q   # 单元测试
PYTHONPATH=. .venv/bin/python scripts/smoke_test.py           # 框架 smoke
```

复现保留结果:进入对应 run 目录,用其中的 `config.yaml` 跑
`scripts/run_benchmark.py` 即可。
