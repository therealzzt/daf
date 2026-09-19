# AGENTS.md —— DAF on CacheBlend 复现项目

本仓库是 **DAF（Disaggregated Attention Fusion，论文 arXiv 2607.21599）** 在 CacheBlend/vLLM 上的复现项目。
基座是 CacheBlend 官方魔改的 vLLM 0.4.1（`vllm_blend/`），DAF 算法以可选开关形式叠加在 CacheBlend 之上。
论文 PDF：`docs/daf2607.21599v1.pdf`。DAF 官方未开源，本项目为按论文描述的自行实现（目标 A：功能复现原型）。

## 目录导览

```
vllm_blend/     ← CacheBlend 魔改 vLLM 0.4.1（DAF 算法代码在此，源码编译安装）
example/        ← 实验驱动脚本（blend.py 为主，含 DAF 开关；blend_{musique,wikimqa,samsum}.py 为原作者评测脚本，无 DAF）
inputs/         ← 2–4k 档样本 1~10.json + 官方数据集 musique_s/samsum/wikimqa_s
experiments/    ← 长上下文实验工作区：样本生成/评分/续跑脚本、分档样本、日志、评分报告
docs/           ← 论文、DAF 改造计划（v2 校正版为准）、复现指南、daf1.0~1.3 实验记录
wsl_setup/      ← WSL/服务器环境搭建与诊断脚本、冒烟测试
```

## 环境（权威步骤见 `docs/服务器复现指南.md`）

- conda 环境名 **`cacheblend`**，Python 必须 3.10；CUDA 12.1（用 `nvidia/label/cuda-12.1.1` 显式钉住，裸 `cuda-toolkit=12.1` 可能解析出 13.x 导致编译失败）。
- 钉版依赖：`torch==2.2.1+cu121`、`xformers==0.0.25`、`transformers==4.41.2`（新版 transformers 会让 vllm 0.4.1 崩溃）。
- 编译安装（10~30 分钟）：

```bash
conda activate cacheblend && cd ~/daf/vllm_blend
export CUDA_HOME=$HOME/miniconda3/envs/cacheblend
export MAX_JOBS=$(nproc)
export TORCH_CUDA_ARCH_LIST="8.0"   # V100=7.0 A100=8.0 3090=8.6 4060/4090=8.9 H100=9.0
export TMPDIR=$HOME/.tmp && mkdir -p $TMPDIR
pip install -r requirements-cuda.txt
pip install -e . --no-build-isolation
python -c "import vllm; print(vllm.__version__)"   # 应输出 0.4.1
```

- 模型由 `BLEND_MODEL_PRESET=bf16|awq|qwen` 切换（`example/blend.py`）：
  - bf16：`mistralai/Mistral-7B-Instruct-v0.2`（需 ≥16GB 显存；gpu_memory_utilization=0.85，max_model_len=16384）
  - AWQ：`TheBloke/Mistral-7B-Instruct-v0.2-AWQ`（8GB 卡可跑；0.75，max_model_len=32768）。**30k 档必须用 AWQ**——bf16 权重 15GB 在 24GB 卡上 KV 池装不下 32768。
  - qwen：`Qwen/Qwen2.5-7B-Instruct`（DAF 论文模型；BF16，0.85，max_model_len=8192——不加 YaRN 上限 32768）。用 ChatML 模板；**Qwen 的 BPE 会在块边界跨块合并 token（decode→encode 不幂等），qwen 预设全程用 `prompt_token_ids` 直传绕过引擎重新分词**，Mistral 预设保持字符串路径不变。
  - 国内下载模型先 `export HF_ENDPOINT=https://hf-mirror.com`；`vllm-nccl-cu12` 的 libnccl 需单独下载放到 `~/.config/vllm/nccl/cu12/`。

## DAF 实现位置（只动了这几个文件）

- **`vllm_blend/vllm/model_executor/models/llama.py`**：层调度与 `cache_fuse_metadata` 控制字典。
  - DAF 字段：`daf`（总开关，默认 False）、`sel_layers=[1,2,3]`（打分层，论文 0-indexed 第 1~3 层）、`dev_scores`、`doc_imp_indices`、`num_imp_tokens`。每次 check 开头必须重置后四项（防跨样本分数污染）。
  - 调度：`select_layer = max(sel_layers)+1 = 4`；层 0 → status=0，层 1~3 → status=3（完整 prefill + 累积 deviation），层 4 → status=1（选择+执行），层 >4 → status=2（compact 续算），-1 = decode。
- **`vllm_blend/vllm/model_executor/models/qwen2.py`**：2026-09-17 已把 llama.py 的整套 cache_fuse/DAF 机制原样移植（白板字段、重置、层调度、status==1 时 residual/positions 裁剪、fake_q 旋转旧 K、hack_kv collect、`kv_scale=1.0`），供 `BLEND_MODEL_PRESET=qwen` 使用。
- **`vllm_blend/vllm/attention/backends/xformers.py`**：算法核心。
  - status=3：V-only deviation `sum((value[:doc_len]-value_old[:doc_len])**2)`，float32 累加，不 topk 不裁剪（需要 `value_old`，加载条件已扩到 status∈[1,2,3]）。
  - status=1：多层分数聚合后 top-k（`recomp_ratio=0.3`，**只在文档段选**，问题 token 无条件保留），写 `doc_imp_indices`/`num_imp_tokens`/`imp_indices`。
  - `_run_daf_attention()`（status=1/2 共用）：① 重要 token 只互看（无掩码自注意力）；② 问题段对 hybrid KV 做因果注意力（`LowerTriangularFromBottomRightMask`）；③ `cat([out_imp, out_suf])` 融合，顺序不可换。hybrid KV 组装时 clone 旧缓存，禁止原地改共享离线 KV。
- **`vllm_blend/vllm/attention/backends/flash_attn.py`**：2026-09-17 已把 xformers.py 的全部 cache_fuse/DAF 机制移植到此（status 0/1/2/3、deviation 累积、选择、hybrid KV、`_run_daf_attention` flash 版）。注意力调用换成 `flash_attn_varlen_func`：重要 token 互看 `causal=False`，问题段 `causal=True`（q_len<kv_len 时 flash 的 causal mask 右下对齐，已数值验证等价于 `LowerTriangularFromBottomRightMask`），GQA 由 flash 原生支持不手动 expand。
- **后端切换**：`VLLM_ATTENTION_BACKEND=FLASH_ATTN`（别名 `FLASH_ATTENTION`）/ `XFORMERS`（`selector.py`）。**默认后端已改为 XFORMERS**（防装 flash_attn 后静默切换）；显式请求 flash 但包未装会大声报错。flash_attn 2.5.8（cu122torch2.2 cp310 wheel，装在 cacheblend 环境）。两个后端的 DAF/CB 输出已交叉验证一致（SM 级别 10/10）。
- **`example/blend.py`**：主驱动。`DAF=0|1` 环境变量切换对照；DAF 模式下 `sel_layers=[1,2,3]`、`recomp_ratio=0.3`、`check_layers=[4]`。原作者 bug 已修：`last_len=len(q_ids+s_end)`（原恒为 1）、`num_layer` 动态获取。

## 运行实验（所有脚本从仓库根 `~/daf` 运行）

冒烟测试：`bash wsl_setup/run_smoke.sh 0|1`（0=CacheBlend，1=DAF）。

2–4k 档：
```bash
DAF=0 python example/blend.py 2>&1 | tee run_cb.log
DAF=1 python example/blend.py 2>&1 | tee run_daf.log
```

6–10k / 30k 档（幂等续跑，可断线；以 30k 为例）：
```bash
python experiments/scripts/make_inputs_long_30k_buriedqa.py      # 生成样本
nohup bash experiments/scripts/run_long30k_resume.sh \
    > experiments/logs/long30k/resume.out 2>&1 &                 # 跑批（GPU 被占用 run_long30k_wait_and_start.sh 看门狗排队）
python experiments/scripts/score_em_long_6k10k.py \
    --dir experiments/inputs_long_30k_buriedqa \
    --logs experiments/logs/long30k/run_cb_long30k_r*.log \
           experiments/logs/long30k/run_daf_long30k_r*.log \
    --out experiments/reports/em_report_long_30k.md              # 评分
```

评测指标：TTFT = `first_token_time - first_scheduled_time`（3 轮取中位数）；QA 主指标 **SM**（substring match，RULER 风格），EM 仅记录（Mistral 恒 0），samsum 用 F1/Rouge-L。评分与归一化逻辑在 `example/utils.py` 与 `experiments/scripts/score_em_long_6k10k.py`。

## 铁律与坑（改动代码/跑实验前必读）

1. **DAF 必须是可选开关**：`daf=False` 时行为与原 CacheBlend 完全一致——这是对照实验的前提。
2. **后端必须显式指定**：DAF/CB 改动已覆盖 xformers 和 flash 两个后端，但**默认后端是 XFORMERS**；用 flash 跑实验必须显式 `VLLM_ATTENTION_BACKEND=FLASH_ATTN` 并确认日志显示 `Using FlashAttention backend`。两后端对比只在 SM/EM 级别对齐（kernel 浮点差可致贪心解码尾部 token 翻转，属正常）。
3. **评测先于跑批**：新实验先跑 1 个样本人工检查输出再放量。教训：`max_tokens=10` 曾把答案截成残句导致 EM 失效，整轮作废（作废数据归档在 `experiments/logs/archived_maxtok10截断作废/`，只归档不删除）。
4. **完成度看日志不看退出码**：vllm 0.4.1 引擎销毁阶段 exit=1 是良性的；续跑脚本以日志中 `===== ` 出现次数（24/12）判断。
5. 对比实验用同一 `recomp_ratio`（DAF 论文配置 0.3），贪心解码 `temperature=0`；测 vLLM 全量基线必须关 prefix caching；计时需 warmup、多轮取中位数；GPU 被其他程序（壁纸引擎/他人任务）占用时速度数据作废。
6. 短上下文（≤10k）DAF 慢于 CacheBlend 是预期现象（论文中 6k/9k 档二者持平，优势在 30k+），不得据此判定实现错误；单次功能测试也不要求 "DAF TTFT ≤ CacheBlend"。
7. 论文未写死的实现点（多层聚合公式、状态融合细节、选择层位置等）属**工程实现假设**，实验记录中必须注明，不得写成论文原方法。
8. vllm_blend 内改动最小化：已有改动限于 `llama.py`、`qwen2.py`、`xformers.py`、`flash_attn.py`、`selector.py`、`example/blend.py`；新改动同样保持最小。开发顺序固定——修基线 → status=3 只加评分 → status=1 改 DAF → status=2 接同一 helper → 验证 → 最后才优化性能。一次只改一件事；调试期保留 shape/NaN/Inf 断言。

## 当前进度速查（详见 `experiments/README.md` 与 `docs/daf1.1~1.3.md`）

**Qwen2.5-7B-Instruct 已支持（`BLEND_MODEL_PRESET=qwen`，xformers + flash 两后端，2026-09-17）**：2–4k 档 10 样本 × {XFORMERS, FLASH} × {DAF=0, DAF=1} 共 40 次运行零断言/NaN，两后端输出逐字一致（仅样本 7 flash-DAF1 答案略长，SM 不受影响）；10/10 答案全部正确。Qwen 答案是简洁精确格式（无前导空格、不啰嗦），EM 不再是恒 0。平均 TTFT（2–4k 档）：CB≈0.16s / DAF≈0.31–0.33s / 全量≈0.63s。

| 档位 | 样本 | 模型 | CacheBlend 加速 | DAF 加速 | DAF/CB 时长比 |
|---|---|---|---|---|---|
| 2–4k（daf1.1） | `inputs/1~10.json` | bf16 | 3.8× | 2.2× | 1.75× |
| 6–10k（daf1.2） | `inputs_long_6k10k_official/`（24） | bf16 | 4.0× | 2.3× | 1.73× |
| 30k（daf1.3） | `inputs_long_30k_buriedqa/`（12） | AWQ | 3.9× | 2.7× | **1.45×**（随长度收窄，符合论文趋势） |

DAF/CB 比值随长度单调下降（1.75→1.73→1.45），外推反超点在 60k–130k；DAF 相对全量加速（30k 档 2.71×）已与论文 2.8× 持平，未复现的是相对 CB 的反超和精度优势。与论文的差距（Qwen2.5-7B/A100/Flash-Attention vs 本项目 Mistral-7B/3090Ti/xformers）在 `docs/daf1.1.md` 有分析。

## 文档索引

- `docs/DAF改造计划_v2_校正版.md` —— 实现方案的权威文档（改动前先读）
- `docs/DAF论文解读与复现指南.md` —— 论文要求（模型/数据集/指标/验收标准）
- `docs/daf1.0.md` / `daf1.1.md` / `daf1.2.md` / `daf1.3.md` —— 各阶段实验记录与教训
- `docs/服务器复现指南.md` —— 环境搭建权威指南
- `experiments/README.md` —— 实验工作区索引与命令链
