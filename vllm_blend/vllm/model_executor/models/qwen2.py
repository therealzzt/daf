# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import Qwen2Config

from vllm.attention import Attention, AttentionMetadata
from vllm.config import LoRAConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput


class Qwen2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            linear_method=linear_method)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           linear_method=linear_method)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen2Attention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 10000,
                 use_sliding_window: bool = False,
                 linear_method: Optional[LinearMethodBase] = None,
                 sliding_window: Optional[int] = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window if use_sliding_window else None
        self.kv_scale = 1.0

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            linear_method=linear_method,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              sliding_window=self.sliding_window)

        self.hack_kv = []

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,

        status,
        cache_fuse_metadata,
        old_kv,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # HACK(Jiayi): Rotate the old K
        # Need to modify the kernel to only take K as input
        if status in [1,2]:
            if cache_fuse_metadata["fake_q"] is None:
                # fake_q 只是给 rotary 算子占位的假 Q（输出丢弃），
                # 用 old_kv[0]（K，[tokens, kv_size]）的形状而不是 q 的：
                # q 是 28 头、k 只有 4 头（GQA），省 ~200MB 显存；
                # rotary 算子对 Q/K 头数没有一致性要求（GQA 本来不同）
                cache_fuse_metadata['fake_q'] = torch.rand_like(old_kv[0])
            _, old_kv[0] = self.rotary_emb(cache_fuse_metadata['org_pos'],
                                        cache_fuse_metadata['fake_q'],
                                        old_kv[0])

        # ===== DAF K+V 打分（可选开关 kv_dev，默认关）=====
        # status=3 打分层默认只对 V 算偏差（V 不带位置信息，新旧可直接比）。
        # 开 kv_dev 后 K 也参与打分，但旧 K 是 collect 阶段存的 RoPE 旋转【前】
        # 原始值（见下方 collect 分支），而本层新鲜 K 稍后会按 check 阶段的
        # 新位置做 RoPE 旋转（RoPE = 旋转位置编码：给每个 token 的 Q/K 按其
        # 绝对位置做旋转变换，位置不同则旋转角不同）——两个坐标系不一致，
        # 直接相减会被位置旋转主导。所以这里用与上面 status∈[1,2] 相同的
        # 对齐方式：把旧 K 按 org_pos（check 阶段全序列的绝对位置）补旋转。
        # 注意：vllm 的 rotary_embedding 算子是【原地】改写输入的，
        # 必须先 clone 再旋转，绝不能污染共享的 old_kv[0]（它还要给别的用途用）；
        # 旋转结果放白板字段 key_old_rot，由本层的注意力后端（xformers/flash）
        # 在 status=3 分支里读取。fake_q 是个形状占位的假 Q——算子要求 Q/K
        # 成对输入，我们只需要 K 的输出，Q 输出丢弃
        if status == 3 and cache_fuse_metadata.get("kv_dev", False):
            if cache_fuse_metadata["fake_q"] is None:
                cache_fuse_metadata['fake_q'] = torch.rand_like(old_kv[0])
            _, key_old_rot = self.rotary_emb(cache_fuse_metadata['org_pos'],
                                             cache_fuse_metadata['fake_q'],
                                             old_kv[0].clone())
            cache_fuse_metadata['key_old_rot'] = key_old_rot

        if cache_fuse_metadata['collect']:
            self.hack_kv = [k.clone(), v.clone()]
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata,
                                status, cache_fuse_metadata, old_kv,
                                self.kv_scale)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        layer_idx: int,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        use_sliding_window = (config.use_sliding_window
                              and layer_idx < config.max_window_layers)
        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            use_sliding_window=use_sliding_window,
            linear_method=linear_method,
            sliding_window=config.sliding_window)
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            linear_method=linear_method,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],

        status: int,
        cache_fuse_metadata: dict,
        old_kv,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,

            # Jiayi: modified start
            status=status,
            cache_fuse_metadata=cache_fuse_metadata,
            old_kv=old_kv,
            # Jiayi: modified end
        )

        if status == 1:
            residual = residual[cache_fuse_metadata["imp_indices"]]

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2Model(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(config, layer_idx, linear_method)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.cache_fuse_metadata = {"check_layers":[1],
                                    "check": False,
                                    "recomp_ratios":[0.16],
                                    "recomp_ratio":0.16,
                                    "original_slot_mapping":None,
                                    "our_slot_mapping":None,
                                    "kv_cache_dtype": None,
                                    "attn_bias": None,
                                    "imp_indices": None,
                                    "org_seq_len": None,
                                    "collect": False,

                                    # ===== 以下为 DAF 新增字段（daf=False 时全部不生效） =====
                                    # 字段语义与 llama.py 完全一致
                                    "daf": False,
                                    "sel_layers": [1, 2, 3],
                                    "dev_scores": None,
                                    "doc_imp_indices": None,
                                    "num_imp_tokens": 0,
                                    # kv_dev：K+V 联合打分开关（默认 False = 只对 V 打分，
                                    # 与历史实验可比）；key_old_rot：打分层里旋转到
                                    # 新坐标系的旧 K（每层覆盖，用完即弃）
                                    "kv_dev": False,
                                    "key_old_rot": None}

        self.old_kvs = [[None,None]] * len(self.layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        if attn_metadata.prefill_metadata:
            temp_status = 0 # full prefill
            if self.cache_fuse_metadata["check"]:
                self.cache_fuse_metadata["org_seq_len"] = input_ids.shape[0]
                check_layer_idx = 0
                self.cache_fuse_metadata["fake_q"] = None
                self.cache_fuse_metadata["attn_bias"] = None
                self.cache_fuse_metadata["imp_indices"] = None
                self.cache_fuse_metadata["original_slot_mapping"] = None
                self.cache_fuse_metadata["our_slot_mapping"] = None
                self.cache_fuse_metadata['org_pos'] = positions[:]
                # DAF：清掉上一次 check 残留的打分和选人结果。
                # 否则下一个样本会在旧分数上继续累加，选人就全错了
                self.cache_fuse_metadata["dev_scores"] = None
                self.cache_fuse_metadata["doc_imp_indices"] = None
                self.cache_fuse_metadata["num_imp_tokens"] = 0
                # key_old_rot 是上一层/上一样本打分层留下的旋转旧 K，
                # 同样不能跨样本残留（每层打分会现算现覆盖，这里清空兜底）
                self.cache_fuse_metadata["key_old_rot"] = None
        else:
            temp_status = -1 # decode
        residual = None


        for i in range(len(self.layers)):

            if self.cache_fuse_metadata["check"]:
                if self.cache_fuse_metadata["daf"]:
                    # ===== DAF 排班（sel_layers=[1,2,3] 时）=====
                    # 第 0 层:      status=0 普通完整 prefill
                    # 第 1~3 层:    status=3 完整 prefill + 累积本层 V 偏差分数
                    # 第 4 层:      status=1 根据累计分数选重要 token，进入压缩模式
                    # 第 5 层及以后: status=2 压缩模式，每层跑 DAF 两路注意力
                    sel_layers = self.cache_fuse_metadata["sel_layers"]
                    # 选择层 = 最后一个打分层 + 1（打分层收完证据，下一层才选人）
                    select_layer = max(sel_layers) + 1
                    assert select_layer < len(self.layers), \
                        "sel_layers 太大，选择层超出了模型层数"
                    if i in sel_layers:
                        temp_status = 3 # 打分层：只收集证据，不做选择
                    elif i == select_layer:
                        temp_status = 1 # 选择层：选人 + 开始 DAF 压缩
                        # check_layer 字段保留原语义（目前无读者，仅作记录）
                        self.cache_fuse_metadata["check_layer"] = select_layer
                    elif i > select_layer:
                        temp_status = 2 # 后续层：继续 DAF 压缩
                    else:
                        temp_status = 0 # 打分层之前的层：普通完整 prefill
                else:
                    # ===== 原版 CacheBlend 排班（保持不变）=====
                    if i in self.cache_fuse_metadata["check_layers"]:
                        temp_status = 1 # check this layer
                        self.cache_fuse_metadata["check_layer"] = self.cache_fuse_metadata["check_layers"][check_layer_idx]
                        check_layer_idx += 1
                    elif i > self.cache_fuse_metadata["check_layers"][0]:
                        temp_status = 2 # after check

            old_kv = self.old_kvs[i]

            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,

                status = temp_status,
                cache_fuse_metadata = self.cache_fuse_metadata,
                old_kv = old_kv
            )

            if temp_status==1:
                positions = positions[self.cache_fuse_metadata["imp_indices"]]
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen2ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(
        self,
        config: Qwen2Config,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        del lora_config
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = Qwen2Model(config, linear_method)

        if config.tie_word_embeddings:
            self.lm_head_weight = self.model.embed_tokens.weight
        else:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.hidden_size)
            self.lm_head_weight = self.lm_head.weight

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head_weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
