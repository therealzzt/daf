"""Attention layer with xFormers and PagedAttention."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (AttentionBias,
                                         BlockDiagonalCausalMask,
                                         LowerTriangularMaskWithTensorBias,
                                         LowerTriangularFromBottomRightMask)

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata,
                                              AttentionMetadataPerStage)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)
from vllm.logger import init_logger

logger = init_logger(__name__)


class XFormersBackend(AttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["XFormersImpl"]:
        return XFormersImpl

    @staticmethod
    def make_metadata(*args, **kwargs) -> "XFormersMetadata":
        return XFormersMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class XFormersMetadata(AttentionMetadataPerStage, PagedAttentionMetadata):
    """Metadata for XFormersbackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    # (batch_size,). The prompt length per sequence. None if it is a decoding.
    prompt_lens: Optional[List[int]]
    # prompt_lens stored as a tensor.
    prompt_lens_tensor: Optional[torch.Tensor]

    # NOTE(sang): Definition of context_len, subquery_len, and seqlen.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seqlen ----------------------|
    #                                   |- subquery_len -|

    # WARNING(sang): context_len has different definition depending on if it is
    # prefill vs decoding. When it is prefill, it doesn't include new tokens.
    # When it is for decoding, it includes a new token.

    # Maximum subquery length in the batch.
    max_subquery_len: Optional[int]
    # FIXME: It is for flash attn.
    # Maximum prompt length in the batch.
    max_prompt_len: Optional[int]
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    subquery_start_loc: Optional[torch.Tensor]
    # FIXME: It is for flash attn.
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: Optional[List[AttentionBias]] = None


class XFormersImpl(AttentionImpl):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|	
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:	
    |<----------------- num_decode_tokens ------------------>|	
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
    |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        suppored_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata[XFormersMetadata],
        kv_scale: float,
        
        status: int,
        cache_fuse_metadata: dict,
        old_kv,
    ) -> torch.Tensor:
        """Forward pass with xFormers and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        num_tokens, hidden_size = query.shape
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        # Jiayi modified start
        # TODO(Jiayi): The following `view`s can be saved
        if status in [1, 2, 3]:
            # DAF status=3（打分层）默认只加载旧 V（V-only 打分）；
            # 开 kv_dev 后旧 K 也参与，但用的是 qwen2.py 里旋转到新坐标系后
            # 放在白板上的 key_old_rot（不是这里的原始 old_kv[0]——
            # 原始旧 K 没做 RoPE，坐标系对不上）
            value_old = old_kv[1].view(-1, self.num_kv_heads, self.head_size)
        if status in [1, 2]:
            key_old = old_kv[0].view(-1, self.num_kv_heads, self.head_size)

        # ===== DAF status=3：打分层，只收集证据，不改变任何计算 =====
        # 每个打分层把"本层新鲜 V"和"旧缓存 V"的偏差累加到白板的 dev_scores 上
        if status == 3:
            suffix_len = cache_fuse_metadata['suffix_len']
            doc_len = value.shape[0] - suffix_len
            assert suffix_len > 0 and doc_len > 0, "suffix_len/doc_len 异常"
            # temp_diff[i] = 第 i 个文档 token 的 ||V_new - V_old||²
            #（对所有 KV 头、所有维度求和，得到一个 token 一个分数）
            # 末尾 suffix_len 个是问题 token：问题没有旧缓存，无条件保留，不参与打分
            temp_diff = torch.sum((value[:doc_len,:,:]-value_old[:doc_len,:,:])**2, dim=[1,2])
            # 用 float32 累加，减少半精度（fp16）多次累加的舍入误差
            temp_diff = temp_diff.float()
            # kv_dev 开：K 也参与打分，dev = dev_V + dev_K（等权——工程假设，
            # 论文未写死 K 是否参与及加权方式，加权系数留作后续消融）。
            # key 是本层新鲜 K（已在 qwen2.py 里按新位置做完 RoPE），
            # key_old_rot 是旧 K 旋转到同一坐标系后的副本，
            # 两者 shape 都是 [doc_len, num_kv_heads, head_size]
            if cache_fuse_metadata.get("kv_dev", False):
                key_old_rot = cache_fuse_metadata["key_old_rot"].view(
                    -1, self.num_kv_heads, self.head_size)
                k_diff = torch.sum(
                    (key[:doc_len,:,:]-key_old_rot[:doc_len,:,:])**2,
                    dim=[1,2]).float()
                assert not torch.isnan(k_diff).any(), "K 偏差出现 NaN"
                temp_diff = temp_diff + k_diff
            if cache_fuse_metadata["dev_scores"] is None:
                cache_fuse_metadata["dev_scores"] = torch.zeros_like(temp_diff)
            cache_fuse_metadata["dev_scores"] += temp_diff
            # 注意：这里不做 topk、不裁剪序列、不改 positions——
            # 之后本层照常走完整因果注意力（status=3 不在 [1,2] 分支里）

            # 临时调试（blend.py 里 DAF_DEBUG_KVDEV=1 时白板带区间）：对照
            # "第一个文档块"对齐前/后的 K 偏差。第一块在 collect 与 check 阶段
            # 位置相同、左侧上下文也相同（因果掩码下它只看得见前缀和自己），
            # 理论上 K 应几乎不变：对齐后偏差≈浮点噪声；未对齐（拿没做 RoPE 的
            # 旧 K 直接减）会被位置旋转主导，应显著更大。验证用，后续可删
            _dbg = cache_fuse_metadata.get("debug_chunk1_range")
            if _dbg is not None and cache_fuse_metadata.get("kv_dev", False):
                _s, _e = _dbg
                _raw_old_k = old_kv[0].view(-1, self.num_kv_heads,
                                            self.head_size)
                _rot_old_k = cache_fuse_metadata["key_old_rot"].view(
                    -1, self.num_kv_heads, self.head_size)
                _d_unal = torch.sum(
                    (key[_s:_e]-_raw_old_k[_s:_e])**2, dim=[1,2]).float().mean()
                _d_alig = torch.sum(
                    (key[_s:_e]-_rot_old_k[_s:_e])**2, dim=[1,2]).float().mean()
                _d_rest = torch.sum(
                    (key[_e:doc_len]-_rot_old_k[_e:doc_len])**2,
                    dim=[1,2]).float().mean()
                _d_v = torch.sum(
                    (value[_s:_e]-value_old[_s:_e])**2, dim=[1,2]).float().mean()
                print(f"[KVDEV-DBG] 块1 K未对齐={_d_unal.item():.4f} "
                      f"K已对齐={_d_alig.item():.6f} V={_d_v.item():.6f} "
                      f"| 后续块K已对齐={_d_rest.item():.4f}")

        if status in [1]:
            if cache_fuse_metadata["daf"]:
                # ===== DAF status=1：选择层，聚合多层分数并选人 =====
                suffix_len = cache_fuse_metadata['suffix_len']
                total_len = value.shape[0]
                doc_len = total_len - suffix_len

                # 多层分数取平均（层数固定时，求和与求平均对 topk 排名等价；
                # 这里取平均只是让数值可读）——这是工程假设，论文未写死聚合公式
                scores = cache_fuse_metadata["dev_scores"] / len(cache_fuse_metadata["sel_layers"])
                assert scores.shape[0] == doc_len, "dev_scores 长度必须等于文档 token 数"

                # 只从文档部分选 topk：recomp_ratio 即论文的重要 token 比例 r
                #（论文实验用 r=0.3，它是可调配置而非固定常数）
                topk_num = int(doc_len * cache_fuse_metadata["recomp_ratio"])
                topk_num = max(1, min(doc_len, topk_num))  # 防止越界
                doc_imp_indices = torch.topk(scores, k=topk_num).indices
                # 按原位置排序，保证压缩后的序列仍按原文顺序排列
                doc_imp_indices = torch.sort(doc_imp_indices).values
                assert doc_imp_indices.ndim == 1 and doc_imp_indices.numel() == topk_num
                assert torch.all(doc_imp_indices < doc_len), "重要 token 不能落在问题段"

                # 问题段（末尾 suffix_len 个）无条件保留，其原始位置是 doc_len..total_len-1
                suffix_indices = torch.arange(doc_len, total_len,
                                              device=doc_imp_indices.device,
                                              dtype=torch.long)

                # 写回白板：深层和 llama.py 都靠这些字段
                cache_fuse_metadata["doc_imp_indices"] = doc_imp_indices
                cache_fuse_metadata["num_imp_tokens"] = doc_imp_indices.numel()
                # imp_indices = [重要文档 token 的原位置 | 问题 token 的原位置]，
                # 供 llama.py 裁剪 residual/positions 复用（顺序与输出拼接顺序一致）
                cache_fuse_metadata["imp_indices"] = torch.cat([doc_imp_indices,
                                                                suffix_indices], dim=0)
            else:
                # ===== 原版 CacheBlend 单层选人（保持不变）=====
                last_len = cache_fuse_metadata['suffix_len']
                total_len = value.shape[0]
                last_indices = [total_len-last_len+l for l in range(last_len)]

                topk_num = int((total_len-last_len)*cache_fuse_metadata["recomp_ratio"])
                temp_diff = torch.sum((value[:-last_len,:,:]-value_old[:-last_len,:,:])**2, dim=[1,2])
                top_indices = torch.topk(temp_diff, k=topk_num).indices

                top_indices, _ = torch.sort(top_indices)
                top_indices = torch.cat([top_indices,
                                            torch.tensor(last_indices, device=top_indices.device)])
                query = query[top_indices]
                cache_fuse_metadata["imp_indices"] = top_indices

                #attn_bias = _make_partial_bias_gqa(cache_fuse_metadata, query.device, self.num_kv_heads, self.num_queries_per_kv)
                attn_bias = LowerTriangularFromBottomRightMask()
                cache_fuse_metadata["attn_bias"] = attn_bias
                attn_metadata.prefill_metadata.attn_bias=None
            
        #if status in [0] and cache_fuse_metadata["original_slot_mapping"]==None and cache_fuse_metadata['check']:
        #    cache_fuse_metadata["original_slot_mapping"] = attn_metadata.slot_mapping#.clone()
        #    cache_fuse_metadata["key_shape"] = key.shape
        #    cache_fuse_metadata["value_shape"] = value.shape
        #    cache_fuse_metadata["kv_cache_string_dtype"] = "auto"
        cache_fuse_metadata["kv_cache_dtype"] = value.dtype
        
        # Jiayi: whether partial update or full update at check layer
        if status in [1]:
            imp_indices = cache_fuse_metadata["imp_indices"]
            #key_old[imp_indices] = key[imp_indices]
            #value_old[imp_indices] = value[imp_indices]
            #key = key_old
            #value = value_old
            
        
        if status in [1, 2] and cache_fuse_metadata["daf"]:
            # ===== DAF：组装"混合 KV"（hybrid KV）=====
            # 混合 KV 的三段来源：
            #   未被选中的文档 token -> 离线旧 KV（不用重算，省钱的关键）
            #   被选中的文档 token   -> 本层 QKV 投影刚算出的新鲜 KV
            #   问题 token           -> 本层的新鲜 KV（问题从不缓存）
            #
            # 注意：clone 出副本再改，不原地修改 old_kv——
            # 否则旧缓存会被"当前问题"污染，下个问题复用时就错了
            #（clone 有性能开销，正确性原型阶段可接受，优化阶段再换成按请求分配的缓冲区）
            key_hybrid = key_old.clone()
            value_hybrid = value_old.clone()
            doc_imp = cache_fuse_metadata["doc_imp_indices"]
            suffix_len = cache_fuse_metadata['suffix_len']
            # key_old 的长度是原始完整序列（文档+问题），文档部分长度 = 总长 - 问题长
            doc_len = key_old.shape[0] - suffix_len

            if status == 1:
                # 选择层：输入还是完整序列，新鲜值按原始位置取
                key_hybrid[doc_imp] = key[doc_imp]
                value_hybrid[doc_imp] = value[doc_imp]
                key_hybrid[doc_len:] = key[doc_len:]
                value_hybrid[doc_len:] = value[doc_len:]
                # Q 砍成压缩序列 [重要文档 token | 问题 token]，
                # 从此之后本层输出和后续所有层都只处理这条短序列
                query = query[cache_fuse_metadata["imp_indices"]]
            else:
                # status == 2：输入已经是压缩序列 [重要文档 token | 问题 token]
                num_imp = cache_fuse_metadata["num_imp_tokens"]
                key_hybrid[doc_imp] = key[:num_imp]
                value_hybrid[doc_imp] = value[:num_imp]
                key_hybrid[doc_len:] = key[num_imp:]
                value_hybrid[doc_len:] = value[num_imp:]

            # key/value 换成混合版：后面写 GPU KV cache 的就是它，
            # 这样 decode 阶段用到的也是融合后的笔记
            key = key_hybrid
            value = value_hybrid
        elif status in [2]:
            # ===== 原版 CacheBlend 的融合（保持不变）=====
            imp_indices = cache_fuse_metadata["imp_indices"]
            key_old[imp_indices] = key
            value_old[imp_indices] = value
            key = key_old
            value = value_old
        
        if kv_cache is not None:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)
            

            #if status in [2]:
                # TODO(Jiayi): This is full update/ Do we need partial update? 
            #    PagedAttention.write_to_paged_cache(key, value, key_cache,
            #                                        value_cache,
            #                                        cache_fuse_metadata["our_slot_mapping"],
            #                                        attn_metadata.kv_cache_dtype,
            #                                        kv_scale)
            #else:
            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            #    PagedAttention.write_to_paged_cache(key, value, key_cache,
            #                                        value_cache,
            #                                        attn_metadata.slot_mapping,
            #                                        attn_metadata.kv_cache_dtype,
            #                                        kv_scale)
            PagedAttention.write_to_paged_cache(key, value, key_cache,
                                                value_cache,
                                                attn_metadata.slot_mapping,
                                                attn_metadata.kv_cache_dtype,
                                                kv_scale)
        
        
        # TODO(Jiayi): need to change num prefill tokens accordingly
        if status in [1,2]:
            num_prefill_tokens = attn_metadata.num_prefill_tokens
            num_decode_tokens = attn_metadata.num_decode_tokens
            output = torch.empty_like(query)
            decode_query = None
            query = query
            key = key[:num_prefill_tokens]
            value = value[:num_prefill_tokens]
            
            assert query.shape[0] == len(cache_fuse_metadata["imp_indices"])
            #assert decode_query.shape[0] == num_decode_tokens
        else:
            num_prefill_tokens = attn_metadata.num_prefill_tokens
            num_decode_tokens = attn_metadata.num_decode_tokens
            assert key.shape[0] == num_prefill_tokens + num_decode_tokens
            assert value.shape[0] == num_prefill_tokens + num_decode_tokens

            output = torch.empty_like(query)
            # Query for decode. KV is not needed because it is already cached.
            decode_query = query[num_prefill_tokens:]
            # QKV for prefill.
            query = query[:num_prefill_tokens]
            key = key[:num_prefill_tokens]
            value = value[:num_prefill_tokens]

            assert query.shape[0] == num_prefill_tokens
            assert decode_query.shape[0] == num_decode_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if kv_cache is None or prefill_meta.block_tables.numel() == 0:
                # normal attention.
                # block tables are empty if the prompt does not have a cached
                # prefix.
                
                if status in [1, 2] and cache_fuse_metadata["daf"]:
                    # DAF 压缩层：走两路注意力（重要 token 互看 + 问题看全文）
                    out = self._run_daf_attention(
                        query, key, value, cache_fuse_metadata)
                else:
                    out = self._run_memory_efficient_xformers_forward(
                        query, key, value, prefill_meta, status, cache_fuse_metadata)
                
                #assert out.shape == output[:num_prefill_tokens].shape
                #output[:num_prefill_tokens] = out
                output = out
            else:
                # prefix-enabled attention
                # TODO(Hai) this triton kernel has regression issue (broke) to
                # deal with different data types between KV and FP8 KV cache,
                # to be addressed separately.
                out = PagedAttention.forward_prefix(
                    query,
                    key,
                    value,
                    key_cache,
                    value_cache,
                    prefill_meta.block_tables,
                    prefill_meta.subquery_start_loc,
                    prefill_meta.prompt_lens_tensor,
                    prefill_meta.context_lens,
                    prefill_meta.max_subquery_len,
                    self.alibi_slopes,
                )
                assert output[:num_prefill_tokens].shape == out.shape
                output[:num_prefill_tokens] = out

        if decode_meta := attn_metadata.decode_metadata:
            output[num_prefill_tokens:] = PagedAttention.forward_decode(
                decode_query,
                key_cache,
                value_cache,
                decode_meta.block_tables,
                decode_meta.context_lens,
                decode_meta.max_context_len,
                attn_metadata.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                kv_scale,
            )

        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)

    def _dense_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias,
    ) -> torch.Tensor:
        """DAF 用的稠密注意力（dense attention，即不算 KV cache 分页、
        直接对连续内存里的 Q/K/V 做一次完整注意力）。

        职责（复用 _run_memory_efficient_xformers_forward 里的同款写法）：
        1. 处理 GQA（Grouped-Query Attention，分组查询注意力：Mistral-7B 的
           Q 有 32 个头，K/V 只有 8 个头，要把 K/V 的头"复制"4 倍对齐 Q）；
        2. 补上 batch 维（xformers 的算子要求输入带 batch 维，这里恒为 1）；
        3. 调 xops.memory_efficient_attention_forward；
        4. 把输出形状还原成 [token 数, Q 头数, 每头维度]。

        attn_bias=None 表示不加任何掩码（双向互相可见）；
        attn_bias=LowerTriangularFromBottomRightMask() 表示因果掩码
        （causal mask：每个位置只能看到自己和之前的 token，不能偷看未来）。
        """
        if self.num_kv_heads != self.num_heads:
            # GQA：把 query 拆出 [KV头数, 每组的Q头数] 两维，
            # 并把 K/V 沿"每组的Q头数"维 expand（逻辑复制，不占额外显存）
            query = query.view(query.shape[0], self.num_kv_heads,
                               self.num_queries_per_kv, query.shape[-1])
            key = key[:, :,
                      None, :].expand(key.shape[0], self.num_kv_heads,
                                      self.num_queries_per_kv, key.shape[-1])
            value = value[:, :,
                          None, :].expand(value.shape[0], self.num_kv_heads,
                                          self.num_queries_per_kv,
                                          value.shape[-1])
        # 加 batch 维：[token数, ...] -> [1, token数, ...]
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
        out = xops.memory_efficient_attention_forward(
            query,
            key,
            value,
            attn_bias=attn_bias,
            p=0.0,
            scale=self.scale,
        )
        # 去掉 batch 维
        out = out.squeeze(0)
        if self.num_kv_heads != self.num_heads:
            # 把 [token数, KV头数, 每组Q头数, 每头维度] 合并回
            # [token数, Q头数, 每头维度]
            out = out.view(out.shape[0], self.num_heads, self.head_size)
        return out

    def _run_daf_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cache_fuse_metadata: dict,
    ) -> torch.Tensor:
        """DAF 的核心：每个压缩层跑一次"两路注意力 + 状态融合"。

        输入约定：
          query: [num_imp + suffix_len, Q头数, 每头维度]，压缩后的 Q
          key/value: [原始完整序列长, KV头数, 每头维度]，上一步组装好的混合 KV
            （未选中的文档 token 用旧缓存，选中的文档 token 和问题 token 用新鲜值）

        数据流（对应论文的三步）：
          A. 重要文档 token 之间互相自注意力（无掩码，双向可见）
             ——复杂度 O((r*Ld)^2)，这是 DAF 比 CacheBlend 省的关键
          B. 问题 token 对"全文混合 KV"做因果注意力（不能偷看后面的问题 token）
          C. State Fusion：把两路输出按 [重要文档 | 问题] 的顺序拼起来，
             作为下一层的压缩 hidden states
        """
        num_imp = cache_fuse_metadata["num_imp_tokens"]
        suffix_len = cache_fuse_metadata["suffix_len"]
        doc_imp = cache_fuse_metadata["doc_imp_indices"]
        doc_len = key.shape[0] - suffix_len

        # 形状断言：宁可立刻报错，也不要静默算错
        assert suffix_len > 0 and doc_len > 0
        assert query.shape[0] == num_imp + suffix_len, \
            "压缩 Q 的长度必须等于 重要token数 + 问题token数"
        assert key.shape[0] == doc_len + suffix_len

        # 切 Q：前 num_imp 个是重要文档 token，其余是问题 token
        q_imp = query[:num_imp]
        q_suf = query[num_imp:]

        # A. 重要 token 自注意力。它们的 K/V 直接从混合 KV 里按原始位置取
        #（上一步已把新鲜值写进这些位置），双向无掩码
        k_imp = key[doc_imp]
        v_imp = value[doc_imp]
        out_imp = self._dense_attention(q_imp, k_imp, v_imp, attn_bias=None)

        # B. 问题对全文（混合 KV：旧文档 + 新重要 + 新问题）做因果注意力。
        # LowerTriangularFromBottomRightMask：Q 是尾部连续段时的因果掩码，
        # 效果 = 每个问题 token 能看全部文档和自己之前的问题 token，
        # 但不能偷看排在后面的问题 token
        out_suf = self._dense_attention(q_suf, key, value,
                                        attn_bias=LowerTriangularFromBottomRightMask())

        # C. State Fusion：拼接两路输出，顺序与 imp_indices 保持一致
        out = torch.cat([out_imp, out_suf], dim=0)

        assert out_imp.shape[0] == num_imp and out_suf.shape[0] == suffix_len
        assert out.shape[0] == num_imp + suffix_len
        assert not torch.isnan(out).any() and not torch.isinf(out).any(), \
            "DAF 注意力输出出现 NaN/Inf"
        return out

    def _run_memory_efficient_xformers_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: XFormersMetadata,
        status,
        cache_fuse_metadata,
    ) -> torch.Tensor:
        """Attention for 1D query of multiple prompts. Multiple prompt
        tokens are flattened in to `query` input.

        See https://facebookresearch.github.io/xformers/components/ops.html
        for API spec.

        Args:
            output: shape = [num_prefill_tokens, num_heads, head_size]
            query: shape = [num_prefill_tokens, num_heads, head_size]
            key: shape = [num_prefill_tokens, num_kv_heads, head_size]
            value: shape = [num_prefill_tokens, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        """
        assert attn_metadata.prompt_lens is not None
        original_query = query
        if self.num_kv_heads != self.num_heads:
            # GQA/MQA requires the shape [B, M, G, H, K].
            # Note that the output also has the same shape (which is different
            # from a spec from the doc).
            query = query.view(query.shape[0], self.num_kv_heads,
                               self.num_queries_per_kv, query.shape[-1])
            key = key[:, :,
                      None, :].expand(key.shape[0], self.num_kv_heads,
                                      self.num_queries_per_kv, key.shape[-1])
            value = value[:, :,
                          None, :].expand(value.shape[0], self.num_kv_heads,
                                          self.num_queries_per_kv,
                                          value.shape[-1])
        # Set attention bias if not provided. This typically happens at
        # the very attention layer of every iteration.
        # FIXME(woosuk): This is a hack.
        if attn_metadata.attn_bias is None:
            if self.alibi_slopes is None:
                # TODO(Jiayi): please pre-allocate mask for faster inference
                #if cache_fuse_metadata["check"]:
                #    attn_metadata.attn_bias = _fetch_maetrailized_mask_gqa(query.shape[0], self.num_kv_heads, self.num_queries_per_kv, query.device,query.dtype)
                #else:
                attn_metadata.attn_bias = BlockDiagonalCausalMask.from_seqlens(
                                                attn_metadata.prompt_lens)
                '''
                attn_bias = BlockDiagonalCausalMask.from_seqlens(
                    attn_metadata.prompt_lens)
                if self.sliding_window is not None:
                    attn_bias = attn_bias.make_local_attention(
                        self.sliding_window)
                attn_metadata.attn_bias = [attn_bias]
                '''
            else:
                attn_metadata.attn_bias = _make_alibi_bias(
                    self.alibi_slopes, self.num_kv_heads, query.dtype,
                    attn_metadata.prompt_lens)

        # No alibi slopes.
        # TODO(woosuk): Too many view operations. Let's try to reduce
        # them in the future for code readability.
        if self.alibi_slopes is None:
            # Add the batch dimension.
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            
            if status in [1,2]:
                #import pdb
                #pdb.set_trace()
                out = xops.memory_efficient_attention_forward(
                        query,
                        key,
                        value,
                        attn_bias=cache_fuse_metadata["attn_bias"],
                        p=0.0,
                        scale=self.scale,
                    )
            else:
                out = xops.memory_efficient_attention_forward(
                query,
                key,
                value,
                attn_bias=attn_metadata.attn_bias,
                p=0.0,
                scale=self.scale)
            '''
            out = xops.memory_efficient_attention_forward(
                query,
                key,
                value,
                attn_bias=attn_metadata.attn_bias[0],
                p=0.0,
                scale=self.scale)
            '''
            
            return out.view_as(original_query)

        # Attention with alibi slopes.
        # FIXME(woosuk): Because xformers does not support dynamic sequence
        # lengths with custom attention bias, we process each prompt one by
        # one. This is inefficient, especially when we have many short prompts.
        output = torch.empty_like(original_query)
        start = 0
        for i, prompt_len in enumerate(attn_metadata.prompt_lens):
            end = start + prompt_len
            out = xops.memory_efficient_attention_forward(
                query[None, start:end],
                key[None, start:end],
                value[None, start:end],
                attn_bias=attn_metadata.attn_bias[i],
                p=0.0,
                scale=self.scale)
            # TODO(woosuk): Unnecessary copy. Optimize.
            output[start:end].copy_(out.view_as(original_query[start:end]))
            start += prompt_len
        return output


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    dtype: torch.dtype,
    prompt_lens: List[int],
) -> LowerTriangularMaskWithTensorBias:
    attn_biases = []
    for prompt_len in prompt_lens:
        bias = torch.arange(prompt_len, dtype=dtype)
        # NOTE(zhuohan): HF uses
        #     `bias = bias[None, :].repeat(prompt_len, 1)`
        # here. We find that both biases give the same results, but
        # the bias below more accurately follows the original ALiBi
        # paper.
        # Calculate a matrix where each element represents ith element- jth
        # element.
        bias = bias[None, :] - bias[:, None]

        padded_len = (prompt_len + 7) // 8 * 8
        num_heads = alibi_slopes.shape[0]
        bias = torch.empty(
            1,  # batch size
            num_heads,
            prompt_len,
            padded_len,
            device=alibi_slopes.device,
            dtype=dtype,
        )[:, :, :, :prompt_len].copy_(bias)
        bias.mul_(alibi_slopes[:, None, None])
        if num_heads != num_kv_heads:
            bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))
        attn_biases.append(LowerTriangularMaskWithTensorBias(bias))

    return attn_biases

def _make_partial_bias_gqa(cache_fuse_metadata, 
                       device,
                       num_kv_heads,
                       num_queries_per_kv,):
    seq_len = cache_fuse_metadata['org_seq_len']
    padded_len = (seq_len + 7) // 8 * 8
    dtype = cache_fuse_metadata['kv_cache_dtype']
    imp_indices = cache_fuse_metadata['imp_indices']
    attn_mask = torch.triu(torch.ones(padded_len,
                                      padded_len,
                                      dtype=dtype,
                                      device=device),
                           diagonal=1)
    #FIXME(Jiayi): The first 1 (bsz) is a hack
    attn_mask = (attn_mask * torch.finfo(dtype).min).view(1, 
                                                          1, 1, padded_len, padded_len) #FIXME(Jiayi): Now only focus on bsz=1
    attn_mask = attn_mask[:,:,:,imp_indices]
    attn_mask = attn_mask.expand(1,
                                 num_kv_heads,num_queries_per_kv,-1,-1)
    #import pdb
    #pdb.set_trace()
    attn_mask_padded = torch.empty(
        1,
        num_kv_heads,
        num_queries_per_kv,
        len(imp_indices),
        padded_len,
        device=device,
        dtype=dtype,
    ).copy_(attn_mask)[:, :, :, :, :seq_len]
    #attn_mask_padded = LowerTriangularMaskWithTensorBias(attn_mask_padded)
    return attn_mask_padded

def _fetch_maetrailized_mask_gqa(q_len,num_kv_heads,num_queries_per_kv,device,dtype):
    seq_len = q_len
    padded_len = (seq_len + 7) // 8 * 8
    attn_mask = torch.triu(torch.ones(seq_len,
                                      padded_len,
                                      dtype=dtype,
                                      device=device),
                           diagonal=1)
    #FIXME(Jiayi): The first 1 (bsz) is a hack
    attn_mask = (attn_mask * torch.finfo(dtype).min).view(#1,
                                                          1, 1, seq_len, padded_len) #FIXME(Jiayi): Now only focus on bsz=1
    attn_mask = attn_mask.expand(#1,
                                 num_kv_heads, num_queries_per_kv,-1,-1)
    
    attn_mask_padded = torch.empty(
        #1,
        num_kv_heads,
        num_queries_per_kv,
        seq_len,
        padded_len,
        device=device,
        dtype=dtype,
    ).copy_(attn_mask)[#:, 
                       :, :, :, :seq_len]
    #attn_mask_padded = LowerTriangularMaskWithTensorBias(attn_mask_padded)
    return attn_mask_padded