"""Attention layer with Flash and PagedAttention.

NOTE(woosuk): At the moment, this file includes a lot of duplicated code from
XFormers backend. The duplicated code will be removed once we use flash-attn or
flashinfer for all the attention operations.

NOTE(DAF port): the CacheBlend/DAF cache_fuse logic is ported from
vllm.attention.backends.xformers. All attention computations that used
xformers ops are replaced by flash_attn_varlen_func:
  - normal causal prefill            -> causal=True（与原生一致）
  - DAF 重要 token 互看（无掩码）     -> causal=False
  - DAF 问题段对 hybrid KV / CB 选择层 -> causal=True，q_len < kv_len 时
    flash-attn 的 causal mask 右下对齐，与 xformers 的
    LowerTriangularFromBottomRightMask 等价（已用 /tmp 小张量数值验证）。
GQA（kv 头数 < q 头数）由 flash-attn 原生支持，不再像 xformers 后端那样
手动 expand KV 头。
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import torch
from flash_attn import flash_attn_varlen_func

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata,
                                              AttentionMetadataPerStage)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)


class FlashAttentionBackend(AttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def make_metadata(*args, **kwargs) -> "FlashAttentionMetadata":
        return FlashAttentionMetadata(*args, **kwargs)

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
class FlashAttentionMetadata(AttentionMetadataPerStage,
                             PagedAttentionMetadata):
    """Metadata for FlashAttentionBackend.

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
    # Maximum prompt length in the batch.
    max_prompt_len: Optional[int]
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    subquery_start_loc: Optional[torch.Tensor]
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length
    # is [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool


class FlashAttentionImpl(AttentionImpl):
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
        self.sliding_window = ((sliding_window, sliding_window)
                               if sliding_window is not None else (-1, -1))
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
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata[FlashAttentionMetadata],
        kv_scale: float,

        status: int,
        cache_fuse_metadata: dict,
        old_kv,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention and PagedAttention.

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
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        # ===== 以下 cache_fuse/DAF 控制流与 xformers 后端逐行对应 =====
        if status in [1, 2, 3]:
            # DAF 的 status=3（打分层）只需要旧 V 来算偏差，不需要旧 K
            value_old = old_kv[1].view(-1, self.num_kv_heads, self.head_size)
        if status in [1, 2]:
            key_old = old_kv[0].view(-1, self.num_kv_heads, self.head_size)

        # ===== DAF status=3：打分层，只收集证据，不改变任何计算 =====
        if status == 3:
            suffix_len = cache_fuse_metadata['suffix_len']
            doc_len = value.shape[0] - suffix_len
            assert suffix_len > 0 and doc_len > 0, "suffix_len/doc_len 异常"
            # temp_diff[i] = 第 i 个文档 token 的 ||V_new - V_old||²
            temp_diff = torch.sum(
                (value[:doc_len, :, :] - value_old[:doc_len, :, :])**2,
                dim=[1, 2])
            # 用 float32 累加，减少半精度多次累加的舍入误差
            temp_diff = temp_diff.float()
            if cache_fuse_metadata["dev_scores"] is None:
                cache_fuse_metadata["dev_scores"] = torch.zeros_like(temp_diff)
            cache_fuse_metadata["dev_scores"] += temp_diff
            # 不做 topk、不裁剪序列——本层之后照常走完整因果注意力

        if status in [1]:
            if cache_fuse_metadata["daf"]:
                # ===== DAF status=1：选择层，聚合多层分数并选人 =====
                suffix_len = cache_fuse_metadata['suffix_len']
                total_len = value.shape[0]
                doc_len = total_len - suffix_len

                # 多层分数取平均（层数固定时与求和对 topk 排名等价）——
                # 工程假设，论文未写死聚合公式
                scores = cache_fuse_metadata["dev_scores"] / len(
                    cache_fuse_metadata["sel_layers"])
                assert scores.shape[0] == doc_len, \
                    "dev_scores 长度必须等于文档 token 数"

                # 只从文档部分选 topk（recomp_ratio 即论文的重要 token 比例 r）
                topk_num = int(doc_len * cache_fuse_metadata["recomp_ratio"])
                topk_num = max(1, min(doc_len, topk_num))  # 防止越界
                doc_imp_indices = torch.topk(scores, k=topk_num).indices
                # 按原位置排序，保证压缩后的序列仍按原文顺序排列
                doc_imp_indices = torch.sort(doc_imp_indices).values
                assert doc_imp_indices.ndim == 1 and \
                    doc_imp_indices.numel() == topk_num
                assert torch.all(doc_imp_indices < doc_len), \
                    "重要 token 不能落在问题段"

                # 问题段（末尾 suffix_len 个）无条件保留
                suffix_indices = torch.arange(doc_len, total_len,
                                              device=doc_imp_indices.device,
                                              dtype=torch.long)

                # 写回白板：深层和 llama.py 都靠这些字段
                cache_fuse_metadata["doc_imp_indices"] = doc_imp_indices
                cache_fuse_metadata["num_imp_tokens"] = doc_imp_indices.numel()
                # imp_indices = [重要文档 token 的原位置 | 问题 token 的原位置]，
                # 供 llama.py 裁剪 residual/positions 复用
                cache_fuse_metadata["imp_indices"] = torch.cat(
                    [doc_imp_indices, suffix_indices], dim=0)
            else:
                # ===== 原版 CacheBlend 单层选人（与 xformers 一致）=====
                last_len = cache_fuse_metadata['suffix_len']
                total_len = value.shape[0]
                last_indices = [total_len - last_len + l
                                for l in range(last_len)]

                topk_num = int((total_len - last_len) *
                               cache_fuse_metadata["recomp_ratio"])
                temp_diff = torch.sum(
                    (value[:-last_len, :, :] - value_old[:-last_len, :, :])**2,
                    dim=[1, 2])
                top_indices = torch.topk(temp_diff, k=topk_num).indices

                top_indices, _ = torch.sort(top_indices)
                top_indices = torch.cat([
                    top_indices,
                    torch.tensor(last_indices, device=top_indices.device)
                ])
                query = query[top_indices]
                cache_fuse_metadata["imp_indices"] = top_indices
                # flash-attn 不需要显式 attn_bias：压缩 Q 对全长 KV 的
                # 因果掩码用 causal=True（右下对齐）表达，见下方调用点

        cache_fuse_metadata["kv_cache_dtype"] = value.dtype

        if status in [1, 2] and cache_fuse_metadata["daf"]:
            # ===== DAF：组装"混合 KV"（hybrid KV）=====
            #   未被选中的文档 token -> 离线旧 KV
            #   被选中的文档 token   -> 本层新鲜 KV
            #   问题 token           -> 本层新鲜 KV（问题从不缓存）
            # clone 出副本再改，禁止原地修改 old_kv（否则旧缓存被当前问题污染）
            key_hybrid = key_old.clone()
            value_hybrid = value_old.clone()
            doc_imp = cache_fuse_metadata["doc_imp_indices"]
            suffix_len = cache_fuse_metadata['suffix_len']
            doc_len = key_old.shape[0] - suffix_len

            if status == 1:
                # 选择层：输入还是完整序列，新鲜值按原始位置取
                key_hybrid[doc_imp] = key[doc_imp]
                value_hybrid[doc_imp] = value[doc_imp]
                key_hybrid[doc_len:] = key[doc_len:]
                value_hybrid[doc_len:] = value[doc_len:]
                # Q 砍成压缩序列 [重要文档 token | 问题 token]
                query = query[cache_fuse_metadata["imp_indices"]]
            else:
                # status == 2：输入已经是压缩序列 [重要文档 token | 问题 token]
                num_imp = cache_fuse_metadata["num_imp_tokens"]
                key_hybrid[doc_imp] = key[:num_imp]
                value_hybrid[doc_imp] = value[:num_imp]
                key_hybrid[doc_len:] = key[num_imp:]
                value_hybrid[doc_len:] = value[num_imp:]

            # key/value 换成混合版：写入 GPU KV cache 的也是它
            key = key_hybrid
            value = value_hybrid
        elif status in [2]:
            # ===== 原版 CacheBlend 的融合（与 xformers 一致）=====
            imp_indices = cache_fuse_metadata["imp_indices"]
            key_old[imp_indices] = key
            value_old[imp_indices] = value
            key = key_old
            value = value_old

        if kv_cache is not None:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            PagedAttention.write_to_paged_cache(key, value, key_cache,
                                                value_cache,
                                                attn_metadata.slot_mapping,
                                                attn_metadata.kv_cache_dtype,
                                                kv_scale)

        if status in [1, 2]:
            num_prefill_tokens = attn_metadata.num_prefill_tokens
            num_decode_tokens = attn_metadata.num_decode_tokens
            output = torch.empty_like(query)
            decode_query = None
            key = key[:num_prefill_tokens]
            value = value[:num_prefill_tokens]

            assert query.shape[0] == len(cache_fuse_metadata["imp_indices"])
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
                # normal attention
                # When block_tables are not filled, it means q and k are the
                # prompt, and they have the same length.
                if status in [1, 2] and cache_fuse_metadata["daf"]:
                    # DAF 压缩层：两路注意力（重要 token 互看 + 问题看全文）
                    out = self._run_daf_attention(query, key, value,
                                                  cache_fuse_metadata)
                    output = out
                elif status in [1, 2]:
                    # 原版 CacheBlend 压缩层：压缩 Q 对全长 KV 的因果注意力。
                    # xformers 用 LowerTriangularFromBottomRightMask；
                    # flash-attn 在 q_len < kv_len 时 causal mask 右下对齐，
                    # 语义等价（已数值验证）。cache_fuse 路径恒为单序列。
                    kv_len = key.shape[0]
                    q_len = query.shape[0]
                    cu_q = torch.tensor([0, q_len], device=query.device,
                                        dtype=torch.int32)
                    cu_k = torch.tensor([0, kv_len], device=query.device,
                                        dtype=torch.int32)
                    out = flash_attn_varlen_func(
                        q=query,
                        k=key,
                        v=value,
                        cu_seqlens_q=cu_q,
                        cu_seqlens_k=cu_k,
                        max_seqlen_q=q_len,
                        max_seqlen_k=kv_len,
                        softmax_scale=self.scale,
                        causal=True,
                    )
                    assert out.shape == output.shape
                    output = out
                else:
                    out = flash_attn_varlen_func(
                        q=query,
                        k=key,
                        v=value,
                        cu_seqlens_q=prefill_meta.seq_start_loc,
                        cu_seqlens_k=prefill_meta.seq_start_loc,
                        max_seqlen_q=prefill_meta.max_prompt_len,
                        max_seqlen_k=prefill_meta.max_prompt_len,
                        softmax_scale=self.scale,
                        causal=True,
                        window_size=self.sliding_window,
                        alibi_slopes=self.alibi_slopes,
                    )
                    assert output[:num_prefill_tokens].shape == out.shape
                    output[:num_prefill_tokens] = out
            else:
                # prefix-enabled attention
                # TODO(Hai) this triton kernel has regression issue (broke) to
                # deal with different data types between KV and FP8 KV cache,
                # to be addressed separately.
                output[:num_prefill_tokens] = PagedAttention.forward_prefix(
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
        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
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
        causal: bool,
    ) -> torch.Tensor:
        """DAF 用的稠密注意力（flash-attn 版，对应 xformers 后端的
        _dense_attention）。

        与 xformers 版的差异：
        - GQA 不需要手动 expand KV 头，flash-attn 原生支持；
        - 掩码用语义参数 causal 表达：causal=False 即无掩码（双向互看），
          causal=True 且 q_len < kv_len 时 flash-attn 的 causal mask
          右下对齐，等价于 xformers 的 LowerTriangularFromBottomRightMask。
        输入/输出形状与 xformers 版一致：[token 数, 头数, 每头维度]。
        """
        q_len = query.shape[0]
        kv_len = key.shape[0]
        cu_seqlens_q = torch.tensor([0, q_len], device=query.device,
                                    dtype=torch.int32)
        cu_seqlens_k = torch.tensor([0, kv_len], device=query.device,
                                    dtype=torch.int32)
        out = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=q_len,
            max_seqlen_k=kv_len,
            softmax_scale=self.scale,
            causal=causal,
        )
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
          key/value: [原始完整序列长, KV头数, 每头维度]，组装好的混合 KV

        数据流（对应论文的三步，与 xformers 后端一致）：
          A. 重要文档 token 之间互相自注意力（无掩码，双向可见）
          B. 问题 token 对"全文混合 KV"做因果注意力（右下对齐）
          C. State Fusion：按 [重要文档 | 问题] 顺序拼接两路输出
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

        # A. 重要 token 自注意力。K/V 直接从混合 KV 里按原始位置取，
        # 双向无掩码（causal=False）
        k_imp = key[doc_imp]
        v_imp = value[doc_imp]
        out_imp = self._dense_attention(q_imp, k_imp, v_imp, causal=False)

        # B. 问题对全文混合 KV 做因果注意力。q_len < kv_len 时
        # flash-attn 的 causal mask 右下对齐，与 xformers 的
        # LowerTriangularFromBottomRightMask 等价
        out_suf = self._dense_attention(q_suf, key, value, causal=True)

        # C. State Fusion：拼接两路输出，顺序与 imp_indices 保持一致
        out = torch.cat([out_imp, out_suf], dim=0)

        assert out_imp.shape[0] == num_imp and out_suf.shape[0] == suffix_len
        assert out.shape[0] == num_imp + suffix_len
        assert not torch.isnan(out).any() and not torch.isinf(out).any(), \
            "DAF 注意力输出出现 NaN/Inf"
        return out
