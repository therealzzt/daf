# =============================================================================
# vLLM 插件：补齐 LMCache 0.5.5 开源 wheel 里 blending 的半截接线
#
# 背景（2026-09-16 排查结论）：
#   LMCache 0.5.5 的 blending 消费端 LMCBlenderBuilder.get_or_create 需要
#   VLLMModelTracker 里注册好的 vllm 模型，但 lmcache 与 vllm 0.29 的
#   lmcache_integration 都没有任何地方调用 register_model —— 功能只接了
#   一半，enable_blending=true 启动即报 "vllm model for vllm-instance not found"。
#
# 本插件做两件事：
#   1) LMCBlenderBuilder 惰性化：连接器 init 时模型尚未加载，先挂起返回惰性代理；
#      首次真正调用 blend() 时模型已加载注册，再创建真身并转发
#   2) 包装 GPUModelRunner.load_model：模型加载完成后自动
#      VLLMModelTracker.register_model("vllm-instance", model)
#
# 安装：pip install -e experiments/lmcache/daf_vllm_plugin_pkg（entry point 注册，
# vllm 启动时自动发现加载，无需环境变量；多进程会重复加载，load() 已做幂等）
#
# 后续 DAF 移植也挂在本插件里（替换 process_qkv 的选人/两路注意力逻辑）。
# =============================================================================

_PATCHED = False


def load():
    """vllm general plugin 入口（可多次调用，幂等）。"""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    from lmcache.v1.compute.blend import utils as blend_utils
    from lmcache.v1.compute.blend.blender import LMCBlender
    from lmcache.v1.compute.models.utils import VLLMModelTracker

    ENGINE_NAME = "vllm-instance"

    class _LazyBlender:
        """惰性 LMCBlender：首次访问真实属性时才创建并替换注册表中的自己。"""

        def __init__(self, builder_cls, instance_id, cache_engine, gpu_connector, config):
            object.__setattr__(self, "_builder_cls", builder_cls)
            object.__setattr__(self, "_instance_id", instance_id)
            object.__setattr__(self, "_args", (cache_engine, gpu_connector, config))
            object.__setattr__(self, "_real", None)

        def _materialize(self):
            real = object.__getattribute__(self, "_real")
            if real is None:
                cls = object.__getattribute__(self, "_builder_cls")
                iid = object.__getattribute__(self, "_instance_id")
                cache_engine, gpu_connector, config = object.__getattribute__(self, "_args")
                vllm_model = VLLMModelTracker.get_model(iid)
                real = LMCBlender(cache_engine, gpu_connector, vllm_model, config)
                cls._blenders[iid] = real
                object.__setattr__(self, "_real", real)
            return real

        def __getattr__(self, name):
            return getattr(self._materialize(), name)

    _orig_get_or_create = blend_utils.LMCBlenderBuilder.get_or_create.__func__

    def _patched_get_or_create(cls, instance_id, cache_engine, gpu_connector, config):
        if instance_id in cls._blenders:
            return cls._blenders[instance_id]
        try:
            return _orig_get_or_create(cls, instance_id, cache_engine, gpu_connector, config)
        except ValueError:
            # 模型尚未注册（连接器 init 早于模型加载）→ 先挂惰性代理
            lazy = _LazyBlender(cls, instance_id, cache_engine, gpu_connector, config)
            cls._blenders[instance_id] = lazy
            return lazy

    blend_utils.LMCBlenderBuilder.get_or_create = classmethod(_patched_get_or_create)

    # 模型加载完成后注册进 VLLMModelTracker（供 blending 创建 LMCBlender）
    from vllm.v1.worker import gpu_model_runner as gmr

    _orig_load_model = gmr.GPUModelRunner.load_model

    def patched_load_model(self, *args, **kwargs):
        ret = _orig_load_model(self, *args, **kwargs)
        VLLMModelTracker.register_model(ENGINE_NAME, self.model)
        return ret

    gmr.GPUModelRunner.load_model = patched_load_model
    print("[daf_vllm_plugin] blending 接线补丁已加载（惰性 Blender + 模型自动注册）")
