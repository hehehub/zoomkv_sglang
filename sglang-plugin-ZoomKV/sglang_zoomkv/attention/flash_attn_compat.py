"""
Flash Attention Compatibility Layer

Adapted from ``ZoomKV-main/attn_hub/flash_attn_compat.py``.

提供跨 GPU 架构的兼容性支持：
- Ampere+ (compute capability >= 8.0): 使用 Flash Attention 2.x
- Turing (compute capability 7.5): 使用 Flash Attention 1.x
"""

import os
import torch

# 设置 Triton 缓存目录到有足够空间的位置；保留用户已设置的值，否则落到项目根 .triton 下
os.environ.setdefault(
    'TRITON_CACHE_DIR',
    os.path.join(
        os.environ.get('PROJECT_ROOT', os.getcwd()),
        '.triton', 'cache',
    ),
)


def get_compute_capability():
    """获取当前 GPU 的计算能力"""
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return major + minor / 10
    return 0.0


def check_flash_attn_support():
    """检查是否支持 Flash Attention 2.x"""
    compute_cap = get_compute_capability()
    # Flash Attention 2.x 需要 compute capability >= 8.0 (Ampere+)
    return compute_cap >= 8.0


# 全局标志：是否支持 Flash Attention 2.x
_FLASH_ATTN_SUPPORTED = check_flash_attn_support()

# 检查是否禁用 Triton 回退（通过环境变量控制）
_DISABLE_TRITON_FALLBACK = os.environ.get('DISABLE_TRITON_FALLBACK', 'false').lower() in ('true', '1', 'yes')


def _resolve_flash_attn_with_kvcache():
    """Prefer SGLang's FA3 kernel. flash-attn 4 does not export this symbol."""
    cached = getattr(_resolve_flash_attn_with_kvcache, "_fn", None)
    if cached is not None:
        return cached
    errors = []
    for module_name in ("sgl_kernel.flash_attn", "flash_attn"):
        try:
            module = __import__(module_name, fromlist=["flash_attn_with_kvcache"])
            fn = getattr(module, "flash_attn_with_kvcache")
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
            continue
        _resolve_flash_attn_with_kvcache._fn = fn
        return fn
    raise ImportError("; ".join(errors))


def flash_attn_with_kvcache_compat(
    q,
    k_cache,
    v_cache,
    causal=False,
    **kwargs
):
    """
    Flash Attention 兼容函数
    
    自动选择最优实现：
    - GPU >= Ampere: SGLang FA3，其次 Flash Attention 2.x
    - 两者都不可用时回退到 PyTorch SDPA
    
    Args:
        q: Query tensor [batch, seqlen_q, num_heads, head_dim]
        k_cache: Key cache [batch, seqlen_k, num_heads, head_dim]
        v_cache: Value cache [batch, seqlen_k, num_heads, head_dim]
        causal: Whether to apply causal mask
        **kwargs: Additional arguments
    
    Returns:
        out: Attention output [batch, seqlen_q, num_heads, head_dim]
    """
    if _FLASH_ATTN_SUPPORTED:
        # 使用 Flash Attention 2.x (最快)
        try:
            flash_attn_with_kvcache = _resolve_flash_attn_with_kvcache()
            return flash_attn_with_kvcache(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                causal=causal,
                **kwargs
            )
        except Exception as e:
            print(f"[Flash Attention] Warning: Flash Attention 2.x failed: {e}")
            if _DISABLE_TRITON_FALLBACK:
                print("[Flash Attention] Triton fallback is disabled. Raising error.")
                raise RuntimeError(
                    f"Flash Attention 2.x failed and Triton fallback is disabled. "
                    f"Error: {e}. "
                    f"To enable Triton fallback, unset DISABLE_TRITON_FALLBACK environment variable."
                )
            print("[Flash Attention] Falling back to Triton implementation")
            return _triton_attention_fallback(q, k_cache, v_cache, causal)
    else:
        # GPU 不支持 Flash Attention 2.x
        if _DISABLE_TRITON_FALLBACK:
            raise RuntimeError(
                f"GPU compute capability {get_compute_capability()} < 8.0 does not support Flash Attention 2.x, "
                f"and Triton fallback is disabled. "
                f"To enable Triton fallback, unset DISABLE_TRITON_FALLBACK environment variable."
            )
        # 使用 Triton 实现
        return _triton_attention_fallback(q, k_cache, v_cache, causal)


def _triton_attention_fallback(q, k, v, causal=False):
    """
    Triton Flash Attention 实现 (用于不支持 Flash Attention 2.x 的 GPU)
    
    使用 PyTorch SDPA 作为可移植回退。CUDA 环境下 PyTorch 会自动选择
    可用的 flash、memory-efficient 或 math kernel。
    
    注意：Turing GPU 不支持 BFloat16，会自动转换为 Float16
    
    Args:
        q: [batch, seqlen_q, num_heads, head_dim]
        k: [batch, seqlen_k, num_heads, head_dim]
        v: [batch, seqlen_k, num_heads, head_dim]
        causal: Whether to apply causal mask
    
    Returns:
        out: [batch, seqlen_q, num_heads, head_dim]
    """
    try:
        # Turing GPU 不支持 BFloat16，需要转换为 Float16
        original_dtype = q.dtype
        need_conversion = (original_dtype == torch.bfloat16)
        
        if need_conversion:
            # print(f"[Flash Attention] Converting BF16 → FP16 for Turing GPU")
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)
        
        from torch.nn.functional import scaled_dot_product_attention

        q_heads = q.shape[2]
        kv_heads = k.shape[2]
        out = scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=causal,
            enable_gqa=q_heads != kv_heads,
        ).transpose(1, 2)
        
        # 转换回原始数据类型
        if need_conversion:
            out = out.to(original_dtype)
        
        return out
    except Exception as e:
        print(f"[Flash Attention] PyTorch SDPA fallback failed: {e}")
        print("[Flash Attention] Error details:")
        import traceback
        traceback.print_exc()
        raise RuntimeError(
            f"PyTorch SDPA fallback failed. GPU compute capability: {get_compute_capability()}. "
            f"Please check the PyTorch/CUDA installation."
        )


# 在模块导入时打印 GPU 兼容性信息
compute_cap = get_compute_capability()
if compute_cap > 0:
    print(f"[Flash Attention Compat] GPU Compute Capability: {compute_cap}")
    if _FLASH_ATTN_SUPPORTED:
        print("[Flash Attention Compat] Using Flash Attention 2.x (native)")
        if _DISABLE_TRITON_FALLBACK:
            print("[Flash Attention Compat] Triton fallback is DISABLED (will raise error if Flash Attention 2.x fails)")
    else:
        print(f"[Flash Attention Compat] GPU compute capability {compute_cap} < 8.0")
        if _DISABLE_TRITON_FALLBACK:
            print("[Flash Attention Compat] Triton fallback is DISABLED (will raise error)")
        else:
            print("[Flash Attention Compat] Using Triton Flash Attention (fallback)")
else:
    print("[Flash Attention Compat] No CUDA device found")
