# 发布阅读说明：处理目标作者环境的必要加载兼容性；不改变目标策略的预测数学或动作维度。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""Small backend/data compatibility fixes, not target-policy mathematics."""
from pathlib import Path
import numpy as np
from contracts import require, sha256_file


def configure_adapter_backends(torch_module, tensorflow_module):
    # The successful OFT evaluator already avoids the cluster's cuDNN handle
    # failure. Keep actual Torch CUDA convolution, never CPU model inference.
    torch_module.backends.cudnn.enabled = False
    # Author TF helpers only resize/crop images. Do not let TF reserve CUDA
    # memory or create another cuDNN stack beside the Torch vision encoder.
    tensorflow_module.config.set_visible_devices([], "GPU")


def load_frozen_initial_states(path, expected_sha):
    """Restricted NumPy allowlist for original, SHA-pinned LIBERO arrays only."""
    path = Path(path).resolve()
    require(path.suffix == ".pruned_init" and isinstance(expected_sha, str)
            and sha256_file(path) == expected_sha, "Initial-state file is not the frozen trusted LIBERO asset")
    import torch
    # Unlike globally disabling weights_only, this local context only permits
    # NumPy array reconstruction and float64 dtype used by these initial states.
    from numpy.core.multiarray import _reconstruct
    allowed = [_reconstruct, np.ndarray, np.dtype, type(np.dtype(np.float64))]
    # NumPy2 renamed numpy.core; serialized LIBERO uses the original qualified name.
    if _reconstruct.__module__ != "numpy.core.multiarray":
        allowed.append((_reconstruct, "numpy.core.multiarray._reconstruct"))
    with torch.serialization.safe_globals(allowed):
        value = torch.load(path, map_location="cpu", weights_only=True)
    array = np.asarray(value)
    require(array.ndim == 2 and array.shape[0] >= 50 and array.shape[1] > 0
            and array.dtype == np.float64 and np.isfinite(array).all(), "Malformed LIBERO initial states")
    return value
