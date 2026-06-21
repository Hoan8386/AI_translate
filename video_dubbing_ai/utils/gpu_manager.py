"""
GPU Manager
============
Quản lý GPU cho RTX 5060 8GB.
Nguyên tắc vàng: Chỉ 1 AI model trên GPU tại 1 thời điểm.
"""

import gc
import torch
from typing import Optional, Any
from utils.logger import get_logger

logger = get_logger(__name__)


class GPUManager:
    """
    Quản lý load/unload model trên GPU.
    
    Đảm bảo chỉ có 1 model nằm trên GPU tại 1 thời điểm.
    Tự động unload model cũ trước khi load model mới.
    
    Usage:
        gpu = GPUManager()
        
        # Load model
        model = gpu.load_model(my_model, "SenseVoice")
        
        # Dùng model...
        
        # Unload khi xong
        gpu.unload_model("SenseVoice")
    """
    
    _instance: Optional['GPUManager'] = None
    
    def __new__(cls):
        """Singleton pattern - chỉ có 1 GPUManager"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self._current_model_name: Optional[str] = None
        self._current_model: Optional[Any] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self._device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            logger.info(f"GPU detected: {gpu_name} ({total_vram:.0f} MB VRAM)")
        else:
            logger.warning("No GPU detected! Running on CPU (rất chậm)")
    
    @property
    def device(self) -> str:
        return self._device
    
    @property
    def current_model_name(self) -> Optional[str]:
        return self._current_model_name
    
    def get_vram_usage(self) -> dict:
        """Lấy thông tin VRAM hiện tại"""
        if self._device != "cuda":
            return {"allocated": 0, "reserved": 0, "total": 0}
        
        return {
            "allocated": torch.cuda.memory_allocated() / (1024**2),
            "reserved": torch.cuda.memory_reserved() / (1024**2),
            "total": torch.cuda.get_device_properties(0).total_memory / (1024**2),
        }
    
    def log_vram(self, context: str = ""):
        """Log VRAM usage"""
        info = self.get_vram_usage()
        prefix = f"[{context}] " if context else ""
        logger.info(
            f"{prefix}VRAM: {info['allocated']:.0f}MB allocated / "
            f"{info['total']:.0f}MB total"
        )
    
    def unload_current(self):
        """
        Unload model hiện tại khỏi GPU.
        Gọi garbage collector và empty CUDA cache.
        """
        if self._current_model is not None:
            model_name = self._current_model_name or "unknown"
            logger.info(f"Unloading model: {model_name}")
            
            # Chuyển model về CPU trước rồi xóa
            try:
                if hasattr(self._current_model, 'cpu'):
                    self._current_model.cpu()
                if hasattr(self._current_model, 'to'):
                    self._current_model.to('cpu')
            except Exception:
                pass
            
            # Xóa reference
            del self._current_model
            self._current_model = None
            self._current_model_name = None
            
            # Force cleanup
            gc.collect()
            if self._device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            self.log_vram(f"After unload {model_name}")
    
    def load_model(self, model: Any, name: str) -> Any:
        """
        Load model lên GPU.
        
        Tự động unload model cũ nếu đang có model trên GPU.
        
        Args:
            model: Model cần load
            name: Tên model (để tracking)
            
        Returns:
            Model đã được chuyển lên GPU
        """
        # Unload model cũ nếu có
        if self._current_model is not None:
            logger.info(
                f"GPU đang có model '{self._current_model_name}'. "
                f"Unloading trước khi load '{name}'..."
            )
            self.unload_current()
        
        logger.info(f"Loading model: {name}")
        self.log_vram(f"Before load {name}")
        
        # Chuyển model lên GPU
        try:
            if hasattr(model, 'to'):
                model = model.to(self._device)
            elif hasattr(model, 'cuda') and self._device == "cuda":
                model = model.cuda()
        except RuntimeError as e:
            logger.error(f"Không thể load {name} lên GPU: {e}")
            logger.info("Thử chạy trên CPU...")
            if hasattr(model, 'to'):
                model = model.to('cpu')
        
        self._current_model = model
        self._current_model_name = name
        
        self.log_vram(f"After load {name}")
        return model
    
    def unload_model(self, name: str = ""):
        """
        Unload model theo tên.
        
        Args:
            name: Tên model cần unload (optional, nếu không truyền sẽ unload model hiện tại)
        """
        if name and self._current_model_name != name:
            logger.warning(
                f"Model '{name}' không phải model hiện tại "
                f"('{self._current_model_name}'). Bỏ qua."
            )
            return
        
        self.unload_current()
    
    def ensure_free(self, required_mb: int = 2000):
        """
        Đảm bảo có đủ VRAM trống.
        
        Args:
            required_mb: Lượng VRAM cần thiết (MB)
        """
        if self._device != "cuda":
            return
        
        info = self.get_vram_usage()
        free = info['total'] - info['allocated']
        
        if free < required_mb:
            logger.warning(
                f"VRAM trống chỉ {free:.0f}MB, cần {required_mb}MB. "
                f"Unloading model hiện tại..."
            )
            self.unload_current()
    
    def __repr__(self):
        info = self.get_vram_usage()
        return (
            f"GPUManager(device={self._device}, "
            f"current_model={self._current_model_name}, "
            f"vram_used={info['allocated']:.0f}MB)"
        )
