"""
Timer
======
Đo thời gian thực thi từng giai đoạn pipeline.
"""

import time
from typing import Dict, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class Timer:
    """
    Context manager để đo thời gian.
    
    Usage:
        with Timer("Stage 1: Video Processor"):
            process_video()
        
        # Hoặc
        timer = Timer()
        timer.start("asr")
        do_asr()
        timer.stop("asr")
        timer.report()
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self._start_time: Optional[float] = None
        self._records: Dict[str, Dict] = {}
    
    # --- Context Manager ---
    
    def __enter__(self):
        self._start_time = time.time()
        if self.name:
            logger.info(f"⏱ [{self.name}] Bắt đầu...")
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self._start_time
        if self.name:
            logger.info(f"⏱ [{self.name}] Hoàn thành trong {self._format_time(elapsed)}")
            self._records[self.name] = {
                "elapsed": elapsed,
                "formatted": self._format_time(elapsed)
            }
    
    # --- Manual timing ---
    
    def start(self, label: str):
        """Bắt đầu đo cho một label"""
        self._records[label] = {"start": time.time()}
    
    def stop(self, label: str) -> float:
        """Kết thúc đo cho một label"""
        if label not in self._records:
            logger.warning(f"Timer '{label}' chưa được start")
            return 0.0
        
        elapsed = time.time() - self._records[label]["start"]
        self._records[label]["elapsed"] = elapsed
        self._records[label]["formatted"] = self._format_time(elapsed)
        
        logger.info(f"⏱ [{label}] {self._format_time(elapsed)}")
        return elapsed
    
    def report(self) -> str:
        """In báo cáo thời gian tất cả stages"""
        if not self._records:
            return "No timing records."
        
        lines = [
            "\n" + "=" * 55,
            "  ⏱ TIMING REPORT",
            "=" * 55,
        ]
        
        total = 0
        for label, data in self._records.items():
            elapsed = data.get("elapsed", 0)
            total += elapsed
            formatted = data.get("formatted", self._format_time(elapsed))
            lines.append(f"  {label:<35} {formatted:>15}")
        
        lines.append("-" * 55)
        lines.append(f"  {'TOTAL':<35} {self._format_time(total):>15}")
        lines.append("=" * 55)
        
        report = "\n".join(lines)
        logger.info(report)
        return report
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format thời gian đẹp"""
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {mins}m {secs:.0f}s"
