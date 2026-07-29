"""
Logger
=======
Structured logging với timestamps và color output.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Color codes
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
        'STAGE': Fore.MAGENTA + Style.BRIGHT,
    }
except ImportError:
    COLORS = {}


class ColorFormatter(logging.Formatter):
    """Formatter với color support cho console"""
    
    FORMAT = "%(asctime)s │ %(levelname)-8s │ %(name)-25s │ %(message)s"
    DATE_FORMAT = "%H:%M:%S"
    
    def format(self, record):
        if COLORS:
            color = COLORS.get(record.levelname, '')
            record.levelname = f"{color}{record.levelname}{Style.RESET_ALL if COLORS else ''}"
            record.msg = f"{color}{record.msg}{Style.RESET_ALL if COLORS else ''}"
        return super().format(record)


class StageFilter(logging.Filter):
    """Filter cho pipeline stage logging"""
    
    def __init__(self, stage_name: str = ""):
        super().__init__()
        self.stage_name = stage_name
    
    def filter(self, record):
        if not hasattr(record, 'stage'):
            record.stage = self.stage_name
        return True


def setup_logger(
    name: str = "dubbing",
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """
    Setup logger cho hệ thống.
    
    Args:
        name: Tên logger
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Có ghi log ra file không
        log_dir: Thư mục chứa log files
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Tránh duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler (có màu)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColorFormatter(
        ColorFormatter.FORMAT, 
        datefmt=ColorFormatter.DATE_FORMAT
    ))
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / "temp" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"dubbing_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            "%(asctime)s │ %(levelname)-8s │ %(name)-25s │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "dubbing") -> logging.Logger:
    """Lấy logger (tạo mới nếu chưa có)"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


def log_stage(stage_num: int, stage_name: str, status: str = "START"):
    """
    Log bắt đầu/kết thúc một pipeline stage.
    
    Args:
        stage_num: Số thứ tự stage (1-10)
        stage_name: Tên stage
        status: START, DONE, ERROR
    """
    logger = get_logger("dubbing.pipeline")
    
    if status == "START":
        logger.info(
            f"\n{'='*60}\n"
            f"  STAGE {stage_num:02d}: {stage_name}\n"
            f"{'='*60}"
        )
    elif status == "DONE":
        logger.info(f"  ✓ Stage {stage_num:02d} ({stage_name}) - HOÀN THÀNH")
    elif status == "ERROR":
        logger.error(f"  ✗ Stage {stage_num:02d} ({stage_name}) - LỖI")
