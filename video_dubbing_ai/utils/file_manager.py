"""
File Manager
==============
Quản lý file tạm, cleanup, và path utilities.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, List
from utils.logger import get_logger

logger = get_logger(__name__)


class FileManager:
    """Quản lý file và thư mục cho pipeline"""
    
    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            project_root = Path(__file__).parent.parent
        self.project_root = project_root
        self.temp_dir = project_root / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def create_job_dir(self, job_id: str) -> Path:
        """
        Tạo thư mục riêng cho mỗi job.
        
        Args:
            job_id: ID của job
            
        Returns:
            Path đến thư mục job
        """
        job_dir = self.temp_dir / job_id
        
        # Tạo sub-directories
        subdirs = [
            "normalized",    # Video chuẩn hóa
            "audio",         # Audio extracted
            "segments",      # Audio segments
            "reference",     # Reference audio cho voice clone
            "generated",     # Generated Vietnamese audio
            "aligned",       # Audio sau khi align
            "lipsync",       # Video sau lip sync
            "merged",        # Merged audio
        ]
        
        for subdir in subdirs:
            (job_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created job directory: {job_dir}")
        return job_dir
    
    def get_job_path(self, job_id: str, category: str, filename: str) -> Path:
        """
        Lấy đường dẫn file trong job directory.
        
        Args:
            job_id: ID của job
            category: Loại file (normalized, audio, segments, etc.)
            filename: Tên file
        """
        return self.temp_dir / job_id / category / filename
    
    def cleanup_job(self, job_id: str, keep_output: bool = True):
        """
        Cleanup file tạm của job.
        
        Args:
            job_id: ID của job
            keep_output: Giữ lại output cuối cùng
        """
        job_dir = self.temp_dir / job_id
        if job_dir.exists():
            if keep_output:
                # Chỉ xóa thư mục temp, giữ output
                for subdir in job_dir.iterdir():
                    if subdir.is_dir() and subdir.name != "output":
                        shutil.rmtree(subdir)
                        logger.debug(f"Cleaned: {subdir}")
            else:
                shutil.rmtree(job_dir)
                logger.info(f"Removed job directory: {job_dir}")
    
    def cleanup_all_temp(self):
        """Xóa toàn bộ file tạm"""
        if self.temp_dir.exists():
            for item in self.temp_dir.iterdir():
                if item.is_dir() and item.name != "logs":
                    shutil.rmtree(item)
                elif item.is_file():
                    item.unlink()
            logger.info("Cleaned all temp files")
    
    def get_file_size_mb(self, path: Path) -> float:
        """Lấy kích thước file (MB)"""
        if path.exists():
            return path.stat().st_size / (1024 * 1024)
        return 0.0
    
    def list_files(self, directory: Path, pattern: str = "*") -> List[Path]:
        """Liệt kê files trong thư mục"""
        if directory.exists():
            return sorted(directory.glob(pattern))
        return []
    
    def ensure_dir(self, path: Path) -> Path:
        """Đảm bảo thư mục tồn tại"""
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def copy_to_output(self, source: Path, job_id: str, output_name: Optional[str] = None) -> Path:
        """
        Copy file kết quả vào thư mục output.
        
        Args:
            source: File nguồn
            job_id: Job ID
            output_name: Tên file output (mặc định giữ tên gốc)
            
        Returns:
            Path đến file output
        """
        output_dir = self.project_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if output_name is None:
            output_name = f"{job_id}_{source.name}"
        
        dest = output_dir / output_name
        shutil.copy2(source, dest)
        logger.info(f"Output saved: {dest}")
        return dest
