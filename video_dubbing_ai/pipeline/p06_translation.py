"""
Stage 6: Translation
======================
Dịch tiếng Trung → tiếng Việt.

Input:  大家好
Output: Xin chào mọi người

API: Gemini API hoặc OpenAI GPT API
KHÔNG chạy model dịch trên máy.
"""

import time
from typing import List, Optional

from models_data.segment import Segment
from utils.logger import get_logger, log_stage
from utils.timer import Timer
from config.settings import get_settings

logger = get_logger(__name__)


class Translator:
    """
    Dịch ZH → VI qua cloud API.
    
    Hỗ trợ cả Gemini và OpenAI GPT.
    Có rate limiting và retry logic.
    """
    
    STAGE_NUM = 6
    STAGE_NAME = "Translation (ZH → VI)"
    
    # System prompt cho translation
    SYSTEM_PROMPT = (
        "Bạn là một dịch giả chuyên nghiệp Trung-Việt. "
        "Dịch câu tiếng Trung sang tiếng Việt tự nhiên, giữ nguyên ý nghĩa. "
        "Chỉ trả về bản dịch tiếng Việt, không giải thích thêm. "
        "Đảm bảo câu dịch ngắn gọn, phù hợp để lồng tiếng (dubbing). "
        "Ưu tiên câu ngắn hơn nếu có thể, vì cần khớp thời lượng âm thanh gốc."
    )
    
    def __init__(self):
        self.settings = get_settings()
        self._client = None
        self._last_request_time = 0
        self._min_interval = 60.0 / self.settings.translation.requests_per_minute
    
    def _init_client(self):
        """Khởi tạo API client"""
        provider = self.settings.translation.provider.lower()
        
        if provider == "gemini":
            self._init_gemini()
        elif provider == "openai":
            self._init_openai()
        else:
            raise ValueError(f"Translation provider không hợp lệ: {provider}")
    
    def _init_gemini(self):
        """Khởi tạo Gemini client"""
        import google.generativeai as genai
        
        api_key = self.settings.translation.gemini_api_key
        if not api_key:
            raise ValueError(
                "Cần GEMINI_API_KEY!\n"
                "1. Lấy key tại https://aistudio.google.com/apikey\n"
                "2. Set GEMINI_API_KEY trong file .env"
            )
        
        genai.configure(api_key=api_key)
        self._client = genai.GenerativeModel(
            self.settings.translation.gemini_model
        )
        logger.info(f"Gemini client initialized (model: {self.settings.translation.gemini_model})")
    
    def _init_openai(self):
        """Khởi tạo OpenAI client"""
        from openai import OpenAI
        
        api_key = self.settings.translation.openai_api_key
        if not api_key:
            raise ValueError(
                "Cần OPENAI_API_KEY!\n"
                "Set OPENAI_API_KEY trong file .env"
            )
        
        self._client = OpenAI(api_key=api_key)
        logger.info(f"OpenAI client initialized (model: {self.settings.translation.openai_model})")
    
    def _rate_limit(self):
        """Rate limiting - đảm bảo không gửi request quá nhanh"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            sleep_time = self._min_interval - elapsed
            time.sleep(sleep_time)
        self._last_request_time = time.time()
    
    def _translate_single(self, text: str) -> str:
        """
        Dịch 1 câu.
        
        Args:
            text: Text tiếng Trung
            
        Returns:
            Text tiếng Việt
        """
        if not text.strip():
            return ""
        
        provider = self.settings.translation.provider.lower()
        
        for attempt in range(self.settings.translation.max_retries):
            try:
                self._rate_limit()
                
                if provider == "gemini":
                    return self._translate_gemini(text)
                else:
                    return self._translate_openai(text)
                    
            except Exception as e:
                logger.warning(
                    f"Translation attempt {attempt+1}/{self.settings.translation.max_retries} "
                    f"failed: {e}"
                )
                if attempt < self.settings.translation.max_retries - 1:
                    wait = self.settings.translation.retry_delay * (attempt + 1)
                    time.sleep(wait)
                else:
                    logger.error(f"Translation failed after {self.settings.translation.max_retries} attempts")
                    return f"[TRANSLATION ERROR: {text}]"
    
    def _translate_gemini(self, text: str) -> str:
        """Dịch bằng Gemini API"""
        prompt = f"{self.SYSTEM_PROMPT}\n\nDịch sang tiếng Việt:\n{text}"
        
        response = self._client.generate_content(prompt)
        translated = response.text.strip()
        return translated
    
    def _translate_openai(self, text: str) -> str:
        """Dịch bằng OpenAI GPT API"""
        response = self._client.chat.completions.create(
            model=self.settings.translation.openai_model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Dịch sang tiếng Việt:\n{text}"},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        translated = response.choices[0].message.content.strip()
        return translated
    
    def _translate_batch_gemini(self, texts: List[str]) -> List[str]:
        """
        Dịch batch bằng Gemini (gửi nhiều câu 1 lần để tiết kiệm API calls).
        """
        if not texts:
            return []
        
        # Format batch
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Dịch từng câu sau sang tiếng Việt. "
            f"Trả về đúng format đánh số, mỗi câu 1 dòng:\n\n"
            f"{numbered}"
        )
        
        self._rate_limit()
        response = self._client.generate_content(prompt)
        result_text = response.text.strip()
        
        # Parse kết quả
        lines = result_text.split("\n")
        results = []
        for line in lines:
            line = line.strip()
            if line:
                # Xóa số thứ tự đầu dòng (1. , 2. , etc.)
                import re
                cleaned = re.sub(r'^\d+[\.\)]\s*', '', line)
                results.append(cleaned)
        
        # Đảm bảo đúng số lượng
        while len(results) < len(texts):
            results.append("")
        
        return results[:len(texts)]
    
    def process(self, segments: List[Segment]) -> List[Segment]:
        """
        Dịch tất cả segments từ ZH → VI.
        
        Args:
            segments: Danh sách Segment đã có zh_text
            
        Returns:
            Danh sách Segment đã có vi_text
        """
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "START")
        
        with Timer(f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}"):
            # Khởi tạo client
            self._init_client()
            
            # Lọc segments có text cần dịch
            to_translate = [s for s in segments if s.zh_text.strip()]
            
            if not to_translate:
                logger.warning("Không có text nào cần dịch!")
                log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
                return segments
            
            logger.info(f"Dịch {len(to_translate)} segments (ZH → VI)")
            
            # Dịch từng segment
            for i, segment in enumerate(to_translate):
                logger.info(
                    f"Translating [{i+1}/{len(to_translate)}]: "
                    f"\"{segment.zh_text}\""
                )
                
                vi_text = self._translate_single(segment.zh_text)
                segment.vi_text = vi_text
                
                logger.info(f"  → \"{vi_text}\"")
            
            # Thống kê
            translated = sum(1 for s in segments if s.vi_text)
            logger.info(
                f"Translation hoàn thành: {translated}/{len(segments)} segments"
            )
        
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
        return segments
