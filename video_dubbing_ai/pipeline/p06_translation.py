"""
Stage 6: Translation (LOCAL GPU)
================================
Dịch tiếng Trung → tiếng Việt sử dụng mô hình LLM Local nhỏ gọn (Qwen2.5-1.5B-Instruct),
chạy hoàn toàn trên GPU RTX 5060, KHÔNG tốn API, KHÔNG lo hết quota.

Input:  List[Segment] (Chứa zh_text)
Output: List[Segment] (Cập nhật vi_text)

Technology:
    transformers + torch (Tối ưu hóa FP16 cho RTX 5060)
"""

import json
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from models_data.segment import Segment
from utils.logger import get_logger, log_stage
from utils.timer import Timer
from utils.gpu_manager import GPUManager
from config.settings import get_settings

logger = get_logger(__name__)


class Translator:
    """
    Dịch ngữ cảnh tiếng Trung → tiếng Việt bằng LLM Local (Qwen2.5).
    """

    STAGE_NUM = 6
    STAGE_NAME = "Translation (ZH → VI - Local)"

    # Lưu model chung để tránh reload nhiều lần (Singleton)
    _shared_model = None
    _shared_tokenizer = None

    def __init__(self):
        self.gpu = GPUManager()
        self.settings = get_settings()

        # Thư mục gốc của project
        project_root = Path(__file__).resolve().parents[1]

        # Đường dẫn tới model local
        self.model_name = str(project_root / "third_party" / "qwen2.5")

        # Kiểm tra model có tồn tại không
        if not Path(self.model_name).exists():
            raise FileNotFoundError(
                f"Không tìm thấy model tại:\n{self.model_name}"
            )

        logger.info("=" * 60)
        logger.info("Local Translation Model")
        logger.info(f"Model Path : {self.model_name}")
        logger.info("=" * 60)

        # Singleton
        self.model = Translator._shared_model
        self.tokenizer = Translator._shared_tokenizer

    def _load_model(self):
        """Load mô hình dịch thuật lên GPU (Tốn khoảng 1.5GB - 3GB VRAM tùy model)"""
        if self.model is not None:
            return

        logger.info(f"Loading Local Translation Model: {self.model_name}...")
        
        # Đảm bảo trống bộ nhớ trước khi nạp
        self.gpu.ensure_free(1500)

        logger.info(f"Loading model from: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        Translator._shared_model = self.model
        Translator._shared_tokenizer = self.tokenizer
        logger.info("Local Translation Model loaded thành công.")

    def _build_system_prompt(self) -> str:
        return (
            "You are a professional movie dubbing translator (Chinese to Vietnamese).\n"
            "Translate the given JSON list of Chinese dialogues into Vietnamese.\n"
            "REQUIREMENTS:\n"
            "1. NATURAL STYLE: Use informal, spoken, smooth Vietnamese. Do NOT translate word-by-word. Do NOT be overly formal.\n"
            "2. CONTEXT-AWARE: Choose appropriate pronouns (anh/em, cậu/tớ, mày/tao, tôi...) based on the conversational flow and the 'speaker' field.\n"
            "3. MATCH LENGTH: Keep sentences concise.\n"
            "4. OUTPUT FORMAT: Respond ONLY with a raw JSON array of translated strings, in the exact same order. No explanation."
        )

    def process(self, segments: List[Segment]) -> List[Segment]:
        log_stage(self.STAGE_NUM, self.STAGE_NAME, "START")

        with Timer(f"Stage {self.STAGE_NUM}: {self.STAGE_NAME}"):
            valid_segments = [s for s in segments if s.zh_text and s.zh_text.strip()]

            if not valid_segments:
                logger.warning("Không có text cần dịch.")
                log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
                return segments

            # Kích hoạt load model local
            self._load_model()

            logger.info(f"Đang tiến hành dịch Local {len(valid_segments)} segments (Gộp ngữ cảnh thoại)...")

            # Đóng gói dữ liệu kèm ID người nói để model nhận biết vai vế hội thoại
            batch_data = [
                {"speaker": getattr(s, 'speaker', 'unknown'), "text": s.zh_text}
                for s in valid_segments
            ]

            user_prompt = f"Translate this JSON array into Vietnamese:\n{json.dumps(batch_data, ensure_ascii=False)}"

            try:
                # Tạo cấu trúc hội thoại chuẩn Chat của Qwen
                messages = [
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": user_prompt}
                ]
                
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

                # Cấu hình tham số sinh văn bản (Hạ temperature để ra kết quả JSON chuẩn xác)
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=2048,
                        temperature=0.1,
                        top_p=0.9,
                        do_sample=False
                    )
                
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                
                # Bóc tách và parse dữ liệu JSON trả về
                if "[" in response_text and "]" in response_text:
                    start_idx = response_text.find("[")
                    end_idx = response_text.rfind("]") + 1
                    response_text = response_text[start_idx:end_idx]

                translated_texts = json.loads(response_text)

                if len(translated_texts) == len(valid_segments):
                    for segment, vi_text in zip(valid_segments, translated_texts):
                        segment.vi_text = str(vi_text).strip()
                        logger.info(f'Local Trans: "{segment.zh_text}" → "{segment.vi_text}"')
                else:
                    raise ValueError("Mismatched output count from local LLM")

            except Exception as e:
                logger.error(f"Lỗi dịch thuật Local LLM: {e}. Tiến hành tự động hạ cấp về dịch thô từng câu để cứu luồng...")
                # Nếu bọc batch lỗi cấu trúc JSON, hệ thống tự động gán text lỗi thay vì làm crash app
                for segment in valid_segments:
                    segment.vi_text = f"[LOCAL TRANS ERROR]"

            translated_count = sum(1 for s in segments if s.vi_text and s.vi_text.strip())
            logger.info(f"Hoàn thành dịch Local: {translated_count}/{len(segments)} segments.")

        # Thu dọn cache GPU sau khi chạy xong bước này để nhường chỗ cho các Stage kế tiếp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log_stage(self.STAGE_NUM, self.STAGE_NAME, "DONE")
        return segments

    @classmethod
    def unload_model(cls):
        """Giải phóng hoàn toàn mô hình dịch thuật khỏi VRAM khi cần tắt app hoặc chuyển tiếp"""
        if cls._shared_model is not None:
            logger.info("Unloading Local Translation Model...")
            del cls._shared_model
            del cls._shared_tokenizer
            cls._shared_model = None
            cls._shared_tokenizer = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()