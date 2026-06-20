"""
╔══════════════════════════════════════════════════════════╗
║  AI Dubbing System - Web Server                         ║
║  ──────────────────────────────────────────────────────  ║
║  Chạy: python run_web.py                                ║
║  Mở browser: http://localhost:8000                       ║
╚══════════════════════════════════════════════════════════╝
"""

import sys
import webbrowser
import threading
from pathlib import Path

# Thêm project root vào path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def open_browser(port: int):
    """Mở browser sau khi server khởi động"""
    import time
    time.sleep(1.5)
    webbrowser.open(f"http://localhost:{port}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🎬 AI Dubbing System - Web Server"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host (mặc định: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8000,
        help="Port (mặc định: 8000)"
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Không tự động mở browser"
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="Auto-reload khi code thay đổi (dev mode)"
    )
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  🎬 AI Dubbing System - Web Server                      ║
║  ──────────────────────────────────────────────────────  ║
║  URL: http://localhost:{args.port:<39s}  ║
║  Nhấn Ctrl+C để dừng                                    ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Mở browser
    if not args.no_browser:
        threading.Thread(
            target=open_browser,
            args=(args.port,),
            daemon=True,
        ).start()
    
    # Chạy uvicorn
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
