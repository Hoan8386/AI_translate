"""
FastAPI Server
================
Khởi tạo FastAPI app, mount static files, WebSocket endpoint.

Chạy: python run_web.py
"""

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.routes import router as api_router
from api.websocket import manager as ws_manager

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
WEB_DIR = PROJECT_ROOT / "web"
OUTPUT_DIR = PROJECT_ROOT / "output"


def create_app() -> FastAPI:
    """Tạo FastAPI application"""
    
    app = FastAPI(
        title="AI Dubbing System",
        description="Chinese Video → Vietnamese AI Dubbing System",
        version="3.0.0",
    )
    
    # CORS - cho phép browser truy cập
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(api_router)
    
    # WebSocket endpoint
    @app.websocket("/ws/jobs/{job_id}")
    async def websocket_endpoint(websocket: WebSocket, job_id: str):
        """WebSocket endpoint để nhận updates real-time cho job"""
        await ws_manager.connect(websocket, job_id)
        try:
            while True:
                # Giữ kết nối mở, đợi client gửi ping hoặc disconnect
                data = await websocket.receive_text()
                # Client có thể gửi ping để giữ kết nối
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket, job_id)
        except Exception:
            ws_manager.disconnect(websocket, job_id)
    
    # Đảm bảo thư mục output tồn tại
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Serve output files (video results)
    app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
    
    # Serve frontend (web/) - phải mount cuối cùng
    if WEB_DIR.exists():
        # Serve index.html for root
        @app.get("/")
        async def serve_index():
            return FileResponse(str(WEB_DIR / "index.html"))
        
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
    
    return app


# Tạo app instance
app = create_app()
