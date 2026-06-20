"""
WebSocket Manager
===================
Quản lý kết nối WebSocket để gửi updates real-time cho client.
"""

import asyncio
import json
from typing import Dict, Set, Any
from fastapi import WebSocket
from datetime import datetime


class ConnectionManager:
    """
    Quản lý các kết nối WebSocket theo job_id.
    
    Mỗi job có thể có nhiều client kết nối cùng lúc
    (ví dụ: mở nhiều tab browser).
    """
    
    def __init__(self):
        # job_id -> set of WebSocket connections
        self._connections: Dict[str, Set[WebSocket]] = {}
        # job_id -> list of messages (buffer để client reconnect)
        self._message_buffer: Dict[str, list] = {}
        self._buffer_max_size = 200
    
    async def connect(self, websocket: WebSocket, job_id: str):
        """Chấp nhận kết nối WebSocket mới"""
        await websocket.accept()
        
        if job_id not in self._connections:
            self._connections[job_id] = set()
        self._connections[job_id].add(websocket)
        
        # Gửi lại buffer messages cho client mới kết nối
        if job_id in self._message_buffer:
            for msg in self._message_buffer[job_id]:
                try:
                    await websocket.send_json(msg)
                except Exception:
                    break
    
    def disconnect(self, websocket: WebSocket, job_id: str):
        """Xóa kết nối WebSocket"""
        if job_id in self._connections:
            self._connections[job_id].discard(websocket)
            if not self._connections[job_id]:
                del self._connections[job_id]
    
    async def send_to_job(self, job_id: str, data: dict):
        """Gửi message đến tất cả client đang theo dõi job"""
        # Thêm timestamp
        data["timestamp"] = datetime.now().strftime("%H:%M:%S")
        
        # Lưu vào buffer
        if job_id not in self._message_buffer:
            self._message_buffer[job_id] = []
        self._message_buffer[job_id].append(data)
        
        # Giới hạn buffer size
        if len(self._message_buffer[job_id]) > self._buffer_max_size:
            self._message_buffer[job_id] = self._message_buffer[job_id][-self._buffer_max_size:]
        
        # Gửi cho tất cả connections
        if job_id in self._connections:
            dead_connections = set()
            for ws in self._connections[job_id]:
                try:
                    await ws.send_json(data)
                except Exception:
                    dead_connections.add(ws)
            
            # Cleanup dead connections
            for ws in dead_connections:
                self._connections[job_id].discard(ws)
    
    def send_to_job_sync(self, job_id: str, data: dict, loop: asyncio.AbstractEventLoop = None):
        """
        Gửi message từ synchronous code (background thread).
        
        Dùng khi pipeline runner (chạy trong thread) cần gửi updates.
        """
        if loop is None:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # Không có event loop trong thread này
                # Chỉ lưu vào buffer
                data["timestamp"] = datetime.now().strftime("%H:%M:%S")
                if job_id not in self._message_buffer:
                    self._message_buffer[job_id] = []
                self._message_buffer[job_id].append(data)
                return
        
        try:
            asyncio.run_coroutine_threadsafe(
                self.send_to_job(job_id, data),
                loop
            )
        except Exception:
            # Fallback: chỉ lưu buffer
            data["timestamp"] = datetime.now().strftime("%H:%M:%S")
            if job_id not in self._message_buffer:
                self._message_buffer[job_id] = []
            self._message_buffer[job_id].append(data)
    
    def clear_buffer(self, job_id: str):
        """Xóa buffer của job"""
        if job_id in self._message_buffer:
            del self._message_buffer[job_id]
    
    def get_connection_count(self, job_id: str) -> int:
        """Đếm số client đang kết nối cho job"""
        if job_id in self._connections:
            return len(self._connections[job_id])
        return 0


# Singleton instance
manager = ConnectionManager()
