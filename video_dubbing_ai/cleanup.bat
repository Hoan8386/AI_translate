@echo off
chcp 65001 >nul
REM ============================================================
REM  cleanup.bat — Xóa các thư mục / file không cần thiết
REM  AI Video Dubbing System
REM ============================================================

echo.
echo ============================================================
echo   AI Dubbing System - Dọn dẹp thư mục không cần thiết
echo ============================================================
echo.
echo Các thư mục/file sẽ bị XÓA:
echo   [1] OpenVoiceV2\       (thư mục trống, không dùng)
echo   [2] scripts\           (thư mục trống)
echo   [3] .ag-kit-backups\   (backup tự động của ag-kit)
echo   [4] package.json       (không cần cho Python project)
echo   [5] freeze.txt         (chỉ tham khảo)
echo   [6] versions.txt       (chỉ tham khảo)
echo   [7] pipeline.md        (đã có ARCHITECTURE.md)
echo   [8] test.py            (file test đơn giản)
echo   [9] __pycache__\       (cache Python)
echo  [10] temp\              (file tạm thời)
echo.
echo Các thư mục KHÔNG bị xóa:
echo   venv\, third_party\, pipeline\, services\, api\
echo   web\, config\, models_data\, utils\, input\, output\
echo.

set /p CONFIRM="Ban co chac muon xoa? (Y/N): "
if /i "%CONFIRM%" NEQ "Y" (
    echo Huy bo. Khong co gi bi xoa.
    pause
    exit /b 0
)

echo.
echo Dang xoa...

REM [1] OpenVoiceV2 - thu muc trong
if exist OpenVoiceV2 (
    rmdir /s /q OpenVoiceV2
    echo   [OK] Xoa OpenVoiceV2\
) else (
    echo   [SKIP] OpenVoiceV2 khong ton tai
)

REM [2] scripts - thu muc trong
if exist scripts (
    rmdir /s /q scripts
    echo   [OK] Xoa scripts\
) else (
    echo   [SKIP] scripts khong ton tai
)

REM [3] .ag-kit-backups
if exist .ag-kit-backups (
    rmdir /s /q .ag-kit-backups
    echo   [OK] Xoa .ag-kit-backups\
) else (
    echo   [SKIP] .ag-kit-backups khong ton tai
)

REM [4] package.json
if exist package.json (
    del /q package.json
    echo   [OK] Xoa package.json
) else (
    echo   [SKIP] package.json khong ton tai
)

REM [5] freeze.txt
if exist freeze.txt (
    del /q freeze.txt
    echo   [OK] Xoa freeze.txt
) else (
    echo   [SKIP] freeze.txt khong ton tai
)

REM [6] versions.txt
if exist versions.txt (
    del /q versions.txt
    echo   [OK] Xoa versions.txt
) else (
    echo   [SKIP] versions.txt khong ton tai
)

REM [7] pipeline.md
if exist pipeline.md (
    del /q pipeline.md
    echo   [OK] Xoa pipeline.md
) else (
    echo   [SKIP] pipeline.md khong ton tai
)

REM [8] test.py
if exist test.py (
    del /q test.py
    echo   [OK] Xoa test.py
) else (
    echo   [SKIP] test.py khong ton tai
)

REM [9] __pycache__ (trong thu muc goc va cac thu muc con)
for /d /r . %%d in (__pycache__) do (
    if exist "%%d" (
        rmdir /s /q "%%d"
        echo   [OK] Xoa "%%d"
    )
)

REM [10] temp
if exist temp (
    rmdir /s /q temp
    mkdir temp
    echo   [OK] Xoa noi dung temp\ (giu lai thu muc trong)
) else (
    echo   [SKIP] temp khong ton tai
)

echo.
echo ============================================================
echo   Don dep hoan tat!
echo ============================================================
echo.
echo Cau truc hien tai:
dir /b
echo.
pause
