/**
 * AI Dubbing System - Dashboard JavaScript
 * ==========================================
 * Kết nối với FastAPI backend qua REST API + WebSocket.
 */

// ============================================
// Configuration
// ============================================
const API_BASE = window.location.origin;
const WS_BASE = `ws://${window.location.host}`;

// ============================================
// State
// ============================================
const state = {
    selectedFile: null,
    currentJobId: null,
    isProcessing: false,
    ws: null,
    wsRetryCount: 0,
    maxWsRetries: 5,
};

// ============================================
// Upload Handling
// ============================================
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const optionsPanel = document.getElementById('optionsPanel');
const uploadProgress = document.getElementById('uploadProgress');

// Click to upload
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// File selected
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Drag & Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

function handleFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    const validExts = ['mp4', 'mkv', 'avi', 'mov', 'webm', 'flv'];

    if (!validExts.includes(ext)) {
        addLog('error', `File không hợp lệ: ${file.name}. Hỗ trợ: MP4, MKV, AVI, MOV, WebM`);
        return;
    }

    if (file.size > 500 * 1024 * 1024) {
        addLog('error', `File quá lớn: ${(file.size / (1024*1024)).toFixed(0)}MB. Giới hạn: 500MB`);
        return;
    }

    state.selectedFile = file;
    
    // Upload to server
    uploadFile(file);
}

async function uploadFile(file) {
    uploadArea.style.display = 'none';
    uploadProgress.style.display = 'block';
    
    const formData = new FormData();
    formData.append('file', file);

    try {
        const xhr = new XMLHttpRequest();
        
        // Track upload progress
        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                document.getElementById('uploadProgressFill').style.width = `${percent}%`;
                document.getElementById('uploadProgressText').textContent = 
                    `Đang upload... ${percent}% (${(e.loaded / (1024*1024)).toFixed(1)}/${(e.total / (1024*1024)).toFixed(1)} MB)`;
            }
        });

        const response = await new Promise((resolve, reject) => {
            xhr.open('POST', `${API_BASE}/api/upload`);
            xhr.onload = () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve(JSON.parse(xhr.responseText));
                } else {
                    const err = JSON.parse(xhr.responseText);
                    reject(new Error(err.detail || 'Upload thất bại'));
                }
            };
            xhr.onerror = () => reject(new Error('Lỗi kết nối server'));
            xhr.send(formData);
        });

        // Upload thành công
        state.currentJobId = response.job_id;
        
        uploadProgress.style.display = 'none';
        optionsPanel.style.display = 'block';
        document.getElementById('selectedFileName').textContent = 
            `${response.filename} (${response.size_mb} MB)`;
        
        addLog('success', `✅ Upload thành công: ${response.filename}`);
        
    } catch (error) {
        uploadProgress.style.display = 'none';
        uploadArea.style.display = 'block';
        addLog('error', `❌ Upload thất bại: ${error.message}`);
    }
}

function clearUpload() {
    state.selectedFile = null;
    state.currentJobId = null;
    fileInput.value = '';
    uploadArea.style.display = 'block';
    optionsPanel.style.display = 'none';
    uploadProgress.style.display = 'none';
}

// ============================================
// Processing (Real API)
// ============================================
async function startProcessing() {
    if (state.isProcessing) return;
    if (!state.currentJobId) {
        addLog('error', 'Chưa upload video!');
        return;
    }

    state.isProcessing = true;
    const startBtn = document.getElementById('startBtn');
    startBtn.disabled = true;
    startBtn.textContent = '⏳ Đang khởi động...';

    // Update monitor badge
    document.getElementById('monitorBadge').textContent = 'LIVE';
    document.getElementById('monitorBadge').classList.add('badge-live');

    // Reset stages
    resetStages();

    // Scroll to monitor
    scrollToSection('monitor');

    try {
        // Connect WebSocket first
        connectWebSocket(state.currentJobId);
        
        // Start pipeline via API
        const skipLipsync = !document.getElementById('lipSyncToggle').checked;
        const translationApi = document.getElementById('translationApi').value;

        const response = await fetch(
            `${API_BASE}/api/jobs/${state.currentJobId}/start?skip_lipsync=${skipLipsync}&translation_provider=${translationApi}`,
            { method: 'POST' }
        );

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Không thể khởi động pipeline');
        }

        addLog('info', `🎬 Pipeline đã bắt đầu...`);
        startBtn.textContent = '⏳ Đang xử lý...';

    } catch (error) {
        state.isProcessing = false;
        startBtn.disabled = false;
        startBtn.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="5,3 19,12 5,21 5,3"/>
            </svg>
            Bắt đầu Dubbing
        `;
        addLog('error', `❌ Lỗi: ${error.message}`);
    }
}

// ============================================
// WebSocket Connection
// ============================================
function connectWebSocket(jobId) {
    if (state.ws) {
        state.ws.close();
    }

    const wsUrl = `${WS_BASE}/ws/jobs/${jobId}`;
    state.ws = new WebSocket(wsUrl);
    state.wsRetryCount = 0;

    state.ws.onopen = () => {
        updateConnectionStatus(true);
        addLog('info', '🔗 Kết nối WebSocket thành công');
    };

    state.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWSMessage(data);
    };

    state.ws.onclose = () => {
        updateConnectionStatus(false);
        
        // Auto reconnect if still processing
        if (state.isProcessing && state.wsRetryCount < state.maxWsRetries) {
            state.wsRetryCount++;
            setTimeout(() => {
                addLog('info', `🔄 Reconnecting WebSocket... (${state.wsRetryCount}/${state.maxWsRetries})`);
                connectWebSocket(jobId);
            }, 2000);
        }
    };

    state.ws.onerror = () => {
        updateConnectionStatus(false);
    };

    // Keep-alive ping
    setInterval(() => {
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send('ping');
        }
    }, 30000);
}

function handleWSMessage(data) {
    if (data.type === 'pong') return;

    if (data.type === 'progress') {
        // Update progress
        updateProgress(data.stage, data.stage_name, data.progress);
        
        // Update stage status
        setStageStatus(data.stage, data.progress);

        // Update log
        if (data.log) {
            addLog(data.log.type || 'info', data.log.message);
        }

        // Update VRAM
        if (data.vram) {
            updateVRAM(data.vram.used, 8192, data.vram.model);
        }

        // Update segments
        if (data.segments) {
            updateSegments(data.segments);
        }
    }

    if (data.type === 'status') {
        if (data.status === 'completed') {
            onPipelineComplete(data);
        } else if (data.status === 'failed') {
            onPipelineFailed(data);
        }
        
        if (data.message) {
            const logType = data.status === 'failed' ? 'error' : 
                           data.status === 'completed' ? 'success' : 'info';
            addLog(logType, data.message);
        }
    }
}

function onPipelineComplete(data) {
    state.isProcessing = false;
    
    // Update UI
    updateProgress(10, 'Hoàn thành!', 100);
    document.getElementById('monitorBadge').textContent = 'DONE';
    document.getElementById('monitorBadge').classList.remove('badge-live');
    document.getElementById('monitorBadge').classList.add('badge-done');

    // Mark all stages as done
    for (let i = 1; i <= 10; i++) {
        markStageDone(i);
    }

    // Show result panel
    const resultPanel = document.getElementById('resultPanel');
    resultPanel.style.display = 'block';
    
    const downloadBtn = document.getElementById('downloadBtn');
    downloadBtn.href = `${API_BASE}/api/jobs/${state.currentJobId}/download`;

    if (data.output_path) {
        const filename = data.output_path.split(/[\\/]/).pop();
        document.getElementById('resultFilename').textContent = filename;
    }

    // Reset start button
    const startBtn = document.getElementById('startBtn');
    startBtn.disabled = false;
    startBtn.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="1,4 1,10 7,10"/>
            <path d="M3.51 15a9 9 0 105.64-11.36L1 10"/>
        </svg>
        Chạy lại
    `;

    // Scroll to result
    resultPanel.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function onPipelineFailed(data) {
    state.isProcessing = false;

    document.getElementById('monitorBadge').textContent = 'ERROR';
    document.getElementById('monitorBadge').classList.remove('badge-live');
    document.getElementById('monitorBadge').classList.add('badge-error');

    const startBtn = document.getElementById('startBtn');
    startBtn.disabled = false;
    startBtn.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="1,4 1,10 7,10"/>
            <path d="M3.51 15a9 9 0 105.64-11.36L1 10"/>
        </svg>
        Thử lại
    `;
}

// ============================================
// UI Updates
// ============================================
function updateProgress(stage, name, percent) {
    // Update ring
    const ring = document.getElementById('progressRing');
    const circumference = 2 * Math.PI * 60;
    const offset = circumference - (percent / 100) * circumference;
    ring.style.strokeDashoffset = offset;
    ring.style.stroke = percent >= 100 ? '#34d399' : 'url(#progressGrad)';

    // Update text
    document.getElementById('progressValue').textContent = Math.round(percent);
    document.getElementById('progressStage').textContent = 
        percent >= 100 ? '✅ Hoàn thành!' : `Stage ${stage}: ${name}`;
}

function setStageStatus(stageNum, progress) {
    const items = document.querySelectorAll('.stage-item');
    items.forEach(item => {
        const num = parseInt(item.dataset.stage);
        if (num < stageNum) {
            markStageDone(num);
        } else if (num === stageNum) {
            item.classList.add('stage-running');
            item.classList.remove('stage-done');
        }
    });
}

function markStageDone(stageNum) {
    const item = document.querySelector(`.stage-item[data-stage="${stageNum}"]`);
    if (item) {
        item.classList.remove('stage-running');
        item.classList.add('stage-done');
    }
}

function resetStages() {
    document.querySelectorAll('.stage-item').forEach(item => {
        item.classList.remove('stage-running', 'stage-done');
    });
    updateProgress(0, '', 0);
    updateVRAM(0, 8192, null);
    document.getElementById('segmentsPanel').style.display = 'none';
    document.getElementById('resultPanel').style.display = 'none';
}

function updateVRAM(used, total, model) {
    const percent = (used / total) * 100;
    const fill = document.getElementById('vramFill');
    fill.style.width = `${percent}%`;

    if (percent > 75) fill.style.background = 'linear-gradient(90deg, #f87171, #ef4444)';
    else if (percent > 50) fill.style.background = 'linear-gradient(90deg, #fbbf24, #f59e0b)';
    else fill.style.background = 'var(--gradient-primary)';

    document.getElementById('vramUsed').textContent = `${used} MB`;
    document.getElementById('vramModel').textContent = 
        model ? `Model: ${model}` : 'Không có model trên GPU';
}

function updateSegments(segments) {
    if (!segments || segments.length === 0) return;

    const panel = document.getElementById('segmentsPanel');
    panel.style.display = 'block';

    const tbody = document.getElementById('segmentsBody');
    tbody.innerHTML = '';

    const speakerColors = {};
    const colors = ['#818cf8', '#34d399', '#fb923c', '#f472b6', '#60a5fa', '#a78bfa'];
    let colorIdx = 0;

    segments.forEach(seg => {
        if (!speakerColors[seg.speaker]) {
            speakerColors[seg.speaker] = colors[colorIdx % colors.length];
            colorIdx++;
        }

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${seg.id}</strong></td>
            <td><span style="color: ${speakerColors[seg.speaker]}">${seg.speaker}</span></td>
            <td>${seg.start.toFixed(1)} - ${seg.end.toFixed(1)}s</td>
            <td>${seg.zh_text || '-'}</td>
            <td>${seg.vi_text || '<em style="opacity:0.5">đang dịch...</em>'}</td>
        `;
        tbody.appendChild(tr);
    });
}

function updateConnectionStatus(connected) {
    const el = document.getElementById('connectionStatus');
    const dot = el.querySelector('.conn-dot');
    const text = el.querySelector('.conn-text');

    if (connected) {
        dot.className = 'conn-dot conn-connected';
        text.textContent = 'Connected';
    } else {
        dot.className = 'conn-dot conn-disconnected';
        text.textContent = 'Offline';
    }
}

// ============================================
// Logging
// ============================================
function addLog(type, message) {
    const container = document.getElementById('logContainer');
    const now = new Date();
    const time = now.toTimeString().slice(0, 8);

    const entry = document.createElement('div');
    entry.className = `log-entry log-${type}`;
    entry.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-msg">${message}</span>
    `;

    container.appendChild(entry);
    container.scrollTop = container.scrollHeight;

    while (container.children.length > 200) {
        container.removeChild(container.firstChild);
    }
}

// ============================================
// Navigation
// ============================================
function scrollToSection(id) {
    document.getElementById(id).scrollIntoView({ behavior: 'smooth' });
}

// Nav active state
const navLinks = document.querySelectorAll('.nav-link');
window.addEventListener('scroll', () => {
    const scrollY = window.scrollY + 100;
    ['hero', 'upload', 'monitor'].forEach(id => {
        const section = document.getElementById(id);
        if (!section) return;
        const top = section.offsetTop;
        const height = section.offsetHeight;
        if (scrollY >= top && scrollY < top + height) {
            navLinks.forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('href') === `#${id}`) {
                    link.classList.add('active');
                }
            });
        }
    });
});

// ============================================
// Intersection Observer for Animations
// ============================================
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

document.querySelectorAll('.feature-card, .monitor-card').forEach((el, i) => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = `opacity 0.5s ease ${i * 0.05}s, transform 0.5s ease ${i * 0.05}s`;
    observer.observe(el);
});

// ============================================
// Init
// ============================================
console.log('🎬 AI Dubbing System Dashboard loaded');
addLog('info', '🔧 Hệ thống sẵn sàng. Upload video để bắt đầu.');
