document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const fileList = document.getElementById('file-list');
    const fileCount = document.getElementById('file-count');
    const btnClear = document.getElementById('btn-clear');
    const btnProcess = document.getElementById('btn-process');

    const consolePanel = document.getElementById('console-panel');
    const consoleOutput = document.getElementById('console-output');
    const statusText = document.getElementById('status-text');
    const statusDot = document.getElementById('status-dot');
    const progressCount = document.getElementById('progress-count');
    const progressBar = document.getElementById('progress-bar');

    const resultPanel = document.getElementById('result-panel');
    const resultTabs = document.getElementById('result-tabs');
    const resultContent = document.getElementById('result-content');
    const impressionBlock = document.getElementById('impression-block');
    const transcriptContent = document.getElementById('transcript-content');
    const jsonContent = document.getElementById('json-content');

    // Must match audio_extensions in config.example.yaml. .mp4 is a recorded
    // session video; ffmpeg strips the audio track during preprocessing.
    const VALID_EXT = ['.wav', '.m4a', '.mp3', '.mp4'];

    let selectedFiles = [];
    let pollInterval = null;
    let currentBatchId = null;
    let batchFiles = [];
    let activeIndex = 0;

    // --- File selection ---

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        addFiles(Array.from(e.dataTransfer.files));
    });

    fileInput.addEventListener('change', (e) => addFiles(Array.from(e.target.files)));

    function addFiles(files) {
        const rejected = [];
        files.forEach(file => {
            const ok = VALID_EXT.some(ext => file.name.toLowerCase().endsWith(ext));
            if (!ok) { rejected.push(file.name); return; }
            if (selectedFiles.some(f => f.name === file.name && f.size === file.size)) return;
            selectedFiles.push(file);
        });

        if (rejected.length) {
            alert(`Skipped ${rejected.length} unsupported file(s):\n${rejected.join('\n')}\n\n` +
                  `Supported: ${VALID_EXT.join(', ')}`);
        }
        renderFileList();
    }

    function renderFileList() {
        fileList.innerHTML = '';
        selectedFiles.forEach((file, i) => {
            const li = document.createElement('li');
            li.className = 'file-row';
            const mb = (file.size / 1048576).toFixed(1);
            li.innerHTML = `<span class="file-row-name"></span>
                            <span class="file-row-size">${mb} MB</span>
                            <button class="btn-remove" data-i="${i}">&times;</button>`;
            li.querySelector('.file-row-name').textContent = file.name;
            fileList.appendChild(li);
        });

        fileCount.textContent = selectedFiles.length === 1
            ? '1 file selected'
            : `${selectedFiles.length} files selected`;

        fileInfo.classList.toggle('hidden', selectedFiles.length === 0);
        dropZone.classList.toggle('hidden', selectedFiles.length > 0);
        btnProcess.disabled = selectedFiles.length === 0;
        btnProcess.textContent = selectedFiles.length > 1
            ? `Process ${selectedFiles.length} Files`
            : 'Process Audio';
    }

    fileList.addEventListener('click', (e) => {
        const btn = e.target.closest('.btn-remove');
        if (!btn) return;
        selectedFiles.splice(parseInt(btn.dataset.i, 10), 1);
        renderFileList();
    });

    btnClear.addEventListener('click', () => {
        selectedFiles = [];
        fileInput.value = '';
        renderFileList();
    });

    // --- Export ---

    document.querySelectorAll('.export-group .btn-secondary').forEach(btn => {
        btn.addEventListener('click', async () => {
            if (!currentBatchId) return;
            const fmt = btn.dataset.fmt;
            const label = btn.textContent;
            try {
                btn.disabled = true;
                btn.textContent = 'Saving...';
                const res = await fetch(`/api/save/${currentBatchId}/${fmt}`, { method: 'POST' });
                const data = await res.json();
                if (data.status === 'success') {
                    appendLog(`Saved ${fmt.toUpperCase()} to ${data.path}`);
                } else if (data.status === 'error') {
                    appendLog(`Save failed: ${data.message}`);
                }
            } catch (err) {
                appendLog(`Save failed: ${err.message}`);
            } finally {
                btn.disabled = false;
                btn.textContent = label;
            }
        });
    });

    // --- Processing ---

    btnProcess.addEventListener('click', async () => {
        if (!selectedFiles.length) return;

        btnProcess.disabled = true;
        btnClear.disabled = true;
        consolePanel.classList.remove('hidden');
        resultPanel.classList.add('hidden');
        consoleOutput.textContent = 'Uploading...';
        statusText.textContent = 'Processing Pipeline...';
        statusDot.className = 'status-indicator processing';
        progressBar.style.width = '0%';
        currentBatchId = null;

        const formData = new FormData();
        selectedFiles.forEach(f => formData.append('files', f));

        try {
            const res = await fetch('/api/upload', { method: 'POST', body: formData });
            const data = await res.json();
            if (res.ok && data.batch_id) {
                consoleOutput.textContent = `Uploaded ${data.count} file(s). Starting...\n`;
                currentBatchId = data.batch_id;
                startPolling(data.batch_id);
            } else {
                handleError(data.message || 'Upload failed');
            }
        } catch (error) {
            handleError(error.message);
        }
    });

    function startPolling(batchId) {
        if (pollInterval) clearInterval(pollInterval);

        pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/api/status/${batchId}`);
                const data = await res.json();

                if (data.log) {
                    consoleOutput.textContent = data.log;
                    consoleOutput.scrollTop = consoleOutput.scrollHeight;
                }

                if (data.total) {
                    progressCount.textContent = `${data.done}/${data.total}`;
                    progressBar.style.width = `${(data.done / data.total) * 100}%`;
                }

                const finished = data.status.startsWith('COMPLETED');
                const errored = data.status.startsWith('ERROR');

                if (finished || errored) {
                    clearInterval(pollInterval);
                    batchFiles = data.files || [];
                    const anyDone = batchFiles.some(f => f.state === 'done');

                    if (errored && !anyDone) {
                        handleError(data.status.replace(/^ERROR:\s*/, ''));
                        return;
                    }

                    const failed = batchFiles.filter(f => f.state === 'error').length;
                    const warned = batchFiles.some(f => (f.warnings || []).length);

                    statusText.textContent = failed || warned
                        ? `Completed — ${failed} failed`.replace(' — 0 failed', ' with warnings')
                        : 'Completed';
                    statusDot.className = 'status-indicator ' + (failed || warned ? 'error' : 'success');

                    activeIndex = Math.max(0, batchFiles.findIndex(f => f.state === 'done'));
                    renderTabs();
                    renderActive();
                    resultPanel.classList.remove('hidden');
                    enableInputs();
                } else if (data.current) {
                    statusText.textContent = `Processing ${data.current}`;
                }
            } catch (error) {
                console.error('Polling error:', error);
            }
        }, 2000);
    }

    // --- Result rendering ---

    function renderTabs() {
        resultTabs.innerHTML = '';
        if (batchFiles.length < 2) return;
        batchFiles.forEach((f, i) => {
            const tab = document.createElement('button');
            tab.className = 'result-tab' + (i === activeIndex ? ' active' : '') +
                            (f.state === 'error' ? ' failed' : '');
            tab.textContent = f.filename;
            tab.addEventListener('click', () => {
                activeIndex = i;
                renderTabs();
                renderActive();
            });
            resultTabs.appendChild(tab);
        });
    }

    const SCORE_KEYS = [
        'word_count', 'duration_minutes', 'hesitancy_score', 'affect_flatness',
        'engagement_level', 'elaboration_positive', 'elaboration_negative',
        'psychomotor_indicators', 'vta', 'pitch_mean_st', 'pitch_cv',
        'loudness_mean_db', 'loudness_cv', 'jitter', 'shimmer'
    ];

    function renderActive() {
        const f = batchFiles[activeIndex];
        if (!f) return;

        resultContent.innerHTML = '';
        impressionBlock.innerHTML = '';

        if (f.state === 'error') {
            const div = document.createElement('div');
            div.className = 'result-warning';
            div.innerHTML = '<strong>This file failed</strong><br>';
            div.appendChild(document.createTextNode(f.error || 'Unknown error'));
            resultContent.appendChild(div);
            transcriptContent.textContent = '';
            jsonContent.textContent = '';
            return;
        }

        if ((f.warnings || []).length) {
            const banner = document.createElement('div');
            banner.className = 'result-warning';
            banner.innerHTML = '<strong>Incomplete analysis</strong><br>' +
                f.warnings.map(escapeHtml).join('<br>') +
                '<br><em>Fields not shown below were not measured.</em>';
            resultContent.appendChild(banner);
        }

        const result = f.result || {};
        SCORE_KEYS.forEach(key => {
            if (result[key] === undefined || result[key] === null) return;
            let val = result[key];
            if (typeof val === 'number') val = Number.isInteger(val) ? val : val.toFixed(3);
            const item = document.createElement('div');
            item.className = 'result-item';
            item.innerHTML = `<span class="result-label">${prettify(key)}</span>
                              <span class="result-val">${val}</span>`;
            resultContent.appendChild(item);
        });

        transcriptContent.textContent =
            f.structured_transcript || f.transcript || '(no transcript produced)';

        fetch(`/api/analysis/${currentBatchId}/${activeIndex}`)
            .then(r => r.json())
            .then(a => {
                jsonContent.textContent = JSON.stringify(a, null, 2);
                const scoring = a.llm_clinical_scoring || {};
                if (scoring.clinical_impression) {
                    const h = document.createElement('h4');
                    h.textContent = 'Clinical Impression';
                    const p = document.createElement('p');
                    p.className = 'impression';
                    p.textContent = scoring.clinical_impression;
                    impressionBlock.appendChild(h);
                    impressionBlock.appendChild(p);
                }
                if ((scoring.key_observations || []).length) {
                    const h = document.createElement('h4');
                    h.textContent = 'Key Observations';
                    const ul = document.createElement('ul');
                    ul.className = 'observations';
                    scoring.key_observations.forEach(o => {
                        const li = document.createElement('li');
                        li.textContent = o;
                        ul.appendChild(li);
                    });
                    impressionBlock.appendChild(h);
                    impressionBlock.appendChild(ul);
                }
            })
            .catch(() => { jsonContent.textContent = '(analysis unavailable)'; });
    }

    document.querySelectorAll('.view-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            ['scores', 'transcript', 'json'].forEach(v => {
                document.getElementById(`view-${v}`).classList.toggle('hidden', v !== btn.dataset.view);
            });
        });
    });

    // --- Helpers ---

    function prettify(key) {
        return key.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
    }

    function escapeHtml(s) {
        return String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
    }

    function appendLog(msg) {
        consoleOutput.textContent += `\n${msg}`;
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
    }

    function handleError(message) {
        appendLog(`Error: ${message}`);
        statusText.textContent = 'Failed';
        statusDot.className = 'status-indicator error';
        enableInputs();
    }

    function enableInputs() {
        btnClear.disabled = false;
        btnProcess.disabled = selectedFiles.length === 0;
    }
});
