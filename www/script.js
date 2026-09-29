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
    const btnCancel = document.getElementById('btn-cancel');
    const stageLabel = document.getElementById('stage-label');
    const stageDetail = document.getElementById('stage-detail');
    const rolesBar = document.getElementById('roles-bar');
    const rolesSummary = document.getElementById('roles-summary');
    const btnSwapRoles = document.getElementById('btn-swap-roles');

    const resultPanel = document.getElementById('result-panel');
    const resultTabs = document.getElementById('result-tabs');
    const resultContent = document.getElementById('result-content');
    const impressionBlock = document.getElementById('impression-block');
    const transcriptContent = document.getElementById('transcript-content');
    const jsonContent = document.getElementById('json-content');

    // Must match AUDIO_EXTENSIONS in gui_server.py. .mp4 is a recorded session
    // video; the decoder uses only its audio track.
    const VALID_EXT = ['.wav', '.m4a', '.mp3', '.mp4', '.ogg', '.opus', '.flac', '.aac'];

    // Inside the app window, pywebview's private bridge can give real file
    // paths, so recordings are read where they are instead of being uploaded
    // and copied. In a plain browser there is no bridge, and files are uploaded.
    const hasBridge = () => !!(window.pywebview && window.pywebview.api &&
                               window.pywebview.api.pick_files);

    let selectedFiles = [];
    let pollInterval = null;
    let currentBatchId = null;
    let batchFiles = [];
    let activeIndex = 0;

    // --- File selection ---

    dropZone.addEventListener('click', async () => {
        if (!hasBridge()) { fileInput.click(); return; }
        try {
            addPathEntries(await window.pywebview.api.pick_files());
        } catch (e) {
            fileInput.click();  // bridge unavailable after all: fall back to upload
        }
    });

    // Called by the app (launcher.py) with the real paths of dropped files.
    // Each replaces the matching upload entry the drop handler just added.
    window.cwAddPaths = (entries) => addPathEntries(entries);

    function addPathEntries(entries) {
        (entries || []).forEach(entry => {
            if (!VALID_EXT.some(ext => entry.name.toLowerCase().endsWith(ext))) return;
            const i = selectedFiles.findIndex(f => f.name === entry.name && f.size === entry.size);
            if (i >= 0) selectedFiles[i] = entry;
            else if (!selectedFiles.some(f => f.path === entry.path)) selectedFiles.push(entry);
        });
        renderFileList();
    }

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

    // --- Stop a running batch ---

    btnCancel.addEventListener('click', async () => {
        if (!currentBatchId) return;
        btnCancel.disabled = true;
        btnCancel.textContent = 'Stopping...';
        try {
            await fetch(`/api/cancel/${currentBatchId}`, { method: 'POST' });
            appendLog('Stopping after the current step...');
        } catch (e) {
            appendLog(`Could not stop: ${e.message}`);
        }
    });

    // --- Correct speaker roles and re-score (no re-transcription) ---

    btnSwapRoles.addEventListener('click', async () => {
        const f = batchFiles[activeIndex];
        if (!f || !currentBatchId) return;

        const roles = f.speaker_roles || {};
        const swapped = {};
        Object.keys(roles).forEach(spk => {
            const r = roles[spk];
            swapped[spk] = r === 'Interviewer' ? 'Subject'
                         : r === 'Subject' ? 'Interviewer' : r;
        });

        btnSwapRoles.disabled = true;
        btnSwapRoles.textContent = 'Re-scoring...';
        try {
            const res = await fetch(`/api/rescore/${currentBatchId}/${activeIndex}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ roles: swapped })
            });
            const data = await res.json();
            if (data.status === 'success') {
                batchFiles[activeIndex] = Object.assign({}, f, {
                    result: data.result,
                    structured_transcript: data.structured_transcript,
                    speaker_roles: data.speaker_roles
                });
                renderActive();
                appendLog('Re-scored with corrected speaker roles.');
            } else {
                appendLog(`Re-score failed: ${data.message}`);
            }
        } catch (e) {
            appendLog(`Re-score failed: ${e.message}`);
        } finally {
            btnSwapRoles.disabled = false;
            btnSwapRoles.textContent = 'Swap roles & re-score';
        }
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
        consoleOutput.textContent = 'Preparing...';
        statusText.textContent = 'Processing Pipeline...';
        statusDot.className = 'status-indicator processing';
        progressBar.style.width = '0%';
        btnCancel.disabled = false;
        btnCancel.textContent = 'Stop';
        currentBatchId = null;

        const meta = {
            // Carried into every analysis and the summary CSV, so results can
            // be grouped by participant rather than by filename.
            participant_id: document.getElementById('participant-id').value || '',
            session_label: document.getElementById('session-label').value || '',
            transcribe_only: document.getElementById('transcribe-only').checked,
            // Criterion measure, recorded alongside the audio so the scores can
            // later be correlated against it. Cannot be added retrospectively.
            criterion_score: document.getElementById('criterion-score').value || '',
        };

        const withPaths = selectedFiles.filter(f => f.path);
        try {
            if (withPaths.length === selectedFiles.length && hasBridge()) {
                // Read in place: nothing is uploaded or copied.
                consoleOutput.textContent = 'Starting...';
                const data = await window.pywebview.api.start_batch(
                    { ...meta, paths: withPaths.map(f => f.path) });
                if (data && data.batch_id) {
                    consoleOutput.textContent = `Processing ${data.count} file(s) in place...\n`;
                    currentBatchId = data.batch_id;
                    startPolling(data.batch_id);
                } else {
                    handleError((data && data.message) || 'Could not start processing');
                }
                return;
            }
            if (withPaths.length) {
                handleError('Some files were added by path and some by upload. ' +
                            'Clear the list and add them again with "click to select".');
                return;
            }
            const formData = new FormData();
            selectedFiles.forEach(f => formData.append('files', f));
            formData.append('participant_id', meta.participant_id);
            formData.append('session_label', meta.session_label);
            formData.append('transcribe_only', meta.transcribe_only ? '1' : '');
            formData.append('criterion_score', meta.criterion_score);
            const { ok, data } = await uploadWithProgress(formData);
            if (ok && data.batch_id) {
                consoleOutput.textContent = `Uploaded ${data.count} file(s). Starting...\n`;
                currentBatchId = data.batch_id;
                startPolling(data.batch_id);
            } else {
                handleError(data.message || 'Upload failed');
            }
        } catch (error) {
            handleError(error.message || String(error));
        }
    });

    // Upload with visible progress (plain-browser fallback). fetch() cannot
    // report upload progress, so this uses XMLHttpRequest.
    function uploadWithProgress(formData) {
        const gb = (n) => (n / 1073741824).toFixed(2) + ' GB';
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.open('POST', '/api/upload');
            xhr.upload.onprogress = (e) => {
                if (!e.lengthComputable) return;
                const pct = (100 * e.loaded) / e.total;
                progressBar.style.width = pct.toFixed(1) + '%';
                consoleOutput.textContent = pct >= 100
                    ? 'Upload complete. Preparing files...'
                    : `Uploading ${pct.toFixed(0)}% (${gb(e.loaded)} of ${gb(e.total)})`;
            };
            xhr.onload = () => {
                try {
                    resolve({ ok: xhr.status >= 200 && xhr.status < 300,
                              data: JSON.parse(xhr.responseText) });
                } catch (err) {
                    reject(new Error(`Upload failed (HTTP ${xhr.status})`));
                }
            };
            xhr.onerror = () => reject(new Error('Upload failed: the app did not respond.'));
            xhr.send(formData);
        });
    }

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
                    // Within-file stage progress, so a long transcription does
                    // not look frozen between files.
                    const perFile = 100 / data.total;
                    const inner = typeof data.stage_fraction === 'number'
                        ? data.stage_fraction * perFile : 0;
                    progressBar.style.width = `${(data.done * perFile) + inner}%`;
                }

                stageLabel.textContent = data.stage || '';
                stageDetail.textContent = data.stage_detail || '';

                const finished = data.status.startsWith('COMPLETED');
                const errored = data.status.startsWith('ERROR');
                const cancelled = data.status.startsWith('CANCELLED');

                if (finished || errored || cancelled) {
                    clearInterval(pollInterval);
                    btnCancel.disabled = true;
                    btnCancel.textContent = 'Stop';
                    stageLabel.textContent = '';
                    stageDetail.textContent = '';
                    batchFiles = data.files || [];
                    const anyDone = batchFiles.some(f => f.state === 'done');

                    if (errored && !anyDone) {
                        handleError(data.status.replace(/^ERROR:\s*/, ''));
                        return;
                    }

                    if (cancelled) {
                        statusText.textContent = 'Stopped';
                        statusDot.className = 'status-indicator';
                        if (!anyDone) { enableInputs(); return; }
                    }

                    const failed = batchFiles.filter(f => f.state === 'error').length;
                    const warned = batchFiles.some(f => (f.warnings || []).length);

                    if (!cancelled) statusText.textContent = failed || warned
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
        'word_count', 'duration_minutes', 'participant_speech_min',
        // Measured from the timestamps: identical on every run.
        'subject_speech_rate_wps', 'subject_response_latency_median_s',
        'subject_pause_mean_s', 'subject_filler_rate',
        // LLM judgments: shown with their measured test-retest reliability.
        'hesitancy_score', 'affect_flatness',
        'engagement_level', 'elaboration_positive', 'elaboration_negative',
        'psychomotor_indicators',
        'subject_pitch_mean_st', 'subject_pitch_cv', 'subject_loudness_mean_db',
        'subject_jitter', 'subject_shimmer',
        'vta', 'pitch_mean_st', 'pitch_cv',
        'loudness_mean_db', 'loudness_cv', 'jitter', 'shimmer'
    ];

    function reliabilityNote(rel) {
        const icc = rel.icc.toFixed(2).replace(/^0/, '');
        const runs = rel.runs === 1 ? 'one run' : `mean of ${rel.runs} runs`;
        if (rel.adequate) return `Test-retest ICC ${icc} (${runs})`;
        let note = `Test-retest ICC ${icc} (${runs}): too unstable to analyse on its own`;
        if (rel.use_instead) note += `. Use the measured ${rel.use_instead.map(k => prettify(k).replace('Participant ', '').toLowerCase()).join(', ')} instead`;
        return note + '.';
    }

    function renderActive() {
        const f = batchFiles[activeIndex];
        if (!f) return;

        resultContent.innerHTML = '';
        impressionBlock.innerHTML = '';

        const roles = f.speaker_roles || {};
        const roleEntries = Object.keys(roles);
        if (roleEntries.length && f.state === 'done') {
            rolesSummary.textContent = roleEntries.map(k => `${k} → ${roles[k]}`).join(', ');
            rolesBar.classList.remove('hidden');
        } else {
            rolesBar.classList.add('hidden');
        }

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

        const flags = (f.quality || {}).flags || [];
        if (flags.length) {
            const block = document.createElement('div');
            block.className = 'quality-block';
            const h = document.createElement('h4');
            h.textContent = 'Check before using these numbers';
            const ul = document.createElement('ul');
            flags.forEach(flag => {
                const li = document.createElement('li');
                li.textContent = flag.message;
                ul.appendChild(li);
            });
            block.appendChild(h);
            block.appendChild(ul);
            resultContent.appendChild(block);
        }

        const result = f.result || {};
        const reliability = f.score_reliability || {};
        SCORE_KEYS.forEach(key => {
            if (result[key] === undefined || result[key] === null || result[key] === '') return;
            let val = result[key];
            if (typeof val === 'number') val = Number.isInteger(val) ? val : val.toFixed(3);
            const item = document.createElement('div');
            item.className = 'result-item';
            const label = document.createElement('span');
            label.className = 'result-label';
            label.textContent = prettify(key);
            const value = document.createElement('span');
            value.className = 'result-val';
            value.textContent = val;
            item.appendChild(label);
            item.appendChild(value);
            const rel = reliability[key];
            if (rel) {
                const note = document.createElement('span');
                note.className = 'result-note';
                note.textContent = reliabilityNote(rel);
                item.appendChild(note);
                if (!rel.adequate) item.classList.add('unreliable');
            }
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

    // Plain names for the measures; anything unlisted falls back to Title Case.
    const LABELS = {
        participant_speech_min: 'Participant speech (min)',
        subject_speech_rate_wps: 'Participant speech rate (words/s)',
        subject_response_latency_median_s: 'Participant response latency (s, median)',
        subject_pause_mean_s: 'Participant mean pause (s)',
        subject_pause_proportion: 'Participant time spent pausing',
        subject_filler_rate: 'Participant fillers per 100 words',
        subject_pitch_mean_st: 'Participant pitch (semitones)',
        subject_pitch_cv: 'Participant pitch variability',
        subject_loudness_mean_db: 'Participant loudness (dB)',
        subject_jitter: 'Participant jitter',
        subject_shimmer: 'Participant shimmer',
        vta: 'VTA (whole recording)',
        pitch_mean_st: 'Pitch, semitones (whole recording)',
        pitch_cv: 'Pitch variability (whole recording)',
        loudness_mean_db: 'Loudness, dB (whole recording)',
        loudness_cv: 'Loudness variability (whole recording)',
        jitter: 'Jitter (whole recording)',
        shimmer: 'Shimmer (whole recording)',
    };

    function prettify(key) {
        if (LABELS[key]) return LABELS[key];
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

    // After a crash the process leaves no visible trace; surface it here so
    // the log reaches whoever is debugging it.
    (async function checkPreviousCrash() {
        try {
            const res = await fetch('/api/diagnostics');
            const data = await res.json();
            if (data.filevault === false) {
                document.getElementById('filevault-notice').classList.remove('hidden');
            }
            const crash = data.previous_crash;
            if (!crash) return;
            const notice = document.getElementById('crash-notice');
            const where = [crash.stage, crash.detail].filter(Boolean).join(' — ');
            document.getElementById('crash-stage').textContent = where ? `while: ${where}.` : '';
            notice.classList.remove('hidden');
            document.getElementById('btn-reveal-logs').addEventListener('click', () => {
                fetch('/api/diagnostics/reveal', { method: 'POST' });
            });
            document.getElementById('btn-dismiss-crash').addEventListener('click', () => {
                fetch('/api/diagnostics/dismiss', { method: 'POST' });
                notice.classList.add('hidden');
            });
        } catch (e) { /* diagnostics are best-effort */ }
    })();

    function enableInputs() {
        btnClear.disabled = false;
        btnProcess.disabled = selectedFiles.length === 0;
    }
});
