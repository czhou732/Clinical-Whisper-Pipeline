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
    const rolesBar = document.getElementById('speakers-panel');
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
    // Expected seconds for the files about to run, for "About 3 min left".
    let lastEstimate = null;
    // Seconds of processing per second of audio on this Mac (from /api/diagnostics).
    let speed = null;
    let pollInterval = null;
    let currentBatchId = null;
    let batchFiles = [];
    let activeIndex = 0;

    // --- File selection ---

    // In the app window the page can open before the bridge is ready. Wait
    // for it briefly: without it, files fall back to an upload, and the app's
    // web view cannot read file contents, so that upload never finishes.
    const inApp = new URLSearchParams(location.search).has('app') || !!window.pywebview;
    function whenBridge(ms = 5000) {
        if (hasBridge()) return Promise.resolve(true);
        return new Promise(resolve => {
            const t = setTimeout(() => resolve(hasBridge()), ms);
            window.addEventListener('pywebviewready', () => {
                clearTimeout(t);
                resolve(hasBridge());
            }, { once: true });
        });
    }

    dropZone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); dropZone.click(); }
    });
    dropZone.addEventListener('click', async () => {
        // In a browser, open the picker straight away: it only opens from a click.
        if (!inApp) { fileInput.click(); return; }
        if (!(await whenBridge())) {
            alert('ClinicalWhisper is still starting. Wait a moment and click again.');
            return;
        }
        try {
            addPathEntries(await window.pywebview.api.pick_files());
        } catch (e) {
            alert('Could not open the file chooser. Quit ClinicalWhisper and open it again.');
        }
    });

    // Called by the app (launcher.py) with the real paths of dropped files.
    let dropTimer = null;
    window.cwAddPaths = (entries) => {
        clearTimeout(dropTimer);
        addPathEntries(entries);
    };

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
        if (!inApp) { addFiles(Array.from(e.dataTransfer.files)); return; }
        // In the app, the dropped files' locations arrive from launcher.py
        // through cwAddPaths. If they don't, say so rather than queue files
        // the app cannot read.
        const names = Array.from(e.dataTransfer.files).map(f => f.name);
        clearTimeout(dropTimer);
        dropTimer = setTimeout(() => alert(
            `ClinicalWhisper couldn't get the location of ${names.join(', ') || 'the dropped files'}. ` +
            'Click the box to choose the files instead.'), 3000);
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

    let editingIndex = -1;

    // --- The picked recording, drawn as its own waveform ---

    const waveCanvas = document.getElementById('file-wave');

    function secondsOf(text) {
        const parts = String(text || '').trim().split(':').map(Number);
        if (!parts.length || parts.some(isNaN)) return null;
        return parts.reduce((a, b) => a * 60 + b, 0);
    }

    // Kept stretches as fractions of the recording, for dimming skipped parts.
    function keptFractions(entry, duration) {
        const e = entry && entry.cwEdits;
        if (!e || !duration) return null;
        const start = secondsOf(e.start) || 0;
        const end = secondsOf(e.end) || duration;
        const skips = String(e.skip || '').split(/[,;\n]+/).map(r => r.split(/\s*[-–]\s*/))
            .filter(r => r.length === 2).map(r => [secondsOf(r[0]), secondsOf(r[1])])
            .filter(r => r[0] !== null && r[1] !== null);
        return x => {
            const t = x * duration;
            return t >= start && t <= end && !skips.some(([a, b]) => t >= a && t < b);
        };
    }

    async function peaksFor(entry) {
        if (entry.cwPeaks) return entry.cwPeaks;
        try {
            if (entry.preview) {
                const res = await fetch(`/api/peaks/${entry.preview}?n=160`);
                if (res.ok) entry.cwPeaks = (await res.json()).peaks;
            } else if (entry instanceof File && entry.size < 60 * 1048576) {
                const ctx = new (window.AudioContext || window.webkitAudioContext)();
                const audio = await ctx.decodeAudioData(await entry.arrayBuffer());
                const data = audio.getChannelData(0);
                const n = 160, step = Math.max(1, Math.floor(data.length / n));
                const rms = Array.from({ length: n }, (_, i) => {
                    let acc = 0;
                    for (let j = i * step; j < Math.min(data.length, (i + 1) * step); j += 8) acc += data[j] * data[j];
                    return Math.sqrt(acc / (step / 8));
                });
                const top = [...rms].sort((a, b) => a - b)[Math.floor(n * 0.98)] || 1;
                entry.cwPeaks = rms.map(v => Math.min(1, v / top));
                ctx.close();
            }
        } catch (e) { /* the picture is optional */ }
        return entry.cwPeaks || null;
    }

    async function drawWave(entry) {
        const c = waveCanvas;
        const r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
        c.width = Math.max(1, Math.round(r.width * dpr));
        c.height = Math.max(1, Math.round(r.height * dpr));
        const ctx = c.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, r.width, r.height);
        if (!entry) return;
        const peaks = await peaksFor(entry);
        if (!entry.duration) entry.duration = await durationOf(entry);
        if (!peaks || entry !== selectedFiles[editingIndex >= 0 ? editingIndex : 0]) return;
        const css = getComputedStyle(document.documentElement);
        const on = css.getPropertyValue('--muted').trim(), off = css.getPropertyValue('--line').trim();
        const kept = keptFractions(entry, entry.duration);
        const step = r.width / peaks.length;
        peaks.forEach((a, i) => {
            const h = Math.max(1.5, a * (r.height - 14));
            ctx.fillStyle = !kept || kept((i + 0.5) / peaks.length) ? on : off;
            ctx.fillRect(i * step + step * 0.2, (r.height - h) / 2, Math.max(1, step * 0.6), h);
        });
    }

    function editSummary(e) {
        if (!e) return '';
        const parts = [];
        if (e.start) parts.push(`from ${e.start}`);
        if (e.end) parts.push(`to ${e.end}`);
        const n = (e.skip || '').split(/[,;\n]+/).filter(x => x.trim()).length;
        if (n) parts.push(`${n} skipped`);
        return parts.join(' · ');
    }

    function renderFileList() {
        fileList.innerHTML = '';
        selectedFiles.forEach((file, i) => {
            const li = document.createElement('li');
            li.className = 'file-row';
            const mb = (file.size / 1048576).toFixed(1);
            li.innerHTML = `<span class="file-row-name"></span>
                            <span class="file-row-edits"></span>
                            <span class="file-row-size">${mb} MB</span>
                            <button class="btn-edit link-button" data-i="${i}">Edit</button>
                            <button class="btn-remove" data-i="${i}" aria-label="Remove">&times;</button>`;
            li.querySelector('.file-row-name').textContent = file.name;
            li.querySelector('.file-row-edits').textContent = editSummary(file.cwEdits);
            fileList.appendChild(li);
            if (i === editingIndex) fileList.appendChild(buildEditor(file));
        });

        fileCount.textContent = selectedFiles.length === 1
            ? '1 file selected'
            : `${selectedFiles.length} files selected`;

        fileInfo.classList.toggle('hidden', selectedFiles.length === 0);
        dropZone.classList.toggle('hidden', selectedFiles.length > 0);
        btnProcess.disabled = selectedFiles.length === 0;
        updateEstimate();
        processLabel();
        drawWave(selectedFiles[editingIndex >= 0 ? editingIndex : 0]);
    }

    // Mark where to start and end, and stretches to leave out. Times stay
    // those of the original recording in every result.
    function buildEditor(file) {
        const li = document.createElement('li');
        li.className = 'file-editor';
        const e = Object.assign({ start: '', end: '', skip: '' }, file.cwEdits || {});
        li.innerHTML = `
            <canvas class="editor-wave" aria-label="Drag across the waveform to select a stretch"></canvas>
            <div class="editor-actions editor-sel">
                <span class="text-sm editor-sel-text">Drag across the waveform to select a stretch, or click to jump there.</span>
                <button class="link-button" data-sel="play" disabled>Play selection</button>
                <button class="link-button" data-sel="cut" disabled>Leave this out</button>
                <button class="link-button" data-sel="only" disabled>Process only this</button>
            </div>
            <audio controls preload="metadata" class="editor-player"></audio>
            <p class="text-sm editor-noplay hidden">This format can't be played here;
                type the times instead.</p>
            <div class="editor-grid">
                <label>Start at <input id="edit-start" placeholder="mm:ss"></label>
                <button class="link-button" data-set="start">Use player time</button>
                <label>End at <input id="edit-end" placeholder="mm:ss"></label>
                <button class="link-button" data-set="end">Use player time</button>
            </div>
            <label class="editor-skip">Leave out
                <input id="edit-skip" placeholder="e.g. 02:30-05:00, 1:10:00-1:12:00"></label>
            <div class="editor-actions">
                <button class="link-button" data-mark="from">Leave out from player time…</button>
                <button class="link-button" data-mark="to" disabled>…to player time</button>
                <span class="text-sm editor-hint"></span>
            </div>
            <div class="editor-actions">
                <button class="btn-secondary" data-act="done">Done</button>
                <button class="link-button" data-act="reset">Use the whole recording</button>
            </div>`;
        const player = li.querySelector('audio');
        const src = file.preview ? `/api/preview/${file.preview}`
            : (file instanceof File ? URL.createObjectURL(file) : '');
        if (src) player.src = src;
        else player.classList.add('hidden');
        player.addEventListener('error', () => {
            player.classList.add('hidden');
            li.querySelector('.editor-noplay').classList.remove('hidden');
        });
        const field = id => li.querySelector('#edit-' + id);
        field('start').value = e.start;
        field('end').value = e.end;
        field('skip').value = e.skip;
        let markFrom = null;

        // Waveform: drag to select, click to seek, skipped stretches shaded.
        const wave = li.querySelector('.editor-wave');
        let sel = null, dragFrom = null, stopAt = null;
        const dur = () => (isFinite(player.duration) && player.duration) || file.duration || 0;
        const tAt = x => {
            const r = wave.getBoundingClientRect();
            return Math.max(0, Math.min(1, (x - r.left) / r.width)) * dur();
        };
        function skips() {
            return field('skip').value.split(/[,;\n]+/).map(r => r.split(/\s*[-–]\s*/))
                .filter(r => r.length === 2).map(r => [secondsOf(r[0]), secondsOf(r[1])])
                .filter(r => r[0] !== null && r[1] !== null);
        }
        async function paint() {
            const r = wave.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
            wave.width = Math.max(1, Math.round(r.width * dpr));
            wave.height = Math.max(1, Math.round(r.height * dpr));
            const ctx = wave.getContext('2d');
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, r.width, r.height);
            const css = getComputedStyle(document.documentElement);
            const D = dur();
            const peaks = await peaksFor(file);
            const x = t => D ? (t / D) * r.width : 0;
            ctx.fillStyle = css.getPropertyValue('--line').trim();
            skips().forEach(([a, b]) => ctx.fillRect(x(a), 0, x(b) - x(a), r.height));
            const st = secondsOf(field('start').value), en = secondsOf(field('end').value);
            if (st) ctx.fillRect(0, 0, x(st), r.height);
            if (en) ctx.fillRect(x(en), 0, r.width - x(en), r.height);
            if (peaks) {
                const step = r.width / peaks.length;
                ctx.fillStyle = css.getPropertyValue('--muted').trim();
                peaks.forEach((a, i) => {
                    const h = Math.max(1.5, a * (r.height - 10));
                    ctx.fillRect(i * step + step * 0.2, (r.height - h) / 2, Math.max(1, step * 0.6), h);
                });
            }
            if (sel) {
                ctx.fillStyle = css.getPropertyValue('--accent-dim').trim() || 'rgba(110,155,255,.2)';
                ctx.fillRect(x(sel[0]), 0, x(sel[1]) - x(sel[0]), r.height);
            }
            if (D && player.currentTime) {
                ctx.fillStyle = css.getPropertyValue('--accent').trim();
                ctx.fillRect(x(player.currentTime), 0, 1.5, r.height);
            }
        }
        function showSel() {
            const has = !!sel && sel[1] - sel[0] >= 0.3;
            li.querySelector('[data-sel="play"]').disabled = !has;
            li.querySelector('[data-sel="cut"]').disabled = !has;
            li.querySelector('[data-sel="only"]').disabled = !has;
            li.querySelector('.editor-sel-text').textContent = has
                ? `Selected ${mmss(sel[0])}–${mmss(sel[1])}`
                : 'Drag across the waveform to select a stretch, or click to jump there.';
            paint();
        }
        wave.addEventListener('pointerdown', ev => {
            if (!dur()) return;
            wave.setPointerCapture(ev.pointerId);
            dragFrom = tAt(ev.clientX);
            sel = null;
        });
        wave.addEventListener('pointermove', ev => {
            if (dragFrom === null) return;
            const t = tAt(ev.clientX);
            sel = [Math.min(dragFrom, t), Math.max(dragFrom, t)];
            showSel();
        });
        wave.addEventListener('pointerup', ev => {
            if (dragFrom === null) return;
            const t = tAt(ev.clientX);
            if (Math.abs(t - dragFrom) < 0.3) {  // a click: jump there
                sel = null;
                player.currentTime = t;
                player.play().catch(() => {});
            }
            dragFrom = null;
            showSel();
        });
        player.addEventListener('timeupdate', () => {
            if (stopAt !== null && player.currentTime >= stopAt) { player.pause(); stopAt = null; }
            paint();
        });
        player.addEventListener('loadedmetadata', paint);
        setTimeout(paint, 0);
        ['start', 'end', 'skip'].forEach(id => field(id).addEventListener('input', paint));

        li.addEventListener('click', ev => {
            const t0 = ev.target;
            if (t0.dataset.sel === 'play' && sel) {
                player.currentTime = sel[0];
                stopAt = sel[1];
                player.play().catch(() => {});
            }
            if (t0.dataset.sel === 'only' && sel) {
                // Transcribe and score just this section; times stay those of the file.
                field('start').value = mmss(sel[0]);
                field('end').value = mmss(sel[1]);
                sel = null;
                showSel();
            }
            if (t0.dataset.sel === 'cut' && sel) {
                const range = `${mmss(sel[0])}-${mmss(sel[1])}`;
                field('skip').value = [field('skip').value.trim(), range].filter(Boolean).join(', ');
                sel = null;
                showSel();
            }
        });
        li.addEventListener('click', ev => {
            const t = ev.target;
            if (t.dataset.set) field(t.dataset.set).value = mmss(player.currentTime);
            if (t.dataset.mark === 'from') {
                markFrom = mmss(player.currentTime);
                li.querySelector('[data-mark="to"]').disabled = false;
                li.querySelector('.editor-hint').textContent = `Leaving out from ${markFrom}…`;
            }
            if (t.dataset.mark === 'to' && markFrom) {
                const range = `${markFrom}-${mmss(player.currentTime)}`;
                field('skip').value = [field('skip').value.trim(), range].filter(Boolean).join(', ');
                markFrom = null;
                t.disabled = true;
                li.querySelector('.editor-hint').textContent = '';
            }
            if (t.dataset.act === 'done' || t.dataset.act === 'reset') {
                const edits = t.dataset.act === 'reset' ? null : {
                    start: field('start').value.trim(),
                    end: field('end').value.trim(),
                    skip: field('skip').value.trim(),
                };
                file.cwEdits = edits && (edits.start || edits.end || edits.skip) ? edits : null;
                if (src.startsWith('blob:')) URL.revokeObjectURL(src);
                editingIndex = -1;
                renderFileList();
            }
        });
        return li;
    }

    // Audio length of a picked file: known for files read in place; for an
    // upload, read from the file's own header by the browser.
    function durationOf(entry) {
        if (entry.duration) return Promise.resolve(entry.duration);
        if (!(entry instanceof File)) return Promise.resolve(null);
        return new Promise(resolve => {
            const audio = new Audio();
            const url = URL.createObjectURL(entry);
            const done = (v) => { URL.revokeObjectURL(url); resolve(v); };
            audio.preload = 'metadata';
            audio.onloadedmetadata = () => done(isFinite(audio.duration) ? audio.duration : null);
            audio.onerror = () => done(null);
            audio.src = url;
        });
    }

    function friendly(seconds) {
        const m = Math.round(seconds / 60);
        if (m < 1) return 'under a minute';
        if (m < 60) return `about ${m} min`;
        return `about ${Math.floor(m / 60)} h ${String(m % 60).padStart(2, '0')} min`;
    }

    async function updateEstimate() {
        const el = document.getElementById('estimate');
        if (!selectedFiles.length || !speed) { el.textContent = ''; return; }
        const durations = await Promise.all(selectedFiles.map(durationOf));
        const known = durations.filter(d => d);
        if (!known.length) { el.textContent = ''; return; }
        const audio = known.reduce((a, b) => a + b, 0);
        const scoring = !document.getElementById('transcribe-only').checked;
        const base = audio * speed.transcribe;
        const total = base + (scoring ? audio * speed.score : 0);
        lastEstimate = total;
        const said = friendly(total);
        let text = `${said.charAt(0).toUpperCase()}${said.slice(1)} on this Mac`;
        if (scoring) text += ` (${friendly(base)} without clinical scores)`;
        if (known.length < durations.length) text += '; some file lengths unknown';
        if (speed.rough) text += '. Rough until this Mac has processed a few files.';
        el.textContent = text;
    }

    const transcribeOnly = document.getElementById('transcribe-only');
    const MODE_HELP = {
        transcript: 'Masked transcript, speaker roles, and voice and timing measures. '
            + 'The fastest choice for long interviews.',
        scored: 'Adds six clinical ratings from a language model, each shown with its measured '
            + 'test-retest ICC. Several times slower, and the least certain part of the output.',
    };
    function syncMode() {
        const mode = transcribeOnly.checked ? 'transcript' : 'scored';
        document.querySelectorAll('.seg-btn').forEach(b => {
            b.classList.toggle('on', b.dataset.mode === mode);
            b.setAttribute('aria-pressed', String(b.dataset.mode === mode));
            if (b.dataset.mode === 'scored') b.disabled = transcribeOnly.disabled;
        });
        document.getElementById('mode-help').textContent = MODE_HELP[mode];
        processLabel();
        updateEstimate();
    }
    document.querySelectorAll('.seg-btn').forEach(b => b.addEventListener('click', () => {
        if (b.disabled) return;
        transcribeOnly.checked = b.dataset.mode === 'transcript';
        transcribeOnly.dispatchEvent(new Event('change'));
    }));
    transcribeOnly.addEventListener('change', syncMode);
    syncMode();

    function processLabel() {
        const n = selectedFiles.length;
        const verb = transcribeOnly.checked ? 'Transcribe' : 'Transcribe and score';
        btnProcess.textContent = n > 1 ? `${verb} ${n} files` : verb;
    }

    fileList.addEventListener('click', (e) => {
        const edit = e.target.closest('.btn-edit');
        if (edit) {
            const i = parseInt(edit.dataset.i, 10);
            editingIndex = editingIndex === i ? -1 : i;
            renderFileList();
            return;
        }
        const btn = e.target.closest('.btn-remove');
        if (!btn) return;
        selectedFiles.splice(parseInt(btn.dataset.i, 10), 1);
        editingIndex = -1;
        renderFileList();
    });

    btnClear.addEventListener('click', () => {
        selectedFiles = [];
        editingIndex = -1;
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

    // --- Passages for clinician review (keyword screen, not a risk score) ---

    const mmss = sec => {
        sec = Math.floor(sec || 0);
        const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), x = sec % 60;
        const two = n => String(n).padStart(2, '0');
        return h ? `${h}:${two(m)}:${two(x)}` : `${two(m)}:${two(x)}`;
    };

    function renderReview(review) {
        if (!review || !review.note) return;
        const items = review.items || [];
        const block = document.createElement('div');
        block.className = 'review-block' + (items.length ? ' has-items' : '');
        const h = document.createElement('h4');
        h.textContent = items.length
            ? `For clinician review (${items.length})` : 'For clinician review';
        const note = document.createElement('p');
        note.className = 'text-sm review-note';
        note.textContent = review.note;
        block.append(h, note);
        if (items.length) {
            const list = document.createElement('ol');
            list.className = 'review-list';
            items.forEach(it => {
                const li = document.createElement('li');
                const head = document.createElement('div');
                head.className = 'review-head';
                head.textContent = `${mmss(it.start)} · ${it.label} · ${it.role}`;
                li.appendChild(head);
                if (it.context) {
                    const q = document.createElement('p');
                    q.className = 'review-context';
                    q.textContent = `Asked: “${it.context}”`;
                    li.appendChild(q);
                }
                const t = document.createElement('p');
                t.className = 'review-text';
                const i = it.text.toLowerCase().indexOf(it.term.toLowerCase());
                if (i >= 0) {
                    const mark = document.createElement('mark');
                    mark.textContent = it.text.slice(i, i + it.term.length);
                    t.append(it.text.slice(0, i), mark, it.text.slice(i + it.term.length));
                } else {
                    t.textContent = it.text;
                }
                li.appendChild(t);
                list.appendChild(li);
            });
            block.appendChild(list);
        }
        resultContent.appendChild(block);
    }

    // --- Speakers: correct roles, add labels (no re-transcription) ---

    const ROLE_CHOICES = {
        interview: [['Interviewer', 'Interviewer'], ['Subject', 'Participant'], ['Other', 'Other']],
        group: [['Moderator', 'Moderator'], ['Participant', 'Participant'], ['Other', 'Other']],
    };
    const roleText = r => r === 'Subject' ? 'Participant'
        : (r || '').startsWith('Other') ? 'Other' : r;
    const baseRole = r => (r || 'Other').split(' ')[0].split('_')[0];
    let speakerMode = 'interview';
    const minutes = s => s >= 60 ? `${Math.round(s / 60)} min` : `${Math.round(s)} s`;

    function renderSpeakers(f, mode) {
        const roles = f.speaker_roles || {};
        const names = f.speaker_names || {};
        const samples = f.speaker_samples || {};
        const assignment = f.speaker_assignment || {};
        const evidence = assignment.evidence || {};
        const canRemember = new Set(f.can_remember || []);
        speakerMode = mode || assignment.mode || 'interview';
        document.getElementById('speakers-mode').value = speakerMode;
        const why = document.getElementById('speakers-why');
        why.textContent = assignment.uncertain
            ? `Please check these roles. ${assignment.why || ''} Participant measures and scores wait until you apply them.`
            : '';
        if (assignment.uncertain) rolesBar.open = true;
        const ids = Object.keys(roles).sort(
            (a, b) => ((samples[b] || {}).talk_s || 0) - ((samples[a] || {}).talk_s || 0));
        rolesSummary.textContent = ids.map(k =>
            `${names[k] ? names[k] + ' · ' : ''}${roleText(roles[k])}`).join(', ');
        const body = document.getElementById('speakers-rows');
        body.innerHTML = '';
        ids.forEach(k => {
            const tr = document.createElement('tr');
            tr.dataset.speaker = k;
            const idCell = document.createElement('td');
            idCell.className = 'spk-id';
            const whoSpan = document.createElement('span');
            whoSpan.className = 'who ' + (roles[k] === 'Interviewer' ? 'who-a' : roles[k] === 'Subject' ? 'who-b' : '');
            whoSpan.textContent = k;
            idCell.appendChild(whoSpan);
            const talk = document.createElement('span');
            talk.className = 'spk-talk';
            talk.textContent = `${minutes((samples[k] || {}).talk_s || 0)} of speech`;
            idCell.appendChild(talk);
            const reasons = (evidence[k] || {}).reasons || [];
            if (reasons.length) {
                const r = document.createElement('span');
                r.className = 'spk-why';
                r.textContent = reasons.join('; ');
                idCell.appendChild(r);
            }
            const at = (samples[k] || {}).sample_s;
            if (f.preview && typeof at === 'number') {
                const play = document.createElement('button');
                play.className = 'link-button spk-play';
                play.textContent = 'Play a sample';
                play.addEventListener('click', () => playSample(f.preview, at, play));
                idCell.appendChild(play);
            }
            const lines = document.createElement('td');
            lines.className = 'spk-lines';
            ((samples[k] || {}).lines || []).forEach(t => {
                const para = document.createElement('p');
                para.textContent = `“${t}”`;
                lines.appendChild(para);
            });
            if (!lines.childNodes.length) lines.textContent = '(only short replies)';
            const roleCell = document.createElement('td');
            const sel = document.createElement('select');
            sel.className = 'spk-role';
            sel.setAttribute('aria-label', `Role for ${k}`);
            let current = baseRole(roles[k]);
            if (speakerMode === 'group' && current === 'Interviewer') current = 'Moderator';
            if (speakerMode === 'group' && current === 'Subject') current = 'Participant';
            if (speakerMode === 'interview' && current === 'Moderator') current = 'Interviewer';
            if (speakerMode === 'interview' && current === 'Participant') current = 'Subject';
            ROLE_CHOICES[speakerMode].forEach(([value, label]) => {
                const o = document.createElement('option');
                o.value = value;
                o.textContent = label;
                if (value === current) o.selected = true;
                sel.appendChild(o);
            });
            roleCell.appendChild(sel);
            const nameCell = document.createElement('td');
            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'spk-name';
            input.maxLength = 40;
            input.placeholder = 'e.g. P01';
            input.value = names[k] || '';
            input.setAttribute('aria-label', `Label for ${k}`);
            nameCell.appendChild(input);
            if (canRemember.has(k)) {
                const lab = document.createElement('label');
                lab.className = 'spk-remember text-sm';
                const box = document.createElement('input');
                box.type = 'checkbox';
                box.className = 'spk-keep';
                lab.append(box, ' Remember this voice (staff only)');
                lab.title = 'Stores a voice signature on this Mac so this person is recognised as '
                    + 'staff in later sessions. Never stored for participants.';
                nameCell.appendChild(lab);
                const sync = () => { lab.classList.toggle('hidden', !['Interviewer', 'Moderator'].includes(sel.value)); };
                sel.addEventListener('change', sync);
                sync();
            }
            tr.append(idCell, lines, roleCell, nameCell);
            body.appendChild(tr);
        });
        document.getElementById('speakers-status').textContent = '';
    }

    // Five seconds of a speaker, from the original recording.
    const samplePlayer = document.getElementById('sample-player');
    let sampleTimer = null;
    function playSample(token, at, btn) {
        clearTimeout(sampleTimer);
        const src = `/api/preview/${token}`;
        const go = () => {
            samplePlayer.currentTime = Math.max(0, at);
            samplePlayer.play().catch(() => { btn.textContent = "Can't play this format"; });
            sampleTimer = setTimeout(() => samplePlayer.pause(), 5000);
        };
        if (!samplePlayer.src.endsWith(src)) {
            samplePlayer.src = src;
            samplePlayer.addEventListener('loadedmetadata', go, { once: true });
            samplePlayer.load();
        } else {
            go();
        }
    }

    document.getElementById('speakers-mode').addEventListener('change', e => {
        renderSpeakers(batchFiles[activeIndex], e.target.value);
    });

    btnSwapRoles.addEventListener('click', async () => {
        const f = batchFiles[activeIndex];
        if (!f || !currentBatchId) return;
        const status = document.getElementById('speakers-status');
        const roles = {};
        const names = {};
        const remember = [];
        document.querySelectorAll('#speakers-rows tr').forEach(tr => {
            const k = tr.dataset.speaker;
            roles[k] = tr.querySelector('.spk-role').value;
            names[k] = tr.querySelector('.spk-name').value.trim();
            const keep = tr.querySelector('.spk-keep');
            if (keep && keep.checked && !keep.closest('.hidden')) {
                if (!names[k]) { status.textContent = `Give ${k} a label before remembering their voice.`; remember.length = 0; return; }
                remember.push(k);
            }
        });
        if (status.textContent.startsWith('Give')) return;
        if (Object.values(roles).filter(r => r === 'Subject').length > 1) {
            status.textContent = 'Only one speaker can be the participant.';
            return;
        }
        btnSwapRoles.disabled = true;
        btnSwapRoles.textContent = 'Applying...';
        status.textContent = '';
        try {
            const res = await fetch(`/api/rescore/${currentBatchId}/${activeIndex}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ roles, names, remember })
            });
            const data = await res.json();
            if (data.status === 'success') {
                batchFiles[activeIndex] = Object.assign({}, f, {
                    result: data.result,
                    structured_transcript: data.structured_transcript,
                    speaker_roles: data.speaker_roles,
                    speaker_names: data.speaker_names,
                    speaker_assignment: data.speaker_assignment || f.speaker_assignment,
                    clinical_review: data.clinical_review || f.clinical_review,
                    quality: data.quality || f.quality,
                    score_reliability: data.score_reliability || f.score_reliability,
                });
                renderActive();
                appendLog(data.rescored ? 'Re-scored with the confirmed roles.'
                                        : 'Speakers updated.');
                if ((data.remembered || []).length) {
                    document.getElementById('speakers-status').textContent =
                        `Remembered: ${data.remembered.join(', ')}.`;
                }
            } else {
                status.textContent = data.message || 'Could not update speakers.';
            }
        } catch (e) {
            status.textContent = `Could not update speakers: ${e.message}`;
        } finally {
            btnSwapRoles.disabled = false;
            btnSwapRoles.textContent = 'Apply';
        }
    });

    // --- Export ---

    // Zip redacted logs into the results folder and show it in Finder.
    document.querySelectorAll('.save-diagnostics').forEach(btn => {
        btn.addEventListener('click', async () => {
            const label = btn.textContent;
            btn.disabled = true;
            try {
                const res = await fetch('/api/diagnostics/save', { method: 'POST' });
                const data = await res.json();
                btn.textContent = res.ok ? `Saved ${data.name}` : 'Could not save diagnostics';
            } catch (e) {
                btn.textContent = 'Could not save diagnostics';
            }
            setTimeout(() => { btn.textContent = label; btn.disabled = false; }, 6000);
        });
    });

    document.getElementById('btn-open-results').addEventListener('click', () => {
        fetch('/api/results/reveal', { method: 'POST' });
    });

    document.querySelectorAll('.export-group [data-fmt]').forEach(btn => {
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
                    if (data.warning) alert(data.warning);
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
            num_speakers: document.getElementById('num-speakers').value || '',
            // Criterion measure, recorded alongside the audio so the scores can
            // later be correlated against it. Cannot be added retrospectively.
            criterion_score: document.getElementById('criterion-score').value || '',
        };

        const withPaths = selectedFiles.filter(f => f.path);
        if (inApp && withPaths.length < selectedFiles.length) {
            // Never upload from the app window: its web view cannot read the files.
            handleError('Some files could not be located. Click "Clear all", then click ' +
                        'the box to choose the files again.');
            return;
        }
        try {
            if (withPaths.length === selectedFiles.length && hasBridge()) {
                // Read in place: nothing is uploaded or copied.
                consoleOutput.textContent = 'Starting...';
                const data = await window.pywebview.api.start_batch(
                    { ...meta, paths: withPaths.map(f => f.path),
                      edits: withPaths.map(f => f.cwEdits || null) });
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
            formData.append('num_speakers', meta.num_speakers);
            formData.append('criterion_score', meta.criterion_score);
            formData.append('edits', JSON.stringify(selectedFiles.map(f => f.cwEdits || null)));
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

    // "45% · about 2 min left" for the current stage, like the command line.
    // Time left is extrapolated from this stage's own pace so far.
    let stageStart = { key: null, t: 0 };
    function stageProgress(data) {
        const f = data.stage_fraction;
        const key = `${data.current}|${data.stage}`;
        if (key !== stageStart.key) stageStart = { key, t: Date.now() };
        // A raw token count means nothing to a reader; the percentage replaces it.
        const detail = /^\d+ tokens$/.test(data.stage_detail || '') ? '' : (data.stage_detail || '');
        if (typeof f !== 'number' || f <= 0) return detail;
        const parts = [detail, `${Math.round(f * 100)}%`].filter(Boolean);
        const elapsed = (Date.now() - stageStart.t) / 1000;
        if (f >= 0.05 && f < 1 && elapsed > 10) {
            const left = elapsed * (1 - f) / f;
            parts.push(left < 60 ? 'under a minute left' : `about ${friendly(left).replace('about ', '')} left`);
        }
        return parts.join(' · ');
    }

    // Stage timeline: each pipeline stage with the time it took.
    const STAGE_NAMES = [
        [/^decod/i, 'Prepare audio'], [/^transcrib/i, 'Transcribe'],
        [/^de-?identif/i, 'Mask identifiers'], [/^acoustic/i, 'Voice measures'],
        [/^clinical scoring/i, 'Clinical scores'],
    ];
    let timeline = [];
    let runStart = 0;
    function stageName(raw) {
        const hit = STAGE_NAMES.find(([rx]) => rx.test(raw || ''));
        return hit ? hit[1] : (raw || '').replace(/\s+\d+.*$/, '');
    }
    function updateTimeline(data) {
        const name = stageName(data.stage);
        const now = Date.now();
        const last = timeline[timeline.length - 1];
        if (name && (!last || last.name !== name)) {
            if (last) last.end = now;
            const again = timeline.find(t => t.name === name);
            if (again) {
                again.spent = (again.spent || 0) + ((again.end || now) - (again.restart || again.start)) / 1000;
                again.end = null;
                again.restart = now;
                timeline = timeline.filter(t => t !== again).concat([again]);
            }
            else timeline.push({ name, start: now, end: null, spent: 0 });
        }
        const cur = timeline[timeline.length - 1];
        const list = document.getElementById('stages');
        list.innerHTML = '';
        timeline.forEach(t => {
            const li = document.createElement('li');
            const running = t === cur && !t.end;
            li.className = 'stage ' + (running ? 'now' : 'done');
            const secs = ((t.end || now) - (t.restart || t.start)) / 1000 + (t.spent || 0);
            const label = document.createElement('span');
            label.textContent = t.name;
            const bar = document.createElement('div');
            bar.className = 'bar';
            const fill = document.createElement('i');
            if (running && typeof data.stage_fraction === 'number') fill.style.width = `${Math.round(data.stage_fraction * 100)}%`;
            bar.appendChild(fill);
            const time = document.createElement('span');
            time.className = 't';
            time.textContent = secs < 60 ? `${Math.round(secs)} s` : `${Math.floor(secs / 60)} min ${String(Math.round(secs % 60)).padStart(2, '0')}`;
            li.append(label, bar, time);
            list.appendChild(li);
        });
        if (lastEstimate) {
            const left = lastEstimate - (now - runStart) / 1000;
            statusText.textContent = left > 60 ? `About ${friendly(left).replace(/^about /, '')} left`
                : left > 0 ? 'About a minute left' : (cur ? cur.name : 'Processing');
        }
    }

    function startPolling(batchId) {
        if (pollInterval) clearInterval(pollInterval);
        timeline = [];
        runStart = Date.now();
        document.getElementById('stages').innerHTML = '';

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

                stageLabel.textContent = '';
                stageDetail.textContent = stageProgress(data);
                if (!data.status.startsWith('COMPLETED')) updateTimeline(data);

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

                    const lastStage = timeline[timeline.length - 1];
                    if (lastStage && !lastStage.end) { lastStage.end = Date.now(); updateTimeline({}); }
                    if (!cancelled) statusText.textContent = failed || warned
                        ? `Finished, ${failed} failed`.replace(', 0 failed', ' with warnings')
                        : 'Finished';
                    statusDot.className = 'status-indicator ' + (failed || warned ? 'error' : 'success');

                    activeIndex = Math.max(0, batchFiles.findIndex(f => f.state === 'done'));
                    renderTabs();
                    renderActive();
                    resultPanel.classList.remove('hidden');
                    enableInputs();
                } else if (data.current) {
                    progressCount.textContent = data.total > 1
                        ? `${data.done} of ${data.total} done · ${data.current}` : data.current;
                    if (!lastEstimate) statusText.textContent = 'Processing';
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

    function renderActive() {
        const f = batchFiles[activeIndex];
        if (!f) return;

        resultContent.innerHTML = '';
        impressionBlock.innerHTML = '';

        const roles = f.speaker_roles || {};
        const roleEntries = Object.keys(roles);
        if (roleEntries.length && f.state === 'done') {
            renderSpeakers(f);
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

        renderReview(f.clinical_review);

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

        renderMeasures(f.result || {}, f.score_reliability || {},
                       (f.speaker_assignment || {}).mode === 'group');

        renderTranscript(f);

        fetch(`/api/analysis/${currentBatchId}/${activeIndex}`)
            .then(r => r.json())
            .then(a => {
                jsonContent.textContent = JSON.stringify(a, null, 2);
                if ((a.speaker_assignment || {}).mode === 'group') renderPerSpeaker(a);
                if (a.kintsugi) renderKintsugi(a.kintsugi);
                const ev = (a.llm_clinical_scoring || {}).evidence;
                if (ev && Object.keys(ev).length) {
                    evidenceFor = ev;
                    resultContent.querySelectorAll('.measure-group').forEach(g => {
                        if (g.querySelector('h4').textContent.startsWith('Clinical scores')) g.remove();
                    });
                    renderClinical(f.result || {}, f.score_reliability || {});
                    evidenceFor = {};
                }
                const scoring = a.llm_clinical_scoring || {};
                // Say plainly when there are no clinical scores, so the
                // measurements are not mistaken for them.
                document.getElementById('scores-note').classList.toggle(
                    'hidden', Object.keys(scoring).length > 0);
                if (scoring.summary || scoring.clinical_impression) {
                    const h = document.createElement('h4');
                    h.textContent = 'Summary of what was said';
                    const p = document.createElement('p');
                    p.className = 'impression';
                    p.textContent = scoring.summary || scoring.clinical_impression;
                    impressionBlock.appendChild(h);
                    impressionBlock.appendChild(p);
                }
                if ((scoring.key_observations || []).length) {
                    const h = document.createElement('h4');
                    h.textContent = 'Evidence';
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

    // Measures grouped the way they are read; the clinical scores as a table
    // with each one's measured reliability beside it.
    const GROUPS = [
        ['Recording', ['language', 'word_count', 'duration_minutes', 'participant_speech_min', 'snr_db']],
        ['Participant timing', ['subject_speech_rate_wps', 'subject_pause_mean_s',
            'subject_filler_rate', 'subject_response_latency_median_s']],
        ['Elaboration, measured', ['subject_words_per_answer_positive',
            'subject_words_per_answer_neutral', 'subject_words_per_answer_negative',
            'subject_positive_to_neutral_elaboration']],
        ['Participant voice', ['subject_pitch_mean_st', 'subject_pitch_cv',
            'subject_loudness_mean_db', 'subject_jitter', 'subject_shimmer']],
        ['Whole recording', ['vta', 'pitch_mean_st', 'pitch_cv', 'loudness_mean_db',
            'loudness_cv', 'jitter', 'shimmer']],
    ];
    // Labels inside a titled group, where "Participant" or "whole recording"
    // would only repeat the heading.
    const SHORT = {
        language: 'Language', word_count: 'Words', duration_minutes: 'Length (min)',
        participant_speech_min: 'Participant speech (min)',
        subject_speech_rate_wps: 'Speech rate (words/s)', subject_pause_mean_s: 'Mean pause (s)',
        subject_filler_rate: 'Fillers per 100 words',
        subject_response_latency_median_s: 'Response latency, median (s)',
        subject_pitch_mean_st: 'Pitch (semitones)', subject_pitch_cv: 'Pitch variability',
        subject_loudness_mean_db: 'Loudness (dB)', subject_jitter: 'Jitter', subject_shimmer: 'Shimmer',
        vta: 'VTA', pitch_mean_st: 'Pitch (semitones)', pitch_cv: 'Pitch variability',
        loudness_mean_db: 'Loudness (dB)', loudness_cv: 'Loudness variability',
        jitter: 'Jitter', shimmer: 'Shimmer',
        anhedonia_content: 'Interest and pleasure (what they say)',
        depressed_mood_content: 'Low mood (what they say)',
        affect_flatness: 'Flat emotional language', engagement_level: 'Engagement',
        subject_words_per_answer_positive: 'Words per answer, positive questions',
        subject_words_per_answer_neutral: 'Words per answer, neutral questions',
        subject_words_per_answer_negative: 'Words per answer, negative questions',
        subject_positive_to_neutral_elaboration: 'Positive vs neutral answers (ratio)',
    };
    const LANGUAGES = { en: 'English', es: 'Spanish', fr: 'French', de: 'German', it: 'Italian',
        pt: 'Portuguese', nl: 'Dutch', tr: 'Turkish', vi: 'Vietnamese', zh: 'Chinese',
        ja: 'Japanese', ko: 'Korean', ar: 'Arabic', hi: 'Hindi', bn: 'Bengali', te: 'Telugu',
        und: 'Unknown' };
    const CLINICAL = ['anhedonia_content', 'depressed_mood_content', 'affect_flatness',
        'engagement_level'];
    const present = v => v !== undefined && v !== null && v !== '';
    const num = v => {
        if (typeof v !== 'number') return String(v);
        if (Number.isInteger(v)) return String(v);
        const a = Math.abs(v);
        return v.toFixed(a >= 100 ? 0 : a >= 10 ? 1 : 2);
    };

    let evidenceFor = {};
    function renderMeasures(result, reliability, group) {
        GROUPS.forEach(([title, keys]) => {
            // A group has no single participant; its measures are per speaker.
            if (group && title.startsWith('Participant')) return;
            const shown = keys.filter(k => present(result[k]) && !(group && k === 'participant_speech_min'));
            if (!shown.length) return;
            const g = document.createElement('div');
            g.className = 'measure-group';
            const h = document.createElement('h4');
            h.textContent = title;
            const dl = document.createElement('dl');
            dl.className = 'kv';
            shown.forEach(k => {
                const dt = document.createElement('dt');
                dt.textContent = SHORT[k] || prettify(k);
                if (k === 'subject_response_latency_median_s') dt.textContent += ' (not reliable yet)';
                const dd = document.createElement('dd');
                dd.textContent = k === 'language' ? (LANGUAGES[result[k]] || result[k]) : num(result[k]);
                dl.append(dt, dd);
            });
            g.append(h, dl);
            resultContent.appendChild(g);
        });

        renderClinical(result, reliability);
    }

    function renderClinical(result, reliability) {
        const scored = CLINICAL.filter(k => present(result[k]) || result.scorer_version);
        if (!scored.length) return;
        const g = document.createElement('div');
        g.className = 'measure-group';
        const h = document.createElement('h4');
        const runs = (reliability[scored[0]] || {}).runs;
        h.textContent = (runs > 1 ? `Clinical scores, 0-3, mean of ${runs} runs` : 'Clinical scores, 0-3')
            + ' · each needs a quote from the participant';
        const table = document.createElement('table');
        table.className = 'scores';
        table.innerHTML = '<thead><tr><th scope="col">Score</th><th scope="col">Value</th>'
            + '<th scope="col">Test-retest ICC</th></tr></thead>';
        const body = document.createElement('tbody');
        scored.forEach(k => {
            const rel = reliability[k];
            const tr = document.createElement('tr');
            const name = document.createElement('td');
            name.textContent = SHORT[k] || prettify(k);
            const quote = ((evidenceFor || {})[k] || [])[0];
            if (quote) {
                const q = document.createElement('span');
                q.className = 'quote';
                q.textContent = `“${quote}”`;
                name.appendChild(q);
            }
            if (rel && rel.measured === false) {
                tr.className = 'weak';
                const why = document.createElement('span');
                why.textContent = 'Reliability of this scorer version not measured yet.';
                name.appendChild(why);
            } else if (rel && !rel.adequate) {
                tr.className = 'weak';
                const why = document.createElement('span');
                why.textContent = rel.use_instead
                    ? `Too unstable to analyse alone. Use ${rel.use_instead.map(x => (SHORT[x] || prettify(x).replace('Participant ', '')).toLowerCase()).join(' and ')} instead.`
                    : 'Too unstable to analyse alone.';
                name.appendChild(why);
            }
            const val = document.createElement('td');
            val.className = 'data';
            val.textContent = present(result[k]) ? num(result[k]) : 'no evidence';
            const icc = document.createElement('td');
            icc.className = 'data';
            icc.textContent = rel && typeof rel.icc === 'number' ? rel.icc.toFixed(2).replace(/^0/, '')
                : 'not yet measured';
            tr.append(name, val, icc);
            body.appendChild(tr);
        });
        table.appendChild(body);
        g.append(h, table);
        resultContent.appendChild(g);
    }

    // Kintsugi's open voice model (add-on): PHQ-9 / GAD-7 bands, research only.
    function renderKintsugi(k) {
        const g = document.createElement('div');
        g.className = 'measure-group';
        const h = document.createElement('h4');
        h.textContent = 'Voice model (Kintsugi, research estimate)';
        g.appendChild(h);
        if (k.skipped) {
            const p = document.createElement('p');
            p.className = 'text-sm';
            p.textContent = `Not run: ${k.skipped}`;
            g.appendChild(p);
        } else {
            const dl = document.createElement('dl');
            dl.className = 'kv';
            [['Depression band', k.depression], ['Anxiety band', k.anxiety]].forEach(([label, r]) => {
                if (!r) return;
                const dt = document.createElement('dt');
                dt.textContent = label;
                const dd = document.createElement('dd');
                dd.textContent = `${r.label} · score ${r.score.toFixed(2)}`;
                dl.append(dt, dd);
            });
            g.appendChild(dl);
        }
        const note = document.createElement('p');
        note.className = 'text-sm';
        note.textContent = k.note || '';
        g.appendChild(note);
        resultContent.appendChild(g);
    }

    // A group has no single participant: one row per speaker instead.
    function renderPerSpeaker(a) {
        const per = (a.timing_features || {}).per_speaker || {};
        const roles = a.speaker_roles || {};
        const names = a.speaker_names || {};
        const ids = Object.keys(per).filter(k => roles[k] && !roles[k].startsWith('Other'))
            .sort((x, y) => (roles[x] || '').localeCompare(roles[y] || ''));
        if (!ids.length) return;
        const g = document.createElement('div');
        g.className = 'measure-group';
        const h = document.createElement('h4');
        h.textContent = 'Each speaker';
        const table = document.createElement('table');
        table.className = 'scores';
        table.innerHTML = '<thead><tr><th scope="col">Speaker</th><th scope="col">Talk (min)</th>'
            + '<th scope="col">Words/s</th><th scope="col">Mean pause (s)</th>'
            + '<th scope="col">Fillers /100 words</th></tr></thead>';
        const body = document.createElement('tbody');
        ids.forEach(k => {
            const t = per[k] || {};
            const tr = document.createElement('tr');
            const cells = [names[k] ? `${names[k]} (${roles[k]})` : roles[k],
                (t.talk_time_s || 0) / 60, t.speech_rate_wps, t.pause_mean_s, t.filler_rate];
            cells.forEach((v, i) => {
                const td = document.createElement('td');
                if (i) td.className = 'data';
                td.textContent = i === 0 ? v : (present(v) ? num(v) : '–');
                tr.appendChild(td);
            });
            body.appendChild(tr);
        });
        table.appendChild(body);
        g.append(h, table);
        resultContent.appendChild(g);
    }

    // Transcript as turns: who and when on the left, what was said on the right.
    const LINE = /^\[(\d[\d:]*) - (\d[\d:]*)\] ([^:]+): (.*)$/;
    const FILLERS = /\b(um+|uh+|erm|hmm+|mm-?hmm|uh-?huh)\b,?/gi;
    function renderTranscript(f) {
        transcriptContent.innerHTML = '';
        if (f.mask_legend) {
            const key = document.createElement('p');
            key.className = 'mask-key';
            key.textContent = f.mask_legend;
            transcriptContent.appendChild(key);
        }
        const text = f.structured_transcript || f.transcript || '';
        if (!text) {
            transcriptContent.appendChild(document.createTextNode('(no transcript produced)'));
            return;
        }
        const roles = f.speaker_roles || {};
        text.split('\n').forEach(line => {
            const m = line.match(LINE);
            const row = document.createElement('div');
            row.className = 'turn';
            const who = document.createElement('div');
            who.className = 'who';
            const said = document.createElement('p');
            if (!m) { said.textContent = line; row.append(who, said); transcriptContent.appendChild(row); return; }
            const label = m[3];
            if (/Interviewer|Moderator/.test(label)) who.classList.add('who-a');
            else if (/Subject|Participant/.test(label)) who.classList.add('who-b');
            who.textContent = label.replace(/\(Subject\)/, '(Participant)').replace(/^Subject$/, 'Participant');
            const time = document.createElement('small');
            time.textContent = m[1];
            who.appendChild(time);
            // Fillers dimmed and masking tags set apart, built as text nodes.
            m[4].split(/(\[[a-z_]+_\d+\])/).forEach(part => {
                if (/^\[[a-z_]+_\d+\]$/.test(part)) {
                    const tag = document.createElement('span');
                    tag.className = 'tag';
                    tag.textContent = part;
                    said.appendChild(tag);
                    return;
                }
                let last = 0;
                part.replace(FILLERS, (hit, _w, at) => {
                    said.appendChild(document.createTextNode(part.slice(last, at)));
                    const span = document.createElement('span');
                    span.className = 'fill';
                    span.textContent = hit;
                    said.appendChild(span);
                    last = at + hit.length;
                    return hit;
                });
                said.appendChild(document.createTextNode(part.slice(last)));
            });
            row.append(who, said);
            transcriptContent.appendChild(row);
        });
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
        snr_db: 'Voice above background noise (SNR, dB)',
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
    // The base app has no scoring model: scoring is a separate add-on. Until
    // it is installed, every run is a transcription run, and the page says so.
    // Footer link for either add-on (Scoring, or Languages for recordings
    // not in English). Only the app window can open the folder picker.
    (async () => {
        if (inApp) await whenBridge();
        if (!(window.pywebview && window.pywebview.api && window.pywebview.api.install_scoring)) return;
        const link = document.getElementById('btn-install-addon');
        const status = document.getElementById('addon-status');
        link.classList.remove('hidden');
        link.addEventListener('click', async () => {
            link.disabled = true;
            status.textContent = 'Choose the add-on folder. Copying takes about a minute.';
            const res = await window.pywebview.api.install_scoring();
            link.disabled = false;
            status.textContent = !res || res.status === 'cancelled' ? ''
                : res.status === 'error' ? res.message
                : res.addon === 'languages' ? 'Languages add-on installed.'
                : res.addon === 'kintsugi' ? 'Kintsugi voice model installed.'
                : 'Clinical scoring installed. Reopen the app to use it.';
        });
    })();

    async function setupScoringAddon() {
        const box = document.getElementById('transcribe-only');
        const note = document.getElementById('addon-note');
        const text = document.getElementById('addon-note-text');
        const btn = document.getElementById('btn-install-scoring');
        const idle = text.textContent;
        box.checked = true;
        box.disabled = true;
        syncMode();
        note.classList.remove('hidden');
        if (inApp) await whenBridge();
        if (!(window.pywebview && window.pywebview.api && window.pywebview.api.install_scoring)) {
            btn.classList.add('hidden');
            text.textContent += ' Open ClinicalWhisper from Applications to add it.';
            return;
        }
        btn.addEventListener('click', async () => {
            btn.disabled = true;
            text.textContent = 'Choose the "ClinicalWhisper Scoring" folder from the add-on '
                + 'download. Copying takes about a minute.';
            const res = await window.pywebview.api.install_scoring();
            btn.disabled = false;
            if (res && res.status === 'ok') {
                box.disabled = false;
                box.checked = false;
                note.classList.add('hidden');
                syncMode();
            } else {
                text.textContent = (res && res.status === 'error') ? res.message : idle;
            }
        });
    }

    (async function checkPreviousCrash() {
        try {
            const res = await fetch('/api/diagnostics');
            const data = await res.json();
            if (data.filevault === false) {
                document.getElementById('filevault-notice').classList.remove('hidden');
            }
            if (data.speed) { speed = data.speed; updateEstimate(); }
            if (data.version) {
                document.getElementById('app-version').textContent = `Version ${data.version}.`;
            }
            const syncText = [data.data_root_synced,
                data.legacy_synced && ('Results from earlier versions: ' + data.legacy_synced)]
                .filter(Boolean).join(' ');
            if (syncText) {
                document.getElementById('sync-notice-text').textContent = syncText;
                document.getElementById('sync-notice').classList.remove('hidden');
            }
            if (data.scoring_available === false) setupScoringAddon();
            // Smaller Macs: scoring's 4.9 GB model would push memory into swap,
            // so start with it off. It can still be ticked back on.
            else if (data.ram_gb && data.ram_gb < 16) {
                document.getElementById('transcribe-only').checked = true;
                syncMode();
                document.getElementById('transcribe-only-hint').textContent =
                    `Recommended on this Mac (${Math.round(data.ram_gb)} GB memory).`;
            }
            const crash = data.previous_crash;
            if (!crash) return;
            const notice = document.getElementById('crash-notice');
            const where = [crash.stage, crash.detail].filter(Boolean).join(' — ');
            document.getElementById('crash-stage').textContent = where ? `while: ${where}.` : '';
            notice.classList.remove('hidden');
            document.getElementById('btn-dismiss-crash').addEventListener('click', () => {
                fetch('/api/diagnostics/dismiss', { method: 'POST' });
                notice.classList.add('hidden');
            });
        } catch (e) { /* diagnostics are best-effort */ }
    })();

    function enableInputs() {
        btnClear.disabled = false;
        btnProcess.disabled = selectedFiles.length === 0;
        updateEstimate();
    }
});
