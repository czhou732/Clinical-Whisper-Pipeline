document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const fileNameDisplay = document.getElementById('file-name');
    const btnClear = document.getElementById('btn-clear');
    const btnProcess = document.getElementById('btn-process');
    
    const consolePanel = document.getElementById('console-panel');
    const consoleOutput = document.getElementById('console-output');
    const statusText = document.getElementById('status-text');
    const statusDot = document.getElementById('status-dot');
    
    const btnSaveOutput = document.getElementById('btn-save-output');
    const resultPanel = document.getElementById('result-panel');
    const resultContent = document.getElementById('result-content');

    let currentFile = null;
    let pollInterval = null;
    let currentTaskId = null;

    // --- Drag and Drop Handlers ---
    
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        // Validate file type
        const validExtensions = ['.wav', '.m4a', '.mp3'];
        const isValid = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
        
        if (!isValid) {
            alert("Invalid file format. Please upload .wav, .m4a, or .mp3");
            return;
        }

        currentFile = file;
        fileNameDisplay.textContent = file.name;
        dropZone.classList.add('hidden');
        fileInfo.classList.remove('hidden');
        checkProcessReady();
    }

    btnClear.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        dropZone.classList.remove('hidden');
        fileInfo.classList.add('hidden');
        checkProcessReady();
    });

    function checkProcessReady() {
        btnProcess.disabled = !currentFile;
    }
    
    // --- Save Output ---
    if(btnSaveOutput) {
        btnSaveOutput.addEventListener('click', async () => {
            if(!currentTaskId) return;
            try {
                btnSaveOutput.disabled = true;
                btnSaveOutput.textContent = "Saving...";
                
                const response = await fetch(`/api/save_output/${currentTaskId}`, {
                    method: 'POST'
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    consoleOutput.textContent += `\nOutput saved successfully to:\n${data.path}`;
                } else if (data.status === 'error') {
                    consoleOutput.textContent += `\nFailed to save output: ${data.message}`;
                }
            } catch(e) {
                console.error("Save output failed", e);
            } finally {
                btnSaveOutput.disabled = false;
                btnSaveOutput.textContent = "Save CSV...";
            }
        });
    }

    // --- Processing Handlers ---

    btnProcess.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI Updates
        btnProcess.disabled = true;
        btnClear.disabled = true;
        consolePanel.classList.remove('hidden');
        resultPanel.classList.add('hidden');
        consoleOutput.textContent = "Initializing upload...";
        statusText.textContent = "Processing Pipeline...";
        statusDot.className = 'status-indicator processing';
        currentTaskId = null;

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                consoleOutput.textContent = "File uploaded. Starting processing...\n";
                currentTaskId = data.task_id;
                startPolling(currentTaskId);
            } else {
                handleError(data.message || "Upload failed");
            }
        } catch (error) {
            handleError(error.message);
        }
    });

    function startPolling(taskId) {
        if (pollInterval) clearInterval(pollInterval);
        
        pollInterval = setInterval(async () => {
            try {
                const response = await fetch(`/api/status/${taskId}`);
                const data = await response.json();
                
                if (data.log) {
                    consoleOutput.textContent = data.log;
                    consoleOutput.scrollTop = consoleOutput.scrollHeight;
                }

                if (data.status.includes('COMPLETED')) {
                    clearInterval(pollInterval);
                    consoleOutput.textContent += `\nPipeline finished successfully.`;
                    statusText.textContent = "Completed";
                    statusDot.className = 'status-indicator success';

                    if(data.result) {
                        renderResults(data.result, data.warnings || []);
                    }

                    // A run with a failed stage still returns a result, just
                    // with blank columns. Say so instead of showing a clean tick.
                    if (data.warnings && data.warnings.length) {
                        statusText.textContent = "Completed with warnings";
                        statusDot.className = 'status-indicator error';
                    }

                    enableInputs();
                } else if (data.status.includes('ERROR')) {
                    clearInterval(pollInterval);
                    handleError(data.status.replace(/^ERROR:\s*/, ''));
                } else if (!data.log) {
                    consoleOutput.textContent = data.status;
                }
            } catch (error) {
                console.error("Polling error:", error);
            }
        }, 2000);
    }
    
    function renderResults(result, warnings) {
        resultContent.innerHTML = '';

        if (warnings && warnings.length) {
            const banner = document.createElement('div');
            banner.className = 'result-warning';
            banner.innerHTML = `<strong>Incomplete analysis</strong><br>` +
                warnings.map(w => w.replace(/</g, '&lt;')).join('<br>') +
                `<br><em>Fields below that are missing were not measured.</em>`;
            resultContent.appendChild(banner);
        }

        const keysToShow = [
            'word_count', 'duration_minutes', 'hesitancy_score', 'affect_flatness',
            'engagement_level', 'elaboration_positive', 'elaboration_negative',
            'psychomotor_indicators', 'vta', 'pitch_cv', 'loudness_cv'
        ];
        
        keysToShow.forEach(key => {
            if (result[key] !== undefined && result[key] !== null) {
                const prettyLabel = key.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
                
                let val = result[key];
                if (typeof val === 'number') {
                    val = Number.isInteger(val) ? val : val.toFixed(3);
                }
                
                const item = document.createElement('div');
                item.className = 'result-item';
                item.innerHTML = `
                    <span class="result-label">${prettyLabel}</span>
                    <span class="result-val">${val}</span>
                `;
                resultContent.appendChild(item);
            }
        });
        
        resultPanel.classList.remove('hidden');
    }

    function handleError(message) {
        consoleOutput.textContent += `\nError: ${message}`;
        statusText.textContent = "Failed";
        statusDot.className = 'status-indicator error';
        enableInputs();
    }

    function enableInputs() {
        btnClear.disabled = false;
        // Keep process disabled unless a new file is added, but for simplicity allow retry if token/file exist
        checkProcessReady();
    }
});
