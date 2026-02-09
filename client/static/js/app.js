/**
 * PDF Form Processing - Client UI JavaScript
 * Shows request/response payloads inline with spinners
 */

// State
let currentMetadata = null;
let currentFilledPdfUrl = null;  // S3 presigned URL for filled PDF
let formsData = [];
let versionsData = [];

// =============================================================================
// Initialization
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Setup tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });

    // Load config and set API toggle
    loadConfig();

    // Check API health
    checkApiHealth();

    // Load initial data
    loadForms();
    loadVersions();
    loadFormStats();
    loadRuns();
});

async function loadConfig() {
    try {
        const response = await fetch('/api/config');
        const config = await response.json();

        // Set the select to match current API URL
        const select = document.getElementById('api-url-select');
        if (select && config.api_urls) {
            for (const [key, url] of Object.entries(config.api_urls)) {
                if (url === config.api_url) {
                    select.value = key;
                    break;
                }
            }
        }
    } catch (e) {
        console.error('Failed to load config:', e);
    }
}

async function switchApiUrl() {
    const select = document.getElementById('api-url-select');
    const key = select.value;

    try {
        const response = await fetch('/api/config/set-api-url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ key })
        });

        const result = await response.json();
        if (result.success) {
            // Refresh data with new API
            checkApiHealth();
            loadForms();
            loadVersions();
            loadFormStats();
            loadRuns();
        } else {
            alert('Failed to switch API: ' + result.error);
        }
    } catch (e) {
        alert('Failed to switch API: ' + e.message);
    }
}

function switchTab(tabId) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    document.querySelector(`.tab[data-tab="${tabId}"]`).classList.add('active');
    document.getElementById(`${tabId}-tab`).classList.add('active');

    // Refresh data when switching tabs
    if (tabId === 'fill') {
        populateFillFormSelects();
    }
}


// =============================================================================
// API Communication & Sidebar Logging
// =============================================================================

let apiCallCounter = 0;

function logApiCall(method, url, requestBody, response, isSuccess, duration) {
    const logContainer = document.getElementById('api-log');
    if (!logContainer) return;

    apiCallCounter++;
    const timestamp = new Date().toLocaleTimeString();
    const statusCode = response.status_code || (isSuccess ? 200 : 500);
    const statusClass = isSuccess ? 'success' : 'error';

    const entryId = `api-log-entry-${apiCallCounter}`;
    const requestJson = requestBody ? JSON.stringify(requestBody, null, 2) : 'null';
    const responseJson = JSON.stringify(response, null, 2);

    const entryHtml = `
        <div class="api-log-entry ${statusClass}" id="${entryId}">
            <div class="api-log-header" onclick="toggleApiLogEntry('${entryId}')">
                <div class="api-log-summary">
                    <span class="api-log-method">${method}</span>
                    <span class="api-log-url">${url}</span>
                </div>
                <div class="api-log-meta">
                    <span class="api-log-status ${statusClass}">${statusCode}</span>
                    <span class="api-log-time">${duration}ms</span>
                    <span class="api-log-timestamp">${timestamp}</span>
                </div>
            </div>
            <div class="api-log-body" style="display: none;">
                <div class="api-log-section">
                    <div class="api-log-section-header">Request</div>
                    <pre class="api-log-json">${requestJson}</pre>
                </div>
                <div class="api-log-section">
                    <div class="api-log-section-header">Response</div>
                    <pre class="api-log-json">${responseJson}</pre>
                </div>
            </div>
        </div>
    `;

    // Insert at the top of the log
    logContainer.insertAdjacentHTML('afterbegin', entryHtml);

    // Limit to last 50 entries
    const entries = logContainer.querySelectorAll('.api-log-entry');
    if (entries.length > 50) {
        entries[entries.length - 1].remove();
    }
}

function toggleApiLogEntry(entryId) {
    const entry = document.getElementById(entryId);
    if (!entry) return;

    const body = entry.querySelector('.api-log-body');
    if (body) {
        body.style.display = body.style.display === 'none' ? 'block' : 'none';
    }
}

function clearApiLog() {
    const logContainer = document.getElementById('api-log');
    if (logContainer) {
        logContainer.innerHTML = '<p class="muted" style="padding: 10px;">No API calls yet.</p>';
    }
    apiCallCounter = 0;
}

async function apiCall(endpoint, options = {}) {
    const method = options.method || 'GET';
    const requestBody = options.body ? JSON.parse(options.body) : null;
    const startTime = Date.now();

    try {
        const response = await fetch(endpoint, {
            headers: { 'Content-Type': 'application/json' },
            ...options
        });
        const data = await response.json();
        const duration = Date.now() - startTime;

        // Log to sidebar
        logApiCall(method, endpoint, requestBody, data, data.success !== false, duration);

        return data;
    } catch (error) {
        const duration = Date.now() - startTime;
        const errorResponse = {
            success: false,
            status_code: 0,
            response: { error: error.message }
        };

        // Log error to sidebar
        logApiCall(method, endpoint, requestBody, errorResponse, false, duration);

        return errorResponse;
    }
}

async function checkApiHealth() {
    const indicator = document.getElementById('api-status-indicator');
    const text = document.getElementById('api-status-text');

    const result = await apiCall('/proxy/health');

    if (result.success && result.response?.status === 'healthy') {
        indicator.className = 'status-indicator connected';
        text.textContent = 'API Connected';
    } else {
        indicator.className = 'status-indicator disconnected';
        text.textContent = 'API Disconnected';
    }
}

// =============================================================================
// Forms Tab
// =============================================================================

async function addPdf() {
    const filePath = document.getElementById('pdf-file-path').value;
    const oppId = document.getElementById('pdf-opp-id').value;

    if (!filePath || !oppId) {
        alert('Please enter file path and opportunity ID');
        return;
    }

    const requestBody = { file_path: filePath, opportunity_id: oppId };

    const result = await apiCall('/proxy/pdf', {
        method: 'POST',
        body: JSON.stringify(requestBody)
    });

    if (result.success) {
        loadForms();
        loadFormStats();
        alert('PDF added successfully!');
    } else {
        alert('Failed to add PDF: ' + (result.response?.error || 'Unknown error'));
    }
}

async function loadForms(versionFilter = null) {
    let endpoint = '/proxy/forms/list';

    if (versionFilter) {
        endpoint += `?version=${versionFilter}`;
    }

    const result = await apiCall(endpoint);

    if (result.success !== false) {
        formsData = result.forms || [];
        renderFormsList();
        populateFillFormSelects();
        populateVersionFilter();
    }
}

function populateVersionFilter() {
    const filter = document.getElementById('forms-version-filter');
    if (!filter) return;

    const currentValue = filter.value;

    // Get unique versions from all forms
    const allVersions = new Set();
    formsData.forEach(f => {
        (f.versions || []).forEach(v => allVersions.add(v));
    });

    filter.innerHTML = '<option value="">All Versions</option>' +
        Array.from(allVersions).sort().map(v =>
            `<option value="${v}" ${v === currentValue ? 'selected' : ''}>${v}</option>`
        ).join('');
}

async function filterFormsByVersion() {
    const filter = document.getElementById('forms-version-filter');
    const version = filter ? filter.value : null;
    await loadForms(version);
    if (version) {
        await loadFormStats(version);
    } else {
        await loadFormStats();
    }
}

function renderFormsList() {
    const container = document.getElementById('forms-list');

    if (formsData.length === 0) {
        container.innerHTML = '<p class="muted">No forms in system. Add a PDF to get started.</p>';
        return;
    }

    container.innerHTML = formsData.map(form => {
        const versions = form.versions || [];
        const versionBadges = versions.length > 0
            ? versions.map(v => `<span class="badge">${v.substring(0, 8)}</span>`).join(' ')
            : '<span class="badge warning">No versions</span>';

        // Stats data
        const questionCount = form.question_count || 0;
        const pdfMappedCount = form.pdf_field_mapped_count || 0;
        const canonicalMappedCount = form.canonical_mapped_count || 0;

        // Calculate percentages
        const pdfMappingPct = questionCount > 0 ? Math.round((pdfMappedCount / questionCount) * 100) : 0;
        const canonicalMappingPct = questionCount > 0 ? Math.round((canonicalMappedCount / questionCount) * 100) : 0;

        // Determine badge classes based on percentages
        const pdfBadgeClass = pdfMappingPct >= 80 ? 'success' : pdfMappingPct >= 50 ? 'warning' : 'error';
        const canonicalBadgeClass = canonicalMappingPct >= 80 ? 'success' : canonicalMappingPct >= 50 ? 'warning' : 'error';

        return `
            <div class="list-item ${form.is_form ? '' : 'not-form'}">
                <div class="list-item-title">
                    ${form.opportunity_id}
                    ${!form.is_form ? '<span class="badge warning">Not a form</span>' : ''}
                </div>
                <div class="list-item-meta">
                    ${form.filename}<br>
                    <code>${(form.checksum || '').substring(0, 16)}...</code>
                </div>
                <div class="list-item-stats-row">
                    <div class="stat-chip">
                        <span class="stat-chip-value">${questionCount}</span>
                        <span class="stat-chip-label">Questions</span>
                    </div>
                    <div class="stat-chip ${pdfBadgeClass}">
                        <span class="stat-chip-value">${pdfMappedCount}/${questionCount}</span>
                        <span class="stat-chip-label">PDF Fields (${pdfMappingPct}%)</span>
                    </div>
                    <div class="stat-chip ${canonicalBadgeClass}">
                        <span class="stat-chip-value">${canonicalMappedCount}/${questionCount}</span>
                        <span class="stat-chip-label">Canonical (${canonicalMappingPct}%)</span>
                    </div>
                </div>
                <div class="list-item-versions">
                    Versions: ${versionBadges}
                </div>
            </div>
        `;
    }).join('');
}

// =============================================================================
// Canonical Tab
// =============================================================================

let canonicalStatsData = [];
let runsData = [];

async function loadRuns() {
    const result = await apiCall('/proxy/canonical/runs');

    if (result.success && result.runs) {
        runsData = result.runs || [];
        renderRunsList();

        // Auto-poll if any run is in progress
        const inProgress = runsData.some(r =>
            ['loading_forms', 'processing', 'calling_llm', 'mapping_forms'].includes(r.status)
        );
        if (inProgress && !canonicalPollInterval) {
            // Find the in-progress run and start polling
            const activeRun = runsData.find(r =>
                ['loading_forms', 'processing', 'calling_llm', 'mapping_forms'].includes(r.status)
            );
            if (activeRun) {
                startPollingCanonicalStatus('versions-payload', activeRun.run_id);
            }
        }
    }
}

function renderRunsList() {
    const container = document.getElementById('runs-list');

    if (!runsData || runsData.length === 0) {
        container.innerHTML = '<p class="muted">No runs yet. Click "Create New Canonical" to start.</p>';
        return;
    }

    // Sort by created_at descending (newest first)
    const sortedRuns = [...runsData].sort((a, b) =>
        new Date(b.created_at || 0) - new Date(a.created_at || 0)
    );

    container.innerHTML = sortedRuns.map(run => {
        const statusClass = getRunStatusClass(run.status);
        const statusIcon = getRunStatusIcon(run.status);

        let details = '';
        if (run.status === 'calling_llm') {
            details = `
                <div class="run-details">
                    Forms: ${run.forms_to_process || run.total_forms || '?'} |
                    Est. tokens: ${(run.estimated_input_tokens || 0).toLocaleString()} in / ${(run.estimated_output_tokens || 0).toLocaleString()} out
                    ${run.forms_truncated > 0 ? `<span class="badge warning">${run.forms_truncated} truncated</span>` : ''}
                </div>
            `;
        } else if (run.status === 'completed' && run.stats) {
            details = `
                <div class="run-details">
                    Canonical Qs: ${run.stats.canonical_questions_count || '?'} |
                    Tokens: ${(run.stats.input_token_count || 0).toLocaleString()} / ${(run.stats.output_token_count || 0).toLocaleString()} |
                    Time: ${run.stats.total_time_seconds || '?'}s
                </div>
            `;
        } else if (run.status === 'failed') {
            details = `<div class="run-details error">${run.error || 'Unknown error'}</div>`;
        } else if (run.total_forms) {
            details = `<div class="run-details">Forms: ${run.total_forms} | Questions: ${run.total_questions || '?'}</div>`;
        }

        return `
            <div class="list-item run-item ${statusClass}">
                <div class="list-item-title">
                    <span class="run-status-icon">${statusIcon}</span>
                    Run ${run.run_id}
                    <span class="badge ${statusClass}">${run.status}</span>
                </div>
                <div class="list-item-meta">
                    Started: ${run.created_at ? new Date(run.created_at).toLocaleString() : 'N/A'}
                    ${run.updated_at ? ` | Updated: ${new Date(run.updated_at).toLocaleString()}` : ''}
                </div>
                ${details}
            </div>
        `;
    }).join('');
}

function getRunStatusClass(status) {
    switch (status) {
        case 'completed': return 'success';
        case 'failed': return 'error';
        case 'calling_llm': return 'warning';
        default: return '';
    }
}

function getRunStatusIcon(status) {
    switch (status) {
        case 'completed': return '✓';
        case 'failed': return '✗';
        case 'calling_llm': return '⏳';
        case 'loading_forms':
        case 'processing':
        case 'mapping_forms':
            return '⟳';
        default: return '○';
    }
}

async function loadVersions() {
    const result = await apiCall('/proxy/canonical/versions');

    if (result.success && result.response?.versions) {
        versionsData = result.response.versions;
        renderVersionsList();
        populateFillFormSelects();
        loadCanonicalStats();
    }
}

async function loadCanonicalStats() {
    const result = await apiCall('/proxy/canonical/stats');

    if (result.success && result.versions) {
        canonicalStatsData = result.versions;
        renderVersionsList();  // Re-render with stats
    }
}

async function loadFormStats(versionFilter = null) {
    let endpoint = '/proxy/forms/stats';
    if (versionFilter) {
        endpoint += `?version=${versionFilter}`;
    }

    const result = await apiCall(endpoint);

    if (result.success) {
        renderFormStats(result);
    }
}

function renderFormStats(stats) {
    const container = document.getElementById('forms-stats');
    if (!container) return;

    const agg = stats.aggregate || {};
    const questions = agg.questions || {};
    const pdfMapping = agg.pdf_field_mapping_pct || {};
    const canonicalMapping = agg.canonical_mapping_pct || {};

    container.innerHTML = `
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">${agg.total_forms || 0}</div>
                <div class="stat-label">Total Forms</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${agg.forms_with_questions || 0}</div>
                <div class="stat-label">Forms with Questions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${questions.total || 0}</div>
                <div class="stat-label">Total Questions</div>
            </div>
        </div>
        <div class="stats-detail-grid">
            <div class="stat-detail-card">
                <div class="stat-detail-header">Questions per Form</div>
                <div class="stat-detail-row">
                    <div class="stat-detail-item">
                        <span class="stat-detail-value">${questions.min || 0}</span>
                        <span class="stat-detail-label">Min</span>
                    </div>
                    <div class="stat-detail-item">
                        <span class="stat-detail-value">${questions.avg || 0}</span>
                        <span class="stat-detail-label">Avg</span>
                    </div>
                    <div class="stat-detail-item">
                        <span class="stat-detail-value">${questions.max || 0}</span>
                        <span class="stat-detail-label">Max</span>
                    </div>
                </div>
            </div>
            <div class="stat-detail-card">
                <div class="stat-detail-header">PDF Field Mapping</div>
                <div class="stat-detail-row">
                    <div class="stat-detail-item ${getStatClass(pdfMapping.min)}">
                        <span class="stat-detail-value">${pdfMapping.min || 0}%</span>
                        <span class="stat-detail-label">Min</span>
                    </div>
                    <div class="stat-detail-item ${getStatClass(pdfMapping.avg)}">
                        <span class="stat-detail-value">${pdfMapping.avg || 0}%</span>
                        <span class="stat-detail-label">Avg</span>
                    </div>
                    <div class="stat-detail-item ${getStatClass(pdfMapping.max)}">
                        <span class="stat-detail-value">${pdfMapping.max || 0}%</span>
                        <span class="stat-detail-label">Max</span>
                    </div>
                </div>
            </div>
            <div class="stat-detail-card">
                <div class="stat-detail-header">Canonical Question Mapping</div>
                <div class="stat-detail-row">
                    <div class="stat-detail-item ${getStatClass(canonicalMapping.min)}">
                        <span class="stat-detail-value">${canonicalMapping.min || 0}%</span>
                        <span class="stat-detail-label">Min</span>
                    </div>
                    <div class="stat-detail-item ${getStatClass(canonicalMapping.avg)}">
                        <span class="stat-detail-value">${canonicalMapping.avg || 0}%</span>
                        <span class="stat-detail-label">Avg</span>
                    </div>
                    <div class="stat-detail-item ${getStatClass(canonicalMapping.max)}">
                        <span class="stat-detail-value">${canonicalMapping.max || 0}%</span>
                        <span class="stat-detail-label">Max</span>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function getStatClass(value) {
    if (value === undefined || value === null) return '';
    if (value >= 80) return 'success';
    if (value >= 50) return 'warning';
    return 'error';
}

function renderVersionsList() {
    const container = document.getElementById('versions-list');

    if (versionsData.length === 0) {
        container.innerHTML = '<p class="muted">No canonical versions. Click "Create New Canonical" to generate.</p>';
        return;
    }

    // Merge stats data if available
    const versionsWithStats = versionsData.map(v => {
        const stats = canonicalStatsData.find(s => s.version_id === v.version_id) || {};
        return { ...v, ...stats };
    });

    container.innerHTML = versionsWithStats.map(v => `
        <div class="list-item ${v.is_latest ? 'selected' : ''}">
            <div class="list-item-title">
                ${v.version_id}
                ${v.is_latest ? '<span class="badge success">Latest</span>' : ''}
            </div>
            <div class="list-item-meta">
                ${v.question_count || v.canonical_questions_count || 0} canonical questions &bull;
                Created: ${new Date(v.created_at).toLocaleString()}
            </div>
            ${v.forms_mapped !== undefined ? `
                <div class="list-item-stats">
                    Forms mapped: ${v.forms_mapped} &bull;
                    Match rate: ${v.avg_match_rate || 0}%
                </div>
            ` : ''}
        </div>
    `).join('');
}

let canonicalPollInterval = null;

async function createCanonical() {
    if (!confirm('This will create a new canonical version from all existing forms. This may take several minutes. Continue?')) {
        return;
    }

    const result = await apiCall('/proxy/canonical/create', {
        method: 'POST',
        body: JSON.stringify({})
    });

    if (result.success && result.response?.run_id) {
        const runId = result.response.run_id;
        alert(`Canonical creation started! Run ID: ${runId}\nCheck the Runs list for progress.`);
        loadRuns();
        startPollingCanonicalStatus(null, runId);
    } else {
        alert('Failed to start canonical creation: ' + (result.response?.error || 'Unknown error'));
    }
}

function showCanonicalProgress(containerId, runId, initialResult) {
    // Progress is now shown in the runs list - just refresh it
    loadRuns();
}

function startPollingCanonicalStatus(containerId, runId) {
    if (canonicalPollInterval) {
        clearInterval(canonicalPollInterval);
    }

    const pollStatus = async () => {
        const result = await apiCall(`/proxy/canonical/run/${runId}`);

        if (result.success && result.run) {
            const run = result.run;
            updateCanonicalProgress(run);

            if (run.status === 'completed' || run.status === 'failed') {
                clearInterval(canonicalPollInterval);
                canonicalPollInterval = null;

                if (run.status === 'completed') {
                    showCanonicalComplete(containerId, run);
                    loadVersions();
                    loadCanonicalStats();
                } else {
                    showCanonicalFailed(containerId, run);
                }
            }
        }
    };

    // Poll immediately, then every 2 seconds
    pollStatus();
    canonicalPollInterval = setInterval(pollStatus, 2000);
}

function updateCanonicalProgress(run) {
    // Progress is now shown in the runs list - just refresh it
    loadRuns();
}

function showCanonicalComplete(containerId, run) {
    const stats = run.stats || {};

    // Refresh runs list and versions
    loadRuns();
    loadVersions();

    // Show success notification
    alert(`Canonical creation completed!\n\nVersion: ${run.version_id}\nCanonical Questions: ${stats.canonical_questions_count || '?'}\nTime: ${stats.total_time_seconds || '?'}s`);
}

function showCanonicalFailed(containerId, run) {
    // Refresh runs list
    loadRuns();

    // Show error notification
    alert(`Canonical creation failed!\n\nError: ${run.error || 'Unknown error'}`);
}

// =============================================================================
// Fill Form Tab
// =============================================================================

function populateFillFormSelects() {
    // Populate forms dropdown
    const formSelect = document.getElementById('fill-opp-id');
    formSelect.innerHTML = '<option value="">-- Select Form --</option>' +
        formsData.map(f => `<option value="${f.opportunity_id}" data-checksum="${f.checksum}">${f.opportunity_id} (${f.filename})</option>`).join('');

    // Populate versions dropdown
    const versionSelect = document.getElementById('fill-version-id');
    versionSelect.innerHTML = '<option value="">-- Select Version --</option>' +
        versionsData.map(v => `<option value="${v.version_id}" ${v.is_latest ? 'selected' : ''}>${v.version_id} ${v.is_latest ? '(Latest)' : ''}</option>`).join('');
}

function onFormSelect() {
    const select = document.getElementById('fill-opp-id');
    const option = select.options[select.selectedIndex];
    const checksum = option?.dataset?.checksum || '';
    document.getElementById('fill-checksum').value = checksum;
}

async function getMetadata() {
    const oppId = document.getElementById('fill-opp-id').value;
    const checksum = document.getElementById('fill-checksum').value;
    const versionId = document.getElementById('fill-version-id').value;
    const minSimilarity = parseInt(document.getElementById('fill-min-similarity').value) || 3;

    if (!oppId || !checksum || !versionId) {
        alert('Please select a form and version');
        return;
    }

    const requestBody = {
        forms: [{ opp_id: oppId, checksum: checksum }],
        version_id: versionId,
        min_similarity: minSimilarity
    };

    const result = await apiCall('/proxy/forms/metadata', {
        method: 'POST',
        body: JSON.stringify(requestBody)
    });

    if (result.success && result.response?.forms?.[0]) {
        currentMetadata = result.response.forms[0];
        renderAnswersForm(currentMetadata.questions);
        document.getElementById('fill-btn').disabled = false;
    } else {
        document.getElementById('answers-form').innerHTML = '<p class="muted">Failed to load metadata. Check API log in sidebar.</p>';
        document.getElementById('fill-btn').disabled = true;
    }
}

function renderAnswersForm(questions) {
    const container = document.getElementById('answers-form');

    if (!questions || questions.length === 0) {
        container.innerHTML = '<p class="muted">No questions found in this form.</p>';
        return;
    }

    container.innerHTML = questions.map((q, idx) => {
        const answerId = q.is_canonical ? q.canonical_question_id : q.question_id;
        const pdfFields = q.debug?.original?.mapped_pdf_fields || [];

        return `
            <div class="answer-field">
                <div class="answer-field-header">
                    <span class="answer-field-id">${answerId}</span>
                    ${q.is_canonical
                        ? `<span class="answer-field-canonical">Canonical (score: ${q.similarity_score})</span>`
                        : '<span class="badge warning">Form-specific</span>'}
                </div>
                <div class="answer-field-label">${q.question_text}</div>
                <div class="answer-field-type">
                    Type: ${q.question_type}
                    ${pdfFields.length > 0
                        ? ` &bull; PDF fields: <code>${pdfFields.join('</code>, <code>')}</code>`
                        : ' &bull; <span style="color: var(--warning);">No PDF field mapped</span>'}
                </div>
                <input type="text"
                       id="answer-${idx}"
                       data-answer-id="${answerId}"
                       placeholder="Enter answer for ${q.question_text}...">
            </div>
        `;
    }).join('');
}

async function fillForm() {
    const oppId = document.getElementById('fill-opp-id').value;
    const checksum = document.getElementById('fill-checksum').value;
    const versionId = document.getElementById('fill-version-id').value;

    // Collect answers
    const answers = {};
    document.querySelectorAll('#answers-form input[data-answer-id]').forEach(input => {
        if (input.value.trim()) {
            answers[input.dataset.answerId] = input.value.trim();
        }
    });

    if (Object.keys(answers).length === 0) {
        alert('Please enter at least one answer');
        return;
    }

    const requestBody = {
        opp_id: oppId,
        checksum: checksum,
        version_id: versionId,
        answers: answers,
        options: { truncate_on_char_limit: true }
    };

    const result = await apiCall('/proxy/forms/fill', {
        method: 'POST',
        body: JSON.stringify(requestBody)
    });

    // Show result card
    document.getElementById('result-card').style.display = 'block';
    renderFillResult(result);

    if (result.success && result.response?.download_url) {
        currentFilledPdfUrl = result.response.download_url;
        previewFilledPdf(currentFilledPdfUrl);
    }
}

function renderFillResult(result) {
    const container = document.getElementById('fill-result');

    if (result.success && result.response?.success) {
        const report = result.response.validation_report;
        const summary = report?.summary || {};
        const s3Key = result.response.output_s3_key || 'N/A';

        container.className = 'success';
        container.innerHTML = `
            <div class="result-header">
                <span class="result-icon">✓</span>
                <span class="result-title">Form Filled Successfully</span>
            </div>
            <div class="result-details">
                <strong>S3 Key:</strong> <code>${s3Key}</code><br>
                Fields filled: ${summary.fields_filled || 0} / ${summary.total_fields || 0} &bull;
                Warnings: ${summary.warnings || 0} &bull;
                Errors: ${summary.errors || 0}
            </div>
        `;
    } else {
        container.className = 'error';
        container.innerHTML = `
            <div class="result-header">
                <span class="result-icon">✗</span>
                <span class="result-title">Failed to Fill Form</span>
            </div>
            <div class="result-details">
                ${result.response?.error || 'Unknown error occurred'}
            </div>
        `;
    }
}

function previewFilledPdf(s3Url) {
    // Use the S3 presigned URL directly in an embed
    const container = document.getElementById('pdf-preview-container');
    container.innerHTML = `<embed src="${s3Url}" type="application/pdf">`;
}

function downloadPdf() {
    if (currentFilledPdfUrl) {
        // Open the S3 presigned URL directly - browser will download
        window.open(currentFilledPdfUrl, '_blank');
    }
}

function openPdfNewTab() {
    if (currentFilledPdfUrl) {
        // Open the S3 presigned URL in a new tab
        window.open(currentFilledPdfUrl, '_blank');
    }
}
