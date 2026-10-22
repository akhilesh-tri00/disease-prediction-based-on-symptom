/**
 * MEDIPREDICT AI - CLIENT APPLICATION LOGIC
 */

let allSymptoms = [];
let symptomCategories = {};
let selectedSymptoms = new Set();
let currentCategory = 'all';
let allDiseases = {};

// Quick Diagnostic Presets
const PRESETS = {
    flu: ['fever', 'chills', 'cough', 'headache', 'fatigue', 'muscle_pain', 'runny_nose'],
    gastro: ['vomiting', 'stomach_pain', 'diarrhoea', 'nausea', 'dehydration'],
    skin: ['itching', 'skin_rash', 'nodal_skin_eruptions', 'dischromic _patches'],
    migraine: ['headache', 'visual_disturbances', 'acidity', 'stiff_neck', 'depression'],
    dengue: ['high_fever', 'chills', 'joint_pain', 'vomiting', 'fatigue', 'headache', 'loss_of_appetite', 'back_pain'],
    diabetes: ['polyuria', 'fatigue', 'weight_loss', 'increased_appetite', 'excessive_hunger']
};

document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

async function initApp() {
    await loadSymptoms();
    await loadDiseases();
}

/**
 * Fetch symptoms list and categories from backend
 */
async function loadSymptoms() {
    try {
        const res = await fetch('/api/symptoms');
        const data = await res.json();
        
        allSymptoms = data.symptoms || [];
        symptomCategories = data.categories || {};
        
        renderCategoryPills();
        renderSymptomsGrid();
    } catch (err) {
        console.error('Failed to load symptoms:', err);
        document.getElementById('symptoms-grid').innerHTML = 
            '<div class="error-msg">Error loading symptoms. Please ensure backend is running.</div>';
    }
}

/**
 * Render category navigation pills
 */
function renderCategoryPills() {
    const container = document.getElementById('category-pills');
    if (!container) return;

    let html = `<button class="cat-pill active" onclick="setCategory('all')">All (${allSymptoms.length})</button>`;
    
    for (const [catName, items] of Object.entries(symptomCategories)) {
        if (items.length > 0) {
            html += `<button class="cat-pill" id="cat-pill-${sanitizeId(catName)}" onclick="setCategory('${catName}')">${catName} (${items.length})</button>`;
        }
    }
    
    container.innerHTML = html;
}

/**
 * Set active category filter
 */
function setCategory(catName) {
    currentCategory = catName;
    document.querySelectorAll('.cat-pill').forEach(btn => btn.classList.remove('active'));
    
    if (catName === 'all') {
        document.querySelector('.cat-pill').classList.add('active');
    } else {
        const target = document.getElementById(`cat-pill-${sanitizeId(catName)}`);
        if (target) target.classList.add('active');
    }
    
    renderSymptomsGrid();
}

/**
 * Render symptom chips based on search and category filters
 */
function renderSymptomsGrid() {
    const container = document.getElementById('symptoms-grid');
    const searchTerm = (document.getElementById('symptom-search')?.value || '').toLowerCase().trim();
    
    let symptomsToDisplay = [];
    
    if (currentCategory === 'all') {
        symptomsToDisplay = allSymptoms.map(s => ({ id: s, label: formatSymptomName(s) }));
    } else if (symptomCategories[currentCategory]) {
        symptomsToDisplay = symptomCategories[currentCategory];
    }
    
    if (searchTerm) {
        symptomsToDisplay = symptomsToDisplay.filter(item => 
            item.label.toLowerCase().includes(searchTerm) || item.id.toLowerCase().includes(searchTerm)
        );
    }
    
    if (symptomsToDisplay.length === 0) {
        container.innerHTML = '<div class="empty-results">No matching symptoms found. Try another search term.</div>';
        return;
    }
    
    container.innerHTML = symptomsToDisplay.map(item => {
        const isSelected = selectedSymptoms.has(item.id);
        return `
            <div class="symptom-chip ${isSelected ? 'selected' : ''}" 
                 id="chip-${sanitizeId(item.id)}"
                 onclick="toggleSymptom('${item.id}')">
                <span>${item.label}</span>
            </div>
        `;
    }).join('');
}

/**
 * Toggle symptom selection
 */
function toggleSymptom(symptomId) {
    if (selectedSymptoms.has(symptomId)) {
        selectedSymptoms.delete(symptomId);
    } else {
        selectedSymptoms.add(symptomId);
    }
    
    updateSelectionUI();
}

/**
 * Update selected tags dock, counters, and button state
 */
function updateSelectionUI() {
    // Update chips
    document.querySelectorAll('.symptom-chip').forEach(chip => {
        const id = chip.id.replace('chip-', '');
        // Re-check state
        const matchingSymptom = Array.from(selectedSymptoms).find(s => sanitizeId(s) === id);
        if (matchingSymptom) {
            chip.classList.add('selected');
        } else {
            chip.classList.remove('selected');
        }
    });

    // Update count
    const countEl = document.getElementById('selected-count');
    if (countEl) countEl.innerText = selectedSymptoms.size;

    // Update tags list
    const tagsContainer = document.getElementById('selected-tags');
    if (tagsContainer) {
        if (selectedSymptoms.size === 0) {
            tagsContainer.innerHTML = '<span class="placeholder-text">No symptoms selected yet. Click any symptom tag above or pick a Quick Preset!</span>';
        } else {
            tagsContainer.innerHTML = Array.from(selectedSymptoms).map(id => `
                <span class="selected-tag">
                    ${formatSymptomName(id)}
                    <span class="tag-remove" onclick="removeSymptom('${id}')">✕</span>
                </span>
            `).join('');
        }
    }

    // Update Diagnose Button
    const btn = document.getElementById('diagnose-btn');
    if (btn) {
        btn.disabled = selectedSymptoms.size === 0;
    }
}

/**
 * Remove specific symptom
 */
function removeSymptom(id) {
    selectedSymptoms.delete(id);
    updateSelectionUI();
}

/**
 * Clear all selected symptoms
 */
function clearAllSymptoms() {
    selectedSymptoms.clear();
    updateSelectionUI();
}

/**
 * Apply quick preset
 */
function applyPreset(presetKey) {
    const list = PRESETS[presetKey];
    if (!list) return;

    selectedSymptoms.clear();
    list.forEach(s => {
        if (allSymptoms.includes(s)) {
            selectedSymptoms.add(s);
        }
    });

    updateSelectionUI();
    renderSymptomsGrid();

    // Auto-run diagnosis for quick user feedback
    runDiagnosis();
}

/**
 * Run diagnosis against /api/predict
 */
async function runDiagnosis() {
    if (selectedSymptoms.size === 0) return;

    const emptyState = document.getElementById('empty-state');
    const loadingPanel = document.getElementById('loading-panel');
    const content = document.getElementById('diagnosis-content');

    if (emptyState) emptyState.style.display = 'none';
    if (loadingPanel) loadingPanel.style.display = 'block';
    if (content) content.style.display = 'none';

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symptoms: Array.from(selectedSymptoms) })
        });

        const data = await response.json();

        if (loadingPanel) loadingPanel.style.display = 'none';

        if (data.success) {
            renderDiagnosisResults(data);
            if (content) content.style.display = 'block';
        } else {
            alert(data.error || 'Prediction could not be completed.');
            if (emptyState) emptyState.style.display = 'block';
        }

    } catch (err) {
        console.error('Diagnosis error:', err);
        if (loadingPanel) loadingPanel.style.display = 'none';
        if (emptyState) emptyState.style.display = 'block';
        alert('Server connection error. Please verify the Flask server is running.');
    }
}

/**
 * Render diagnostic outputs into DOM
 */
function renderDiagnosisResults(data) {
    // Primary Disease
    document.getElementById('res-disease').innerText = data.primary_prediction;
    document.getElementById('res-confidence-text').innerText = `${data.confidence}%`;
    document.getElementById('res-progress-bar').style.width = `${Math.min(100, data.confidence)}%`;
    document.getElementById('res-desc').innerText = data.description || 'Clinical condition predicted based on symptom correlation.';
    
    // Severity badge
    const sevEl = document.getElementById('res-severity');
    sevEl.innerText = data.severity;
    sevEl.className = 'badge severity-badge';
    if (data.severity.toLowerCase().includes('emergency') || data.severity.toLowerCase().includes('critical')) {
        sevEl.classList.add('emergency');
    }

    // Specialist
    document.getElementById('res-specialist').innerText = data.specialist || 'General Physician';

    // Differential Probabilities Bar Chart
    const probList = document.getElementById('prob-list');
    if (probList && data.top_candidates) {
        probList.innerHTML = data.top_candidates.map(c => `
            <div class="prob-row">
                <div class="prob-info">
                    <span class="prob-name">${c.disease}</span>
                    <span class="prob-val">${c.probability}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-bar" style="width: ${c.probability}%"></div>
                </div>
            </div>
        `).join('');
    }

    // Precautions
    const precList = document.getElementById('res-precautions');
    if (precList && data.precautions) {
        precList.innerHTML = data.precautions.map(p => `<li>${p}</li>`).join('');
    }
}

/**
 * Filter symptoms by search input
 */
function filterSymptoms() {
    const input = document.getElementById('symptom-search');
    const clearBtn = document.getElementById('search-clear');
    
    if (clearBtn) {
        clearBtn.style.display = input.value ? 'flex' : 'none';
    }
    
    renderSymptomsGrid();
}

function clearSearch() {
    const input = document.getElementById('symptom-search');
    if (input) {
        input.value = '';
        filterSymptoms();
    }
}

/**
 * Switch main navigation tabs
 */
function switchMainTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));

    const btn = document.getElementById(`tab-${tabId}`);
    const pane = document.getElementById(`pane-${tabId}`);

    if (btn) btn.classList.add('active');
    if (pane) pane.classList.add('active');
}

/**
 * Load Diseases Library
 */
async function loadDiseases() {
    try {
        const res = await fetch('/api/diseases');
        allDiseases = await res.json();
        renderDiseasesDirectory();
    } catch (err) {
        console.error('Failed to load diseases directory:', err);
    }
}

function renderDiseasesDirectory(filterText = '') {
    const container = document.getElementById('diseases-directory');
    if (!container) return;

    const entries = Object.entries(allDiseases);
    const filtered = entries.filter(([name, info]) => {
        if (!filterText) return true;
        const q = filterText.toLowerCase();
        return name.toLowerCase().includes(q) || 
               info.description.toLowerCase().includes(q) ||
               info.specialist.toLowerCase().includes(q);
    });

    if (filtered.length === 0) {
        container.innerHTML = '<div class="empty-results">No diseases matched your search.</div>';
        return;
    }

    container.innerHTML = filtered.map(([name, info]) => `
        <div class="disease-card">
            <div>
                <div class="disease-card-header">
                    <h4 class="disease-card-name">${name}</h4>
                    <span class="badge severity-badge ${info.severity.toLowerCase().includes('emergency') ? 'emergency' : ''}">${info.severity}</span>
                </div>
                <p class="disease-card-desc">${info.description}</p>
            </div>
            <div class="disease-card-footer">
                <span>Specialist:</span>
                <span class="disease-specialist">👨‍⚕️ ${info.specialist}</span>
            </div>
        </div>
    `).join('');
}

function filterDiseases() {
    const searchVal = document.getElementById('disease-search')?.value.trim();
    renderDiseasesDirectory(searchVal);
}

/**
 * Graph modal helpers
 */
function openModal(imageSrc, title) {
    const modal = document.getElementById('graph-modal');
    const modalImg = document.getElementById('modal-img');
    const modalTitle = document.getElementById('modal-title');

    if (modal && modalImg) {
        modalImg.src = imageSrc;
        if (modalTitle) modalTitle.innerText = title;
        modal.classList.add('active');
    }
}

function closeModal() {
    const modal = document.getElementById('graph-modal');
    if (modal) modal.classList.remove('active');
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
});

/**
 * Utilities
 */
function formatSymptomName(str) {
    return str.replace(/_/g, ' ').replace(/\s+/g, ' ').trim().replace(/\b\w/g, l => l.toUpperCase());
}

function sanitizeId(str) {
    return str.replace(/[^a-zA-Z0-9]/g, '-').toLowerCase();
}
