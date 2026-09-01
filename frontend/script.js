// SmartCart Customer Clustering — Data Science UI Script
const API_BASE = "https://smartcart-customer-clustering.onrender.com";

let selectedFile = null;
let currentClusteringData = null;
let scatterChartInstance = null;

document.addEventListener("DOMContentLoaded", () => {
    setupUploadHandlers();

    const btnRunClustering = document.getElementById("btn-run-clustering");
    if (btnRunClustering) {
        btnRunClustering.addEventListener("click", executeClustering);
    }

    const clusterSelectFilter = document.getElementById("cluster-select-filter");
    if (clusterSelectFilter) {
        clusterSelectFilter.addEventListener("change", (e) => {
            if (currentClusteringData) {
                renderExploreTable(currentClusteringData.customers, e.target.value);
                highlightClusterSummaryCard(e.target.value);
            }
        });
    }

    // Auto-load sample dataset preview on initial load
    loadSampleDatasetPreview();
});

function setupUploadHandlers() {
    const dropzone = document.getElementById("upload-dropzone");
    const fileInput = document.getElementById("file-input");
    const btnChoose = document.getElementById("btn-choose-file");
    const btnSample = document.getElementById("btn-sample-data");
    const btnChange = document.getElementById("btn-change-file");

    if (btnChoose && fileInput) {
        btnChoose.addEventListener("click", () => fileInput.click());
    }

    if (fileInput) {
        fileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });
    }

    if (btnChange) {
        btnChange.addEventListener("click", () => {
            selectedFile = null;
            showPromptState();
        });
    }

    if (btnSample) {
        btnSample.addEventListener("click", loadSampleDatasetPreview);
    }

    if (dropzone) {
        ["dragenter", "dragover"].forEach(evt => {
            dropzone.addEventListener(evt, (e) => {
                e.preventDefault();
                dropzone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach(evt => {
            dropzone.addEventListener(evt, (e) => {
                e.preventDefault();
                dropzone.classList.remove("dragover");
            });
        });

        dropzone.addEventListener("drop", (e) => {
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleFileSelect(e.dataTransfer.files[0]);
            }
        });
    }
}

function showPromptState() {
    document.getElementById("dropzone-prompt").classList.remove("hidden");
    document.getElementById("file-loaded-state").classList.add("hidden");
    document.getElementById("dataset-preview-container").classList.add("hidden");
    hideValidationAlert();
}

function handleFileSelect(file) {
    if (!file.name.endsWith(".csv")) {
        showValidationAlert("Please select a valid CSV file (.csv).");
        return;
    }

    selectedFile = file;
    hideValidationAlert();

    const reader = new FileReader();
    reader.onload = (e) => {
        const text = e.target.result;
        parseAndPreviewCSV(text, file.name);
    };
    reader.readAsText(file);
}

async function loadSampleDatasetPreview() {
    selectedFile = null;
    hideValidationAlert();

    try {
        const res = await fetch(`${API_BASE}/api/analytics`);
        if (!res.ok) throw new Error("Failed to load sample dataset");
        const data = await res.json();
        
        showFileLoadedState("smartcart_customers.csv (Sample Dataset)", `${data.total_customers.toLocaleString()} rows · 22 columns`);

        const resClusters = await fetch(`${API_BASE}/api/cluster`, { method: "POST" });
        if (resClusters.ok) {
            const clusterData = await resClusters.json();
            renderPreviewTableFromData(clusterData.customers);
        }
    } catch (err) {
        console.error("Sample preview fetch error:", err);
    }
}

function parseAndPreviewCSV(csvText, filename) {
    const lines = csvText.split("\n").filter(l => l.trim().length > 0);
    if (lines.length <= 1) {
        showValidationAlert("CSV file appears empty or invalid.");
        return;
    }

    const headers = lines[0].split(",").map(h => h.trim().replace(/^"|"$/g, ''));
    const rows = lines.slice(1, 6).map(line => line.split(",").map(cell => cell.trim().replace(/^"|"$/g, '')));

    showFileLoadedState(filename, `${(lines.length - 1).toLocaleString()} rows · ${headers.length} columns`);
    renderPreviewTableFromParsed(headers, rows, lines.length - 1, headers.length);
}

function showFileLoadedState(filename, metadataStr) {
    document.getElementById("loaded-filename").textContent = filename;
    document.getElementById("loaded-metadata").textContent = metadataStr;

    document.getElementById("dropzone-prompt").classList.add("hidden");
    document.getElementById("file-loaded-state").classList.remove("hidden");
}

function renderPreviewTableFromParsed(headers, rows, rowCount, colCount) {
    const previewContainer = document.getElementById("dataset-preview-container");
    const previewSub = document.getElementById("preview-sub-text");
    const thead = document.getElementById("preview-thead");
    const tbody = document.getElementById("preview-tbody");

    previewSub.textContent = `${rowCount.toLocaleString()} customers · ${colCount} columns`;
    
    const previewCols = headers.slice(0, 8);

    thead.innerHTML = `<tr>${previewCols.map(h => `<th>${h}</th>`).join("")}</tr>`;
    tbody.innerHTML = rows.map(r => `
        <tr>${previewCols.map((_, idx) => `<td>${r[idx] || "--"}</td>`).join("")}</tr>
    `).join("");

    previewContainer.classList.remove("hidden");
}

function renderPreviewTableFromData(customers) {
    const previewContainer = document.getElementById("dataset-preview-container");
    const thead = document.getElementById("preview-thead");
    const tbody = document.getElementById("preview-tbody");

    const headers = ["ID", "Year_Birth", "Education", "Marital_Status", "Income ($)", "Recency (Days)", "Web Purchases", "Store Purchases"];
    thead.innerHTML = `<tr>${headers.map(h => `<th>${h}</th>`).join("")}</tr>`;

    const sampleRows = customers.slice(0, 5);
    tbody.innerHTML = sampleRows.map(c => `
        <tr>
            <td>#${c.id}</td>
            <td>${c.year_birth}</td>
            <td>${c.education}</td>
            <td>${c.marital_status}</td>
            <td>$${c.income.toLocaleString()}</td>
            <td>${c.recency} days</td>
            <td>${c.web_purchases}</td>
            <td>${c.store_purchases}</td>
        </tr>
    `).join("");

    previewContainer.classList.remove("hidden");
}

function showValidationAlert(msg) {
    const alertBox = document.getElementById("validation-alert");
    const msgSpan = document.getElementById("validation-error-msg");
    if (alertBox && msgSpan) {
        msgSpan.textContent = msg;
        alertBox.classList.remove("hidden");
    }
}

function hideValidationAlert() {
    const alertBox = document.getElementById("validation-alert");
    if (alertBox) alertBox.classList.add("hidden");
}

// Execute Clustering Action
async function executeClustering() {
    const statusBox = document.getElementById("clustering-status-box");
    const resultsContainer = document.getElementById("results-section");

    statusBox.classList.remove("hidden");
    resultsContainer.classList.add("hidden");

    try {
        const formData = new FormData();
        if (selectedFile) {
            formData.append("file", selectedFile);
        }

        const res = await fetch(`${API_BASE}/api/cluster`, {
            method: "POST",
            body: selectedFile ? formData : undefined
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Clustering execution failed.");
        }

        currentClusteringData = await res.json();
        renderClusteringResults(currentClusteringData);

    } catch (err) {
        showValidationAlert(err.message);
        console.error(err);
    } finally {
        statusBox.classList.add("hidden");
    }
}

// Render Complete Results View
function renderClusteringResults(data) {
    const resultsContainer = document.getElementById("results-section");
    resultsContainer.classList.remove("hidden");

    renderAuditPill(data);
    renderHorizontalClusterSummary(data.cluster_summaries);
    renderScatterChart(data.pca_points);
    renderProfileComparisonMatrix(data.cluster_summaries);
    renderExploreTable(data.customers, "all");
    renderInsightsGrid(data.cluster_summaries);

    resultsContainer.scrollIntoView({ behavior: "smooth", block: "start" });
}

// Render Audit Pill (2,240 uploaded vs 2,236 clustered explanation)
function renderAuditPill(data) {
    const pill = document.getElementById("audit-summary-pill");
    if (!pill) return;

    const raw = data.total_raw || 2240;
    const proc = data.total_processed || 2236;
    const excl = data.total_excluded !== undefined ? data.total_excluded : (raw - proc);

    if (excl > 0) {
        pill.textContent = `${raw.toLocaleString()} records uploaded · ${proc.toLocaleString()} customers clustered · ${excl} records excluded during preprocessing (outliers: Age ≥ 90 or Income ≥ $600,000)`;
        pill.classList.remove("hidden");
    } else {
        pill.textContent = `${proc.toLocaleString()} customers clustered`;
        pill.classList.remove("hidden");
    }
}

// 1. Clickable Horizontal Cluster Summary Bar with Dynamic Percentages
function renderHorizontalClusterSummary(summaries) {
    const bar = document.getElementById("cluster-summary-bar");
    if (!bar) return;

    bar.innerHTML = summaries.map(c => `
        <div class="cluster-bar-item c-item-${c.cluster_id}" data-cluster-id="${c.cluster_id}" role="button" tabindex="0">
            <span class="c-tag">CLUSTER 0${c.cluster_id}</span>
            <div class="c-count">${c.count.toLocaleString()} customers</div>
            <div class="c-pct">${c.percentage}%</div>
            <div class="c-name">${c.cluster_name}</div>
        </div>
    `).join("");

    const cards = bar.querySelectorAll(".cluster-bar-item");
    cards.forEach(card => {
        card.addEventListener("click", () => {
            const clusterId = card.getAttribute("data-cluster-id");
            
            const filterSelect = document.getElementById("cluster-select-filter");
            if (filterSelect) {
                filterSelect.value = clusterId;
            }

            if (currentClusteringData) {
                renderExploreTable(currentClusteringData.customers, clusterId);
            }

            cards.forEach(c => c.classList.remove("active-filter"));
            card.classList.add("active-filter");

            const exploreSec = document.getElementById("explore-section");
            if (exploreSec) {
                exploreSec.scrollIntoView({ behavior: "smooth", block: "nearest" });
            }
        });
    });
}

function highlightClusterSummaryCard(clusterId) {
    const bar = document.getElementById("cluster-summary-bar");
    if (!bar) return;

    const cards = bar.querySelectorAll(".cluster-bar-item");
    cards.forEach(c => {
        if (c.getAttribute("data-cluster-id") === clusterId.toString()) {
            c.classList.add("active-filter");
        } else {
            c.classList.remove("active-filter");
        }
    });
}

// 2. PCA Scatter Plot (Visual Centerpiece)
function renderScatterChart(pcaPoints) {
    const ctx = document.getElementById("pca-scatter-chart");
    if (!ctx) return;

    if (scatterChartInstance) {
        scatterChartInstance.destroy();
    }

    const clusterGroups = { 0: [], 1: [], 2: [], 3: [] };
    pcaPoints.forEach(pt => {
        if (clusterGroups[pt.cluster]) {
            clusterGroups[pt.cluster].push({ x: pt.x, y: pt.y, id: pt.id, cluster: pt.cluster });
        }
    });

    const colors = {
        0: "#2563eb",
        1: "#059669",
        2: "#d97706",
        3: "#dc2626"
    };

    const datasets = Object.keys(clusterGroups).map(c => ({
        label: `Cluster 0${c}`,
        data: clusterGroups[c],
        backgroundColor: colors[c],
        borderColor: colors[c],
        pointRadius: 5,
        pointHoverRadius: 7,
        pointStyle: 'circle'
    }));

    scatterChartInstance = new Chart(ctx, {
        type: "scatter",
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: {
                    position: "top",
                    labels: {
                        usePointStyle: true,
                        font: { size: 12, weight: "600" }
                    }
                },
                tooltip: {
                    backgroundColor: "rgba(15, 23, 42, 0.9)",
                    padding: 10,
                    titleFont: { size: 12, weight: "700" },
                    bodyFont: { size: 12 },
                    callbacks: {
                        label: (ctx) => {
                            const raw = ctx.raw;
                            return `Customer #${raw.id} (Cluster ${raw.cluster}): PCA1 = ${raw.x}, PCA2 = ${raw.y}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: "Principal Component 1 (PCA 1)", font: { weight: "600", size: 13 } },
                    grid: { color: "#f1f5f9" }
                },
                y: {
                    title: { display: true, text: "Principal Component 2 (PCA 2)", font: { weight: "600", size: 13 } },
                    grid: { color: "#f1f5f9" }
                }
            }
        }
    });
}

// 3. Cluster Profile Comparison Matrix Table
function renderProfileComparisonMatrix(summaries) {
    const thead = document.getElementById("comparison-thead");
    const tbody = document.getElementById("comparison-tbody");
    if (!thead || !tbody) return;

    thead.innerHTML = `
        <tr>
            <th>Attribute Metric</th>
            ${summaries.map(c => `<th>Cluster 0${c.cluster_id}</th>`).join("")}
        </tr>
    `;

    const metrics = [
        { label: "Customers Count", key: c => `${c.count.toLocaleString()} (${c.percentage}%)` },
        { label: "Avg Annual Income", key: c => `$${c.avg_income.toLocaleString()}` },
        { label: "Avg Total Spending", key: c => `$${c.avg_spending.toLocaleString()}` },
        { label: "Avg Purchase Recency", key: c => `${c.avg_recency} days` },
        { label: "Avg Customer Age", key: c => `${c.avg_age} years` },
        { label: "Avg Total Children", key: c => `${c.avg_children}` },
        { label: "Dominant Living Situation", key: c => c.dominant_living_with },
        { label: "Dominant Education", key: c => c.dominant_education }
    ];

    tbody.innerHTML = metrics.map(m => `
        <tr>
            <td>${m.label}</td>
            ${summaries.map(c => `<td>${m.key(c)}</td>`).join("")}
        </tr>
    `).join("");
}

// 4. Explore Customer Table with Filter
function renderExploreTable(customers, filterVal) {
    const tbody = document.getElementById("customer-explore-tbody");
    const footer = document.getElementById("pagination-footer");
    if (!tbody) return;

    let filtered = customers;
    if (filterVal !== "all") {
        filtered = customers.filter(c => c.cluster.toString() === filterVal.toString());
    }

    const displayRows = filtered.slice(0, 100);

    tbody.innerHTML = displayRows.map(c => `
        <tr>
            <td>#${c.id}</td>
            <td>${c.year_birth}</td>
            <td>${c.education}</td>
            <td>$${c.income.toLocaleString()}</td>
            <td>${c.recency} days</td>
            <td>$${c.total_spending.toLocaleString()}</td>
            <td>${c.web_purchases}</td>
            <td>${c.store_purchases}</td>
            <td><span class="cluster-pill pill-${c.cluster}">Cluster ${c.cluster}</span></td>
        </tr>
    `).join("");

    if (footer) {
        footer.textContent = `Showing 1 to ${displayRows.length} of ${filtered.length.toLocaleString()} customer records.`;
    }
}

// 5. Dynamic Cluster Insights Grid
function renderInsightsGrid(summaries) {
    const grid = document.getElementById("insights-grid");
    if (!grid) return;

    grid.innerHTML = summaries.map(c => `
        <div class="insight-card">
            <div class="insight-header">
                <span class="cluster-pill pill-${c.cluster_id}">Cluster ${c.cluster_id}</span>
                <span class="insight-title">${c.cluster_name}</span>
            </div>
            <p class="insight-body">${c.interpretation}</p>
        </div>
    `).join("");
}
