// Global variables
let inventoryData = null;

// PillPilot branding
const BRAND_NAME = 'PillPilot';
const BRAND_TAGLINE = 'Medicine Inventory Management';

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const csvFile = document.getElementById('csvFile');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileStats = document.getElementById('fileStats');
const dashboard = document.getElementById('dashboard');

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 DOM Content Loaded - Setting up application...');
    setupEventListeners();
    setupNavigation();
    console.log('✅ Application setup complete');
    
    // Add a test function to verify navigation works
    window.testNavigation = function() {
        console.log('🧪 Testing navigation...');
        console.log('Transfer card element:', document.getElementById('transferNavCard'));
        console.log('Analytics card element:', document.getElementById('analyticsNavCard'));
        console.log('showPage function:', typeof window.showPage);
        return 'Navigation test complete - check console for details';
    };
});

// Business Critical Chart Loading Functions

async function loadStockAging() {
    console.log('📊 Loading stock aging chart...');
    try {
        const response = await fetch('/api/charts/stock-aging');
        if (!response.ok) throw new Error('Failed to fetch stock aging data');
        
        const data = await response.json();
        console.log('📊 Stock aging data received:', data);
        
        // Update metrics
        document.getElementById('valueAtRisk').textContent = `$${data.total_value_at_risk.toLocaleString()}`;
        
        // Create timeline chart
        const trace = {
            x: data.timeline_data.map(d => d.bucket),
            y: data.timeline_data.map(d => d.quantity),
            type: 'bar',
            marker: {
                color: data.timeline_data.map(d => d.color),
                line: { color: 'rgba(255,255,255,0.2)', width: 1 }
            },
            text: data.timeline_data.map(d => `${d.quantity} items<br>$${d.value.toLocaleString()} value`),
            textposition: 'auto',
            hovertemplate: '<b>%{x}</b><br>Quantity: %{y}<br>Value: $%{customdata}<br><extra></extra>',
            customdata: data.timeline_data.map(d => d.value.toLocaleString())
        };
        
        const layout = {
            title: false,
            xaxis: {
                title: 'Aging Buckets',
                color: '#d1d5db',
                gridcolor: 'rgba(255,255,255,0.1)'
            },
            yaxis: {
                title: 'Quantity',
                color: '#d1d5db',
                gridcolor: 'rgba(255,255,255,0.1)'
            },
            plot_bgcolor: 'transparent',
            paper_bgcolor: 'transparent',
            font: { color: '#ffffff', family: 'Inter' },
            margin: { t: 20, r: 20, b: 60, l: 60 },
            autosize: true,
            responsive: true
        };
        
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d', 'resetScale2d'],
            displaylogo: false
        };
        
        Plotly.newPlot('stockAgingChart', [trace], layout, config);
        console.log('✅ Stock aging chart loaded successfully');
        
    } catch (error) {
        console.error('❌ Error loading stock aging chart:', error);
        document.getElementById('stockAgingChart').innerHTML = 
            '<div class="chart-error">Failed to load stock aging data</div>';
    }
}

async function loadStockoutRisk() {
    console.log('⚠️ Loading stockout risk chart...');
    try {
        const response = await fetch('/api/charts/stockout-risk');
        if (!response.ok) throw new Error('Failed to fetch stockout risk data');
        
        const data = await response.json();
        
        // Update metrics
        document.getElementById('criticalItemsCount').textContent = data.critical_items.length;
        
        // Create risk level pie chart
        const riskColors = {
            'Critical': '#ef4444',
            'High': '#f97316', 
            'Medium': '#eab308',
            'Low': '#22c55e'
        };
        
        const trace = {
            values: data.risk_summary.map(d => d.Medicine),
            labels: data.risk_summary.map(d => d.RiskLevel),
            type: 'pie',
            marker: {
                colors: data.risk_summary.map(d => riskColors[d.RiskLevel])
            },
            textinfo: 'label+percent+value',
            texttemplate: '%{label}<br>%{value} items<br>%{percent}',
            hovertemplate: '<b>%{label} Risk</b><br>Items: %{value}<br>Percentage: %{percent}<br><extra></extra>'
        };
        
        const layout = {
            title: false,
            font: { color: '#ffffff', family: 'Inter' },
            plot_bgcolor: 'transparent',
            paper_bgcolor: 'transparent',
            margin: { t: 20, r: 20, b: 20, l: 20 },
            autosize: true,
            responsive: true,
            legend: {
                orientation: 'h',
                y: -0.1,
                font: { color: '#d1d5db' }
            }
        };
        
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d', 'resetScale2d'],
            displaylogo: false
        };
        
        Plotly.newPlot('stockoutRiskChart', [trace], layout, config);
        
    } catch (error) {
        console.error('Error loading stockout risk chart:', error);
        document.getElementById('stockoutRiskChart').innerHTML = 
            '<div class="chart-error">Failed to load stockout risk data</div>';
    }
}

async function loadTransferOpportunities() {
    console.log('🔄 Loading transfer opportunities chart...');
    try {
        const response = await fetch('/api/charts/transfer-opportunities');
        if (!response.ok) throw new Error('Failed to fetch transfer opportunities data');
        
        const data = await response.json();
        
        // Update metrics
        document.getElementById('savingsPotential').textContent = `$${data.total_savings_potential.toLocaleString()}`;
        
        if (data.opportunities.length === 0) {
            document.getElementById('transferOpportunityChart').innerHTML = 
                '<div class="chart-message">No transfer opportunities identified at this time</div>';
            return;
        }
        
        // Create scatter plot of transfer opportunities
        const trace = {
            x: data.opportunities.map(d => d.TransferQuantity),
            y: data.opportunities.map(d => d.EstimatedSavings),
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: data.opportunities.map(d => Math.min(d.TransferQuantity / 5, 20)),
                color: data.opportunities.map(d => d.Priority === 'High' ? '#ef4444' : '#eab308'),
                opacity: 0.7,
                line: { color: 'rgba(255,255,255,0.3)', width: 1 }
            },
            text: data.opportunities.map(d => 
                `${d.Medicine}<br>${d.FromStore} → ${d.ToStore}<br>Transfer: ${d.TransferQuantity} units<br>Savings: $${d.EstimatedSavings}`
            ),
            hovertemplate: '<b>%{text}</b><br><extra></extra>'
        };
        
        const layout = {
            title: false,
            xaxis: {
                title: 'Transfer Quantity',
                color: '#d1d5db',
                gridcolor: 'rgba(255,255,255,0.1)'
            },
            yaxis: {
                title: 'Estimated Savings ($)',
                color: '#d1d5db',
                gridcolor: 'rgba(255,255,255,0.1)'
            },
            plot_bgcolor: 'transparent',
            paper_bgcolor: 'transparent',
            font: { color: '#ffffff', family: 'Inter' },
            margin: { t: 20, r: 20, b: 60, l: 80 },
            autosize: true,
            responsive: true
        };
        
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d', 'resetScale2d'],
            displaylogo: false
        };
        
        Plotly.newPlot('transferOpportunityChart', [trace], layout, config);
        
    } catch (error) {
        console.error('Error loading transfer opportunities chart:', error);
        document.getElementById('transferOpportunityChart').innerHTML = 
            '<div class="chart-error">Failed to load transfer opportunities data</div>';
    }
}

function setupEventListeners() {
    // File upload events
    csvFile.addEventListener('change', handleFileSelect);
    
    // Drag and drop events
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('click', () => csvFile.click());
}

function setupNavigation() {
    console.log('🧭 Setting up navigation...');
    
    // Get navigation card elements
    const transferNavCard = document.getElementById('transferNavCard');
    const analyticsNavCard = document.getElementById('analyticsNavCard');
    const backFromTransfer = document.getElementById('backFromTransfer');
    const backFromAnalytics = document.getElementById('backFromAnalytics');
    
    console.log('Navigation elements found:', {
        transferNavCard: !!transferNavCard,
        analyticsNavCard: !!analyticsNavCard,
        backFromTransfer: !!backFromTransfer,
        backFromAnalytics: !!backFromAnalytics
    });
    
    // Set up navigation card click handlers as backup (in case inline onclick fails)
    if (transferNavCard) {
        transferNavCard.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('🔄 Transfer nav card clicked via event listener');
            showPage('transfer');
            return false;
        });
    } else {
        console.error('❌ Transfer navigation card not found!');
    }
    
    if (analyticsNavCard) {
        analyticsNavCard.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('📈 Analytics nav card clicked via event listener');
            showPage('analytics');
            return false;
        });
    } else {
        console.error('❌ Analytics navigation card not found!');
    }
    
    // Set up back button handlers
    if (backFromTransfer) {
        backFromTransfer.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('⬅️ Back from transfer clicked');
            showPage('dashboard');
            return false;
        });
    }
    
    if (backFromAnalytics) {
        backFromAnalytics.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('⬅️ Back from analytics clicked');
            showPage('dashboard');
            return false;
        });
    }
    
    console.log('✅ Navigation setup complete');
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        csvFile.files = files;
        handleFileSelect({ target: { files: files } });
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file && file.type === 'text/csv') {
        uploadFile(file);
    } else {
        alert('Please select a valid CSV file.');
    }
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        showLoading();
        
        const response = await fetch('/api/upload-csv', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            console.log('🎯 CSV upload successful!', result);
            showFileInfo(file.name, result);
            console.log('📊 About to load dashboard...');
            await loadDashboard();
            console.log('✅ Dashboard loading completed!');
        } else {
            console.error('❌ Upload failed:', result.error);
            alert('Error: ' + result.error);
        }
    } catch (error) {
        alert('Error uploading file: ' + error.message);
    } finally {
        hideLoading();
    }
}

function showFileInfo(name, stats) {
    fileName.textContent = `File: ${name}`;
    fileStats.textContent = `Rows: ${stats.rows} | Stores: ${stats.stores} | Medicines: ${stats.medicines}`;
    fileInfo.style.display = 'block';
    fileInfo.classList.add('fade-in');
}

function showLoading() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading';
    loadingDiv.id = 'loadingIndicator';
    uploadArea.appendChild(loadingDiv);
}

function hideLoading() {
    const loadingDiv = document.getElementById('loadingIndicator');
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

async function loadDashboard() {
    console.log('🚀 Starting loadDashboard...');
    try {
        // Load all dashboard data
        console.log('📡 Fetching dashboard data...');
        const [summaryResponse, stockLevelsResponse, expiredResponse] = await Promise.all([
            fetch('/api/inventory-summary'),
            fetch('/api/stock-levels'),
            fetch('/api/expired-medicines')
        ]);
        console.log('📥 API responses received');
        
        const summary = await summaryResponse.json();
        const stockLevels = await stockLevelsResponse.json();
        const expired = await expiredResponse.json();
        
        // Store inventory data globally for search functionality
        window.inventoryData = stockLevels;
        
        if (summaryResponse.ok) {
            updateStatsCards(summary.overall_stats);
            
            // TODO: Setup ML controls and load status when functions exist
            // setupMLControls();
            // await loadMLStatus();
            
            // Show dashboard
            const dashboard = document.querySelector('.dashboard');
            if (dashboard) {
                dashboard.style.display = 'block';
                dashboard.scrollIntoView({ behavior: 'smooth' });
                console.log('✅ Dashboard is now visible and scrolled into view');
            }
        } else {
            alert('Error loading dashboard: ' + summary.error);
        }
    } catch (error) {
        alert('Error loading dashboard: ' + error.message);
    }
}

function updateStatsCards(stats) {
    // Add null checks to prevent errors
    const totalQuantityEl = document.getElementById('totalQuantity');
    const expiredCountEl = document.getElementById('expiredCount');
    const outOfStockCountEl = document.getElementById('outOfStockCount');
    const expiringSoonCountEl = document.getElementById('expiringSoonCount');
    
    if (totalQuantityEl && stats.total_quantity !== undefined) {
        totalQuantityEl.textContent = stats.total_quantity.toLocaleString();
    }
    if (expiredCountEl && stats.expired_count !== undefined) {
        expiredCountEl.textContent = stats.expired_count;
    }
    if (outOfStockCountEl && stats.out_of_stock_count !== undefined) {
        outOfStockCountEl.textContent = stats.out_of_stock_count;
    }
    if (expiringSoonCountEl && stats.expiring_soon_count !== undefined) {
        expiringSoonCountEl.textContent = stats.expiring_soon_count;
    }
}

function updateStoreSummaryTable(storeSummary) {
    const tbody = document.querySelector('#storeSummaryTable tbody');
    tbody.innerHTML = '';
    
    Object.entries(storeSummary).forEach(([store, data]) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${store}</strong></td>
            <td>${data.Quantity.toLocaleString()}</td>
            <td>${data.UniqueMedicines}</td>
            <td><span class="status-badge status-out-of-stock">${data.OutOfStockCount}</span></td>
        `;
        tbody.appendChild(row);
    });
}

function updateMedicineSummaryTable(medicineSummary) {
    const tbody = document.querySelector('#medicineSummaryTable tbody');
    tbody.innerHTML = '';
    
    Object.entries(medicineSummary).forEach(([medicine, data]) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${medicine}</strong></td>
            <td>${data.Quantity.toLocaleString()}</td>
            <td>${data.StoresWithStock}</td>
            <td><span class="status-badge status-out-of-stock">${data.OutOfStockCount}</span></td>
        `;
        tbody.appendChild(row);
    });
}

function updateExpiredTable(expired) {
    const tbody = document.querySelector('#expiredTable tbody');
    tbody.innerHTML = '';
    
    expired.forEach(item => {
        const daysOverdue = Math.abs(item.DaysUntilExpiry);
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${item.Store}</strong></td>
            <td>${item.Medicine}</td>
            <td>${item.Quantity}</td>
            <td>${new Date(item.ExpiryDate).toLocaleDateString()}</td>
            <td><span class="status-badge status-expired">${daysOverdue} days</span></td>
        `;
        tbody.appendChild(row);
    });
    
    document.getElementById('expiredAlert').style.display = 'block';
}

async function loadCharts() {
    try {
        // Load enterprise charts
        await loadEnterpriseCharts();
        
        // Load simplified legacy charts
        const [stockByStoreResponse, stockStatusResponse] = await Promise.all([
            fetch('/api/charts/stock-by-store'),
            fetch('/api/charts/stock-status')
        ]);
        
        const stockByStoreData = await stockByStoreResponse.json();
        const stockStatusData = await stockStatusResponse.json();
        
        // Render simplified charts with enhanced responsive config
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d', 'resetScale2d'],
            displaylogo: false
        };
        
        if (stockByStoreResponse.ok) {
            const stockData = JSON.parse(stockByStoreData);
            Plotly.newPlot('stockByStoreChart', stockData.data, {
                ...stockData.layout,
                autosize: true,
                responsive: true
            }, config);
        }
        
        if (stockStatusResponse.ok) {
            const statusData = JSON.parse(stockStatusData);
            Plotly.newPlot('stockStatusChart', statusData.data, {
                ...statusData.layout,
                autosize: true,
                responsive: true
            }, config);
        }
        
        // Make all charts responsive after render
        setTimeout(() => {
            makeChartsResponsive();
        }, 100);
        
    } catch (error) {
        console.error('Error loading charts:', error);
    }
}

function makeChartsResponsive() {
    // Ensure all Plotly charts are responsive
    const chartIds = ['stockByStoreChart', 'stockStatusChart', 'riskMatrixHeatmap'];
    chartIds.forEach(id => {
        const element = document.getElementById(id);
        if (element && element.querySelector('.plotly-graph-div')) {
            Plotly.Plots.resize(element);
        }
    });
    
    // Also handle any other Plotly charts
    document.querySelectorAll('.plotly-graph-div').forEach(chart => {
        Plotly.Plots.resize(chart.parentElement);
    });
}

async function loadEnterpriseCharts() {
    try {
        // Load business critical charts
        await Promise.all([
            loadStockAging(),
            loadStockoutRisk(),
            loadTransferOpportunities()
        ]);
        
        // Load Risk Matrix Heatmap
        await loadRiskMatrixHeatmap();
        
        // Setup interactive controls
        setupChartControls();
        
    } catch (error) {
        console.error('Error loading enterprise charts:', error);
    }
}

async function loadRiskMatrixHeatmap(view = 'category', metric = 'risk') {
    try {
        const params = new URLSearchParams();
        params.append('view', view);
        
        const response = await fetch(`/api/charts/risk-matrix?${params}`);
        const chartData = await response.json();
        
        if (response.ok) {
            const config = {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d', 'resetScale2d', 'zoom2d'],
                displaylogo: false
            };
            
            const heatmapData = JSON.parse(chartData);
            Plotly.newPlot('riskMatrixHeatmap', heatmapData.data, {
                ...heatmapData.layout,
                autosize: true,
                responsive: true
            }, config);
            
            // Ensure this chart is also responsive
            setTimeout(() => {
                const element = document.getElementById('riskMatrixHeatmap');
                if (element && element.querySelector('.plotly-graph-div')) {
                    Plotly.Plots.resize(element);
                }
            }, 100);
        }
    } catch (error) {
        console.error('Error loading risk matrix heatmap:', error);
    }
}

function setupChartControls() {
    // Risk matrix controls
    const matrixView = document.getElementById('matrixView');
    const matrixMetric = document.getElementById('matrixMetric');
    
    if (matrixView) {
        matrixView.addEventListener('change', (e) => {
            const view = e.target.value;
            const metric = matrixMetric ? matrixMetric.value : 'risk';
            loadRiskMatrixHeatmap(view, metric);
        });
    }
    
    if (matrixMetric) {
        matrixMetric.addEventListener('change', (e) => {
            const metric = e.target.value;
            const view = matrixView ? matrixView.value : 'category';
            loadRiskMatrixHeatmap(view, metric);
        });
    }
}

// Utility functions
function formatNumber(num) {
    return num.toLocaleString();
}

function formatDate(dateString) {
    return new Date(dateString).toLocaleDateString();
}

async function loadTransferSuggestions() {
    const suggestionsLoading = document.getElementById('suggestionsLoading');
    const suggestionsList = document.getElementById('suggestionsList');
    const suggestionsActions = document.getElementById('suggestionsActions');
    const allSuggestions = document.getElementById('allSuggestions');
    const showMoreBtn = document.getElementById('showMoreBtn');
    const totalCount = document.getElementById('totalCount');
    
    try {
        suggestionsLoading.style.display = 'flex';
        
        const response = await fetch('/api/transfer-suggestions');
        const data = await response.json();
        
        if (response.ok) {
            suggestionsLoading.style.display = 'none';
            
            if (data.suggestions.length === 0) {
                showNoSuggestions();
            } else {
                // Show top 3 suggestions
                displaySuggestions(data.top_3, suggestionsList);
                suggestionsList.style.display = 'block';
                
                // Setup show more functionality if there are more than 3
                if (data.total_count > 3) {
                    totalCount.textContent = data.total_count;
                    suggestionsActions.style.display = 'block';
                    
                    // Store all suggestions for show more
                    showMoreBtn.suggestionsData = data.suggestions;
                    
                    showMoreBtn.onclick = toggleAllSuggestions;
                }
            }
        } else {
            console.error('Error loading suggestions:', data.error);
            showNoSuggestions();
        }
    } catch (error) {
        console.error('Error loading suggestions:', error);
        suggestionsLoading.style.display = 'none';
        showNoSuggestions();
    }
}

function displaySuggestions(suggestions, container) {
    container.innerHTML = '';
    
    suggestions.forEach(suggestion => {
        const suggestionCard = createSuggestionCard(suggestion);
        container.appendChild(suggestionCard);
    });
}

function createSuggestionCard(suggestion) {
    const card = document.createElement('div');
    card.className = 'suggestion-card';
    
    card.innerHTML = `
        <div class="suggestion-header">
            <div class="suggestion-main">
                <div class="suggestion-medicine">${suggestion.medicine}</div>
                <div class="suggestion-route">
                    <div class="transfer-arrow">
                        <span>${suggestion.from_store}</span>
                        <i class="fas fa-arrow-right"></i>
                        <span>${suggestion.to_store}</span>
                    </div>
                </div>
                <div class="suggestion-reason">${suggestion.reason}</div>
            </div>
            <div class="urgency-badge urgency-${suggestion.urgency.toLowerCase()}">
                ${suggestion.urgency}
            </div>
        </div>
        
        <div class="suggestion-details">
            <div class="detail-item">
                <div class="detail-label">Transfer Qty</div>
                <div class="detail-value">${suggestion.suggested_quantity}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">From Stock</div>
                <div class="detail-value">${suggestion.from_current_stock}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">To Stock</div>
                <div class="detail-value">${suggestion.to_current_stock}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Expires In</div>
                <div class="detail-value">${suggestion.days_until_expiry}d</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Score</div>
                <div class="detail-value">${suggestion.score}</div>
            </div>
        </div>
    `;
    
    return card;
}

function showNoSuggestions() {
    const suggestionsList = document.getElementById('suggestionsList');
    suggestionsList.innerHTML = `
        <div class="no-suggestions">
            <i class="fas fa-check-circle"></i>
            <h4>All Stores Optimally Stocked</h4>
            <p>No transfer suggestions needed at this time. Stock levels are well balanced across all stores.</p>
        </div>
    `;
    suggestionsList.style.display = 'block';
}

function toggleAllSuggestions() {
    const showMoreBtn = document.getElementById('showMoreBtn');
    const allSuggestions = document.getElementById('allSuggestions');
    const allSuggestionsList = allSuggestions.querySelector('.suggestions-list') || 
                                document.createElement('div');
    
    if (!allSuggestions.querySelector('.suggestions-list')) {
        allSuggestionsList.className = 'suggestions-list';
        allSuggestions.appendChild(allSuggestionsList);
    }
    
    if (allSuggestions.style.display === 'none' || !allSuggestions.style.display) {
        // Show all suggestions
        displaySuggestions(showMoreBtn.suggestionsData.slice(3), allSuggestionsList);
        allSuggestions.style.display = 'block';
        showMoreBtn.classList.add('expanded');
        showMoreBtn.innerHTML = `
            <i class="fas fa-chevron-up"></i>
            Show Less
        `;
    } else {
        // Hide all suggestions
        allSuggestions.style.display = 'none';
        showMoreBtn.classList.remove('expanded');
        showMoreBtn.innerHTML = `
            <i class="fas fa-chevron-down"></i>
            Show All Suggestions (<span id="totalCount">${showMoreBtn.suggestionsData.length}</span>)
        `;
    }
}

// ============================================================================
// MACHINE LEARNING FUNCTIONALITY
// ============================================================================

async function loadMLStatus() {
    try {
        const response = await fetch('/api/ml/model-stats');
        const data = await response.json();
        
        const statusIndicator = document.getElementById('mlStatusIndicator');
        const statusText = document.getElementById('mlStatusText');
        
        if (response.ok && data.ml_enabled) {
            statusIndicator.className = 'status-indicator';
            statusText.textContent = `ML Ready - ${data.model_statistics.total_models} models trained`;
        } else {
            statusIndicator.className = 'status-indicator error';
            statusText.textContent = 'ML not available';
        }
    } catch (error) {
        const statusIndicator = document.getElementById('mlStatusIndicator');
        const statusText = document.getElementById('mlStatusText');
        statusIndicator.className = 'status-indicator error';
        statusText.textContent = 'ML service unavailable';
    }
}

function setupMLControls() {
    // Train Models Button
    document.getElementById('trainModelsBtn').addEventListener('click', async () => {
        const btn = document.getElementById('trainModelsBtn');
        
        setButtonLoading(btn, true);
        showMLResults('Training ML models...');
        
        try {
            const response = await fetch('/api/ml/train-models', { method: 'POST' });
            const data = await response.json();
            
            if (response.ok) {
                showMLResults(`
                    <div class="ml-result-card">
                        <div class="ml-result-header">✅ Training Complete</div>
                        <div class="ml-result-content">
                            ${data.message}<br>
                            <strong>Models trained:</strong> ${data.trained_models}<br>
                            <strong>Methods:</strong> ${Object.entries(data.statistics.methods || {}).map(([k,v]) => `${k}: ${v}`).join(', ')}
                        </div>
                    </div>
                `);
                loadMLStatus(); // Refresh status
            } else {
                showMLResults(`
                    <div class="ml-result-card">
                        <div class="ml-result-header">❌ Training Failed</div>
                        <div class="ml-result-content">${data.error}</div>
                    </div>
                `);
            }
        } catch (error) {
            showMLResults(`
                <div class="ml-result-card">
                    <div class="ml-result-header">❌ Error</div>
                    <div class="ml-result-content">Failed to train models: ${error.message}</div>
                </div>
            `);
        } finally {
            setButtonLoading(btn, false);
        }
    });

    // Demand Forecast Button
    document.getElementById('demandForecastBtn').addEventListener('click', async () => {
        const btn = document.getElementById('demandForecastBtn');
        
        setButtonLoading(btn, true);
        showMLResults('Generating demand forecasts...');
        
        try {
            const response = await fetch('/api/ml/demand-forecast');
            const data = await response.json();
            
            if (response.ok) {
                const highRisk = data.forecasts.filter(f => f.stockout_probability_7d > 0.7);
                
                let tableHtml = `
                    <div class="ml-result-card">
                        <div class="ml-result-header">🔮 Demand Forecasts Generated</div>
                        <div class="ml-result-content">
                            <strong>Summary:</strong> ${data.summary.total_predictions} predictions generated<br>
                            <strong>High Risk Items:</strong> ${data.summary.high_risk_stockouts}<br>
                            <strong>Medium Risk Items:</strong> ${data.summary.medium_risk_stockouts}<br>
                            <strong>Avg Days of Cover:</strong> ${data.summary.avg_days_of_cover} days
                        </div>
                    </div>
                `;
                
                if (highRisk.length > 0) {
                    tableHtml += `
                        <div class="ml-result-card">
                            <div class="ml-result-header">⚠️ High Risk Items (Top 10)</div>
                            <div class="ml-result-content">
                                <table class="forecast-table">
                                    <thead>
                                        <tr>
                                            <th>Store</th>
                                            <th>Medicine</th>
                                            <th>7-Day Demand</th>
                                            <th>Stockout Risk</th>
                                            <th>Days of Cover</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${highRisk.slice(0, 10).map(f => `
                                            <tr class="high-risk">
                                                <td>${f.store_id}</td>
                                                <td>${f.medicine}</td>
                                                <td>${f.predicted_demand_7d}</td>
                                                <td>${(f.stockout_probability_7d * 100).toFixed(1)}%</td>
                                                <td>${f.days_of_cover}</td>
                                            </tr>
                                        `).join('')}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    `;
                }
                
                showMLResults(tableHtml);
            } else {
                showMLResults(`
                    <div class="ml-result-card">
                        <div class="ml-result-header">❌ Forecast Failed</div>
                        <div class="ml-result-content">${data.error}</div>
                    </div>
                `);
            }
        } catch (error) {
            showMLResults(`
                <div class="ml-result-card">
                    <div class="ml-result-header">❌ Error</div>
                    <div class="ml-result-content">Failed to generate forecasts: ${error.message}</div>
                </div>
            `);
        } finally {
            setButtonLoading(btn, false);
        }
    });

    // ML Transfer Optimization Button
    document.getElementById('mlTransferBtn').addEventListener('click', async () => {
        const btn = document.getElementById('mlTransferBtn');
        
        setButtonLoading(btn, true);
        showMLResults('Optimizing transfers with ML...');
        
        try {
            const response = await fetch('/api/ml/transfer-optimization');
            const data = await response.json();
            
            if (response.ok) {
                let resultHtml = `
                    <div class="ml-result-card">
                        <div class="ml-result-header">🚚 ML Transfer Optimization Complete</div>
                        <div class="ml-result-content">
                            <strong>Total Suggestions:</strong> ${data.summary.total_suggestions}<br>
                            <strong>High Urgency:</strong> ${data.summary.high_urgency}<br>
                            <strong>Total Units:</strong> ${data.summary.total_units}<br>
                            <strong>Estimated Savings:</strong> $${data.summary.estimated_savings.toFixed(2)}
                        </div>
                    </div>
                `;
                
                if (data.ml_transfer_suggestions.length > 0) {
                    resultHtml += `
                        <div class="ml-result-card">
                            <div class="ml-result-header">📋 Top ML Transfer Suggestions</div>
                            <div class="ml-result-content">
                                <table class="forecast-table">
                                    <thead>
                                        <tr>
                                            <th>From Store</th>
                                            <th>To Store</th>
                                            <th>Medicine</th>
                                            <th>Quantity</th>
                                            <th>Urgency</th>
                                            <th>Savings</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${data.ml_transfer_suggestions.slice(0, 10).map(s => `
                                            <tr class="${s.urgency_score > 70 ? 'high-risk' : s.urgency_score > 30 ? 'medium-risk' : 'low-risk'}">
                                                <td>${s.from_store}</td>
                                                <td>${s.to_store}</td>
                                                <td>${s.medicine}</td>
                                                <td>${s.quantity}</td>
                                                <td>${s.urgency_score.toFixed(1)}</td>
                                                <td>$${s.cost_savings.toFixed(2)}</td>
                                            </tr>
                                        `).join('')}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    `;
                }
                
                showMLResults(resultHtml);
            } else {
                showMLResults(`
                    <div class="ml-result-card">
                        <div class="ml-result-header">❌ Optimization Failed</div>
                        <div class="ml-result-content">${data.error}</div>
                    </div>
                `);
            }
        } catch (error) {
            showMLResults(`
                <div class="ml-result-card">
                    <div class="ml-result-header">❌ Error</div>
                    <div class="ml-result-content">Failed to optimize transfers: ${error.message}</div>
                </div>
            `);
        } finally {
            setButtonLoading(btn, false);
        }
    });

    // Risk Analysis Button
    document.getElementById('riskAnalysisBtn').addEventListener('click', async () => {
        const btn = document.getElementById('riskAnalysisBtn');
        
        setButtonLoading(btn, true);
        showMLResults('Analyzing stockout risks...');
        
        try {
            const response = await fetch('/api/ml/stockout-risk-analysis');
            const data = await response.json();
            
            if (response.ok) {
                const analysis = data.risk_analysis;
                
                let resultHtml = `
                    <div class="ml-result-card">
                        <div class="ml-result-header">📊 Risk Analysis Summary</div>
                        <div class="ml-result-content">
                            <strong>Items Analyzed:</strong> ${analysis.overall.total_items_analyzed}<br>
                            <strong>High Risk Items:</strong> ${analysis.overall.high_risk_items}<br>
                            <strong>Medium Risk Items:</strong> ${analysis.overall.medium_risk_items}<br>
                            <strong>Global Average Risk:</strong> ${(analysis.overall.avg_global_risk * 100).toFixed(1)}%
                        </div>
                    </div>
                `;
                
                // Risk by category
                const topRiskCategories = Object.entries(analysis.by_category)
                    .sort(([,a], [,b]) => b.avg_risk - a.avg_risk)
                    .slice(0, 5);
                
                resultHtml += `
                    <div class="ml-result-card">
                        <div class="ml-result-header">📈 Risk by Category (Top 5)</div>
                        <div class="ml-result-content">
                            <table class="forecast-table">
                                <thead>
                                    <tr>
                                        <th>Category</th>
                                        <th>Avg Risk</th>
                                        <th>Max Risk</th>
                                        <th>Items at Risk</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${topRiskCategories.map(([cat, data]) => `
                                        <tr class="${data.avg_risk > 0.5 ? 'high-risk' : data.avg_risk > 0.2 ? 'medium-risk' : 'low-risk'}">
                                            <td>${cat}</td>
                                            <td>${(data.avg_risk * 100).toFixed(1)}%</td>
                                            <td>${(data.max_risk * 100).toFixed(1)}%</td>
                                            <td>${data.items_at_risk}</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                `;
                
                showMLResults(resultHtml);
            } else {
                showMLResults(`
                    <div class="ml-result-card">
                        <div class="ml-result-header">❌ Analysis Failed</div>
                        <div class="ml-result-content">${data.error}</div>
                    </div>
                `);
            }
        } catch (error) {
            showMLResults(`
                <div class="ml-result-card">
                    <div class="ml-result-header">❌ Error</div>
                    <div class="ml-result-content">Failed to analyze risks: ${error.message}</div>
                </div>
            `);
        } finally {
            setButtonLoading(btn, false);
        }
    });
}

function setButtonLoading(button, isLoading) {
    if (isLoading) {
        button.classList.add('loading');
        button.disabled = true;
    } else {
        button.classList.remove('loading');
        button.disabled = false;
    }
}

function showMLResults(html) {
    const resultsDiv = document.getElementById('mlResults');
    resultsDiv.innerHTML = html;
    resultsDiv.classList.add('show');
}

// Auto-refresh functionality (optional)
function startAutoRefresh() {
    setInterval(async () => {
        if (inventoryData) {
            await loadDashboard();
        }
    }, 30000); // Refresh every 30 seconds
}

// Initialize auto-refresh if needed
// startAutoRefresh();

// Event listeners for responsive charts and navigation
document.addEventListener('DOMContentLoaded', () => {
    // Add window resize listener for chart responsiveness
    window.addEventListener('resize', debounce(() => {
        makeChartsResponsive();
    }, 250));
    
    // Setup navigation
    setupNavigation();
    
    // Additional debugging - force navigation setup after a delay
    setTimeout(() => {
        console.log('Setting up direct navigation handlers as backup...');
        
        // Direct handler for transfer nav card
        const transferCard = document.getElementById('transferNavCard');
        if (transferCard) {
            transferCard.onclick = function(e) {
                e.preventDefault();
                e.stopPropagation();
                console.log('Direct transfer card click handler triggered');
                showPage('transfer');
                return false;
            };
        }
        
        // Direct handler for analytics nav card  
        const analyticsCard = document.getElementById('analyticsNavCard');
        if (analyticsCard) {
            analyticsCard.onclick = function(e) {
                e.preventDefault();
                e.stopPropagation();
                console.log('Direct analytics card click handler triggered');
                showPage('analytics');
                return false;
            };
        }
    }, 1000);
});

// Debounce function to limit resize event calls
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Duplicate navigation function removed - using the one at the top of the file

// Make showPage globally available
window.showPage = function showPage(page) {
    console.log('🎯 showPage called with:', page);
    
    const dashboardSection = document.querySelector('.dashboard');
    const transferPage = document.getElementById('transferPage');
    const analyticsPage = document.getElementById('analyticsPage');
    
    console.log('Page elements found:', {
        dashboardSection: !!dashboardSection,
        transferPage: !!transferPage,
        analyticsPage: !!analyticsPage
    });
    
    if (!transferPage) {
        console.error('transferPage element not found!');
        return;
    }
    
    if (!analyticsPage) {
        console.error('analyticsPage element not found!');
        return;
    }
    
    // Hide all pages
    if (dashboardSection) dashboardSection.style.display = 'none';
    if (transferPage) transferPage.style.display = 'none';
    if (analyticsPage) analyticsPage.style.display = 'none';
    
    // Show selected page
    switch(page) {
        case 'dashboard':
            console.log('📊 Showing dashboard');
            if (dashboardSection) {
                dashboardSection.style.display = 'block';
                console.log('✅ Dashboard is now visible');
            }
            break;
        case 'transfer':
            console.log('🔄 Showing transfer page');
            if (transferPage) {
                transferPage.style.display = 'block';
                console.log('✅ Transfer page is now visible');
                // Load all transfer suggestions and setup search
                setTimeout(async () => {
                    // TODO: Implement loadAllTransferSuggestions function
                    console.log('🔄 Loading transfer suggestions...');
                    if (window.inventoryData && Array.isArray(window.inventoryData)) {
                        console.log('💾 Inventory data available for search:', window.inventoryData.length, 'items');
                        // TODO: Implement setupTransferSearch function
                        // setupTransferSearch();
                    } else {
                        console.log('⚠️ No inventory data available for search');
                    }
                }, 100);
            } else {
                console.error('❌ Transfer page element not found!');
            }
            break;
        case 'analytics':
            console.log('📈 Showing analytics page');
            if (analyticsPage) {
                analyticsPage.style.display = 'block';
                console.log('✅ Analytics page is now visible');
                // Load charts when analytics page is shown
                setTimeout(() => {
                    // TODO: Implement loadCharts and loadEnterpriseCharts functions
                    console.log('📈 Loading analytics charts...');
                }, 100);
            } else {
                console.error('❌ Analytics page element not found!');
            }
            break;
        default:
            console.error('❌ Unknown page:', page);
    }
    
    // Scroll to top
    window.scrollTo(0, 0);
}

function setupTransferSearch() {
    const searchInput = document.getElementById('medicineSearch');
    const searchButton = document.getElementById('searchButton');
    const medicineResults = document.getElementById('medicineResults');
    
    if (!searchInput || !searchButton || !medicineResults) {
        console.log('Transfer search elements not found, skipping setup');
        return;
    }
    
    let allMedicines = [];
    
    // Get all unique medicines from inventory data
    if (window.inventoryData && Array.isArray(window.inventoryData)) {
        allMedicines = [...new Set(window.inventoryData.map(item => item.Medicine))].sort();
    }
    
    // Refresh medicines when page is shown
    function refreshMedicines() {
        if (window.inventoryData && Array.isArray(window.inventoryData)) {
            allMedicines = [...new Set(window.inventoryData.map(item => item.Medicine))].sort();
        }
    }
    
    function performSearch() {
        const query = searchInput.value.toLowerCase().trim();
        if (query.length < 2) {
            medicineResults.innerHTML = '';
            return;
        }
        
        const filteredMedicines = allMedicines.filter(medicine => 
            medicine.toLowerCase().includes(query)
        ).slice(0, 10); // Limit to 10 results
        
        displaySearchResults(filteredMedicines);
    }
    
    function displaySearchResults(medicines) {
        if (medicines.length === 0) {
            medicineResults.innerHTML = '<div style="padding: 16px; color: #d1d5db; text-align: center;">No medicines found</div>';
            return;
        }
        
        const resultsHTML = medicines.map(medicine => {
            const medicineData = window.inventoryData.filter(item => item.Medicine === medicine);
            const totalQuantity = medicineData.reduce((sum, item) => sum + item.Quantity, 0);
            const storeCount = medicineData.length;
            
            return `
                <div class="medicine-option" data-medicine="${medicine}">
                    <div class="medicine-name">${medicine}</div>
                    <div class="medicine-info">${totalQuantity} units across ${storeCount} stores</div>
                </div>
            `;
        }).join('');
        
        medicineResults.innerHTML = resultsHTML;
        
        // Add click handlers to medicine options
        document.querySelectorAll('.medicine-option').forEach(option => {
            option.addEventListener('click', () => {
                const selectedMedicine = option.dataset.medicine;
                selectMedicine(selectedMedicine);
            });
        });
    }
    
    function selectMedicine(medicine) {
        searchInput.value = medicine;
        medicineResults.innerHTML = '';
        loadTransferSuggestionsForMedicine(medicine);
    }
    
    // Event listeners
    searchInput.addEventListener('input', debounce(performSearch, 300));
    searchButton.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            performSearch();
        }
    });
}

async function loadTransferSuggestionsForMedicine(medicine) {
    const transferResults = document.getElementById('transferResults');
    const selectedMedicineInfo = document.getElementById('selectedMedicineInfo');
    const transferSuggestions = document.getElementById('transferSuggestions');
    
    // Show loading state
    transferResults.style.display = 'block';
    selectedMedicineInfo.innerHTML = '<div class="loading">Loading medicine information...</div>';
    transferSuggestions.innerHTML = '';
    
    try {
        // Get medicine data
        const medicineData = window.inventoryData.filter(item => item.Medicine === medicine);
        
        // Display medicine info
        const totalQuantity = medicineData.reduce((sum, item) => sum + item.Quantity, 0);
        const storeCount = medicineData.length;
        const averageQuantity = Math.round(totalQuantity / storeCount);
        
        selectedMedicineInfo.innerHTML = `
            <div class="selected-medicine-title">${medicine}</div>
            <div class="selected-medicine-stats">
                Total Quantity: ${formatNumber(totalQuantity)} units • 
                Available in ${storeCount} stores • 
                Average per store: ${formatNumber(averageQuantity)} units
            </div>
        `;
        
        // Load transfer suggestions
        const response = await fetch('/api/transfer-suggestions');
        const allSuggestions = await response.json();
        
        // Filter suggestions for this medicine
        const medicineSuggestions = allSuggestions.filter(suggestion => 
            suggestion.medicine === medicine
        );
        
        if (medicineSuggestions.length === 0) {
            transferSuggestions.innerHTML = '<div style="color: #d1d5db; text-align: center; padding: 32px;">No transfer suggestions available for this medicine.</div>';
            return;
        }
        
        // Display suggestions
        const suggestionsHTML = medicineSuggestions.map(suggestion => `
            <div class="suggestion-card">
                <div class="suggestion-header">
                    <div class="suggestion-route">
                        <span class="store-from">${suggestion.from_store}</span>
                        <i class="fas fa-arrow-right" style="color: #4ade80; margin: 0 12px;"></i>
                        <span class="store-to">${suggestion.to_store}</span>
                    </div>
                    <div class="suggestion-priority priority-${suggestion.priority.toLowerCase()}">${suggestion.priority}</div>
                </div>
                <div class="suggestion-details">
                    <div class="suggestion-quantity">${formatNumber(suggestion.quantity)} units</div>
                    <div class="suggestion-reason">${suggestion.reason}</div>
                    <div class="suggestion-impact">Expected impact: ${suggestion.impact}</div>
                </div>
            </div>
        `).join('');
        
        transferSuggestions.innerHTML = suggestionsHTML;
        
    } catch (error) {
        console.error('Error loading transfer suggestions:', error);
        transferSuggestions.innerHTML = '<div style="color: #ef4444; text-align: center; padding: 32px;">Error loading transfer suggestions</div>';
    }
}

// Function to load all transfer suggestions for the dedicated transfer page
async function loadAllTransferSuggestions() {
    const allSuggestionsLoading = document.getElementById('allSuggestionsLoading');
    const allSuggestionsDisplay = document.getElementById('allSuggestionsDisplay');

    try {
        // Show loading
        if (allSuggestionsLoading) allSuggestionsLoading.style.display = 'flex';
        if (allSuggestionsDisplay) allSuggestionsDisplay.style.display = 'none';

        // Fetch suggestions
        const response = await fetch('/api/transfer-suggestions');
        const data = await response.json();

        if (response.ok) {
            const suggestions = data.suggestions || [];
            
            // Hide loading
            if (allSuggestionsLoading) allSuggestionsLoading.style.display = 'none';

            if (suggestions.length === 0) {
                if (allSuggestionsDisplay) {
                    allSuggestionsDisplay.innerHTML = '<div style="color: #d1d5db; text-align: center; padding: 24px;">No transfer suggestions available at this time.</div>';
                    allSuggestionsDisplay.style.display = 'block';
                }
                return;
            }

            // Display all suggestions using the existing displaySuggestions function
            if (allSuggestionsDisplay) {
                displaySuggestions(suggestions, allSuggestionsDisplay);
                allSuggestionsDisplay.style.display = 'block';
            }

        } else {
            throw new Error(data.error || 'Failed to load transfer suggestions');
        }

    } catch (error) {
        console.error('Error loading transfer suggestions:', error);
        if (allSuggestionsLoading) allSuggestionsLoading.style.display = 'none';
        if (allSuggestionsDisplay) {
            allSuggestionsDisplay.innerHTML = '<div style="color: #ef4444; text-align: center; padding: 24px;">Error loading transfer suggestions</div>';
            allSuggestionsDisplay.style.display = 'block';
        }
    }
}
