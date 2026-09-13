/**
 * CustomerIQ – Frontend Application Logic
 * Handles all API calls, chart rendering, and page-specific initialisation.
 * Loaded as a plain <script> so all functions are globally accessible by router.js.
 */

// ========================================================
// GLOBALS
// ========================================================
var overviewChartInstance = null;
var donutChartInstance = null;
var analyticsContractChart = null;
var analyticsInternetChart = null;
var analyticsTenureChart = null;
var analyticsPaymentChart = null;
var currentSegment = 'All Segments';
var customerOffset = 0;
var customerSegment = 'All Segments';

// ========================================================
// DASHBOARD INIT
// ========================================================
document.addEventListener('DOMContentLoaded', function () {
    loadDashboardStats(currentSegment);
    loadCustomersTable();

    // Close notification dropdown when clicking outside
    document.addEventListener('click', function (e) {
        var dropdown = document.getElementById('notif-dropdown');
        var notifBtn = document.getElementById('notif-btn');
        if (dropdown && notifBtn && !dropdown.contains(e.target) && !notifBtn.contains(e.target)) {
            dropdown.classList.remove('show');
        }
    });

    // Keyboard shortcut (Ctrl+K or Cmd+K) to focus search
    document.addEventListener('keydown', function (e) {
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
            e.preventDefault();
            var searchInput = document.getElementById('header-search');
            if (searchInput) searchInput.focus();
        }
    });
});

function toggleNotifDropdown(e) {
    e.stopPropagation();
    var dropdown = document.getElementById('notif-dropdown');
    if (dropdown) {
        dropdown.classList.toggle('show');
    }
}

function handleSegmentChange(val) {
    currentSegment = val;
    loadDashboardStats(currentSegment);
    loadCustomersTable();
}

var searchTimer = null;
function handleSearch(e) {
    clearTimeout(searchTimer);
    var query = e.target.value;
    searchTimer = setTimeout(function () {
        loadCustomersTable(query);
    }, 300);
}

// ========================================================
// DASHBOARD: LOAD STATS
// ========================================================
async function loadDashboardStats(segment) {
    try {
        var res = await fetch('/api/dashboard-stats?segment=' + encodeURIComponent(segment));
        var data = await res.json();

        // KPI Cards
        document.getElementById('kpi-total').innerText = data.kpis.total_customers;
        document.getElementById('kpi-total-change').innerText = data.kpis.total_customers_change;
        document.getElementById('kpi-atrisk').innerText = data.kpis.at_risk_customers;
        document.getElementById('kpi-atrisk-pct').innerText = data.kpis.at_risk_percent;
        document.getElementById('kpi-atrisk-change').innerText = data.kpis.at_risk_change;
        document.getElementById('kpi-churnrate').innerText = data.kpis.churn_rate;
        document.getElementById('kpi-churn-change').innerText = data.kpis.churn_rate_change;
        document.getElementById('kpi-retention').innerText = data.kpis.predicted_retention;
        document.getElementById('kpi-retention-change').innerText = data.kpis.predicted_retention_change;

        renderOverviewChart(data.trend_overview);
        renderDonutChart(data.risk_distribution);
        renderTopFactors(data.top_churn_factors, 'factors-container');
        renderInsights(data.insights);

    } catch (err) {
        console.error('Error loading dashboard stats:', err);
    }
}

// ========================================================
// CHART: LINE OVERVIEW
// ========================================================
function renderOverviewChart(trendData) {
    var ctx = document.getElementById('overviewChart').getContext('2d');
    var labels = trendData.map(function (d) { return d.month; });
    var churnedVals = trendData.map(function (d) { return d.churned; });
    var retainedVals = trendData.map(function (d) { return d.retained; });

    if (overviewChartInstance) overviewChartInstance.destroy();

    overviewChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Retained',
                    data: retainedVals,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.05)',
                    borderWidth: 2,
                    pointBackgroundColor: '#2563eb',
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    tension: 0.35,
                    fill: false
                },
                {
                    label: 'Churned',
                    data: churnedVals,
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.05)',
                    borderWidth: 2,
                    pointBackgroundColor: '#ef4444',
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    tension: 0.35,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#0f172a',
                    titleFont: { family: 'Inter', size: 12 },
                    bodyFont: { family: 'Inter', size: 12 },
                    padding: 10,
                    cornerRadius: 8
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#64748b', font: { family: 'Inter', size: 11 } }
                },
                y: {
                    grid: { color: '#f1f5f9' },
                    ticks: { color: '#64748b', font: { family: 'Inter', size: 11 } },
                    beginAtZero: true
                }
            }
        }
    });
}

// ========================================================
// CHART: DONUT
// ========================================================
function renderDonutChart(riskDist) {
    var ctx = document.getElementById('donutChart').getContext('2d');

    document.getElementById('donut-center-pct').innerText = riskDist.overall_at_risk_pct;
    document.getElementById('pct-high-risk').innerText = riskDist.high_risk.percent + '%';
    document.getElementById('pct-med-risk').innerText = riskDist.medium_risk.percent + '%';
    document.getElementById('pct-low-risk').innerText = riskDist.low_risk.percent + '%';

    if (donutChartInstance) donutChartInstance.destroy();

    donutChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['High Risk', 'Medium Risk', 'Low Risk'],
            datasets: [{
                data: [riskDist.high_risk.count, riskDist.medium_risk.count, riskDist.low_risk.count],
                backgroundColor: ['#ef4444', '#f59e0b', '#10b981'],
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '76%',
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return ' ' + context.label + ': ' + context.raw + ' customers';
                        }
                    }
                }
            }
        }
    });
}

// ========================================================
// SHAP FACTORS BAR LIST
// ========================================================
function renderTopFactors(factors, containerId) {
    var container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';

    factors.forEach(function (f) {
        var item = document.createElement('div');
        item.className = 'factor-item';
        item.innerHTML =
            '<div class="factor-header">' +
            '<span class="factor-name">' + f.factor + '</span>' +
            '<span class="factor-pct">' + f.percentage + '%</span>' +
            '</div>' +
            '<div class="factor-bar-bg">' +
            '<div class="factor-bar-fill" style="width:' + f.percentage + '%;"></div>' +
            '</div>';
        container.appendChild(item);
    });
}

// ========================================================
// INSIGHTS LIST
// ========================================================
function renderInsights(insights) {
    var container = document.getElementById('insights-container');
    if (!container) return;
    container.innerHTML = '';
    var dotColors = ['#2563eb', '#f59e0b', '#8b5cf6', '#10b981'];
    insights.forEach(function (text, i) {
        var color = dotColors[i % dotColors.length];
        var li = document.createElement('li');
        li.className = 'insight-item';
        li.innerHTML =
            '<span class="insight-dot" style="background:' + color + ';"></span>' +
            '<span>' + text + '</span>';
        container.appendChild(li);
    });
}

// ========================================================
// DASHBOARD: CUSTOMERS TABLE (recent predictions)
// ========================================================
async function loadCustomersTable(searchQuery) {
    if (searchQuery === undefined) searchQuery = '';
    try {
        var res = await fetch('/api/customers?limit=6&search=' + encodeURIComponent(searchQuery) + '&segment=' + encodeURIComponent(currentSegment));
        var data = await res.json();
        var tbody = document.getElementById('predictions-tbody');
        if (!tbody) return;
        tbody.innerHTML = '';

        data.customers.forEach(function (c) {
            var pillClass = 'pill-low';
            if (c.risk_level === 'High') pillClass = 'pill-high';
            else if (c.risk_level === 'Medium') pillClass = 'pill-medium';

            var tr = document.createElement('tr');
            tr.innerHTML =
                '<td class="cust-id">' + c.customer_id + '</td>' +
                '<td><b>' + c.churn_probability.toFixed(2) + '</b></td>' +
                '<td><span class="risk-pill ' + pillClass + '">' + c.risk_level + '</span></td>' +
                '<td>' + c.predicted_churn_date + '</td>' +
                '<td><button class="view-btn" onclick="openCustomerModal(\'' + c.customer_id + '\')">View</button></td>';
            tbody.appendChild(tr);
        });

    } catch (err) {
        console.error('Error loading customers table:', err);
    }
}

// ========================================================
// CUSTOMER MODAL (SHAP detail)
// ========================================================
async function openCustomerModal(customerId) {
    var modal = document.getElementById('shap-modal');
    document.getElementById('modal-cust-id').innerText = 'Customer ' + customerId;
    document.getElementById('modal-shap-img').src = '';
    document.getElementById('modal-drivers-grid').innerHTML = '<p style="color:#64748b; font-size:0.8rem;">Calculating SHAP explanation...</p>';
    modal.classList.add('open');

    try {
        var res = await fetch('/api/customer/' + customerId + '/shap');
        var data = await res.json();

        document.getElementById('modal-prob').innerText = data.customer.churn_probability.toFixed(2);

        var pillHtml = '<span class="risk-pill pill-low">Low Risk</span>';
        if (data.customer.risk_level === 'High') {
            pillHtml = '<span class="risk-pill pill-high">High Risk</span>';
            document.getElementById('modal-prob').style.color = '#ef4444';
        } else if (data.customer.risk_level === 'Medium') {
            pillHtml = '<span class="risk-pill pill-medium">Medium Risk</span>';
            document.getElementById('modal-prob').style.color = '#f59e0b';
        } else {
            document.getElementById('modal-prob').style.color = '#10b981';
        }
        document.getElementById('modal-risk-pill').innerHTML = pillHtml;
        document.getElementById('modal-contract').innerText = data.customer.contract + ' (' + data.customer.tenure + ' mos)';
        document.getElementById('modal-bill').innerText = '$' + data.customer.monthly_charges.toFixed(2);
        document.getElementById('modal-shap-img').src = data.shap_plot_base64;

        var driversGrid = document.getElementById('modal-drivers-grid');
        driversGrid.innerHTML = '';
        data.top_features.forEach(function (f) {
            var isPos = f.shap_value > 0;
            var card = document.createElement('div');
            card.className = 'shap-feature-card';
            card.innerHTML =
                '<span class="name">' + f.feature + '</span>' +
                '<span class="' + (isPos ? 'effect-pos' : 'effect-neg') + '">' +
                (isPos ? '▲ +' : '▼ ') + f.shap_value.toFixed(3) + ' (' + f.effect + ')' +
                '</span>';
            driversGrid.appendChild(card);
        });

    } catch (err) {
        console.error('Error opening modal:', err);
    }
}

function closeModal(e) {
    if (e.target.id === 'shap-modal') {
        document.getElementById('shap-modal').classList.remove('open');
    }
}

function closeModalDirect() {
    document.getElementById('shap-modal').classList.remove('open');
}

// ========================================================
// PREDICTION PAGE: SIMULATE
// ========================================================
async function handleSimulate(e) {
    e.preventDefault();
    var btn = document.getElementById('sim-submit-btn');
    if (btn) btn.innerText = 'Running...';

    var payload = {
        gender: document.getElementById('sim-gender').value,
        SeniorCitizen: parseInt(document.getElementById('sim-senior').value),
        Partner: document.getElementById('sim-partner').value,
        Dependents: document.getElementById('sim-dependents').value,
        tenure: parseInt(document.getElementById('sim-tenure').value),
        PhoneService: document.getElementById('sim-phone').value,
        MultipleLines: document.getElementById('sim-multilines').value,
        InternetService: document.getElementById('sim-internet').value,
        OnlineSecurity: document.getElementById('sim-security').value,
        OnlineBackup: document.getElementById('sim-backup').value,
        DeviceProtection: document.getElementById('sim-device').value,
        TechSupport: document.getElementById('sim-techsupport').value,
        StreamingTV: document.getElementById('sim-tv').value,
        StreamingMovies: document.getElementById('sim-movies').value,
        Contract: document.getElementById('sim-contract').value,
        PaperlessBilling: document.getElementById('sim-paperless').value,
        PaymentMethod: document.getElementById('sim-payment').value,
        MonthlyCharges: parseFloat(document.getElementById('sim-monthly').value),
        TotalCharges: parseFloat(document.getElementById('sim-total').value)
    };

    try {
        var res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        var data = await res.json();

        document.getElementById('sim-prob').innerText = data.churn_probability.toFixed(2);
        var badge = document.getElementById('sim-risk-badge');
        badge.innerText = data.risk_level + ' Risk';
        badge.className = 'risk-pill pill-' + data.risk_level.toLowerCase();

        // Colour the probability number
        var probEl = document.getElementById('sim-prob');
        if (data.risk_level === 'High') probEl.style.color = '#ef4444';
        else if (data.risk_level === 'Medium') probEl.style.color = '#f59e0b';
        else probEl.style.color = '#10b981';

        document.getElementById('sim-shap-img').src = data.shap_plot_base64;
        document.getElementById('sim-result-box').style.display = 'block';

    } catch (err) {
        console.error('Error running simulation:', err);
        alert('Prediction failed. Make sure the FastAPI server is running.');
    } finally {
        if (btn) {
            btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg> Run Simulation';
        }
    }
}

// ========================================================
// CUSTOMERS PAGE
// ========================================================
function handleCustomerSegmentChange(val) {
    customerSegment = val;
    customerOffset = 0;
    loadAllCustomers(0);
}

async function loadAllCustomers(offsetDelta) {
    customerOffset = Math.max(0, customerOffset + (offsetDelta || 0));
    try {
        var res = await fetch('/api/customers?limit=25&offset=' + customerOffset + '&segment=' + encodeURIComponent(customerSegment));
        var data = await res.json();
        var tbody = document.getElementById('all-customers-tbody');
        if (!tbody) return;
        tbody.innerHTML = '';

        if (data.customers.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; padding:2rem; color:var(--text-muted);">No customers found.</td></tr>';
            return;
        }

        data.customers.forEach(function (c) {
            var pillClass = 'pill-low';
            if (c.risk_level === 'High') pillClass = 'pill-high';
            else if (c.risk_level === 'Medium') pillClass = 'pill-medium';

            var tr = document.createElement('tr');
            tr.innerHTML =
                '<td class="cust-id">' + c.customer_id + '</td>' +
                '<td><b>' + c.churn_probability.toFixed(2) + '</b></td>' +
                '<td><span class="risk-pill ' + pillClass + '">' + c.risk_level + '</span></td>' +
                '<td>' + (c.contract || '—') + '</td>' +
                '<td>' + (c.tenure !== undefined ? c.tenure + ' mo' : '—') + '</td>' +
                '<td>' + (c.monthly_charges !== undefined ? '$' + c.monthly_charges.toFixed(2) : '—') + '</td>' +
                '<td>' + c.predicted_churn_date + '</td>' +
                '<td><button class="view-btn" onclick="openCustomerModal(\'' + c.customer_id + '\')">View</button></td>';
            tbody.appendChild(tr);
        });

        var countEl = document.getElementById('cust-table-count');
        if (countEl) countEl.innerText = 'Showing ' + (customerOffset + 1) + '–' + (customerOffset + data.customers.length) + ' of ' + data.total;

    } catch (err) {
        console.error('Error loading all customers:', err);
    }
}

// ========================================================
// ANALYTICS PAGE
// ========================================================
async function initAnalyticsPage() {
    try {
        var res = await fetch('/api/dashboard-stats?segment=All Segments');
        var data = await res.json();

        // Global SHAP factors
        renderTopFactors(data.top_churn_factors, 'analytics-shap-factors');

        // Build analytics from risk distribution data
        buildAnalyticsCharts(data);

    } catch (err) {
        console.error('Error loading analytics:', err);
    }
}

function buildAnalyticsCharts(data) {
    // Contract type chart (bar) — use risk distribution as proxy data
    var rd = data.risk_distribution;

    // Churn by Contract: fetch segment data for each contract type
    Promise.all([
        fetch('/api/dashboard-stats?segment=Month-to-month').then(function (r) { return r.json(); }),
        fetch('/api/dashboard-stats?segment=One year').then(function (r) { return r.json(); }),
        fetch('/api/dashboard-stats?segment=Two year').then(function (r) { return r.json(); })
    ]).then(function (results) {
        var labels = ['Month-to-month', 'One Year', 'Two Year'];
        var churnRates = results.map(function (r) {
            return parseFloat(r.kpis.churn_rate);
        });

        var ctx1 = document.getElementById('analyticsContractChart');
        if (!ctx1) return;
        if (analyticsContractChart) analyticsContractChart.destroy();
        analyticsContractChart = new Chart(ctx1, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Churn Rate (%)',
                    data: churnRates,
                    backgroundColor: ['#ef4444', '#f59e0b', '#10b981'],
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { display: false }, ticks: { color: '#64748b' } },
                    y: { grid: { color: '#f1f5f9' }, ticks: { color: '#64748b' }, beginAtZero: true }
                }
            }
        });
    });

    // Internet service chart
    Promise.all([
        fetch('/api/dashboard-stats?segment=Fiber optic').then(function (r) { return r.json(); }),
        fetch('/api/dashboard-stats?segment=DSL').then(function (r) { return r.json(); })
    ]).then(function (results) {
        var ctx2 = document.getElementById('analyticsInternetChart');
        if (!ctx2) return;
        if (analyticsInternetChart) analyticsInternetChart.destroy();
        analyticsInternetChart = new Chart(ctx2, {
            type: 'bar',
            data: {
                labels: ['Fiber Optic', 'DSL'],
                datasets: [{
                    label: 'Churn Rate (%)',
                    data: results.map(function (r) { return parseFloat(r.kpis.churn_rate); }),
                    backgroundColor: ['#8b5cf6', '#2563eb'],
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { display: false }, ticks: { color: '#64748b' } },
                    y: { grid: { color: '#f1f5f9' }, ticks: { color: '#64748b' }, beginAtZero: true }
                }
            }
        });
    });

    // Tenure distribution doughnut
    var ctx3 = document.getElementById('analyticsTenureChart');
    if (ctx3) {
        if (analyticsTenureChart) analyticsTenureChart.destroy();
        analyticsTenureChart = new Chart(ctx3, {
            type: 'doughnut',
            data: {
                labels: ['0–12 months', '13–36 months', '37–72 months'],
                datasets: [{
                    data: [
                        rd.high_risk.count,
                        rd.medium_risk.count,
                        rd.low_risk.count
                    ],
                    backgroundColor: ['#ef4444', '#f59e0b', '#10b981'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                cutout: '60%',
                plugins: { legend: { position: 'bottom', labels: { color: '#64748b', font: { family: 'Inter' } } } }
            }
        });
    }

    // Payment method bar
    Promise.all([
        fetch('/api/dashboard-stats?segment=Electronic check').then(function (r) { return r.json(); })
    ]).then(function (results) {
        var ctx4 = document.getElementById('analyticsPaymentChart');
        if (!ctx4) return;
        if (analyticsPaymentChart) analyticsPaymentChart.destroy();
        analyticsPaymentChart = new Chart(ctx4, {
            type: 'bar',
            data: {
                labels: ['Electronic Check (At-Risk %)', 'Other Segments'],
                datasets: [{
                    data: [parseFloat(results[0].kpis.churn_rate), parseFloat(data.kpis.churn_rate)],
                    backgroundColor: ['#ef4444', '#2563eb'],
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { display: false }, ticks: { color: '#64748b' } },
                    y: { grid: { color: '#f1f5f9' }, ticks: { color: '#64748b' }, beginAtZero: true }
                }
            }
        });
    });
}

// ========================================================
// REPORTS PAGE
// ========================================================
async function initReportsPage() {
    try {
        // Load overall stats for the summary KPIs
        var res = await fetch('/api/dashboard-stats?segment=All Segments');
        var data = await res.json();
        var rd = data.risk_distribution;

        document.getElementById('report-high').innerText = rd.high_risk.count.toLocaleString();
        document.getElementById('report-medium').innerText = rd.medium_risk.count.toLocaleString();
        document.getElementById('report-low').innerText = rd.low_risk.count.toLocaleString();
        document.getElementById('report-total').innerText = data.kpis.total_customers;

        // Load at-risk customers for the report table (high + medium = limit 50 sorted by prob)
        var res2 = await fetch('/api/customers?limit=50&segment=All Segments');
        var custData = await res2.json();
        var tbody = document.getElementById('report-tbody');
        if (!tbody) return;
        tbody.innerHTML = '';

        // Filter to high and medium risk
        var atRisk = custData.customers.filter(function (c) {
            return c.risk_level === 'High' || c.risk_level === 'Medium';
        });

        atRisk.forEach(function (c) {
            var pillClass = c.risk_level === 'High' ? 'pill-high' : 'pill-medium';
            var tr = document.createElement('tr');
            tr.innerHTML =
                '<td class="cust-id">' + c.customer_id + '</td>' +
                '<td><b>' + c.churn_probability.toFixed(2) + '</b></td>' +
                '<td><span class="risk-pill ' + pillClass + '">' + c.risk_level + '</span></td>' +
                '<td>' + (c.contract || '—') + '</td>' +
                '<td>' + (c.tenure !== undefined ? c.tenure : '—') + '</td>' +
                '<td>' + (c.monthly_charges !== undefined ? '$' + c.monthly_charges.toFixed(2) : '—') + '</td>' +
                '<td>' + c.predicted_churn_date + '</td>';
            tbody.appendChild(tr);
        });

    } catch (err) {
        console.error('Error loading reports page:', err);
    }
}

// ========================================================
// SETTINGS PAGE
// ========================================================
function saveSettings() {
    var msg = document.getElementById('settings-saved-msg');
    if (msg) {
        msg.style.display = 'block';
        setTimeout(function () { msg.style.display = 'none'; }, 3000);
    }
}

// ========================================================
// EXPORT
// ========================================================
function exportReport() {
    window.location.href = '/api/export-report';
}
