/**
 * router.js – Hash-based client-side router for CustomerIQ
 * Switches visible page sections and highlights the active sidebar item.
 * Loaded as a plain <script> (not module) so all functions are global.
 */

const PAGE_INIT = {
    dashboard: null,
    prediction: null,
    customers: function () { loadAllCustomers(0); },
    analytics: function () { initAnalyticsPage(); },
    reports: function () { initReportsPage(); },
    settings: null
};

function hideAllSections() {
    document.querySelectorAll('.page-section').forEach(function (sec) {
        sec.classList.remove('active');
    });
}

function showSection(id) {
    hideAllSections();
    var el = document.getElementById(id);
    if (el) el.classList.add('active');

    // Highlight active sidebar nav item
    document.querySelectorAll('.nav-item').forEach(function (item) {
        item.classList.remove('active');
    });
    var navItem = document.getElementById('nav-' + id);
    if (navItem) navItem.classList.add('active');
}

function router() {
    var hash = window.location.hash.replace('#', '') || 'dashboard';
    var validPages = Object.keys(PAGE_INIT);
    if (validPages.indexOf(hash) === -1) hash = 'dashboard';

    showSection(hash);

    // Run page-specific initializer if it exists and hasn't been run yet
    if (PAGE_INIT[hash]) {
        PAGE_INIT[hash]();
    }
}

window.addEventListener('hashchange', router);
window.addEventListener('load', router);
