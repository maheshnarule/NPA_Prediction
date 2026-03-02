// NPA Predictor - JavaScript Functions

document.addEventListener('DOMContentLoaded', function () {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Form validation
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function (event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });

    // Auto-format numbers
    const numberInputs = document.querySelectorAll('input[type="number"]');
    numberInputs.forEach(input => {
        input.addEventListener('blur', function () {
            if (this.value) {
                // Format with commas for large numbers
                if (this.name.includes('Income') || this.name.includes('LoanAmount')) {
                    const value = parseFloat(this.value.replace(/,/g, ''));
                    if (!isNaN(value)) {
                        this.value = value.toLocaleString('en-US');
                    }
                }
            }
        });

        input.addEventListener('focus', function () {
            // Remove commas for editing
            this.value = this.value.replace(/,/g, '');
        });
    });

    // Risk score animation
    const riskScoreElement = document.querySelector('.risk-score-circle');
    if (riskScoreElement) {
        const score = parseFloat(riskScoreElement.querySelector('.score-number').textContent);
        animateRiskScore(score);
    }

    // Copy API endpoint
    const copyApiBtn = document.getElementById('copyApiBtn');
    if (copyApiBtn) {
        copyApiBtn.addEventListener('click', function () {
            const apiUrl = window.location.origin + '/api/predict';
            navigator.clipboard.writeText(apiUrl).then(() => {
                const originalText = this.innerHTML;
                this.innerHTML = '<i class="fas fa-check"></i> Copied!';
                setTimeout(() => {
                    this.innerHTML = originalText;
                }, 2000);
            });
        });
    }

    // Auto-calculate DTI ratio
    const incomeInput = document.querySelector('input[name="Income"]');
    const dtiInput = document.querySelector('input[name="DTIRatio"]');
    if (incomeInput && dtiInput) {
        incomeInput.addEventListener('input', calculateSuggestedDTI);
    }

    // Chart theme toggle
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', function () {
            document.body.classList.toggle('dark-mode');
            this.innerHTML = document.body.classList.contains('dark-mode')
                ? '<i class="fas fa-sun"></i> Light Mode'
                : '<i class="fas fa-moon"></i> Dark Mode';
        });
    }
});

function animateRiskScore(targetScore) {
    const scoreElement = document.querySelector('.score-number');
    if (!scoreElement) return;

    const duration = 2000; // 2 seconds
    const steps = 60;
    const increment = targetScore / steps;
    let current = 0;
    let step = 0;

    const timer = setInterval(() => {
        step++;
        current += increment;
        if (step >= steps) {
            current = targetScore;
            clearInterval(timer);
        }
        scoreElement.textContent = Math.round(current) + '%';
    }, duration / steps);
}

function calculateSuggestedDTI() {
    const income = parseFloat(document.querySelector('input[name="Income"]').value);
    const dtiInput = document.querySelector('input[name="DTIRatio"]');

    if (income && dtiInput && !dtiInput.value) {
        // Suggest a DTI based on income (lower income = lower suggested DTI)
        let suggestedDTI;
        if (income < 30000) {
            suggestedDTI = 0.3;
        } else if (income < 60000) {
            suggestedDTI = 0.35;
        } else if (income < 100000) {
            suggestedDTI = 0.4;
        } else {
            suggestedDTI = 0.45;
        }

        dtiInput.placeholder = `Suggested: ${suggestedDTI}`;
    }
}

// Export data function
function exportData(format = 'csv') {
    const data = {
        timestamp: new Date().toISOString(),
        page: window.location.pathname,
        data: {}
    };

    if (format === 'json') {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        downloadBlob(blob, `npa-data-${Date.now()}.json`);
    } else {
        // CSV export logic
        const csvContent = "data:text/csv;charset=utf-8,";
        const blob = new Blob([csvContent], { type: 'text/csv' });
        downloadBlob(blob, `npa-data-${Date.now()}.csv`);
    }
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Keyboard shortcuts
document.addEventListener('keydown', function (e) {
    // Ctrl/Cmd + P to print
    if ((e.ctrlKey || e.metaKey) && e.key === 'p') {
        e.preventDefault();
        window.print();
    }

    // Ctrl/Cmd + N for new prediction
    if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
        e.preventDefault();
        window.location.href = '/predict';
    }

    // Escape to go home
    if (e.key === 'Escape') {
        window.location.href = '/';
    }
});

// Form auto-save
function setupAutoSave() {
    const form = document.getElementById('predictionForm');
    if (!form) return;

    const inputs = form.querySelectorAll('input, select, textarea');
    const saveKey = 'npa-form-autosave';

    // Load saved data
    const savedData = localStorage.getItem(saveKey);
    if (savedData) {
        try {
            const data = JSON.parse(savedData);
            inputs.forEach(input => {
                if (data[input.name] !== undefined) {
                    input.value = data[input.name];
                }
            });

            // Show restore notification
            showNotification('Form data restored from last session', 'info');
        } catch (e) {
            console.error('Failed to restore form data:', e);
        }
    }

    // Auto-save on input
    inputs.forEach(input => {
        input.addEventListener('input', saveFormData);
    });

    function saveFormData() {
        const formData = {};
        inputs.forEach(input => {
            if (input.name) {
                formData[input.name] = input.value;
            }
        });
        localStorage.setItem(saveKey, JSON.stringify(formData));
    }

    // Clear saved data on submit
    form.addEventListener('submit', function () {
        localStorage.removeItem(saveKey);
    });
}

function showNotification(message, type = 'info') {
    const alertClass = {
        'info': 'alert-info',
        'success': 'alert-success',
        'warning': 'alert-warning',
        'error': 'alert-danger'
    }[type];

    const notification = document.createElement('div');
    notification.className = `alert ${alertClass} alert-dismissible fade show position-fixed`;
    notification.style.cssText = `
        top: 20px;
        right: 20px;
        z-index: 9999;
        min-width: 300px;
        max-width: 400px;
    `;
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    document.body.appendChild(notification);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

// Initialize when page loads
window.onload = function () {
    setupAutoSave();

    // Add loading spinner for form submissions
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function () {
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn) {
                const originalText = submitBtn.innerHTML;
                submitBtn.innerHTML = `
                    <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
                    Processing...
                `;
                submitBtn.disabled = true;

                // Restore button after 10 seconds (fallback)
                setTimeout(() => {
                    submitBtn.innerHTML = originalText;
                    submitBtn.disabled = false;
                }, 10000);
            }
        });
    });
};