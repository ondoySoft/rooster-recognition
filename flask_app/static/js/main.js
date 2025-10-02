// Custom JavaScript for Rooster Recognition System

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // File upload drag and drop functionality
    initializeDragAndDrop();
    
    // Form validation
    initializeFormValidation();
    
    // Auto-hide alerts
    autoHideAlerts();
    
    // Smooth scrolling for anchor links
    initializeSmoothScrolling();
});

function initializeDragAndDrop() {
    const fileInput = document.getElementById('file');
    const uploadArea = document.querySelector('.file-upload-area');
    
    if (fileInput && uploadArea) {
        // Create file upload area if it doesn't exist
        if (!uploadArea) {
            const form = fileInput.closest('form');
            const fileUploadDiv = document.createElement('div');
            fileUploadDiv.className = 'file-upload-area mb-3';
            fileUploadDiv.innerHTML = `
                <i class="fas fa-cloud-upload-alt fa-3x text-muted mb-3"></i>
                <p class="mb-2">Drag and drop your rooster image here</p>
                <p class="text-muted">or click to browse</p>
            `;
            fileInput.parentNode.insertBefore(fileUploadDiv, fileInput);
        }
        
        // Drag and drop events
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                fileInput.dispatchEvent(new Event('change'));
            }
        });
        
        uploadArea.addEventListener('click', function() {
            fileInput.click();
        });
    }
}

function initializeFormValidation() {
    const forms = document.querySelectorAll('.needs-validation');
    
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            
            form.classList.add('was-validated');
        });
    });
}

function autoHideAlerts() {
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        if (!alert.classList.contains('alert-danger')) {
            setTimeout(() => {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            }, 5000);
        }
    });
}

function initializeSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                e.preventDefault();
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Image preview functionality
function previewImage(input) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            const previewImg = document.getElementById('previewImg');
            const previewCard = document.getElementById('previewCard');
            
            if (previewImg && previewCard) {
                previewImg.src = e.target.result;
                previewCard.style.display = 'block';
                previewCard.classList.add('fade-in');
            }
        };
        
        reader.readAsDataURL(input.files[0]);
    }
}

// File size validation
function validateFileSize(input) {
    const maxSize = 16 * 1024 * 1024; // 16MB
    const file = input.files[0];
    
    if (file && file.size > maxSize) {
        alert('File size must be less than 16MB');
        input.value = '';
        return false;
    }
    return true;
}

// File type validation
function validateFileType(input) {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
    const file = input.files[0];
    
    if (file && !allowedTypes.includes(file.type)) {
        alert('Please select a valid image file (JPG, PNG, GIF, BMP)');
        input.value = '';
        return false;
    }
    return true;
}

// Progress bar animation
function animateProgressBar(element, targetValue) {
    const progressBar = element.querySelector('.progress-bar');
    if (!progressBar) return;
    
    let currentValue = 0;
    const increment = targetValue / 100;
    
    const timer = setInterval(() => {
        currentValue += increment;
        if (currentValue >= targetValue) {
            currentValue = targetValue;
            clearInterval(timer);
        }
        
        progressBar.style.width = currentValue + '%';
        progressBar.setAttribute('aria-valuenow', currentValue);
        progressBar.textContent = Math.round(currentValue) + '%';
    }, 20);
}

// API functions
async function uploadImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Upload failed');
        }
        
        return await response.json();
    } catch (error) {
        console.error('Error uploading image:', error);
        throw error;
    }
}

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

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

// Search functionality
function initializeSearch() {
    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
        const debouncedSearch = debounce(performSearch, 300);
        searchInput.addEventListener('input', debouncedSearch);
    }
}

function performSearch(query) {
    // Implement search functionality here
    console.log('Searching for:', query);
}

// Export functions for global use
window.RoosterApp = {
    previewImage,
    validateFileSize,
    validateFileType,
    animateProgressBar,
    uploadImage,
    formatFileSize,
    formatDate,
    debounce
};
