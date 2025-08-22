/**
 * SafetyVision AI - Main JavaScript Functions
 * 탱크로리 안전상태 체크 시스템 메인 스크립트
 */

// 전역 변수
let currentUploadXHR = null;
let analysisPollingInterval = null;

// DOM 로드 완료 시 초기화
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

/**
 * 애플리케이션 초기화
 */
function initializeApp() {
    // Bootstrap 툴팁 초기화
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // 애니메이션 효과 추가
    addScrollAnimations();
    
    // 폼 검증 초기화
    initializeFormValidation();
    
    // 키보드 단축키 설정
    setupKeyboardShortcuts();
}

/**
 * 스크롤 애니메이션 추가
 */
function addScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-up');
            }
        });
    }, observerOptions);
    
    // 애니메이션을 적용할 요소들 관찰
    const animatedElements = document.querySelectorAll('.card, .metric-card, .detection-item');
    animatedElements.forEach(el => observer.observe(el));
}

/**
 * 폼 검증 초기화
 */
function initializeFormValidation() {
    const forms = document.querySelectorAll('.needs-validation');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });
}

/**
 * 키보드 단축키 설정
 */
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function(event) {
        // Ctrl + U: 업로드 페이지로 이동
        if (event.ctrlKey && event.key === 'u') {
            event.preventDefault();
            window.location.href = '/upload';
        }
        
        // Ctrl + H: 홈페이지로 이동
        if (event.ctrlKey && event.key === 'h') {
            event.preventDefault();
            window.location.href = '/';
        }
        
        // ESC: 모달 닫기
        if (event.key === 'Escape') {
            const openModals = document.querySelectorAll('.modal.show');
            openModals.forEach(modal => {
                bootstrap.Modal.getInstance(modal)?.hide();
            });
        }
    });
}

/**
 * 파일 크기 포맷팅
 * @param {number} bytes - 바이트 단위 크기
 * @returns {string} 포맷된 크기 문자열
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * 날짜 시간 포맷팅
 * @param {string} isoString - ISO 형식 날짜 문자열
 * @returns {string} 포맷된 날짜 시간
 */
function formatDateTime(isoString) {
    if (!isoString) return '';
    
    const date = new Date(isoString);
    const options = {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    };
    
    return date.toLocaleDateString('ko-KR', options);
}

/**
 * 성공 토스트 메시지 표시
 * @param {string} message - 표시할 메시지
 */
function showSuccessToast(message) {
    showToast(message, 'success');
}

/**
 * 오류 토스트 메시지 표시
 * @param {string} message - 표시할 메시지
 */
function showErrorToast(message) {
    showToast(message, 'error');
}

/**
 * 정보 토스트 메시지 표시
 * @param {string} message - 표시할 메시지
 */
function showInfoToast(message) {
    showToast(message, 'info');
}

/**
 * 토스트 메시지 표시
 * @param {string} message - 표시할 메시지
 * @param {string} type - 토스트 타입 (success, error, info, warning)
 */
function showToast(message, type = 'info') {
    // 토스트 컨테이너가 없으면 생성
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        toastContainer.style.zIndex = '9999';
        document.body.appendChild(toastContainer);
    }
    
    // 토스트 색상 설정
    const colorClass = {
        'success': 'bg-success',
        'error': 'bg-danger',
        'info': 'bg-info',
        'warning': 'bg-warning'
    }[type] || 'bg-info';
    
    // 토스트 HTML 생성
    const toastId = 'toast-' + Date.now();
    const toastHtml = `
        <div id="${toastId}" class="toast ${colorClass} text-white" role="alert">
            <div class="toast-header ${colorClass} text-white border-0">
                <i class="fas fa-info-circle me-2"></i>
                <strong class="me-auto">SafetyVision AI</strong>
                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        </div>
    `;
    
    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    
    // 토스트 표시
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, {
        autohide: true,
        delay: 5000
    });
    
    toast.show();
    
    // 토스트가 숨겨진 후 제거
    toastElement.addEventListener('hidden.bs.toast', function() {
        toastElement.remove();
    });
}

/**
 * 로딩 스피너 표시
 * @param {HTMLElement} element - 스피너를 표시할 요소
 * @param {string} text - 로딩 텍스트
 */
function showLoadingSpinner(element, text = '로딩 중...') {
    const spinnerHtml = `
        <div class="d-flex justify-content-center align-items-center p-4">
            <div class="spinner-border text-primary me-3" role="status">
                <span class="visually-hidden">로딩 중...</span>
            </div>
            <span>${text}</span>
        </div>
    `;
    
    element.innerHTML = spinnerHtml;
}

/**
 * 확인 다이얼로그 표시
 * @param {string} message - 확인 메시지
 * @param {string} title - 다이얼로그 제목
 * @returns {Promise<boolean>} 사용자 선택 결과
 */
function showConfirmDialog(message, title = '확인') {
    return new Promise((resolve) => {
        // 모달 HTML 생성
        const modalId = 'confirm-modal-' + Date.now();
        const modalHtml = `
            <div class="modal fade" id="${modalId}" tabindex="-1">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">${title}</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <p>${message}</p>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">취소</button>
                            <button type="button" class="btn btn-primary" id="confirm-btn">확인</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        
        const modalElement = document.getElementById(modalId);
        const modal = new bootstrap.Modal(modalElement);
        
        // 확인 버튼 클릭 시
        modalElement.querySelector('#confirm-btn').addEventListener('click', function() {
            modal.hide();
            resolve(true);
        });
        
        // 모달이 숨겨진 후 제거
        modalElement.addEventListener('hidden.bs.modal', function() {
            modalElement.remove();
            resolve(false);
        });
        
        modal.show();
    });
}

/**
 * 파일 다운로드
 * @param {string} url - 다운로드 URL
 * @param {string} filename - 파일명
 */
function downloadFile(url, filename) {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename || '';
    link.style.display = 'none';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

/**
 * 클립보드에 텍스트 복사
 * @param {string} text - 복사할 텍스트
 * @returns {Promise<boolean>} 복사 성공 여부
 */
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showSuccessToast('클립보드에 복사되었습니다.');
        return true;
    } catch (err) {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-9999px';
        
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            document.execCommand('copy');
            showSuccessToast('클립보드에 복사되었습니다.');
            return true;
        } catch (err) {
            showErrorToast('클립보드 복사에 실패했습니다.');
            return false;
        } finally {
            document.body.removeChild(textArea);
        }
    }
}

/**
 * URL 파라미터 가져오기
 * @param {string} name - 파라미터 이름
 * @returns {string|null} 파라미터 값
 */
function getUrlParameter(name) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
}

/**
 * 페이지 스크롤을 맨 위로
 */
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

/**
 * 요소가 뷰포트에 보이는지 확인
 * @param {HTMLElement} element - 확인할 요소
 * @returns {boolean} 보이는지 여부
 */
function isElementInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

/**
 * 디바운스 함수
 * @param {Function} func - 실행할 함수
 * @param {number} wait - 대기 시간 (밀리초)
 * @returns {Function} 디바운스된 함수
 */
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

/**
 * 스로틀 함수
 * @param {Function} func - 실행할 함수
 * @param {number} limit - 제한 시간 (밀리초)
 * @returns {Function} 스로틀된 함수
 */
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

/**
 * 페이지 성능 측정
 */
function measurePagePerformance() {
    if ('performance' in window) {
        window.addEventListener('load', function() {
            setTimeout(function() {
                const perfData = performance.getEntriesByType('navigation')[0];
                const loadTime = perfData.loadEventEnd - perfData.loadEventStart;
                
                if (loadTime > 3000) { // 3초 이상인 경우
                    console.warn('페이지 로딩이 느립니다:', loadTime + 'ms');
                }
            }, 0);
        });
    }
}

/**
 * 오류 추적
 */
function setupErrorTracking() {
    window.addEventListener('error', function(event) {
        console.error('JavaScript 오류:', {
            message: event.message,
            source: event.filename,
            line: event.lineno,
            column: event.colno,
            stack: event.error?.stack
        });
    });
    
    window.addEventListener('unhandledrejection', function(event) {
        console.error('처리되지 않은 Promise 거부:', event.reason);
    });
}

// 성능 측정 및 오류 추적 초기화
measurePagePerformance();
setupErrorTracking();

// 전역 함수들을 window 객체에 추가 (다른 스크립트에서 사용 가능)
window.SafetyVisionAI = {
    formatFileSize,
    formatDateTime,
    showSuccessToast,
    showErrorToast,
    showInfoToast,
    showToast,
    showLoadingSpinner,
    showConfirmDialog,
    downloadFile,
    copyToClipboard,
    getUrlParameter,
    scrollToTop,
    isElementInViewport,
    debounce,
    throttle
};