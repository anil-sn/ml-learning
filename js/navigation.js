/**
 * ML Learning Portfolio - Navigation and Interactive Features
 * Industry-standard JavaScript following best practices
 * 
 * Features:
 * - Smooth navigation handling
 * - Active page highlighting
 * - Responsive menu behavior
 * - Accessibility enhancements
 * - Performance optimizations
 */

(function() {
    'use strict';

    // ==========================================================================
    // Configuration and Constants
    // ==========================================================================
    
    const CONFIG = {
        activeClass: 'active',
        navbarSelector: '.navbar',
        linkSelector: '.navbar a',
        sidebarSelector: '.sidebar',
        transitionDuration: 300,
        debounceDelay: 100
    };

    // ==========================================================================
    // Utility Functions
    // ==========================================================================
    
    /**
     * Debounce function to limit the rate of function execution
     * @param {Function} func - Function to debounce
     * @param {number} wait - Wait time in milliseconds
     * @returns {Function} Debounced function
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
     * Get current page name from URL
     * @returns {string} Current page name
     */
    function getCurrentPage() {
        const path = window.location.pathname;
        const segments = path.split('/');
        const filename = segments[segments.length - 1];
        
        // Handle index.html or empty path
        if (filename === 'index.html' || filename === '' || filename === 'ml-learning') {
            return 'index.html';
        }
        
        return filename;
    }

    /**
     * Check if element is in viewport
     * @param {Element} element - Element to check
     * @returns {boolean} True if element is in viewport
     */
    function isInViewport(element) {
        const rect = element.getBoundingClientRect();
        return (
            rect.top >= 0 &&
            rect.left >= 0 &&
            rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
            rect.right <= (window.innerWidth || document.documentElement.clientWidth)
        );
    }

    // ==========================================================================
    // Navigation Management
    // ==========================================================================
    
    /**
     * Initialize navigation functionality
     */
    function initNavigation() {
        const navbar = document.querySelector(CONFIG.navbarSelector);
        if (!navbar) {
            console.warn('Navigation bar not found');
            return;
        }

        // Set active page
        setActivePage();
        
        // Add smooth scrolling to anchor links
        addSmoothScrolling();
        
        // Add keyboard navigation support
        addKeyboardNavigation();
        
        // Add mobile menu enhancements
        addMobileEnhancements();
        
        // Add sidebar enhancements
        addSidebarEnhancements();
        
        console.log('Navigation initialized successfully');
    }

    /**
     * Set active page in navigation
     */
    function setActivePage() {
        const currentPage = getCurrentPage();
        const navLinks = document.querySelectorAll(CONFIG.linkSelector);
        
        navLinks.forEach(link => {
            const href = link.getAttribute('href');
            const linkPage = href.split('/').pop();
            
            // Remove existing active class
            link.classList.remove(CONFIG.activeClass);
            
            // Add active class to current page
            if (linkPage === currentPage || 
                (currentPage === 'index.html' && (linkPage === 'index.html' || linkPage === '../index.html'))) {
                link.classList.add(CONFIG.activeClass);
            }
        });
    }

    /**
     * Add smooth scrolling behavior to anchor links
     */
    function addSmoothScrolling() {
        const anchorLinks = document.querySelectorAll('a[href^="#"]');
        
        anchorLinks.forEach(link => {
            link.addEventListener('click', function(e) {
                const targetId = this.getAttribute('href').substring(1);
                const targetElement = document.getElementById(targetId);
                
                if (targetElement) {
                    e.preventDefault();
                    
                    targetElement.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                    
                    // Update URL without jumping
                    history.pushState(null, null, `#${targetId}`);
                }
            });
        });
    }

    /**
     * Add keyboard navigation support
     */
    function addKeyboardNavigation() {
        const navLinks = document.querySelectorAll(CONFIG.linkSelector);
        
        navLinks.forEach((link, index) => {
            link.addEventListener('keydown', function(e) {
                let targetIndex;
                
                switch(e.key) {
                    case 'ArrowRight':
                    case 'ArrowDown':
                        e.preventDefault();
                        targetIndex = (index + 1) % navLinks.length;
                        navLinks[targetIndex].focus();
                        break;
                        
                    case 'ArrowLeft':
                    case 'ArrowUp':
                        e.preventDefault();
                        targetIndex = (index - 1 + navLinks.length) % navLinks.length;
                        navLinks[targetIndex].focus();
                        break;
                        
                    case 'Home':
                        e.preventDefault();
                        navLinks[0].focus();
                        break;
                        
                    case 'End':
                        e.preventDefault();
                        navLinks[navLinks.length - 1].focus();
                        break;
                }
            });
        });
    }

    /**
     * Add mobile-specific enhancements
     */
    function addMobileEnhancements() {
        const navbar = document.querySelector(CONFIG.navbarSelector);
        if (!navbar) return;

        // Add touch-friendly hover effects
        if ('ontouchstart' in window) {
            navbar.classList.add('touch-device');
        }

        // Handle window resize
        const handleResize = debounce(() => {
            // Recalculate navigation layout if needed
            setActivePage();
        }, CONFIG.debounceDelay);

        window.addEventListener('resize', handleResize);
    }

    /**
     * Add sidebar-specific enhancements
     */
    function addSidebarEnhancements() {
        const sidebar = document.querySelector(CONFIG.sidebarSelector);
        if (!sidebar) return;

        // Add smooth scroll behavior for sidebar links
        const sidebarLinks = sidebar.querySelectorAll(CONFIG.linkSelector);
        sidebarLinks.forEach(link => {
            link.addEventListener('click', function() {
                // Add a subtle animation to the sidebar
                sidebar.style.transform = 'translateX(-2px)';
                setTimeout(() => {
                    sidebar.style.transform = 'translateX(0)';
                }, 150);
            });
        });

        // Add hover effect to sidebar
        sidebar.addEventListener('mouseenter', function() {
            this.style.boxShadow = '0 8px 30px rgba(74, 108, 247, 0.12)';
        });

        sidebar.addEventListener('mouseleave', function() {
            this.style.boxShadow = 'var(--shadow-md)';
        });

        // Add focus management for sidebar
        const firstLink = sidebarLinks[0];
        const lastLink = sidebarLinks[sidebarLinks.length - 1];

        if (firstLink && lastLink) {
            firstLink.addEventListener('keydown', function(e) {
                if (e.key === 'Tab' && e.shiftKey) {
                    e.preventDefault();
                    lastLink.focus();
                }
            });

            lastLink.addEventListener('keydown', function(e) {
                if (e.key === 'Tab' && !e.shiftKey) {
                    e.preventDefault();
                    firstLink.focus();
                }
            });
        }
    }

    // ==========================================================================
    // Interactive Features
    // ==========================================================================
    
    /**
     * Initialize interactive features
     */
    function initInteractiveFeatures() {
        // Add hover effects to cards
        addCardHoverEffects();
        
        // Add loading states to external links
        addExternalLinkHandling();
        
        // Add scroll-based animations
        addScrollAnimations();
        
        console.log('Interactive features initialized');
    }

    /**
     * Add enhanced hover effects to cards
     */
    function addCardHoverEffects() {
        const cards = document.querySelectorAll('.card, .concept-card, .resource-card');
        
        cards.forEach(card => {
            card.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-5px)';
            });
            
            card.addEventListener('mouseleave', function() {
                this.style.transform = 'translateY(0)';
            });
        });
    }

    /**
     * Add handling for external links
     */
    function addExternalLinkHandling() {
        const externalLinks = document.querySelectorAll('a[href^="http"]');
        
        externalLinks.forEach(link => {
            // Add external link indicator
            if (!link.querySelector('.external-indicator')) {
                const indicator = document.createElement('span');
                indicator.className = 'external-indicator';
                indicator.innerHTML = '↗';
                indicator.setAttribute('aria-label', 'Opens in new tab');
                link.appendChild(indicator);
            }
            
            // Add click tracking (optional)
            link.addEventListener('click', function() {
                console.log(`External link clicked: ${this.href}`);
            });
        });
    }

    /**
     * Add scroll-based animations
     */
    function addScrollAnimations() {
        const animatedElements = document.querySelectorAll('.card, .topic-section');
        
        const handleScroll = debounce(() => {
            animatedElements.forEach(element => {
                if (isInViewport(element)) {
                    element.classList.add('animate-in');
                }
            });
        }, CONFIG.debounceDelay);

        window.addEventListener('scroll', handleScroll);
        
        // Initial check
        handleScroll();
    }

    // ==========================================================================
    // Performance Optimizations
    // ==========================================================================
    
    /**
     * Initialize performance optimizations
     */
    function initPerformanceOptimizations() {
        // Lazy load images
        addLazyLoading();
        
        // Preload critical resources
        preloadCriticalResources();
        
        console.log('Performance optimizations initialized');
    }

    /**
     * Add lazy loading for images
     */
    function addLazyLoading() {
        if ('IntersectionObserver' in window) {
            const imageObserver = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        img.src = img.dataset.src;
                        img.classList.remove('lazy');
                        observer.unobserve(img);
                    }
                });
            });

            const lazyImages = document.querySelectorAll('img[data-src]');
            lazyImages.forEach(img => imageObserver.observe(img));
        }
    }

    /**
     * Preload critical resources
     */
    function preloadCriticalResources() {
        // Preload next likely page
        const currentPage = getCurrentPage();
        const preloadLinks = {
            'index.html': ['html/Projects.html', 'html/Environments.html'],
            'Projects.html': ['../index.html', 'Environments.html'],
            'Environments.html': ['../index.html', 'Projects.html'],
            'prerequisites.html': ['../index.html', 'Projects.html', 'Environments.html']
        };

        const linksToPreload = preloadLinks[currentPage] || [];
        
        linksToPreload.forEach(href => {
            const link = document.createElement('link');
            link.rel = 'prefetch';
            link.href = href;
            document.head.appendChild(link);
        });
    }

    // ==========================================================================
    // Accessibility Enhancements
    // ==========================================================================
    
    /**
     * Initialize accessibility features
     */
    function initAccessibility() {
        // Add skip navigation link
        addSkipNavigation();
        
        // Enhance focus management
        enhanceFocusManagement();
        
        // Add ARIA labels where needed
        addAriaLabels();
        
        console.log('Accessibility features initialized');
    }

    /**
     * Add skip navigation link
     */
    function addSkipNavigation() {
        const skipLink = document.createElement('a');
        skipLink.href = '#main-content';
        skipLink.textContent = 'Skip to main content';
        skipLink.className = 'skip-link';
        skipLink.style.cssText = `
            position: absolute;
            top: -40px;
            left: 6px;
            background: var(--accent-1);
            color: white;
            padding: 8px;
            text-decoration: none;
            border-radius: 4px;
            z-index: 1000;
            transition: top 0.3s;
        `;
        
        skipLink.addEventListener('focus', function() {
            this.style.top = '6px';
        });
        
        skipLink.addEventListener('blur', function() {
            this.style.top = '-40px';
        });
        
        document.body.insertBefore(skipLink, document.body.firstChild);
    }

    /**
     * Enhance focus management
     */
    function enhanceFocusManagement() {
        // Add focus trap for modals (if any)
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Tab') {
                const focusableElements = document.querySelectorAll(
                    'a[href], button, textarea, input[type="text"], input[type="radio"], input[type="checkbox"], select'
                );
                
                const firstElement = focusableElements[0];
                const lastElement = focusableElements[focusableElements.length - 1];
                
                if (e.shiftKey && document.activeElement === firstElement) {
                    e.preventDefault();
                    lastElement.focus();
                } else if (!e.shiftKey && document.activeElement === lastElement) {
                    e.preventDefault();
                    firstElement.focus();
                }
            }
        });
    }

    /**
     * Add ARIA labels where needed
     */
    function addAriaLabels() {
        // Add ARIA labels to navigation
        const navbar = document.querySelector(CONFIG.navbarSelector);
        if (navbar) {
            navbar.setAttribute('role', 'navigation');
            navbar.setAttribute('aria-label', 'Main navigation');
        }
        
        // Add ARIA labels to external links
        const externalLinks = document.querySelectorAll('a[href^="http"]');
        externalLinks.forEach(link => {
            if (!link.getAttribute('aria-label')) {
                link.setAttribute('aria-label', `${link.textContent} (opens in new tab)`);
            }
        });
    }

    // ==========================================================================
    // Error Handling and Logging
    // ==========================================================================
    
    /**
     * Global error handler
     */
    function handleGlobalError(event) {
        console.error('Global error:', event.error);
        
        // In production, you might want to send this to an error tracking service
        // Example: sendToErrorTracking(event.error);
    }

    /**
     * Initialize error handling
     */
    function initErrorHandling() {
        window.addEventListener('error', handleGlobalError);
        window.addEventListener('unhandledrejection', handleGlobalError);
    }

    // ==========================================================================
    // Initialization
    // ==========================================================================
    
    /**
     * Initialize all features when DOM is ready
     */
    function init() {
        try {
            initNavigation();
            initInteractiveFeatures();
            initPerformanceOptimizations();
            initAccessibility();
            initErrorHandling();
            
            console.log('ML Learning Portfolio initialized successfully');
        } catch (error) {
            console.error('Initialization error:', error);
        }
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // Export for testing (if needed)
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = {
            init,
            getCurrentPage,
            setActivePage
        };
    }

})();
