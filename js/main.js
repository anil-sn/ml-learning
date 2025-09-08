document.addEventListener('DOMContentLoaded', () => {

    // --- Feature: Dark Mode Toggle ---
    const themeToggle = document.getElementById('theme-toggle');
    const htmlElement = document.documentElement;

    // Function to apply the saved theme on page load
    const applySavedTheme = () => {
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'dark') {
            htmlElement.classList.add('dark-mode');
            if (themeToggle) themeToggle.checked = true;
        } else {
            htmlElement.classList.remove('dark-mode');
            if (themeToggle) themeToggle.checked = false;
        }
    };

    // Event listener for the toggle switch
    if (themeToggle) {
        themeToggle.addEventListener('change', () => {
            if (themeToggle.checked) {
                htmlElement.classList.add('dark-mode');
                localStorage.setItem('theme', 'dark');
            } else {
                htmlElement.classList.remove('dark-mode');
                localStorage.setItem('theme', 'light');
            }
        });
    }

    // --- Feature: Back to Top Button ---
    const backToTopBtn = document.getElementById('backToTopBtn');

    // Show or hide the button based on scroll position
    window.addEventListener('scroll', () => {
        if (window.scrollY > 300) {
            if (backToTopBtn) backToTopBtn.style.display = 'block';
        } else {
            if (backToTopBtn) backToTopBtn.style.display = 'none';
        }
    });

    // Scroll to the top when the button is clicked
    if (backToTopBtn) {
        backToTopBtn.addEventListener('click', () => {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }

    // --- Feature: Video Resources Interaction ---
    const videoCards = document.querySelectorAll('.video-card');
    
    // Function to handle video opening
    const openVideo = (videoUrl) => {
        if (!videoUrl) {
            console.warn('No video URL provided');
            return;
        }

        // Clean URL and ensure it's properly formatted
        let cleanUrl = videoUrl.trim();
        
        // Convert youtu.be URLs to youtube.com/watch format for better compatibility
        if (cleanUrl.includes('youtu.be/')) {
            const videoId = cleanUrl.split('youtu.be/')[1].split('?')[0];
            cleanUrl = `https://www.youtube.com/watch?v=${videoId}`;
        }
        
        console.log('Opening video:', cleanUrl);
        
        // Simple and reliable approach - just open in new tab
        // This avoids the double-opening issue caused by multiple fallbacks
        window.open(cleanUrl, '_blank', 'noopener,noreferrer');
    };
    
    videoCards.forEach(card => {
        // Click handler
        card.addEventListener('click', (e) => {
            e.preventDefault();
            const videoUrl = card.getAttribute('data-video-url');
            openVideo(videoUrl);
        });

        // Keyboard navigation support
        card.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                const videoUrl = card.getAttribute('data-video-url');
                openVideo(videoUrl);
            }
        });

        // Make video cards focusable for accessibility
        if (!card.hasAttribute('tabindex')) {
            card.setAttribute('tabindex', '0');
        }

        // Add aria-label for accessibility
        const videoTitle = card.querySelector('.video-title')?.textContent || 'Video';
        card.setAttribute('aria-label', `Play video: ${videoTitle}`);
        card.setAttribute('role', 'button');
    });

    // Enhanced hover effects for video cards
    videoCards.forEach(card => {
        const playOverlay = card.querySelector('.video-play-overlay');
        
        card.addEventListener('mouseenter', () => {
            card.style.cursor = 'pointer';
            if (playOverlay) {
                playOverlay.style.opacity = '1';
                playOverlay.style.transform = 'translate(-50%, -50%) scale(1.1)';
            }
        });

        card.addEventListener('mouseleave', () => {
            if (playOverlay) {
                playOverlay.style.opacity = '0.9';
                playOverlay.style.transform = 'translate(-50%, -50%) scale(1)';
            }
        });

        // Focus handling for keyboard navigation
        card.addEventListener('focus', () => {
            card.style.outline = '2px solid var(--accent-1)';
            card.style.outlineOffset = '2px';
        });

        card.addEventListener('blur', () => {
            card.style.outline = 'none';
        });
    });
    
    // Apply theme on initial load
    applySavedTheme();
});