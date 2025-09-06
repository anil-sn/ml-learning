document.addEventListener('DOMContentLoaded', () => {

    // --- Feature: Dark Mode Toggle ---
    const themeToggle = document.getElementById('theme-toggle');
    const body = document.body;

    // Function to apply the saved theme on page load
    const applySavedTheme = () => {
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'dark') {
            body.classList.add('dark-mode');
            if (themeToggle) themeToggle.checked = true;
        } else {
            body.classList.remove('dark-mode');
            if (themeToggle) themeToggle.checked = false;
        }
    };

    // Event listener for the toggle switch
    if (themeToggle) {
        themeToggle.addEventListener('change', () => {
            if (themeToggle.checked) {
                body.classList.add('dark-mode');
                localStorage.setItem('theme', 'dark');
            } else {
                body.classList.remove('dark-mode');
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
    
    // Apply theme on initial load
    applySavedTheme();
});