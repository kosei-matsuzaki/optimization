    document.addEventListener("DOMContentLoaded", function () {
      if (typeof renderMathInElement !== "function") return;
      renderMathInElement(document.body, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false }
        ],
        throwOnError: false,
        errorColor: "#dc2626",
        strict: "ignore"
      });
    });

// Scroll reveal
const revealObs = new IntersectionObserver((entries) => {
  entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); });
}, { threshold: 0.07, rootMargin: '0px 0px -40px 0px' });
document.querySelectorAll('.reveal').forEach(el => revealObs.observe(el));

// Scroll-spy for on-page nav
const sections = ['vso','baselines','differences'];
const navLinks  = document.querySelectorAll('.page-nav-link');
const offset    = 56 + 42 + 60; // header + page-nav + buffer

function updateNav() {
  let current = sections[0];
  sections.forEach(id => {
    const el = document.getElementById(id);
    if (el && el.getBoundingClientRect().top <= offset) current = id;
  });
  navLinks.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + current));
}
window.addEventListener('scroll', updateNav, { passive: true });
updateNav();

// Smooth scroll
navLinks.forEach(a => {
  a.addEventListener('click', e => {
    e.preventDefault();
    const t = document.getElementById(a.getAttribute('href').slice(1));
    if (t) t.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
});
