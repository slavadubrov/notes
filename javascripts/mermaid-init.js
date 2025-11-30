// Initialize Mermaid on page load for client-side rendering
// Keeps security tight and avoids server-side Puppeteer in restricted envs
(function () {
  function initMermaid() {
    if (window.mermaid && typeof window.mermaid.initialize === "function") {
      try {
        window.mermaid.initialize({
          startOnLoad: true,
          securityLevel: "strict",
          theme: "default",
        });
      } catch (e) {
        // Swallow errors to avoid breaking the page if Mermaid fails
        console.warn("Mermaid initialization failed:", e);
      }
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initMermaid);
  } else {
    initMermaid();
  }
})();
