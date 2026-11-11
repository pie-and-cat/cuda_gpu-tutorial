  // Apply the saved theme (or the system preference) before the page paints, so there is no light flash.
  try {
    const t = localStorage.getItem("threadlab-theme");
    document.documentElement.dataset.theme = t === "dark" || t === "light" ? t : (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  } catch (e) { /* storage unavailable: stay light */ }
