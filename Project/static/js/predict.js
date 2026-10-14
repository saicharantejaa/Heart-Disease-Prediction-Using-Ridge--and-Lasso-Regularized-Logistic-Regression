document.addEventListener("DOMContentLoaded", () => { const form = document.querySelector("#patient-form"); if (form) form.addEventListener("submit", () => form.classList.add("is-loading")); });
