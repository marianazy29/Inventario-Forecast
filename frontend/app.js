const API_URL = location.protocol === "file:" ? "http://127.0.0.1:8000" : "";

// Escape text before inserting it into HTML
const escapeHtml = (text) => String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// Readable error message from a FastAPI error response
const errorMessage = (data, fallback) =>
    typeof data.detail === "string" ? data.detail : (data.detail?.[0]?.msg || fallback);

// --- TAB NAVIGATION ---
document.querySelectorAll(".nav-link").forEach((button) => {
    button.addEventListener("click", () => {
        document.querySelectorAll(".nav-link").forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".content-section").forEach((s) => s.classList.remove("active"));
        button.classList.add("active");
        document.getElementById(button.getAttribute("data-target")).classList.add("active");
    });
});

// --- API STATUS ---
async function loadStatus() {
    const statusBox = document.getElementById("api-status");
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        statusBox.textContent = `● Sistema activo · ${data.products} productos · datos hasta ${data.last_date_with_data}`;
        statusBox.classList.remove("offline");
    } catch {
        statusBox.textContent = "● Sin conexión con el servidor (ejecuta run.py)";
        statusBox.classList.add("offline");
    }
}

// --- MODEL METRICS (artifacts/metrics.json) ---
async function loadMetrics() {
    try {
        const response = await fetch(`${API_URL}/metrics`);
        if (!response.ok) return;
        const m = await response.json();
        const setText = (id, value) => (document.getElementById(id).textContent = value);
        setText("eval-period", `Evaluado sobre ${m.test_rows.toLocaleString("es")} registros (${m.test_period}).`);
        setText("m-base-mae", m.baseline.mae.toFixed(2));
        setText("m-base-mase", m.baseline.mase.toFixed(4));
        setText("m-ridge-mae", m.ridge.mae.toFixed(2));
        setText("m-ridge-mase", m.ridge.mase.toFixed(4));
        setText("m-xgb-mae", m.xgboost.mae.toFixed(2));
        setText("m-xgb-mase", m.xgboost.mase.toFixed(4));
        // Error reduction in units compared with the baseline
        const improvement = (1 - m.xgboost.relative_mae) * 100;
        setText("m-xgb-gain", `(${improvement.toFixed(1)}% menos error que la línea base)`);
        // How often the real sales fell inside the predicted range in the final test
        if (m.weekly !== undefined) {
            setText("weekly-results", `Pronóstico semanal (total de 7 días por producto): el modelo se equivoca en promedio un ${m.weekly.xgboost_error_pct} % (la línea base, un ${m.weekly.baseline_error_pct} %). El rango semanal acertó en el ${(m.weekly.interval_coverage * 100).toFixed(1)} % de las semanas.`);
        }
        if (m.interval_coverage !== undefined) {
            setText("interval-coverage", `Rango diario: en la prueba final, la venta real de un día cayó dentro del rango indicado en el ${(m.interval_coverage * 100).toFixed(1)} % de los casos (objetivo: ${(m.interval_target * 100).toFixed(0)} %).`);
        }
    } catch { /* keep the "–" placeholders */ }
}

// --- PRODUCT SEARCH BY NAME (autocomplete) ---
const searchInput = document.getElementById("product-search");
const productIdInput = document.getElementById("product-id");
const suggestionList = document.getElementById("product-suggestions");
const selectionHint = document.getElementById("product-selected");
let searchTimer = null;
let suggestions = [];
let activeIndex = -1;

function closeSuggestions() {
    suggestionList.classList.add("hidden");
    suggestionList.innerHTML = "";
    activeIndex = -1;
}

function selectProduct(product) {
    searchInput.value = product.name;
    productIdInput.value = product.product_id;
    selectionHint.textContent = `${product.name} · ${product.category} · código ${product.product_id}`;
    selectionHint.classList.add("selected");
    closeSuggestions();
}

function renderSuggestions() {
    if (!suggestions.length) {
        suggestionList.innerHTML = '<li class="empty">Sin coincidencias</li>';
    } else {
        suggestionList.innerHTML = suggestions
            .map((p, i) => `<li role="option" data-i="${i}" class="${i === activeIndex ? "active" : ""}">${escapeHtml(p.name)}<small>${escapeHtml(p.category)} · ${escapeHtml(p.product_id)}</small></li>`)
            .join("");
    }
    suggestionList.classList.remove("hidden");
}

searchInput.addEventListener("input", () => {
    productIdInput.value = ""; 
    selectionHint.classList.remove("selected");
    selectionHint.textContent = "Escribe al menos 2 letras y elige de la lista.";
    clearTimeout(searchTimer);
    const query = searchInput.value.trim();
    if (query.length < 2) return closeSuggestions();
    searchTimer = setTimeout(async () => {
        try {
            const response = await fetch(`${API_URL}/products?q=${encodeURIComponent(query)}&limit=8`);
            suggestions = (await response.json()).results;
            activeIndex = -1;
            renderSuggestions();
        } catch { closeSuggestions(); }
    }, 200);
});

searchInput.addEventListener("keydown", (e) => {
    if (suggestionList.classList.contains("hidden") || !suggestions.length) return;
    if (e.key === "ArrowDown") { e.preventDefault(); activeIndex = (activeIndex + 1) % suggestions.length; renderSuggestions(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); activeIndex = (activeIndex - 1 + suggestions.length) % suggestions.length; renderSuggestions(); }
    else if (e.key === "Enter" && activeIndex >= 0) { e.preventDefault(); selectProduct(suggestions[activeIndex]); }
    else if (e.key === "Escape") closeSuggestions();
});

suggestionList.addEventListener("mousedown", (e) => { // mousedown fires before the input loses focus
    const item = e.target.closest("li[data-i]");
    if (item) selectProduct(suggestions[Number(item.dataset.i)]);
});
document.addEventListener("click", (e) => { if (!e.target.closest(".autocomplete")) closeSuggestions(); });

// --- HANDLER 1: WEEKLY FORECAST ---
const DAY_NAMES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"];

// "2026-10-01" -> "01/10" (built from the text, so the time zone cannot shift the day)
const shortDate = (isoDate) => `${isoDate.slice(8, 10)}/${isoDate.slice(5, 7)}`;

const LAG_SOURCE_TEXT = {
    history: () => "ventas reales de las semanas anteriores",
    recursive_forecast: (data) => `pronóstico encadenado desde el último dato real (${data.days_ahead} días antes de la fecha elegida)`,
};

function renderWeek(data) {
    const p = data.product;
    const hasActual = data.days.some((d) => d.actual !== undefined);
    const rows = data.days.map((d) => `
        <tr>
            <td>${DAY_NAMES[d.day_of_week]} ${shortDate(d.date)}</td>
            <td class="num"><strong>${d.predicted}</strong></td>
            ${hasActual ? `<td class="num">${d.actual !== undefined ? d.actual : "–"}</td>` : ""}
        </tr>`).join("");
    const source = (LAG_SOURCE_TEXT[data.lag_source] || (() => data.lag_source))(data);
    return `
        <strong>${escapeHtml(p.name)}</strong> <small>(${escapeHtml(p.category)} · ${escapeHtml(p.product_id)})</small>
        <table class="week-table">
            <thead>
                <tr><th>Día</th><th class="num">Pronóstico</th>${hasActual ? '<th class="num">Venta real</th>' : ""}</tr>
            </thead>
            <tbody>${rows}</tbody>
            <tfoot>
                <tr>
                    <td>Total de la semana</td>
                    <td class="num">${data.week_total}</td>
                    ${hasActual ? `<td class="num">${data.actual_total !== undefined ? data.actual_total : "–"}</td>` : ""}
                </tr>
            </tfoot>
        </table>
        <small>Los valores de cada día son orientativos: un día puntual varía mucho por azar. El total semanal es mucho más confiable.
        Base del cálculo: ${escapeHtml(source)}.</small>
        ${data.warning ? `<br><small>⚠ ${escapeHtml(data.warning)}</small>` : ""}`;
}

document.getElementById("prediction-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const resultBox = document.getElementById("prediction-result");
    resultBox.classList.remove("hidden");

    // If the user typed a name but did not pick from the list, the backend tries to resolve it
    const payload = { date: document.getElementById("prediction-date").value };
    if (productIdInput.value) payload.product_id = productIdInput.value;
    else payload.name = searchInput.value.trim();

    resultBox.innerHTML = "Consultando el modelo...";
    try {
        const response = await fetch(`${API_URL}/forecast-week`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await response.json();
        resultBox.innerHTML = response.ok
            ? renderWeek(data)
            : `<strong>No se pudo calcular:</strong> ${escapeHtml(errorMessage(data, "Error interno."))}`;
    } catch {
        resultBox.innerHTML = "<strong>Error de red:</strong> no se pudo conectar con el servidor. Asegúrate de ejecutar run.py.";
    }
});

// --- HANDLER 2: RECOMMENDATIONS ---
document.getElementById("btn-recommend").addEventListener("click", async () => {
    const container = document.getElementById("recommendations-container");
    const date = document.getElementById("recommendation-date").value;
    container.innerHTML = "<div class='empty-state-pastel'><p>Calculando la demanda de todos los productos...</p></div>";

    try {
        const response = await fetch(`${API_URL}/recommend`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ date, top: 3 }),
        });
        const data = await response.json();

        if (response.ok && data.top_recommendations.length > 0) {
            container.innerHTML = "";
            data.top_recommendations.forEach((item, index) => {
                const highDemand = item.change_vs_recent_pct > 25;
                const sign = item.change_vs_recent_pct >= 0 ? "+" : "";
                const card = document.createElement("div");
                card.className = `rec-card-pastel top-${index + 1}`;
                card.innerHTML = `
                    <div class="badge-rec ${highDemand ? "bg-light-rose" : "bg-light-blue"}">${highDemand ? "Alta demanda" : "Buen margen"}</div>
                    <h3>${escapeHtml(item.name)}</h3>
                    <p class="rec-meta">Código: ${escapeHtml(item.product_id)} · Categoría: ${escapeHtml(item.category)}</p>
                    <div class="rec-metrics">
                        <p>Demanda esperada: <strong>${item.expected_demand} u.</strong></p>
                        <p>Vs. últimas 4 semanas: <strong style="color:${item.change_vs_recent_pct >= 0 ? "#0ca678" : "#d6336c"};">${sign}${item.change_vs_recent_pct}%</strong></p>
                        <p>Ganancia estimada: <strong style="color:#4c6ef5;">${item.estimated_profit_bs} Bs.</strong></p>
                    </div>`;
                container.appendChild(card);
            });
            if (data.warning) {
                const note = document.createElement("p");
                note.className = "hint";
                note.textContent = `${data.warning}`;
                container.appendChild(note);
            }
        } else {
            container.innerHTML = `<div class='empty-state-pastel'><p style='color: var(--pastel-rose-text);'>${escapeHtml(errorMessage(data, "No se encontraron resultados."))}</p></div>`;
        }
    } catch {
        container.innerHTML = "<div class='empty-state-pastel'><p style='color: var(--pastel-rose-text);'>🔌 Error de comunicación con el servidor.</p></div>";
    }
});

loadStatus();
loadMetrics();