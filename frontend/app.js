// Si abres index.html con doble clic (file://) apunta al backend local;
// si lo abres desde http://127.0.0.1:8000/ usa el mismo servidor.
const API_URL = location.protocol === "file:" ? "http://127.0.0.1:8000" : "";

const esc = (t) => String(t).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// --- NAVEGACIÓN ENTRE PESTAÑAS ---
document.querySelectorAll(".nav-link").forEach((button) => {
    button.addEventListener("click", () => {
        document.querySelectorAll(".nav-link").forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".content-section").forEach((s) => s.classList.remove("active"));
        button.classList.add("active");
        document.getElementById(button.getAttribute("data-target")).classList.add("active");
    });
});

// --- ESTADO REAL DE LA API ---
async function cargarEstado() {
    const el = document.getElementById("api-status");
    try {
        const r = await fetch(`${API_URL}/health`);
        const d = await r.json();
        el.textContent = `● Sistema activo · ${d.productos} productos · datos hasta ${d.ultima_fecha_con_datos}`;
        el.classList.remove("offline");
    } catch {
        el.textContent = "● Sin conexión con el servidor (ejecuta run.py)";
        el.classList.add("offline");
    }
}

// --- MÉTRICAS REALES (artifacts/metricas.json) ---
async function cargarMetricas() {
    try {
        const r = await fetch(`${API_URL}/metrics`);
        if (!r.ok) return;
        const m = await r.json();
        const set = (id, v) => (document.getElementById(id).textContent = v);
        set("eval-periodo", `Evaluado sobre ${m.filas_prueba.toLocaleString("es")} registros reales (${m.periodo_prueba}).`);
        set("m-base-mae", m.linea_base.mae.toFixed(2));
        set("m-base-mase", m.linea_base.mase.toFixed(4));
        set("m-ridge-mae", m.ridge.mae.toFixed(2));
        set("m-ridge-mase", m.ridge.mase.toFixed(4));
        set("m-xgb-mae", m.xgboost.mae.toFixed(2));
        set("m-xgb-mase", m.xgboost.mase.toFixed(4));
        // Error reduction in units compared with the baseline
        const mejora = (1 - m.xgboost.mae_relativo) * 100;
        set("m-xgb-gain", `(${mejora.toFixed(1)}% menos error que la línea base)`);
    } catch { /* se queda en "–" */ }
}

// --- BUSCADOR DE PRODUCTOS POR NOMBRE ---
const inputBuscar = document.getElementById("producto-buscar");
const inputId = document.getElementById("producto-id");
const lista = document.getElementById("producto-sugerencias");
const hint = document.getElementById("producto-seleccionado");
let temporizador = null;
let sugerencias = [];
let indiceActivo = -1;

function cerrarLista() {
    lista.classList.add("hidden");
    lista.innerHTML = "";
    indiceActivo = -1;
}

function elegir(p) {
    inputBuscar.value = p.nombre;
    inputId.value = p.producto_id;
    hint.textContent = `✔ ${p.nombre} · ${p.categoria} · código ${p.producto_id}`;
    hint.classList.add("ok");
    cerrarLista();
}

function pintarLista() {
    if (!sugerencias.length) {
        lista.innerHTML = '<li class="vacio">Sin coincidencias</li>';
    } else {
        lista.innerHTML = sugerencias
            .map((p, i) => `<li role="option" data-i="${i}" class="${i === indiceActivo ? "activo" : ""}">${esc(p.nombre)}<small>${esc(p.categoria)} · ${esc(p.producto_id)}</small></li>`)
            .join("");
    }
    lista.classList.remove("hidden");
}

inputBuscar.addEventListener("input", () => {
    inputId.value = ""; // al editar el texto, se invalida la selección anterior
    hint.classList.remove("ok");
    hint.textContent = "Escribe al menos 2 letras y elige de la lista.";
    clearTimeout(temporizador);
    const q = inputBuscar.value.trim();
    if (q.length < 2) return cerrarLista();
    temporizador = setTimeout(async () => {
        try {
            const r = await fetch(`${API_URL}/productos?q=${encodeURIComponent(q)}&limite=8`);
            sugerencias = (await r.json()).resultados;
            indiceActivo = -1;
            pintarLista();
        } catch { cerrarLista(); }
    }, 200);
});

inputBuscar.addEventListener("keydown", (e) => {
    if (lista.classList.contains("hidden") || !sugerencias.length) return;
    if (e.key === "ArrowDown") { e.preventDefault(); indiceActivo = (indiceActivo + 1) % sugerencias.length; pintarLista(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); indiceActivo = (indiceActivo - 1 + sugerencias.length) % sugerencias.length; pintarLista(); }
    else if (e.key === "Enter" && indiceActivo >= 0) { e.preventDefault(); elegir(sugerencias[indiceActivo]); }
    else if (e.key === "Escape") cerrarLista();
});

lista.addEventListener("mousedown", (e) => { // mousedown: ocurre antes de que el input pierda el foco
    const li = e.target.closest("li[data-i]");
    if (li) elegir(sugerencias[Number(li.dataset.i)]);
});
document.addEventListener("click", (e) => { if (!e.target.closest(".autocomplete")) cerrarLista(); });

// --- CONTROLADOR 1: PRONÓSTICO ---
document.getElementById("prediction-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const resultBox = document.getElementById("prediction-result");
    resultBox.classList.remove("hidden");

    // Si escribió el nombre pero no eligió de la lista, el backend intenta resolverlo igual
    const payload = { fecha: document.getElementById("fecha-pred").value };
    if (inputId.value) payload.producto_id = inputId.value;
    else payload.nombre = inputBuscar.value.trim();
    const lag = document.getElementById("venta-anterior").value;
    if (lag !== "") payload.venta_semana_anterior = parseFloat(lag);

    resultBox.innerHTML = "🔮 Consultando el modelo...";
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await response.json();

        if (response.ok) {
            const p = data.producto;
            const origen = {
                "manual": "dato ingresado por ti",
                "historial real": "venta real de hace 7 días",
                "pronóstico recursivo": `pronóstico encadenado ${data.dias_hacia_el_futuro} días desde el último dato real`,
                "promedio histórico": "promedio histórico",
            }[data.fuente_lag] || data.fuente_lag;
            resultBox.innerHTML = `
                📈 <strong>${esc(p.nombre)}</strong> <small>(${esc(p.categoria)} · ${esc(p.producto_id)})</small><br>
                El modelo predice una venta de <strong>${data.demanda_predicha} unidades</strong> para el ${esc(data.fecha)}.
                ${data.venta_real !== undefined ? `<br><small>Venta real registrada ese día: <strong>${data.venta_real}</strong> u.</small>` : ""}
                <br><small>Promedio diario últimas 4 semanas: ${data.promedio_diario_28d} u. · Base del cálculo: ${esc(origen)}.</small>
                ${data.aviso ? `<br><small>⚠️ ${esc(data.aviso)}</small>` : ""}`;
        } else {
            const detalle = typeof data.detail === "string" ? data.detail : (data.detail?.[0]?.msg || "Error interno.");
            resultBox.innerHTML = `⚠️ <strong>Inconveniente en la consulta:</strong> ${esc(detalle)}`;
        }
    } catch {
        resultBox.innerHTML = "🔌 <strong>Error de red:</strong> No se pudo conectar al Backend. Asegúrate de ejecutar 'run.py'.";
    }
});

// --- CONTROLADOR 2: RECOMENDACIONES ---
document.getElementById("btn-recommend").addEventListener("click", async () => {
    const container = document.getElementById("recommendations-container");
    const fecha = document.getElementById("fecha-rec").value;
    container.innerHTML = "<div class='empty-state-pastel'><p>⚙️ Procesando estacionalidad y márgenes de ganancia...</p></div>";

    try {
        const response = await fetch(`${API_URL}/recommend`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ fecha, top: 3 }),
        });
        const data = await response.json();

        if (response.ok && data.top_recomendaciones.length > 0) {
            container.innerHTML = "";
            data.top_recomendaciones.forEach((item, index) => {
                const alta = item.incremento_estacional > 25;
                const signo = item.incremento_estacional >= 0 ? "+" : "";
                const card = document.createElement("div");
                card.className = `rec-card-pastel top-${index + 1}`;
                card.innerHTML = `
                    <div class="badge-rec ${alta ? "bg-light-rose" : "bg-light-blue"}">${alta ? "🔥 Alta Demanda" : "💎 Margen Comercial"}</div>
                    <h3>${esc(item.nombre)}</h3>
                    <p class="rec-meta">ID: ${esc(item.producto_id)} · Categoría: ${esc(item.categoria)}</p>
                    <div class="rec-metrics">
                        <p>Demanda esperada: <strong>${item.demanda_esperada} u.</strong></p>
                        <p>Vs. últimas 4 semanas: <strong style="color:${item.incremento_estacional >= 0 ? "#0ca678" : "#d6336c"};">${signo}${item.incremento_estacional}%</strong></p>
                        <p>Retorno estimado: <strong style="color:#4c6ef5;">${item.ganancia_estimada_bs} Bs.</strong></p>
                    </div>
                    <div style="margin-top:0.8rem; font-size:0.8rem; color: var(--text-light); font-style:italic;">
                        💡 Acción sugerida: Posicionar en el mostrador central y sugerir venta cruzada.
                    </div>`;
                container.appendChild(card);
            });
            if (data.aviso) {
                const nota = document.createElement("p");
                nota.className = "hint";
                nota.textContent = `⚠️ ${data.aviso}`;
                container.appendChild(nota);
            }
        } else {
            const detalle = typeof data.detail === "string" ? data.detail : (data.detail?.[0]?.msg || "No se hallaron registros.");
            container.innerHTML = `<div class='empty-state-pastel'><p style='color: var(--pastel-rose-text);'>⚠️ ${esc(detalle)}</p></div>`;
        }
    } catch {
        container.innerHTML = "<div class='empty-state-pastel'><p style='color: var(--pastel-rose-text);'>🔌 Error de comunicación con la API del recomendador.</p></div>";
    }
});

cargarEstado();
cargarMetricas();