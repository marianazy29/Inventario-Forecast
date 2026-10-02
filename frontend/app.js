const API_URL = "http://127.0.0.1:8000";

// --- LOGICA DE NAVEGACIÓN ENTRE PESTAÑAS ---
document.querySelectorAll(".nav-link").forEach(button => {
    button.addEventListener("click", () => {
        // Desactivar botones y secciones anteriores
        document.querySelectorAll(".nav-link").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".content-section").forEach(s => s.classList.remove("active"));
        
        // Activar pestaña elegida
        button.classList.add("active");
        const targetSection = button.getAttribute("data-target");
        document.getElementById(targetSection).classList.add("active");
    });
});

// Sincronizar el valor numérico del slider en pantalla
const slider = document.getElementById("eval-range");
if(slider) {
    slider.addEventListener("input", (e) => {
        document.getElementById("range-val").innerText = e.target.value;
    });
}

// --- CONTROLADOR 1: CÁLCULO DE PRONÓSTICO ---
document.getElementById("prediction-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    
    const resultBox = document.getElementById("prediction-result");
    resultBox.classList.remove("hidden");
    resultBox.innerHTML = "🔮 Consultando árboles de decisión en el servidor...";

    const payload = {
        fecha: document.getElementById("fecha-pred").value,
        producto_id: document.getElementById("producto-id").value.trim(),
        venta_semana_anterior: parseFloat(document.getElementById("venta-anterior").value || 0)
    };

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await response.json();

        if (response.ok) {
            resultBox.innerHTML = `📈 <strong>Stock sugerido:</strong> El modelo unificado predice una venta de <strong>${data.demanda_predicha} unidades</strong> para la fecha consultada.<br><br><small>✅ Se recomienda asegurar este inventario base en góndolas para evitar pérdidas operacionales.</small>`;
        } else {
            resultBox.innerHTML = `⚠️ <strong>Inconveniente en la consulta:</strong> ${data.detail || "Error interno."}`;
        }
    } catch (error) {
        resultBox.innerHTML = `🔌 <strong>Error de red:</strong> No se pudo conectar al Backend. Asegúrate de ejecutar 'run.py'.`;
    }
});

// --- CONTROLADOR 2: RECOMENDACIONES PRESCRIPTIVAS ---
document.getElementById("btn-recommend").addEventListener("click", async () => {
    const container = document.getElementById("recommendations-container");
    const fecha = document.getElementById("fecha-rec").value;

    container.innerHTML = "<div class='empty-state-pastel'><p>⚙️ Procesando estacionalidad y márgenes de ganancia en vivo...</p></div>";

    try {
        const response = await fetch(`${API_URL}/recommend`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ fecha: fecha })
        });
        const data = await response.json();

        if (response.ok && data.top_recomendaciones.length > 0) {
            container.innerHTML = ""; // Limpiar estado vacío
            
            data.top_recomendaciones.forEach((item, index) => {
                const card = document.createElement("div");
                card.className = `rec-card-pastel top-${index + 1}`;
                
                let insignia = item.incremento_estacional > 25 ? "🔥 Alta Demanda" : "💎 Margen Comercial";
                let badgeClass = item.incremento_estacional > 25 ? "bg-light-rose" : "bg-light-blue";

                card.innerHTML = `
                    <div class="badge-rec ${badgeClass}">${insignia}</div>
                    <h3>${item.nombre}</h3>
                    <p class="rec-meta">ID: ${item.producto_id} · Categoría: ${item.categoria}</p>
                    <div class="rec-metrics">
                        <p>Demanda esperada: <strong>${item.demanda_esperada} u.</strong></p>
                        <p>Pico estacional: <strong style="color: #0ca678;">+${item.incremento_estacional}%</strong></p>
                        <p>Retorno estimado: <strong style="color: #4c6ef5;">${item.ganancia_estimada_bs} Bs.</strong></p>
                    </div>
                    <div style="margin-top:0.8rem; font-size:0.8rem; color: var(--text-light); font-style:italic;">
                        💡 Acción sugerida: Posicionar en el mostrador central y sugerir venta cruzada.
                    </div>
                `;
                container.appendChild(card);
            });
        } else {
            container.innerHTML = `<div class='empty-state-pastel'><p style='color: var(--pastel-rose-text);'>⚠️ Error: ${data.detail || "No se hallaron registros."}</p></div>`;
        }
    } catch (error) {
        container.innerHTML = "<div class='empty-state-pastel'><p style='color: var(--pastel-rose-text);'>🔌 Error de comunicación con la API del recomendador.</p></div>";
    }
});
