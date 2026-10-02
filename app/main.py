import os
import pandas as pd
import holidays
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import ConsultaPrediccion, ConsultaRecomendacion
from app.settings import MODEL_PATH, ENCODER_PATH, DATASET_INTELIGENTE_PATH

app = FastAPI(
    title="Sistema Inteligente de Inventario y Ventas - Licorería",
    description="API integrada para predicciones de stock y analítica prescriptiva de recomendaciones.",
    version="1.0.0"
)

# Inyección de Seguridad CORS rescatada de los demos profesionales
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Validar existencia de los artefactos
if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    raise RuntimeError("Error Crítico: No se encontraron los archivos .pkl en la carpeta 'artifacts/'.")

modelo_xgb = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

festividades_comerciales = {(5, 27), (6, 23), (9, 21), (11, 1), (12, 24), (12, 31)}
margen_ganancia = {"VINOS": 25.0, "WHISKIES": 45.0, "CREMAS": 20.0, "SODA": 3.0, "AGUAS": 1.5}

# --- ENDPOINT 1: PREDICCIÓN DE DEMANDA ---
@app.post("/predict")
def predecir_demanda(consulta: ConsultaPrediccion):
    try:
        fecha_dt = pd.to_datetime(consulta.fecha)
        if consulta.producto_id not in encoder.classes_:
            raise HTTPException(status_code=404, detail=f"Código de producto '{consulta.producto_id}' no registrado.")
        
        mes = fecha_dt.month
        dia_semana = fecha_dt.dayofweek
        es_fin_semana = 1 if dia_semana in [4, 5, 6] else 0
        es_feriado = 1 if fecha_dt in holidays.Bolivia(years=[fecha_dt.year]) else 0
        es_festivo = 1 if (fecha_dt.month, fecha_dt.day) in festividades_comerciales else 0
        
        prod_encoded = encoder.transform([consulta.producto_id])
        
        columnas = ['Producto_Codificado', 'Mes', 'Día_Semana', 'Es_Fin_De_Semana', 'Es_Feriado', 'Es_Evento_Festivo', 'Venta_Semana_Anterior']
        entrada = pd.DataFrame([[prod_encoded[0], mes, dia_semana, es_fin_semana, es_feriado, es_festivo, consulta.venta_semana_anterior]], columns=columnas)
        
        prediccion = int(max(0, round(modelo_xgb.predict(entrada)[0])))
        return {"status": "success", "demanda_predicha": prediccion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT 2: RECOMENDADOR INTELIGENTE (ANALÍTICA PRESCRIPTIVA) ---
@app.post("/recommend")
def recomendar_productos(consulta: ConsultaRecomendacion):
    try:
        fecha_dt = pd.to_datetime(consulta.fecha)
        df_base = pd.read_csv(DATASET_INTELIGENTE_PATH)
        
        mes = fecha_dt.month
        dia_semana = fecha_dt.dayofweek
        es_fin_semana = 1 if dia_semana in [4, 5, 6] else 0
        es_feriado = 1 if fecha_dt in holidays.Bolivia(years=[fecha_dt.year]) else 0
        es_festivo = 1 if (fecha_dt.month, fecha_dt.day) in festividades_comerciales else 0
        
        lista_recomendaciones = []
        
        for p_id in encoder.classes_:
            datos_prod = df_base[df_base['ProductoId'] == p_id]
            if datos_prod.empty: continue
            
            nombre = datos_prod['NombreProducto'].iloc[0]
            cat = datos_prod['Categoría'].iloc[0]
            promedio_hist = datos_prod['Cantidad_Vendida'].mean()
            
            prod_encoded = encoder.transform([p_id])
            columnas = ['Producto_Codificado', 'Mes', 'Día_Semana', 'Es_Fin_De_Semana', 'Es_Feriado', 'Es_Evento_Festivo', 'Venta_Semana_Anterior']
            entrada = pd.DataFrame([[prod_encoded[0], mes, dia_semana, es_fin_semana, es_feriado, es_festivo, round(promedio_hist)]], columns=columnas)
            
            pred = int(max(0, round(modelo_xgb.predict(entrada)[0])))
            incremento = ((pred - promedio_hist) / promedio_hist) * 100 if promedio_hist > 0 else 0
            ganancia = pred * margen_ganancia.get(cat, 5.0)
            
            lista_recomendaciones.append({
                "producto_id": p_id,
                "nombre": nombre,
                "categoria": cat,
                "demanda_esperada": pred,
                "incremento_estacional": round(incremento, 1),
                "ganancia_estimada_bs": round(ganancia, 2)
            })
            
        df_rec = pd.DataFrame(lista_recomendaciones).sort_values(by="ganancia_estimada_bs", ascending=False)
        return {"status": "success", "top_recomendaciones": df_rec.head(3).to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
