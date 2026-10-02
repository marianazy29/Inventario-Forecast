from pydantic import BaseModel, Field, field_validator
from datetime import datetime

class ConsultaPrediccion(BaseModel):
    fecha: str = Field(..., description="Fecha a consultar (AAAA-MM-DD)", example="2026-11-01")
    producto_id: str = Field(..., description="ID alfanumérico del producto", example="VNS-012")
    venta_semana_anterior: float = Field(..., ge=0, description="Cantidad física vendida hace 7 días", example=10.0)

    @field_validator('fecha')
    @classmethod
    def validar_formato_fecha(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("El formato de fecha debe ser estrictamente AAAA-MM-DD.")

class ConsultaRecomendacion(BaseModel):
    fecha: str = Field(..., description="Fecha para la que se desean recomendaciones (AAAA-MM-DD)", example="2026-06-23")
