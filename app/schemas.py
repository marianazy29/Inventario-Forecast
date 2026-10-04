from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


def _validar_fecha(v: str) -> str:
    try:
        datetime.strptime(v, "%Y-%m-%d")
        return v
    except ValueError:
        raise ValueError("El formato de fecha debe ser estrictamente AAAA-MM-DD.")


class ConsultaPrediccion(BaseModel):
    fecha: str = Field(..., description="Fecha a consultar (AAAA-MM-DD)", examples=["2026-11-01"])
    producto_id: Optional[str] = Field(None, description="Código del producto (opcional si se envía nombre)", examples=["VNS-012"])
    nombre: Optional[str] = Field(None, description="Nombre del producto (opcional si se envía producto_id)", examples=["PEPSI LATA"])
    venta_semana_anterior: Optional[float] = Field(
        None, ge=0,
        description="Venta física de hace 7 días. Si se omite, se calcula del historial / pronóstico.",
    )

    _val_fecha = field_validator("fecha")(_validar_fecha)

    @model_validator(mode="after")
    def requerir_producto(self):
        if not (self.producto_id or "").strip() and not (self.nombre or "").strip():
            raise ValueError("Debes enviar 'producto_id' o 'nombre'.")
        return self


class ConsultaRecomendacion(BaseModel):
    fecha: str = Field(..., description="Fecha para la que se desean recomendaciones (AAAA-MM-DD)", examples=["2026-11-01"])
    top: int = Field(3, ge=1, le=10, description="Cantidad de productos a recomendar")

    _val_fecha = field_validator("fecha")(_validar_fecha)
