from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

def _validate_date(value: str) -> str:
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return value
    except ValueError:
        raise ValueError("El formato de fecha debe ser AAAA-MM-DD.")


class PredictionRequest(BaseModel):
    date: str = Field(..., description="Fecha a consultar (AAAA-MM-DD)", examples=["2026-11-01"])
    product_id: Optional[str] = Field(None, description="Código del producto (opcional si se envía el nombre)", examples=["VNS-012"])
    name: Optional[str] = Field(None, description="Nombre del producto (opcional si se envía el código)", examples=["PEPSI LATA"])
    sales_last_week: Optional[float] = Field(
        None, ge=0,
        description="Venta de hace 7 días. Si se omite, se calcula del historial o del pronóstico.",
    )

    _check_date = field_validator("date")(_validate_date)

    @model_validator(mode="after")
    def require_product(self):
        if not (self.product_id or "").strip() and not (self.name or "").strip():
            raise ValueError("Debes enviar 'product_id' o 'name'.")
        return self


class RecommendationRequest(BaseModel):
    date: str = Field(..., description="Fecha para la que se desean recomendaciones (AAAA-MM-DD)", examples=["2026-11-01"])
    top: int = Field(3, ge=1, le=10, description="Cantidad de productos a recomendar")

    _check_date = field_validator("date")(_validate_date)


class WeekForecastRequest(BaseModel):
    date: str = Field(..., description="Primer día de la semana a pronosticar (AAAA-MM-DD)", examples=["2026-10-01"])
    product_id: Optional[str] = Field(None, description="Código del producto (opcional si se envía el nombre)", examples=["CVZ-001"])
    name: Optional[str] = Field(None, description="Nombre del producto (opcional si se envía el código)", examples=["PACEÑA LATA GRANDE"])

    _check_date = field_validator("date")(_validate_date)

    @model_validator(mode="after")
    def require_product(self):
        if not (self.product_id or "").strip() and not (self.name or "").strip():
            raise ValueError("Debes enviar 'product_id' o 'name'.")
        return self
