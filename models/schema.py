from typing import Any
from pydantic import BaseModel

class MarketResearchData(BaseModel):
    historical_data: dict[str, Any] | None = None
    stock_info: dict[str, Any] | None = None
    analyst_recommendations: dict[str, Any] | None = None
