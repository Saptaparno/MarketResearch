from typing import Any

from pydantic import BaseModel, Field


class MarketResearchContent(BaseModel):
    """Session/tool payloads aggregated under ``market_research``."""

    historical_data: dict[str, Any] | None = None
    stock_info: dict[str, Any] | None = None
    analyst_recommendations: dict[str, Any] | None = None


class MarketResearchAgentOutput(BaseModel):
    market_research: MarketResearchContent = Field(default_factory=MarketResearchContent)


class PortfolioHealthContent(BaseModel):
    last_quote: dict[str, Any] | None = None
    last_return: dict[str, Any] | None = None
    compare_returns: dict[str, Any] | None = None


class PortfolioHealthAgentOutput(BaseModel):
    portfolio_health: PortfolioHealthContent = Field(default_factory=PortfolioHealthContent)


class RiskAnalystContent(BaseModel):
    last_vol: dict[str, Any] | None = None
    last_dd: dict[str, Any] | None = None
    last_corr: dict[str, Any] | None = None
    disclaimer: str | None = None


class RiskAnalystAgentOutput(BaseModel):
    risk_analyst: RiskAnalystContent = Field(default_factory=RiskAnalystContent)


class FinancialPlanningContent(BaseModel):
    last_fv_lump: dict[str, Any] | None = None
    last_fv_monthly: dict[str, Any] | None = None
    last_payment: dict[str, Any] | None = None


class FinancialPlanningAgentOutput(BaseModel):
    financial_planning: FinancialPlanningContent = Field(default_factory=FinancialPlanningContent)


class GeneralEnquiryContent(BaseModel):
    last_utc: str | None = None
    capabilities: str | None = None
    normalized_question: str | None = None
    text_response: str | None = None


class GeneralEnquiryAgentOutput(BaseModel):
    general_enquiry: GeneralEnquiryContent = Field(default_factory=GeneralEnquiryContent)


class RankedSymbolEntry(BaseModel):
    symbol: str
    exploratory_rank: int
    rough_fit_weight: float


class GoalMapContent(BaseModel):
    goal: str
    time_horizon_years: float
    categories: list[str]


class FeeCompareIllustration(BaseModel):
    error: str | None = None
    starting_notional: float | None = None
    years: float | None = None
    assumed_annual_growth_pct: float | None = None
    fee_pct_a: float | None = None
    fee_pct_b: float | None = None
    ending_a: float | None = None
    ending_b: float | None = None


class ProductRecommendationContent(BaseModel):
    last_ranked: list[RankedSymbolEntry] = Field(default_factory=list)
    last_goal_map: GoalMapContent | None = None
    last_fee_compare: FeeCompareIllustration | None = None
    text_response: str | None = None


class ProductRecommendationAgentOutput(BaseModel):
    product_recommendation: ProductRecommendationContent = Field(default_factory=ProductRecommendationContent)


class IngestOutput(BaseModel):
    file: str | None = None
    ticker: str | None = None
    year: int | None = None
    doc_type: str | None = None
    error: str | None = None


class TranscriptOutput(BaseModel):
    ticker: str | None = None
    text: str | None = None
    source: str | None = None
    error: str | None = None


class ChartOutput(BaseModel):
    ticker: str | None = None
    range: str | None = None
    bytes_len: int | None = None
    error: str | None = None


class DescriptionOutput(BaseModel):
    image: str | None = None
    label: str | None = None
    description: str | None = None
    error: str | None = None


class MultimodalContent(BaseModel):
    last_ingest: IngestOutput | None = None
    transcript: TranscriptOutput | None = None
    last_chart: ChartOutput | None = None
    last_description: DescriptionOutput | None = None


class MultimodalAgentOutput(BaseModel):
    multimodal: MultimodalContent = Field(default_factory=MultimodalContent)


class MultimodalSessionInput(BaseModel):
    pdf_path: str | None = None
    audio_path: str | None = None
    audio_ticker: str | None = None
    chart_ticker: str = "SPY"
    chart_range: str = "1y"
    screenshot_path: str | None = None
    screenshot_label: str = ""


class TradeContent(BaseModel):
    last_propose: dict[str, Any] | None = None
    last_large_propose: dict[str, Any] | None = None


class TradeAgentOutput(BaseModel):
    trade: TradeContent = Field(default_factory=TradeContent)

