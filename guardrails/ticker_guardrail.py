import re
from typing import Union

from agno.exceptions import CheckTrigger, InputCheckError
from agno.guardrails.base import BaseGuardrail
from agno.run.agent import RunInput
from agno.run.team import TeamRunInput

# Uppercase tokens that are common English words, not tickers.
_STOP_WORDS = frozenset({
    "A", "AN", "THE", "IS", "ARE", "WAS", "BE", "AT", "BY", "DO",
    "GO", "IF", "IN", "IT", "ME", "MY", "NO", "OF", "OK", "ON", "OR",
    "SO", "TO", "UP", "US", "WE", "BUY", "FOR", "NOT", "ALL",
    "AND", "GET", "HAS", "HOW", "LET", "PUT", "SET", "USE", "CAN",
    "NEW", "NOW", "OWN", "RUN", "SEE", "TRY", "ETF", "IPO", "CEO",
    "CFO", "SEC", "FED", "GDP", "CPI", "PPI", "YOY", "QOQ", "YTD",
    "NAV", "AUM", "ROE", "ROI", "EPS", "PE", "PB", "EV", "I",
})

# Match tokens that are *already* all-uppercase in the original text.
# This avoids flagging normal English words that happen to be in the input.
_TOKEN_RE = re.compile(r'\b[A-Z]{1,5}\b')


class TickerPermissionGuardrail(BaseGuardrail):
    """Block messages that mention tickers the user is not permitted to trade.

    Initialise with ``permitted_tickers`` drawn from session_state at agent
    creation time.  Pass ``None`` (default) to allow all tickers — useful when
    the agent is used without per-user restrictions.
    """

    def __init__(self, permitted_tickers: list[str] | None = None):
        self.permitted_tickers: frozenset[str] | None = (
            frozenset(t.upper() for t in permitted_tickers) if permitted_tickers is not None else None
        )

    def _denied_tickers(self, text: str) -> list[str]:
        if self.permitted_tickers is None:
            return []
        # Only inspect tokens that are already all-uppercase in the original text,
        # so normal English words (which are mixed-case) are never treated as tickers.
        candidates = _TOKEN_RE.findall(text)
        return sorted({t for t in candidates if t not in _STOP_WORDS and t not in self.permitted_tickers})

    def check(self, run_input: Union[RunInput, TeamRunInput]) -> None:
        denied = self._denied_tickers(run_input.input_content_string())
        if denied:
            raise InputCheckError(
                f"Ticker(s) not permitted for this user: {denied}. "
                "Contact your administrator to update your trading allowlist.",
                check_trigger=CheckTrigger.INPUT_NOT_ALLOWED,
            )

    async def async_check(self, run_input: Union[RunInput, TeamRunInput]) -> None:
        self.check(run_input)
