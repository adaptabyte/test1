"""Plain data types shared across the app."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Offer:
    """One merchant offer from a Google Shopping result, normalized across providers."""

    title: str
    merchant: str
    price: float | None
    price_text: str = ""
    currency: str = "USD"
    shipping: float | None = None  # None means the listing did not say
    shipping_text: str = ""
    link: str = ""  # best link for a human: merchant page if known, else Google product page
    merchant_url: str = ""  # direct merchant URL, when the provider returned one
    product_id: str = ""
    condition: str = ""
    snippet: str = ""
    extensions: list[str] = field(default_factory=list)
    thumbnail: str = ""
    multiple_sources: bool = False
    position: int = 0
    raw: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Links come from third-party search results; never render javascript: or data: URLs.
        if not self.link.lower().startswith(("http://", "https://")):
            self.link = ""
        if not self.merchant_url.lower().startswith(("http://", "https://")):
            self.merchant_url = ""

    @property
    def spec_text(self) -> str:
        """Title plus Google's structured attribute chips; what the rules parse specs from."""
        return " | ".join(part for part in [self.title, *self.extensions] if part)

    @property
    def full_text(self) -> str:
        """Everything we know about the listing, including the free-text snippet."""
        return " | ".join(part for part in [self.spec_text, self.snippet] if part)


@dataclass
class Reason:
    """Why a listing was rejected. `code` is stable and used for grouping in the UI."""

    code: str
    message: str


@dataclass
class Evaluation:
    """The outcome of running a tracker's rules over one offer."""

    reasons: list[Reason] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)
    units: int = 1
    total_price: float | None = None
    unit_price: float | None = None
    tier: str = ""
    merchant: str = ""  # canonical name of the trusted merchant, if matched
    merchant_domain: str = ""

    @property
    def accepted(self) -> bool:
        return not self.reasons

    @property
    def decision(self) -> str:
        return "accepted" if self.accepted else "rejected"

    def reject(self, code: str, message: str) -> None:
        self.reasons.append(Reason(code, message))

    def note(self, message: str) -> None:
        self.notes.append(message)
