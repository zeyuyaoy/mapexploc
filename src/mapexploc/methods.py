"""Versioned, explicit experimental methods; the legacy default is immutable."""

from typing import Any, Literal

from pydantic import Field, model_validator

from .report_v2 import AnalysisConfiguration


class MethodConfiguration(AnalysisConfiguration):
    method_profile: Literal["v2x-legacy", "v2x-development"] = "v2x-legacy"
    reference_strategy: Literal["whole_shuffle", "block_shuffle", "positional_pool"] = (
        "whole_shuffle"
    )
    region_strategy: Literal["legacy", "windows_10", "windows_5"] = "legacy"
    reference_seed: int = Field(default=42, ge=0, le=2**32 - 1)
    diagnostic_seed: int = Field(default=2026, ge=0, le=2**32 - 1)
    reference_pool: list[str] = Field(default_factory=list)
    reference_pool_id: str | None = None
    reference_sensitivity: bool = False
    faithfulness: bool = False
    diagnostic_draws: int = Field(default=4, ge=1, le=32)
    bootstrap_replicates: int = Field(default=1000, ge=100, le=10000)

    @model_validator(mode="after")
    def validate_profile(self) -> "MethodConfiguration":
        if self.method_profile == "v2x-legacy":
            if "reference_seed" not in self.model_fields_set:
                self.reference_seed = self.seed
            elif self.reference_seed != self.seed:
                raise ValueError(
                    "The legacy profile couples reference and coalition seeds; use"
                    " v2x-development to vary them independently"
                )
        if self.method_profile == "v2x-legacy" and (
            self.reference_strategy != "whole_shuffle"
            or self.region_strategy != "legacy"
        ):
            raise ValueError("The legacy profile fixes its reference game and regions")
        if self.reference_strategy == "positional_pool":
            if not self.reference_pool or not self.reference_pool_id:
                raise ValueError(
                    "Positional references require a frozen development pool and"
                    " identity"
                )
        elif self.reference_pool or self.reference_pool_id:
            raise ValueError("Reference pools only apply to positional_pool")
        return self


def parse_configuration(value: dict[str, Any]) -> AnalysisConfiguration:
    """Absence of a profile keeps the original schema and numerical behavior."""
    cls = MethodConfiguration if "method_profile" in value else AnalysisConfiguration
    return cls.model_validate(value)
