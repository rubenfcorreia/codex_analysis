from __future__ import annotations

from typing import Any, Dict, Optional


def observation_compartment(
    cache: Dict[str, Any],
    exp_id: Any,
    observation: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Return the basal/apical compartment for a dendrite or spine observation."""
    if not isinstance(observation, dict):
        return None

    compartment = observation.get("compartment")
    if compartment is not None and str(compartment).strip():
        return str(compartment)

    exp_meta = cache.get("experiments", {}).get(str(exp_id), {}) if isinstance(cache, dict) else {}
    if isinstance(exp_meta, dict):
        for key in ("compartment", "compartment_label", "dendrite_compartment"):
            compartment = exp_meta.get(key)
            if compartment is not None and str(compartment).strip():
                return str(compartment)

    for key in ("conversion_compartment", "source_compartment"):
        compartment = observation.get(key)
        if compartment is not None and str(compartment).strip():
            return str(compartment)
    return None


__all__ = ["observation_compartment"]
