from .contracts import AnalysisFamilyResult, AnalysisScope, FigureContext, attach_scope, validate_family_payload
from .registry import family_names, register_family, run_analysis_family, run_family

ANALYSIS_FAMILIES = ["state", "mixed_model", "calcium_events", "visual_response", "coincidence", "correlation", "lag", "transitions", "roi_split", "coactivity", "trial_type"]
from .splits import SPLIT_BRANCHES, require_split_groups, scope_split_rows, split_groups

__all__ = [
    "ANALYSIS_FAMILIES", "AnalysisFamilyResult", "AnalysisScope",
    "SPLIT_BRANCHES", "FigureContext", "attach_scope", "family_names",
    "register_family", "require_split_groups", "run_analysis_family", "run_family",
    "scope_split_rows", "split_groups", "validate_family_payload",
]
