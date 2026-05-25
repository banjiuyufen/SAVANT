"""Endpoint ontology checks for effect support."""


from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONTOLOGY_PATH = (
    PROJECT_ROOT
    / "theorem_library"
    / "ontologies"
    / "stage2_effect_endpoint_ontology.json"
)


def _norm_text(value: Any) -> str:
    text = str(value or "").lower()
    text = (
        text.replace("µ", "u")
        .replace("μ", "u")
        .replace("γ", "gamma")
        .replace("α", "alpha")
        .replace("β", "beta")
        .replace("κ", "kappa")
        .replace("_", " ")
        .replace("/", " ")
        .replace("-", " ")
    )
    text = re.sub(r"[^a-z0-9.+\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _predicate_text(pred: Any) -> str:
    parts: List[str] = [str(getattr(pred, "functor", "") or "")]
    for arg in getattr(pred, "args", []) or []:
        value = getattr(arg, "value", "")
        if value in ("?", "", None):
            continue
        try:
            if arg.is_variable():
                continue
        except Exception:
            pass
        role = getattr(arg, "role", "")
        if role:
            parts.append(str(role))
        parts.append(str(value))
    return " ".join(parts)


class Stage2EndpointOntology:


    def __init__(self, ontology_path: Optional[Path] = None):
        self.ontology_path = Path(ontology_path or DEFAULT_ONTOLOGY_PATH)
        self.metadata: Dict[str, Any] = {}
        self.categories: List[Dict[str, Any]] = []
        self._compiled: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.ontology_path.exists():
            return
        data = json.loads(self.ontology_path.read_text(encoding="utf-8"))
        self.metadata = data.get("metadata", {})
        self.categories = data.get("categories", [])
        compiled: List[Dict[str, Any]] = []
        for category in self.categories:
            patterns = []
            for pattern in category.get("patterns", []) or []:
                try:
                    patterns.append(re.compile(str(pattern), re.IGNORECASE))
                except re.error:
                    continue
            if patterns:
                item = dict(category)
                item["_patterns"] = patterns
                compiled.append(item)
        self._compiled = compiled

    def score(self, pred: Any) -> Dict[str, Any]:
        raw_text = _predicate_text(pred)
        norm = _norm_text(raw_text)
        if not norm:
            return {
                "score": 0.0,
                "matched_categories": [],
                "query_text": raw_text,
                "normalized_text": norm,
                "detail": "no endpoint text",
            }

        matches: List[Dict[str, Any]] = []
        for category in self._compiled:
            matched_patterns = []
            for pattern in category.get("_patterns", []):
                if pattern.search(norm):
                    matched_patterns.append(pattern.pattern)
            if matched_patterns:
                matches.append(
                    {
                        "id": category.get("id", ""),
                        "label": category.get("label", ""),
                        "level": category.get("level", ""),
                        "score": float(category.get("score", 0.0)),
                        "patterns": matched_patterns[:3],
                    }
                )

        if not matches:
            return {
                "score": 0.0,
                "matched_categories": [],
                "query_text": raw_text,
                "normalized_text": norm,
                "detail": f"no Stage2 endpoint ontology match for: {norm}",
            }

        matches.sort(key=lambda item: item["score"], reverse=True)
        best = matches[0]
        return {
            "score": best["score"],
            "best_category": best["id"],
            "best_level": best["level"],
            "matched_categories": [item["id"] for item in matches],
            "matches": matches,
            "query_text": raw_text,
            "normalized_text": norm,
            "detail": (
                f"Stage2 endpoint ontology matched {best['id']}"
                f"({best['score']:.2f}) for: {norm}"
            ),
        }
