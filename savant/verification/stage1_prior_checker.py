"""Ontology-prior checks for construction support."""


from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class Stage1PriorResult:
    score: float
    cap_5level: int = 5
    warnings: List[str] = field(default_factory=list)
    details: List[str] = field(default_factory=list)
    sub_scores: Dict[str, float] = field(default_factory=dict)
    matched_type: str = "unknown"


def _as_list(value: Any) -> List[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    return [value]


def _norm_text(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("µ", "u").replace("μ", "u").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(_flatten_text(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_text(v) for v in value)
    return str(value)


def _present(props: Dict[str, Any], field_name: str) -> bool:
    value = props.get(field_name)
    if value is None or value == "":
        return False
    if isinstance(value, list):
        return any(v not in (None, "") for v in value)
    if isinstance(value, dict):
        return any(v not in (None, "", [], {}) for v in value.values())
    return True


def _first_present(props: Dict[str, Any], fields: Iterable[str]) -> Optional[Any]:
    for field_name in fields:
        if _present(props, field_name):
            return props.get(field_name)
    return None


def _parse_numeric_range(text: str) -> Optional[Tuple[float, float]]:
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    if not nums:
        return None
    values = [float(n) for n in nums[:2]]
    if len(values) == 1:
        return values[0], values[0]
    return min(values), max(values)


def _parse_size_nm(value: Any) -> Tuple[Optional[Tuple[float, float, float]], str]:
    text = _norm_text(value)
    if not text:
        return None, ""

    qualitative = ""
    if "nano" in text:
        qualitative = "nano"
    elif "micro" in text:
        qualitative = "micro"

    rng = _parse_numeric_range(text)
    if rng is None:
        return None, qualitative

    lo, hi = rng
    unit = "nm"
    if re.search(r"\b(mm|millimeter|millimetre)s?\b", text):
        unit = "mm"
    elif re.search(r"\b(um|micron|micrometer|micrometre)s?\b", text):
        unit = "um"
    elif re.search(r"\b(nm|nanometer|nanometre)s?\b", text):
        unit = "nm"

    factor = {"nm": 1.0, "um": 1000.0, "mm": 1000000.0}[unit]
    lo_nm, hi_nm = lo * factor, hi * factor
    return (lo_nm, hi_nm, (lo_nm + hi_nm) / 2.0), qualitative


def _parse_zeta_mv(value: Any) -> Optional[float]:
    text = _norm_text(value)
    if not text:
        return None
    rng = _parse_numeric_range(text)
    if rng is None:
        return None
    return max(abs(rng[0]), abs(rng[1]))


def _score_range(mean_value: float, ideal: List[float], acceptable: List[float]) -> float:
    if ideal[0] <= mean_value <= ideal[1]:
        return 1.0
    if acceptable[0] <= mean_value <= acceptable[1]:
        return 0.75
    return 0.25


class Stage1PriorChecker:


    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = (
                Path(__file__).resolve().parents[2]
                / "theorem_library"
                / "ontologies"
                / "stage1_priors.json"
            )
        with open(config_path, encoding="utf-8") as f:
            self.config = json.load(f)

        self.alias_to_type: Dict[str, str] = {}
        for type_name, prior in self.config["types"].items():
            self.alias_to_type[type_name.lower()] = type_name
            for alias in prior.get("aliases", []):
                self.alias_to_type[_norm_text(alias)] = type_name

    def check(self, props: Dict[str, Any]) -> Stage1PriorResult:
        props = props or {}
        matched_type, type_score, type_warning = self._match_type(props.get("Type"))
        prior = self.config["types"].get(matched_type)

        if prior is None:
            warnings = []
            if type_warning:
                warnings.append(type_warning)
            return Stage1PriorResult(
                score=0.45,
                cap_5level=3,
                warnings=warnings or ["Unknown adjuvant type; ontology prior cannot fully validate construction."],
                details=["type=unknown"],
                sub_scores={"type": type_score},
                matched_type="unknown",
            )

        cap = 5
        warnings: List[str] = []
        details: List[str] = [f"type={matched_type}"]
        if type_warning:
            warnings.append(type_warning)

        completeness_score, completeness_details, completeness_cap = self._score_completeness(props, prior)
        cap = min(cap, completeness_cap)
        details.extend(completeness_details)

        size_score, size_warnings, size_cap = self._score_size(props, prior, matched_type)
        warnings.extend(size_warnings)
        cap = min(cap, size_cap)

        method_score, method_warnings, method_cap = self._score_method(props, prior)
        warnings.extend(method_warnings)
        cap = min(cap, method_cap)

        component_score, component_warnings, component_cap = self._score_component_role(props, prior, matched_type)
        warnings.extend(component_warnings)
        cap = min(cap, component_cap)

        zeta_score, zeta_warnings = self._score_zeta(props)
        warnings.extend(zeta_warnings)

        pattern_warnings, pattern_cap = self._check_global_patterns(props, matched_type)
        warnings.extend(pattern_warnings)
        cap = min(cap, pattern_cap)

        sub_scores = {
            "type": type_score,
            "completeness": completeness_score,
            "size": size_score,
            "method": method_score,
            "component_role": component_score,
            "zeta": zeta_score,
        }

        weights = self.config["score_weights"]
        total_weight = sum(weights[k] for k in sub_scores)
        score = sum(sub_scores[k] * weights[k] for k in sub_scores) / total_weight
        score = max(0.0, min(1.0, score))

        details.append(
            "sub_scores="
            + ",".join(f"{k}:{v:.2f}" for k, v in sorted(sub_scores.items()))
        )
        if cap < 5:
            details.append(f"cap={cap}")

        return Stage1PriorResult(
            score=score,
            cap_5level=cap,
            warnings=warnings,
            details=details,
            sub_scores=sub_scores,
            matched_type=matched_type,
        )

    def _match_type(self, raw_type: Any) -> Tuple[str, float, str]:
        text = _norm_text(raw_type)
        if not text:
            return "unknown", 0.0, "Missing adjuvant type."
        if text in self.alias_to_type:
            return self.alias_to_type[text], 1.0, ""
        compact = text.replace(" ", "_")
        if compact in self.alias_to_type:
            return self.alias_to_type[compact], 1.0, ""
        for alias, type_name in self.alias_to_type.items():
            if alias and alias in text:
                return type_name, 0.85, f"Adjuvant type '{raw_type}' was normalized to '{type_name}'."
        return "unknown", 0.35, f"Adjuvant type '{raw_type}' is outside the ontology prior table."

    def _score_completeness(self, props: Dict[str, Any], prior: Dict[str, Any]) -> Tuple[float, List[str], int]:
        groups = prior.get("required_any", [])
        recommended = prior.get("recommended_any", [])
        required_hits = sum(1 for group in groups if any(_present(props, f) for f in group))
        recommended_hits = sum(1 for group in recommended if any(_present(props, f) for f in group))

        required_score = required_hits / len(groups) if groups else 1.0
        recommended_score = recommended_hits / len(recommended) if recommended else 1.0
        score = 0.75 * required_score + 0.25 * recommended_score

        cap = 5
        missing_required = [
            "/".join(group) for group in groups if not any(_present(props, f) for f in group)
        ]
        details = [f"completeness={required_hits}/{len(groups)} required, {recommended_hits}/{len(recommended)} recommended"]
        if missing_required:
            cap = 4 if required_hits else 3
            details.append("missing_required=" + ";".join(missing_required))
        return score, details, cap

    def _score_size(self, props: Dict[str, Any], prior: Dict[str, Any], matched_type: str) -> Tuple[float, List[str], int]:
        size_prior = prior.get("size_nm")
        if not size_prior:
            return 1.0, [], 5

        size_value = _first_present(props, ["ParticleSize", "Size"])
        if not size_value:
            cap = int(size_prior.get("missing_cap", 5))
            return 0.65, [f"{matched_type} lacks explicit size/particle_size."], cap

        parsed, qualitative = _parse_size_nm(size_value)
        warnings: List[str] = []
        cap = 5

        if qualitative in size_prior.get("qualitative_bad", []):
            warnings.append(f"{matched_type} has incompatible qualitative size label '{qualitative}'.")
            cap = min(cap, 2)
            return 0.1, warnings, cap
        if parsed is None:
            if qualitative in size_prior.get("qualitative_ok", []):
                return 0.75, [], 5
            warnings.append(f"Could not parse size '{size_value}' for {matched_type}.")
            return 0.55, warnings, int(size_prior.get("missing_cap", 5))

        lo, hi, mean_nm = parsed
        score = _score_range(mean_nm, size_prior["ideal"], size_prior["acceptable"])
        if "hard_cap_above" in size_prior and lo > size_prior["acceptable"][1]:
            cap = min(cap, int(size_prior["hard_cap_above"]))
            warnings.append(f"{matched_type} size {lo:.1f}-{hi:.1f} nm exceeds acceptable range.")
            score = 0.0
        if "hard_cap_below" in size_prior and hi < size_prior["acceptable"][0]:
            cap = min(cap, int(size_prior["hard_cap_below"]))
            warnings.append(f"{matched_type} size {lo:.1f}-{hi:.1f} nm is below acceptable range.")
            score = 0.0
        return score, warnings, cap

    def _score_method(self, props: Dict[str, Any], prior: Dict[str, Any]) -> Tuple[float, List[str], int]:
        method_text = _norm_text(
            " ".join(str(v) for v in _as_list(_first_present(props, ["Method", "PreparationName"])))
            + " "
            + _flatten_text(props.get("PreparationDetails"))
        )
        if not method_text:
            return 0.45, ["Missing preparation method."], 4

        compatible = [_norm_text(m) for m in prior.get("compatible_methods", [])]
        vague = [_norm_text(m) for m in prior.get("vague_methods", [])]
        if any(m in method_text for m in compatible):
            if any(m in method_text for m in vague):
                return 0.65, ["Preparation method is vague/new_prepare; construction is only weakly specified."], 5
            return 1.0, [], 5
        return 0.30, [f"Preparation method '{method_text}' is not typical for this adjuvant type."], 3

    def _score_component_role(self, props: Dict[str, Any], prior: Dict[str, Any], matched_type: str) -> Tuple[float, List[str], int]:
        role_prior = prior.get("component_role", {})
        if not role_prior:
            return 1.0, [], 5

        text = _norm_text(" ".join([
            _flatten_text(props.get("Contains")),
            _flatten_text(props.get("ComponentTypes")),
            _flatten_text(props.get("ComponentDetails")),
            _flatten_text(props.get("OilPhase")),
            _flatten_text(props.get("WaterPhase")),
            _flatten_text(props.get("Salt")),
            _flatten_text(props.get("Polymer")),
            _flatten_text(props.get("Crosslinker")),
            _flatten_text(props.get("NeedleMaterial")),
            _flatten_text(props.get("Target")),
            _flatten_text(props.get("FunctionalGroups")),
        ]))
        component_types = {_norm_text(t) for t in _as_list(props.get("ComponentTypes"))}

        for field_name in role_prior.get("also_accept_fields", []):
            if _present(props, field_name):
                return 1.0, [], 5

        accepted_types = {_norm_text(t) for t in role_prior.get("also_accept_component_types", [])}
        if accepted_types and component_types.intersection(accepted_types):
            return 1.0, [], 5

        for token_set_name in role_prior.get("requires_any_token_set", []):
            tokens = self.config["component_token_sets"].get(token_set_name, [])
            if any(_norm_text(token) in text for token in tokens):
                return 1.0, [], 5

        cap = int(role_prior.get("missing_cap", 4))
        return (
            0.35,
            [f"{matched_type} lacks expected component-role evidence for ontology prior."],
            cap,
        )

    def _score_zeta(self, props: Dict[str, Any]) -> Tuple[float, List[str]]:
        zeta = _first_present(props, ["ZetaPotential", "Zeta"])
        if not zeta:
            return 0.75, []
        abs_mv = _parse_zeta_mv(zeta)
        if abs_mv is None:
            return 0.60, [f"Could not parse zeta potential '{zeta}'."]
        if abs_mv <= 40:
            return 1.0, []
        if abs_mv <= 60:
            return 0.70, [f"Zeta potential magnitude {abs_mv:.1f} mV is high but still plausible."]
        return 0.40, [f"Zeta potential magnitude {abs_mv:.1f} mV is unusually high."]

    def _check_global_patterns(self, props: Dict[str, Any], matched_type: str) -> Tuple[List[str], int]:
        warnings: List[str] = []
        cap = 5
        text = _norm_text(" ".join([
            _flatten_text(props.get("Contains")),
            _flatten_text(props.get("ComponentDetails")),
        ]))
        method_text = _norm_text(" ".join([
            _flatten_text(props.get("Method")),
            _flatten_text(props.get("PreparationName")),
            _flatten_text(props.get("PreparationDetails")),
        ]))

        for pattern in self.config.get("global_incompatible_patterns", []):
            if matched_type not in pattern.get("type_any", []):
                continue
            if not all(_norm_text(tok) in text for tok in pattern.get("component_all_tokens", [])):
                continue
            method_any = [_norm_text(m) for m in pattern.get("method_any", [])]
            if method_any and not any(m in method_text for m in method_any):
                continue
            warnings.append(pattern.get("warning", pattern.get("name", "Global incompatible pattern matched.")))
            cap = min(cap, int(pattern.get("cap", 3)))
        return warnings, cap
