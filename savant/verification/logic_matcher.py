"""Three-stage evidence matcher for SAVANT."""


import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from pathlib import Path

import sys
import os
import math
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from savant.grounding.predicate_core import Predicate, Argument, create_argument
from savant.grounding.unit_utils import PhysicalQuantity, UnitParser
from savant.grounding.functor_vocabulary import STANDARD_FUNCTORS, FUNCTOR_SYNONYMS
from savant.verification.predicate_schema import PredicateSchemaRegistry
from savant.verification.stage1_prior_checker import Stage1PriorChecker
from savant.verification.stage2_endpoint_ontology import Stage2EndpointOntology
from savant.verification.vaxjo_knowledge import VaxjoKnowledge

logger = logging.getLogger(__name__)


def _pred_repr(pred_dict: Dict) -> str:

    if not pred_dict:
        return "?"
    functor = pred_dict.get("functor", "?")
    args = pred_dict.get("args", [])
    arg_parts = []
    for a in args:
        role  = a.get("role", "")
        value = a.get("value", "?")
        if value == "?":
            continue
        arg_parts.append(f"{role}={value}" if role else str(value))
    return f"{functor}({', '.join(arg_parts)})" if arg_parts else functor


def _rule_to_evidence(rule: Dict, **kwargs) -> "Evidence":

    conds = rule.get("conditions", [])
    cond_reprs = [_pred_repr(c) for c in conds] if conds else []
    if not cond_reprs:

        single_cond = rule.get("condition")
        if single_cond:
            cond_reprs = [_pred_repr(single_cond)]
    conditions_repr = " AND ".join(cond_reprs) if cond_reprs else "?"
    effect_repr     = _pred_repr(rule.get("effect", {}))
    arrow_type      = rule.get("arrow_type", ">>")
    evidence_text   = rule.get("evidence_text", "")

    import re as _re

    evidence_text = _re.sub(r'\s*\[co_conds=\[[^\]]*\]\]', '', evidence_text).strip()


    evidence_text = _re.sub(
        r'\s*&\s*\w[\w\-/]*\s*\([^)]*\)\s*$', '', evidence_text
    ).strip()
    evidence_text = _re.sub(r'\s*&\s*\S.*$', '', evidence_text).strip()

    evidence_text = _re.sub(r'[\s,;]+$', '', evidence_text).strip()

    if len(evidence_text) > 180:
        evidence_text = evidence_text[:177] + "..."

    return Evidence(
        rule_id          = rule.get("rule_id", ""),
        source_paper     = rule.get("paper_id", ""),
        confidence       = rule.get("confidence", 0.9),
        similarity_score = rule.get("_similarity_score", 0.0),
        match_type       = kwargs.get("match_type", "semantic"),
        matched_content  = kwargs.get("matched_content", ""),
        chain_id         = rule.get("chain_id", ""),
        experiment_id    = rule.get("experiment_id", ""),
        step_in_chain    = rule.get("step_in_chain", 0),
        conditions_repr  = conditions_repr,
        effect_repr      = effect_repr,
        arrow_type       = arrow_type,
        evidence_text    = evidence_text,
        source_type      = rule.get("source_type", ""),
    )


def _split_dsl_chain(chain_dsl: str) -> List[str]:
    norm = str(chain_dsl).replace('→', '>>').replace('-->', '>>')
    norm = re.sub(r'(?<!-)->', '>>', norm)
    return [p.strip() for p in norm.split('>>') if p.strip()]


def _node_text_to_pred_dict(node_text: str) -> Optional[Dict[str, Any]]:
    text = str(node_text).strip()
    m = re.match(r'^(\w[\w\-/]*)\s*\((.*)?\)\s*$', text)
    if not m:
        return None
    value = re.sub(r"\s+", " ", (m.group(2) or "").strip())
    return {
        "functor": m.group(1),
        "args": [
            {"role": "agent", "value": "?", "value_type": "variable"},
            {"role": "target", "value": value or "?", "value_type": "concept"},
        ],
    }


@dataclass
class Evidence:

    rule_id: str
    source_paper: str
    confidence: float
    similarity_score: float
    match_type: str
    matched_content: str = ""
    chain_id: str = ""
    experiment_id: str = ""
    step_in_chain: int = 0

    conditions_repr: str = ""
    effect_repr: str = ""
    arrow_type: str = ">>"
    evidence_text: str = ""
    source_type: str = ""


@dataclass
class StageResult:

    stage: int
    score: float
    evidence: List[Evidence] = field(default_factory=list)
    status: str = "unknown"
    detail: str = ""
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchResult:

    stage1: StageResult
    stage2: StageResult
    stage3: StageResult
    overall_score: float = 0.0
    verdict: str = "unknown"

    def compute_overall(self) -> None:


        w1, w2, w3 = _stage_weights()
        self.overall_score = (
            w1 * self.stage1.score
            + w2 * self.stage2.score
            + w3 * self.stage3.score
        )
        if self.overall_score >= 4.0:
            self.verdict = "VERIFIED"
        elif self.overall_score >= 3.0:
            self.verdict = "PARTIAL"
        elif self.overall_score >= 2.0:
            self.verdict = "HYPOTHESIS"
        else:
            self.verdict = "UNVERIFIED"


def _stage_weights() -> Tuple[float, float, float]:

    raw = os.getenv("SAVANT_STAGE_WEIGHTS", "0.30,0.30,0.40")
    try:
        parts = [float(x.strip()) for x in raw.split(",")]
        if len(parts) != 3 or any(x < 0 for x in parts):
            raise ValueError
        total = sum(parts)
        if total <= 0:
            raise ValueError
        return tuple(x / total for x in parts)
    except Exception:
        return (0.30, 0.30, 0.40)


class ProxyFunctorResolver:


    THRESHOLD = 0.75


    SYNONYMS: Dict[str, str] = FUNCTOR_SYNONYMS

    def __init__(self, embedding_model=None, kb_functors: Optional[List[str]] = None):
        self._model = embedding_model
        self._standard_embeddings: Optional[np.ndarray] = None
        self._cache: Dict[str, Tuple[str, float, str]] = {}

        kb_functors = kb_functors or []
        self._kb_functors_set: set = set(kb_functors)


        case_map: Dict[str, str] = {}
        for f in STANDARD_FUNCTORS:
            case_map[f.lower()] = f
        for f in kb_functors:
            case_map[f.lower()] = f

        self._case_map = case_map
        self._standard_list: List[str] = sorted(set(case_map.values()))

        if self._model:
            self._build_standard_embeddings()

    def _build_standard_embeddings(self):

        try:
            vecs = self._model.encode(
                self._standard_list, normalize_embeddings=True,
                show_progress_bar=False
            )
            self._standard_embeddings = np.array(vecs)
            logger.info('Runtime diagnostic.')
        except Exception as e:
            logger.warning('Runtime diagnostic.')
            self._standard_embeddings = None

    def _canonical(self, functor: str) -> str:

        return self._case_map.get(functor.lower(), functor)

    def resolve(self, functor: str) -> Tuple[str, float, str]:

        if functor in self._cache:
            return self._cache[functor]

        canonical = self._canonical(functor)


        if canonical in self._standard_list:

            if canonical in self._kb_functors_set:
                self._cache[functor] = (canonical, 1.0, "exact")
                return canonical, 1.0, "exact"

            lower = functor.lower()
            if lower in self.SYNONYMS:
                mapped = self._canonical(self.SYNONYMS[lower])
                self._cache[functor] = (mapped, 0.9, "synonym")
                return mapped, 0.9, "synonym"
            self._cache[functor] = (canonical, 1.0, "exact")
            return canonical, 1.0, "exact"


        lower = functor.lower()
        if lower in self.SYNONYMS:
            mapped = self._canonical(self.SYNONYMS[lower])
            self._cache[functor] = (mapped, 0.9, "synonym")
            return mapped, 0.9, "synonym"


        if self._model and self._standard_embeddings is not None:
            try:
                query_vec = self._model.encode(
                    [functor], normalize_embeddings=True, show_progress_bar=False
                )
                scores = (self._standard_embeddings @ query_vec.T).flatten()
                best_idx = int(np.argmax(scores))
                best_score = float(scores[best_idx])
                if best_score >= self.THRESHOLD:
                    best_functor = self._standard_list[best_idx]
                    self._cache[functor] = (best_functor, best_score, "semantic")
                    return best_functor, best_score, "semantic"
            except Exception as e:
                logger.debug('Runtime diagnostic.')


        self._cache[functor] = (functor, 0.0, "unknown")
        return functor, 0.0, "unknown"


def _dict_to_predicate(pred_dict: Dict) -> Optional["Predicate"]:

    if not pred_dict or 'functor' not in pred_dict:
        return None
    functor = pred_dict['functor']
    args_data = pred_dict.get('args', [])
    args = []
    for a in args_data:
        role  = a.get('role', 'arg')
        value = a.get('value', '?')
        vtype = a.get('value_type', 'concept')
        args.append(create_argument(role, str(value), vtype))
    return Predicate(functor=functor, args=args, source="knowledge_base")


def _extract_functor_and_values(pred_dict: Dict) -> tuple:

    if not pred_dict:
        return ("", [])
    functor = pred_dict.get('functor', '')
    values  = [str(a.get('value', '')) for a in pred_dict.get('args', [])
               if a.get('value') not in ('?', '', None)
               and a.get('value_type') != 'variable']
    return (functor, values)


def _extract_functor_and_roles(pred_dict: Dict) -> Tuple[str, Dict[str, str]]:

    if not pred_dict:
        return ("", {})
    functor = pred_dict.get('functor', '')
    roles: Dict[str, str] = {}
    for a in pred_dict.get('args', []):
        val = a.get('value')
        role = a.get('role', '')
        if val not in ('?', '', None) and a.get('value_type') != 'variable' and role:
            roles[role] = str(val)
    return (functor, roles)


def _roles_to_flat(roles: Dict[str, str]) -> List[str]:

    return list(roles.values())


class SymmetricMatcher:


    def __init__(self, knowledge_db, proxy_resolver: "ProxyFunctorResolver",
                 embedding_model=None,
                 knowledge_registry: Optional[PredicateSchemaRegistry] = None,
                 rule_effect_instance: Optional[Dict[str, Any]] = None,
                 rule_cond_instance: Optional[Dict[str, Any]] = None):
        self.kb      = knowledge_db
        self.proxy   = proxy_resolver
        self.model   = embedding_model
        self.knowledge_registry = knowledge_registry
        self._rule_effect_instance = rule_effect_instance or {}
        self._rule_cond_instance = rule_cond_instance or {}


        self._value_embeddings: Dict[str, np.ndarray] = {}
        if self.model is not None and hasattr(knowledge_db, 'rules_metadata'):
            self._precompute_kb_embeddings()

    def _precompute_kb_embeddings(self):

        values_to_embed: set = set()
        for rule in self.kb.rules_metadata.values():

            for arg in rule.get('effect', {}).get('args', []):
                v = str(arg.get('value', '')).strip()
                if v and v != '?' and arg.get('value_type') != 'variable':
                    qty = self._try_parse_quantity(v)
                    if qty is None and not self._is_biological_entity(v):
                        values_to_embed.add(v)

            for cond in rule.get('conditions', []):
                for arg in cond.get('args', []):
                    v = str(arg.get('value', '')).strip()
                    if v and v != '?' and arg.get('value_type') != 'variable':
                        qty = self._try_parse_quantity(v)
                        if qty is None and not self._is_biological_entity(v):
                            values_to_embed.add(v)

        if values_to_embed:
            values_list = sorted(values_to_embed)
            logger.info('Runtime diagnostic.')
            try:
                vecs = self.model.encode(
                    values_list, normalize_embeddings=True,
                    show_progress_bar=False
                )
                for i, v in enumerate(values_list):
                    self._value_embeddings[v] = vecs[i]
                logger.info('Runtime diagnostic.')
            except Exception as e:
                logger.warning('Runtime diagnostic.')
                self._value_embeddings = {}


    def _positional_match(self, query_pred: Dict, rule_pred: Dict) -> float:

        if not query_pred or not rule_pred:
            return 0.0

        q_functor = query_pred.get('functor', '')
        r_functor = rule_pred.get('functor', '')


        if q_functor != r_functor:
            fs = self._functor_score(q_functor, r_functor)
            if fs > 0.2:
                _, q_vals = _extract_functor_and_values(query_pred)
                _, r_vals = _extract_functor_and_values(rule_pred)
                vs = self._value_sim(q_vals, r_vals) if q_vals and r_vals else 0.0
                return fs * vs
            return 0.0


        schema_cls = None
        if self.knowledge_registry is not None:
            schema_cls = self.knowledge_registry.get_schema(q_functor)
            if schema_cls is None:
                schema_cls = self.knowledge_registry.ensure_schema_from_dict(query_pred)
            if schema_cls is None and rule_pred:
                schema_cls = self.knowledge_registry.ensure_schema_from_dict(rule_pred)
        if schema_cls is None:
            from savant.verification.auto_schemas import get_schema
            schema_cls = get_schema(q_functor)

        if schema_cls is None:

            _, q_vals = _extract_functor_and_values(query_pred)
            _, r_vals = _extract_functor_and_values(rule_pred)
            return self._value_sim(q_vals, r_vals) if q_vals and r_vals else 0.0


        from dataclasses import fields
        field_names = [f.name for f in fields(schema_cls)]


        q_args = query_pred.get('args', [])
        q_values = [str(a.get('value', '')) for a in q_args
                    if a.get('value') not in ('?', '', None)
                    and a.get('value_type') != 'variable']


        r_args = rule_pred.get('args', [])
        r_values = [str(a.get('value', '')) for a in r_args
                    if a.get('value') not in ('?', '', None)
                    and a.get('value_type') != 'variable']


        sims = []
        value_idx = 0
        for field_name in field_names:
            if field_name == 'agent':
                continue

            qv = q_values[value_idx] if value_idx < len(q_values) else None
            rv = r_values[value_idx] if value_idx < len(r_values) else None
            value_idx += 1

            if qv is None or rv is None:
                continue

            qv_str = str(qv).strip()
            rv_str = str(rv).strip()
            if not qv_str or not rv_str:
                continue

            if qv_str.lower() == rv_str.lower():
                sims.append(1.0)
            else:
                sim = self._value_sim([qv_str], [rv_str])
                sims.append(sim)

        if not sims:
            return 0.0
        return sum(sims) / len(sims)


    def match_step(self,
                   cond_text: str,
                   eff_text:  str,
                   source_types: List[str] = None) -> Tuple[float, Optional[dict]]:

        source_types = source_types or ['causal_chain', 'immune_effect']


        q_cond_functor, q_cond_values = self._parse_node(cond_text)
        q_eff_functor,  q_eff_values  = self._parse_node(eff_text)


        resolved_eff,  eff_proxy_score,  _ = self.proxy.resolve(q_eff_functor)
        resolved_cond, cond_proxy_score, _ = self.proxy.resolve(q_cond_functor)


        candidates = []


        if q_eff_values:
            c = self.kb.search_by_eff_value(resolved_eff, q_eff_values[0])
            candidates.extend([r for r in c if r.get('source_type') in source_types])
        if len(candidates) < 3:
            candidates.extend(self._get_candidates_by_eff(resolved_eff, source_types))


        if q_cond_values:
            c = self.kb.search_by_cond_value(resolved_cond, q_cond_values[0])
            candidates.extend([r for r in c if r.get('source_type') in source_types])
        if len(candidates) < 6:
            candidates.extend(self._get_candidates_by_cond(resolved_cond, source_types))


        seen = set()
        unique_candidates = []
        for r in candidates:
            rid = r.get('rule_id') or r.get('id')
            if rid is not None and rid not in seen:
                seen.add(rid)
                unique_candidates.append(r)
            elif rid is None:
                unique_candidates.append(r)

        if not unique_candidates:
            return 0.0, None


        scored = []
        for rule in unique_candidates:
            s_forward = self._score_rule(rule,
                                         resolved_eff,  eff_proxy_score,  q_eff_values,
                                         resolved_cond, cond_proxy_score, q_cond_values)
            s_reverse = self._score_rule(rule,
                                         resolved_cond, cond_proxy_score, q_cond_values,
                                         resolved_eff,  eff_proxy_score,  q_eff_values)
            s = max(s_forward, s_reverse)
            scored.append((s, rule))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_rule = scored[0]
        return best_score, best_rule

    def match_predicate(self,
                        query_pred: "Predicate",
                        source_types: List[str] = None) -> Tuple[float, Optional[dict]]:

        source_types = source_types or ['immune_effect', 'causal_chain']

        resolved_eff, eff_proxy_score, _ = self.proxy.resolve(query_pred.functor)
        q_eff_values = [str(a.value) for a in query_pred.args
                        if a.value not in ('?', '') and not a.is_variable()]


        if q_eff_values:
            candidates = self.kb.search_by_eff_value(resolved_eff, q_eff_values[0])
            candidates = [r for r in candidates if r.get('source_type') in source_types]
        else:
            candidates = []


        if not candidates:
            candidates = self._get_candidates_by_eff(resolved_eff, source_types)

        if not candidates:
            return 0.0, None

        scored = []
        for rule in candidates:
            r_eff_dict = rule.get('effect', {})
            r_eff_functor = r_eff_dict.get('functor', '')

            ef_score = self._functor_score(resolved_eff, r_eff_functor)
            if ef_score == 0.0:
                continue


            q_eff_pred = {
                'functor': resolved_eff,
                'args': [
                    {'role': 'agent', 'value': '?', 'value_type': 'variable'},
                ] + [{'role': 'value', 'value': v, 'value_type': 'concept'} for v in q_eff_values]
            }
            ev_score = self._positional_match(q_eff_pred, r_eff_dict)


            total = ef_score * ev_score
            scored.append((total, rule))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0] if scored else (0.0, None)


    def _get_candidates_by_eff(self, eff_functor: str,
                                source_types: List[str]) -> List[dict]:

        rules = self.kb.get_rules_by_eff_functor(eff_functor)
        if not rules:

            all_keys = self.kb.get_all_eff_functors()
            key_map = {k.lower(): k for k in all_keys}
            actual = key_map.get(eff_functor.lower())
            if actual:
                rules = self.kb.get_rules_by_eff_functor(actual)
        return [r for r in rules if r.get('source_type') in source_types]

    def _get_candidates_by_cond(self, cond_functor: str,
                                 source_types: List[str]) -> List[dict]:

        rules = self.kb.get_rules_by_cond_functor(cond_functor)
        if not rules:
            all_keys = self.kb.get_all_cond_functors()
            key_map = {k.lower(): k for k in all_keys}
            actual = key_map.get(cond_functor.lower())
            if actual:
                rules = self.kb.get_rules_by_cond_functor(actual)
        return [r for r in rules if r.get('source_type') in source_types]

    def _faiss_fallback(self, query_text: str,
                        source_types: List[str], top_k: int = 8) -> List[dict]:

        raw = self.kb.search_relevant_rules(query_text, top_k=top_k * 2, threshold=0.30)
        return [r for r in raw if r.get('source_type') in source_types][:top_k]

    def _score_rule(self, rule: dict,
                    resolved_eff: str,  eff_proxy_score: float,
                    q_eff_values: Union[List[str], Dict[str, str]],
                    resolved_cond: str, cond_proxy_score: float,
                    q_cond_values: Union[List[str], Dict[str, str]]
                    ) -> float:


        conds = rule.get('conditions') or []
        r_cond_dict = conds[0] if conds else {}
        r_eff_dict = rule.get('effect', {})


        r_eff_functor = r_eff_dict.get('functor', '')
        r_cond_functor = r_cond_dict.get('functor', '')

        ef_score = self._functor_score(resolved_eff, r_eff_functor)
        if ef_score == 0.0:
            return 0.0

        cf_score = self._functor_score(resolved_cond, r_cond_functor)
        if cf_score == 0.0:
            return 0.0


        q_eff_flat = list(q_eff_values.values()) if isinstance(q_eff_values, dict) else q_eff_values
        q_eff_pred = {
            'functor': resolved_eff,
            'args': [
                {'role': 'agent', 'value': '?', 'value_type': 'variable'},
            ] + [{'role': 'value', 'value': v, 'value_type': 'concept'} for v in q_eff_flat]
        }
        ev_score = self._positional_match(q_eff_pred, r_eff_dict)


        q_cond_flat = list(q_cond_values.values()) if isinstance(q_cond_values, dict) else q_cond_values
        q_cond_pred = {
            'functor': resolved_cond,
            'args': [
                {'role': 'agent', 'value': '?', 'value_type': 'variable'},
            ] + [{'role': 'value', 'value': v, 'value_type': 'concept'} for v in q_cond_flat]
        }
        cv_score = self._positional_match(q_cond_pred, r_cond_dict)


        value_score = (0.5 * ev_score + 0.5 * cv_score)

        functor_decay = min(eff_proxy_score, cond_proxy_score)
        return value_score * functor_decay

    def _functor_score(self, q_functor: str, r_functor: str) -> float:

        if not r_functor:
            return 0.0
        if q_functor == r_functor:
            return 1.0
        if q_functor.lower() == r_functor.lower():
            return 0.95

        mapped, score, _ = self.proxy.resolve(r_functor)
        if mapped == q_functor and score > 0:
            return score
        return 0.2


    _BIO_ENTITIES: Set[str] = {
        'TLR2', 'TLR3', 'TLR4', 'TLR5', 'TLR7', 'TLR8', 'TLR9', 'TLR7_8', 'TLR7/8', 'TLR7-8',
        'CD3', 'CD4', 'CD8', 'CD11b', 'CD11c', 'CD14', 'CD16',
        'CD19', 'CD20', 'CD28', 'CD40', 'CD45', 'CD56', 'CD69',
        'CD80', 'CD86', 'CD163',
        'IL-1', 'IL-2', 'IL-4', 'IL-6', 'IL-10', 'IL-12', 'IL-17',
        'IL-1β', 'TNF-α', 'IFN-γ', 'IFN-α', 'IFN-β', 'TGF-β',
        'GM-CSF', 'G-CSF', 'M-CSF',
        'IgG', 'IgA', 'IgM', 'IgE', 'IgD',
        'Th1', 'Th2', 'Th17', 'Treg', 'Tregs',
        'C3a', 'C3b', 'C4a', 'C4b', 'C5a', 'C5b', 'C1q',
        'SARS-CoV-2', 'COVID-19', 'HIV-1', 'HIV-2',
        'H1N1', 'H3N2', 'H5N1', 'H7N9',
        'HPV-16', 'HPV-18', 'HSV-1', 'HSV-2',
        'MHC-I', 'MHC-II', 'HLA-A', 'HLA-B', 'HLA-C', 'HLA-DR',
        'B7-1', 'B7-2', 'CTLA-4', 'PD-1', 'PD-L1', 'PD-L2',
        'LAG-3', 'TIM-3', 'ICOS', 'ICOSL', 'OX40', 'OX40L',
        '4-1BB', 'GITR', 'GITRL', 'TNF-R1', 'TNF-R2',
        'Fas', 'FasL', 'TRAIL', 'TRAIL-R',
        'CXCL8', 'CCL2', 'CCL5', 'CXCL10',
    }

    @classmethod
    def _is_biological_entity(cls, text: str) -> bool:

        if not text:
            return False
        text_proc = text.upper().replace('_', '-')
        for entity in cls._BIO_ENTITIES:
            idx = text_proc.find(entity.upper())
            if idx == -1:
                continue

            before_ok = (idx == 0) or (not text_proc[idx - 1].isalnum())
            after_pos = idx + len(entity)
            after_ok = (after_pos >= len(text_proc)) or (not text_proc[after_pos].isalnum())
            if before_ok and after_ok:
                return True
        return False


    _tokenizer = None

    @classmethod
    def _get_tokenizer(cls):
        if cls._tokenizer is None:
            from transformers import AutoTokenizer
            model_name_or_path = os.getenv(
                "SAVANT_TOKENIZER_MODEL",
                os.getenv("SAVANT_EMBEDDING_MODEL", "FremyCompany/BioLORD-2023-M"),
            )
            try:
                kwargs = {"local_files_only": True} if Path(model_name_or_path).exists() else {}
                cls._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
            except Exception:
                cls._tokenizer = False
        return None if cls._tokenizer is False else cls._tokenizer

    @classmethod
    def _filter_biological_numbers(cls, text: str) -> str:

        if not text or not any(c.isdigit() for c in text):
            return text

        try:
            tokenizer = cls._get_tokenizer()
            encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        except Exception:
            return text

        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        offsets = encoding['offset_mapping']

        mask_regions = []
        for tok, (start, end) in zip(tokens, offsets):
            clean = tok.replace('##', '').replace('▁', '')


            if not re.match(r'^-?\d+(\.\d+)?$', clean):
                continue

            char_before = text[start - 1] if start > 0 else ''
            char_after = text[end] if end < len(text) else ''


            if re.match(r'[a-zA-Z]', char_before) or re.match(r'[a-zA-Z]', char_after):

                unit_chars = {'µ', 'μ', 'g', 'm', 'l', 'n', '%', '°', 'c', 'h'}
                if char_after.lower() not in unit_chars:
                    mask_regions.append((start, end))
                    continue


            if char_before == '-':
                if start >= 2 and re.match(r'[a-zA-Z]', text[start - 2]):
                    mask_regions.append((start, end))
                    continue

        if not mask_regions:
            return text


        mask_regions.sort()
        merged = [mask_regions[0]]
        for s, e in mask_regions[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))


        result = list(text)
        for s, e in merged:
            for j in range(s, min(e, len(result))):
                if result[j].isdigit() or result[j] == '.':
                    result[j] = 'X'
        return ''.join(result)

    @staticmethod
    def _try_parse_quantity(text: str):

        if not text:
            return None
        if SymmetricMatcher._is_biological_entity(text):
            return None
        try:
            from savant.grounding.unit_utils import UnitParser, PhysicalQuantity
            clean = text.strip()

            if clean.lower().startswith('quantity('):
                clean = clean[len('quantity('):].rstrip(')')


            normalized = SymmetricMatcher._normalize_numeric_text(clean)
            working = normalized if normalized != clean else clean


            result = SymmetricMatcher._extract_quantity_substring(working)
            if result is not None:
                return result


            if any(c.isdigit() for c in working):
                result = UnitParser.parse(working)
                if result is not None:
                    return result

            return None
        except Exception:
            return None

    @staticmethod
    def _extract_quantity_substring(text: str):

        import re
        from savant.grounding.unit_utils import PhysicalQuantity

        clean = text.lower().strip()


        UNIT_CATEGORIES = {

            'kg': ('Mass', 'ug', 1e9), 'g': ('Mass', 'ug', 1e6),
            'mg': ('Mass', 'ug', 1000.0), 'µg': ('Mass', 'ug', 1.0),
            'ug': ('Mass', 'ug', 1.0), 'ng': ('Mass', 'ug', 0.001),

            'm': ('Length', 'nm', 1e9), 'cm': ('Length', 'nm', 1e7),
            'mm': ('Length', 'nm', 1e6), 'µm': ('Length', 'nm', 1000.0),
            'um': ('Length', 'nm', 1000.0), 'nm': ('Length', 'nm', 1.0),

            'years': ('Time', 'h', 8760.0), 'year': ('Time', 'h', 8760.0),
            'months': ('Time', 'h', 720.0), 'month': ('Time', 'h', 720.0),
            'weeks': ('Time', 'h', 168.0), 'week': ('Time', 'h', 168.0),
            'days': ('Time', 'h', 24.0), 'day': ('Time', 'h', 24.0),
            'd': ('Time', 'h', 24.0),
            'hours': ('Time', 'h', 1.0), 'hour': ('Time', 'h', 1.0),
            'hr': ('Time', 'h', 1.0), 'h': ('Time', 'h', 1.0),
            'min': ('Time', 'h', 1/60),
            'sec': ('Time', 'h', 1/3600),

            'fold': ('Ratio', 'fold', 1.0), 'times': ('Ratio', 'fold', 1.0),
            'x': ('Ratio', 'fold', 1.0),

            '%': ('Percentage', '%', 1.0), 'percent': ('Percentage', '%', 1.0),

            'celsius': ('Temperature', 'celsius', 1.0), '°c': ('Temperature', 'celsius', 1.0),
        }


        sorted_units = sorted(UNIT_CATEGORIES.keys(), key=len, reverse=True)

        found_unit = None
        found_pos = -1
        for unit in sorted_units:
            pattern = r'(?<![a-zA-Z0-9])' + re.escape(unit) + r'(?![a-zA-Z])'
            m = re.search(pattern, clean)
            if m:
                found_unit = unit
                found_pos = m.start()
                break

        if not found_unit:
            return None


        prefix = clean[:found_pos]


        filtered_prefix = SymmetricMatcher._filter_biological_numbers(prefix)


        last_region_match = re.search(
            r'(\d+(?:\.\d+)?(?:\s*[-–~]\s*\d+(?:\.\d+)?)?)\s*$',
            filtered_prefix
        )
        if last_region_match:
            numbers = re.findall(r'(\d+(?:\.\d+)?)', last_region_match.group(1))
        else:
            numbers = []
        if not numbers:
            return None

        nums = [float(n) for n in numbers]
        avg = sum(nums) / len(nums)

        category, base_unit, multiplier = UNIT_CATEGORIES[found_unit]


        if category == 'Temperature':
            from savant.grounding.unit_utils import UnitParser
            simplified = f"{avg} {found_unit}"
            return UnitParser.parse_temperature(simplified)

        norm_avg = avg * multiplier
        if len(nums) >= 2:
            min_val = min(nums) * multiplier
            max_val = max(nums) * multiplier
        else:
            min_val = max_val = norm_avg

        return PhysicalQuantity(
            original_text=text,
            min_val=min_val, max_val=max_val, avg_val=norm_avg,
            unit=base_unit, category=category
        )

    @staticmethod
    def _normalize_numeric_text(text: str) -> str:

        import re
        result = text


        WORD_NUMS = {
            'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        }
        for word, num in WORD_NUMS.items():
            result = re.sub(rf'\b{word}\b', num, result, flags=re.IGNORECASE)


        result = result.replace('cmonths', 'months')
        result = result.replace('cmonth', 'month')


        result = re.sub(r'\+\s*(\d)', r'\1', result)

        return result

    @staticmethod
    def _numeric_log_overlap(q_qty, r_qty) -> float:

        try:

            if q_qty.category != r_qty.category:
                return 0.05
            qv = q_qty.avg_val
            rv = r_qty.avg_val
            if qv <= 0 or rv <= 0:
                return 0.5
            log_q = math.log10(qv)
            log_r = math.log10(rv)
            gap = abs(log_q - log_r)
            return math.exp(-2.0 * gap)
        except Exception:
            return 0.5

    def _value_sim_by_role(self, q_roles: Dict[str, str],
                           r_roles: Dict[str, str]) -> float:

        shared_roles = set(q_roles.keys()) & set(r_roles.keys())
        if not shared_roles:
            return self._value_sim(
                _roles_to_flat(q_roles),
                _roles_to_flat(r_roles)
            )
        sims = []
        for role in shared_roles:
            s = self._value_sim([q_roles[role]], [r_roles[role]])
            sims.append(s)
        return float(sum(sims) / len(sims))

    @staticmethod
    def _extract_numbers(text: str) -> List[float]:

        import re
        nums = re.findall(r'\d+\.?\d*', text)
        return [float(n) for n in nums if float(n) > 0]

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:

        if len(s1) < len(s2):
            return SymmetricMatcher._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    @classmethod
    def _string_similarity(cls, s1: str, s2: str) -> float:

        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        distance = cls._levenshtein_distance(s1, s2)
        return 1.0 - distance / max_len


    _BIOLOGY_ONTOLOGY = {

        "toll_like_receptor":       {"members": {"TLR1","TLR2","TLR3","TLR4","TLR5","TLR6","TLR7","TLR8","TLR9","TLR10","TLR11","TLR12","TLR13"}, "penalty": 0.35},
        "nod_like_receptor":        {"members": {"NOD1","NOD2","NLRP1","NLRP3","NLRC4","NAIP"}, "penalty": 0.30},
        "rig_i_like_receptor":      {"members": {"RIG-I","MDA5","LGP2"}, "penalty": 0.30},
        "c_type_lectin_receptor":   {"members": {"Dectin-1","Dectin-2","Mincle"}, "penalty": 0.25},

        "immunoglobulin_class":     {"members": {"IgG","IgA","IgM","IgE","IgD"}, "penalty": 0.35},
        "igg_subclass":             {"members": {"IgG1","IgG2","IgG2a","IgG2b","IgG2c","IgG3","IgG4"}, "penalty": 0.30},
        "iga_subclass":             {"members": {"IgA1","IgA2"}, "penalty": 0.25},

        "t_cell_coreceptor":        {"members": {"CD4","CD8"}, "penalty": 0.35},
        "t_helper_subset":          {"members": {"Th1","Th2","Th17","Treg","Tfh","Th9","Th22"}, "penalty": 0.40},
        "t_cell_activation_marker": {"members": {"CD28","CTLA-4","PD-1","ICOS","CD40L"}, "penalty": 0.25},

        "b_cell_marker":            {"members": {"CD19","CD20","CD21","CD22","CD40","CD79"}, "penalty": 0.25},

        "dendritic_cell_subset":    {"members": {"cDC1","cDC2","pDC","mDC","DC"}, "penalty": 0.30},
        "macrophage_subset":        {"members": {"M1","M2","TAM","Kupffer cell","microglia"}, "penalty": 0.25},
        "professional_apc":         {"members": {"DC","B cell","macrophage","monocyte"}, "penalty": 0.30},

        "granulocyte":              {"members": {"neutrophil","eosinophil","basophil","mast cell"}, "penalty": 0.25},
        "innate_lymphoid_cell":     {"members": {"NK cell","ILC1","ILC2","ILC3","LTi"}, "penalty": 0.25},
        "myeloid_cell":             {"members": {"monocyte","macrophage","neutrophil","eosinophil","basophil","mast cell","DC"}, "penalty": 0.25},
        "lymphocyte":               {"members": {"T cell","B cell","NK cell"}, "penalty": 0.25},

        "interferon":               {"members": {"IFN-α","IFN-β","IFN-γ","IFN-λ"}, "penalty": 0.30},
        "interleukin_th1":          {"members": {"IL-2","IL-12","IL-15","IL-18","IL-27"}, "penalty": 0.30},
        "interleukin_th2":          {"members": {"IL-4","IL-5","IL-10","IL-13","IL-25","IL-33"}, "penalty": 0.30},
        "interleukin_th17":         {"members": {"IL-17A","IL-17F","IL-21","IL-22","IL-23"}, "penalty": 0.30},
        "interleukin_proinflammatory": {"members": {"IL-1β","IL-1α","IL-6","IL-8","TNF-α","TNF-β"}, "penalty": 0.30},
        "growth_factor_cytokine":   {"members": {"TGF-β","GM-CSF","G-CSF","M-CSF","SCF"}, "penalty": 0.25},

        "chemokine_cxcl":           {"members": {"CXCL8","CXCL9","CXCL10","CXCL11","CXCL12","CXCL13"}, "penalty": 0.25},
        "chemokine_ccl":            {"members": {"CCL2","CCL3","CCL4","CCL5","CCL19","CCL21","CCL22"}, "penalty": 0.25},

        "mineral_salt_adjuvant":    {"members": {"alum","aluminum salt","aluminum hydroxide","aluminum phosphate","calcium phosphate"}, "penalty": 0.25},
        "oil_in_water_emulsion":    {"members": {"MF59","AS03","AF03"}, "penalty": 0.25},
        "tlr_agonist_adjuvant":     {"members": {"CpG","Poly(I:C)","MPLA","imiquimod","resiquimod","Pam3CSK4"}, "penalty": 0.30},
        "saponin_adjuvant":         {"members": {"QS-21","ISCOM","Matrix-M"}, "penalty": 0.25},

        "particulate_carrier":      {"members": {"liposome","nanoparticle","micelle","emulsion","virosome","niosome"}, "penalty": 0.25},
        "polymeric_carrier":        {"members": {"PLGA","PLA","PEG","chitosan","dextran","hyaluronic acid"}, "penalty": 0.25},
        "inorganic_nanoparticle":   {"members": {"gold nanoparticle","silica nanoparticle","iron oxide nanoparticle","mesoporous silica"}, "penalty": 0.25},

        "lymphoid_organ":           {"members": {"lymph nodes","spleen","thymus","bone marrow","Peyer patch","tonsil","appendix"}, "penalty": 0.25},
        "mucosal_tissue":           {"members": {"gut","intestine","colon","lung","skin","nasal cavity","oral cavity","vagina"}, "penalty": 0.25},
        "systemic_organ":           {"members": {"liver","kidney","heart","brain","muscle","blood"}, "penalty": 0.20},

        "mapk_subfamily":           {"members": {"ERK","JNK","p38"}, "penalty": 0.25},
        "pi3k_akt_axis":            {"members": {"PI3K","Akt","mTOR"}, "penalty": 0.25},
        "jak_stat_pathway":         {"members": {"JAK1","JAK2","JAK3","TYK2","STAT1","STAT3","STAT4","STAT6"}, "penalty": 0.25},
        "innate_sensor_pathway":    {"members": {"NF-κB","IRF3","IRF7","cGAS-STING"}, "penalty": 0.25},

        "programmed_cell_death":    {"members": {"apoptosis","necrosis","pyroptosis","autophagy","ferroptosis","necroptosis"}, "penalty": 0.25},

        "protein_antigen":          {"members": {"protein","peptide","fusion protein","recombinant protein"}, "penalty": 0.20},
        "subunit_vaccine_platform": {"members": {"VLP","RBD","spike protein","subunit","virus-like particle"}, "penalty": 0.25},
        "model_antigen":            {"members": {"OVA","ovalbumin","HBsAg","HEL","KLH"}, "penalty": 0.20},
    }

    _VALUE_TO_FAMILIES: Dict[str, Set[str]] = {}
    for _fam_name, _info in _BIOLOGY_ONTOLOGY.items():
        for _member in _info["members"]:
            _VALUE_TO_FAMILIES.setdefault(_member.lower(), set()).add(_fam_name)

    @classmethod
    def _ontology_penalty(cls, val_a: str, val_b: str) -> float:

        a = val_a.strip().lower()
        b = val_b.strip().lower()
        if a == b:
            return 0.0
        fams_a = cls._VALUE_TO_FAMILIES.get(a, set())
        if not fams_a:
            return 0.0
        fams_b = cls._VALUE_TO_FAMILIES.get(b, set())
        shared = fams_a & fams_b
        if not shared:
            return 0.0
        return max(cls._BIOLOGY_ONTOLOGY[f]["penalty"] for f in shared)

    def _value_sim(self, q_values: List[str], r_values: List[str]) -> float:

        if not q_values or not r_values:
            return 0.5


        if len(q_values) == 1 and len(r_values) == 1:
            q_qty = self._try_parse_quantity(q_values[0])
            r_qty = self._try_parse_quantity(r_values[0])
            if q_qty is not None and r_qty is not None:
                return self._numeric_log_overlap(q_qty, r_qty)
            if (q_qty is None) != (r_qty is None):

                return 0.3

        if self.model is None:
            q_str = ' '.join(q_values).lower()
            r_str = ' '.join(r_values).lower()
            common = sum(1 for w in q_str.split() if w in r_str)
            return min(common / max(len(q_str.split()), 1), 1.0)


        if len(q_values) == 1 and len(r_values) == 1:
            qv = q_values[0].strip()
            rv = r_values[0].strip()
            q_is_bio = self._is_biological_entity(qv)
            r_is_bio = self._is_biological_entity(rv)
            if q_is_bio and r_is_bio:
                return 1.0 if qv == rv else 0.0


        try:
            q_vecs = []
            for v in q_values:
                v = v.strip()
                if v in self._value_embeddings:
                    q_vecs.append(self._value_embeddings[v])
                else:
                    vec = self.model.encode(
                        [v], normalize_embeddings=True,
                        show_progress_bar=False
                    )[0]
                    self._value_embeddings[v] = vec
                    q_vecs.append(vec)

            r_vecs = []
            for v in r_values:
                v = v.strip()
                if v in self._value_embeddings:
                    r_vecs.append(self._value_embeddings[v])
                else:
                    vec = self.model.encode(
                        [v], normalize_embeddings=True,
                        show_progress_bar=False
                    )[0]
                    self._value_embeddings[v] = vec
                    r_vecs.append(vec)

            q_vecs = np.array(q_vecs)
            r_vecs = np.array(r_vecs)
            sims = q_vecs @ r_vecs.T
            best_per_q = sims.max(axis=1)
            emb_score = float(best_per_q.mean())


            if len(q_values) == 1 and len(r_values) == 1 and emb_score > 0.4:
                q_qty = self._try_parse_quantity(q_values[0])
                r_qty = self._try_parse_quantity(r_values[0])
                if q_qty is None and r_qty is None:
                    q_nums = self._extract_numbers(q_values[0])
                    r_nums = self._extract_numbers(r_values[0])
                    if q_nums and r_nums:
                        q_set = set(round(n, 3) for n in q_nums)
                        r_set = set(round(n, 3) for n in r_nums)

                        if not q_set.intersection(r_set):
                            return 0.2

                        if q_set != r_set:
                            jaccard = len(q_set & r_set) / len(q_set | r_set)
                            penalty_score = 0.2 + 0.4 * jaccard
                            if penalty_score < emb_score:
                                return penalty_score


            if len(q_values) == 1 and len(r_values) == 1 and emb_score > 0.70:
                morph = self._string_similarity(q_values[0], r_values[0])
                if 0.35 < morph < 0.92:
                    penalty = 0.40 * (1.0 - morph)
                    emb_score = max(0.0, emb_score - penalty)


            if emb_score > 0.45:
                if len(q_values) == 1 and len(r_values) == 1:
                    onto_penalty = self._ontology_penalty(q_values[0], r_values[0])
                    if onto_penalty > 0:
                        emb_score = max(0.0, emb_score - onto_penalty)
                else:
                    max_penalty = 0.0
                    for qi, q_val in enumerate(q_values):
                        best_ri = int(sims[qi].argmax())
                        r_val = r_values[best_ri]
                        penalty = self._ontology_penalty(q_val, r_val)
                        if penalty > max_penalty:
                            max_penalty = penalty
                    if max_penalty > 0:
                        emb_score = max(0.0, emb_score - max_penalty)

            return emb_score
        except Exception as e:
            logger.debug('Runtime diagnostic.')
            return 0.5

    @staticmethod
    def _parse_node(text: str) -> Tuple[str, List[str]]:

        text = text.strip()

        m_assign = re.match(r'^(\w[\w_]*)\s*=\s*\((.*)\)$', text)
        if m_assign:
            functor = m_assign.group(1).strip()
            val     = m_assign.group(2).strip()
            return (functor, [val] if val else [])

        m_assign = re.match(r'^(\w[\w_]*)\s*=\s*(.+)$', text)
        if m_assign:
            functor = m_assign.group(1).strip()
            val     = m_assign.group(2).strip()
            return (functor, [val] if val else [])

        m_pred = re.match(r'^(\w[\w\-/]*)\s*\((.*)?\)\s*$', text)
        if m_pred:
            functor  = m_pred.group(1).strip()
            args_str = m_pred.group(2) or ''
            values   = [v.strip() for v in args_str.split(',') if v.strip()]
            return (functor, values)

        return ("", [text] if text else [])


class LogicMatcher:


    @staticmethod
    def _build_ontology_literal_constraints() -> Dict[str, Set[str]]:

        import typing
        from dataclasses import fields
        from savant.grounding.ontology import (
            Adjuvant, NanoparticleAdjuvant, Antigen, Vaccine, DiseaseModel
        )

        def _extract_literal(type_hint):
            origin = typing.get_origin(type_hint)
            if origin is typing.Union:
                for arg in typing.get_args(type_hint):
                    result = _extract_literal(arg)
                    if result:
                        return result
            if origin is typing.Literal:
                return set(typing.get_args(type_hint))
            return None


        PLACEHOLDERS = {'Unknown', 'new_prepare'}


        CLASS_FIELD_MAP = {
            Adjuvant: {
                'type': {'Type'},
                'composition_type': {'Method'},
            },
            NanoparticleAdjuvant: {
                'type': {'Type'},
                'composition_type': {'Method'},
                'shape': {'Shape'},


            },
            Antigen: {
                'composition_type': {'AgMethod'},
            },
            Vaccine: {
                'composition_type': {'Method'},
            },
            DiseaseModel: {
                'type': {'DiseaseType'},
            },
        }

        constraints: Dict[str, Set[str]] = {}
        for cls, field_map in CLASS_FIELD_MAP.items():
            for f in fields(cls):
                raw_vals = _extract_literal(f.type)
                if not raw_vals:
                    continue

                vals = {v.lower() for v in raw_vals if v not in PLACEHOLDERS}
                if not vals:
                    continue
                functors = field_map.get(f.name)
                if functors:
                    for func in functors:
                        constraints.setdefault(func, set()).update(vals)


        if 'Type' in constraints:
            constraints['AgType'] = set(constraints['Type'])
        if 'Size' in constraints:
            constraints['AgSize'] = set(constraints['Size'])

        return constraints

    _ONTOLOGY_LITERAL_VALUES: Dict[str, Set[str]] = _build_ontology_literal_constraints()

    def __init__(self,
                 knowledge_db,
                 verifier=None,
                 embedding_model=None,
                 stage3_knowledge_db=None):
        self.kb = knowledge_db
        self.verifier = verifier

        if embedding_model is None and hasattr(knowledge_db, 'model'):
            embedding_model = knowledge_db.model


        kb_functors = []
        if hasattr(knowledge_db, 'get_all_functors'):
            kb_functors.extend(knowledge_db.get_all_functors())
        else:
            if hasattr(knowledge_db, 'get_all_eff_functors'):
                kb_functors.extend(knowledge_db.get_all_eff_functors())
            if hasattr(knowledge_db, 'get_all_cond_functors'):
                kb_functors.extend(knowledge_db.get_all_cond_functors())
        kb_functors = sorted(set(kb_functors))


        self.knowledge_registry = PredicateSchemaRegistry()
        self._rule_effect_instance: Dict[str, Any] = {}
        self._rule_cond_instance: Dict[str, Any] = {}
        if hasattr(knowledge_db, 'rules_metadata'):
            for rule in knowledge_db.rules_metadata.values():
                effect_pred = _dict_to_predicate(rule.get('effect'))
                if effect_pred:
                    inst = self.knowledge_registry.create_instance(effect_pred)
                    if inst:
                        self._rule_effect_instance[rule['rule_id']] = inst
                conds = rule.get('conditions') or []
                if conds:
                    cond_pred = _dict_to_predicate(conds[0])
                    if cond_pred:
                        inst = self.knowledge_registry.create_instance(cond_pred)
                        if inst:
                            self._rule_cond_instance[rule['rule_id']] = inst
        logger.info(
            'Runtime diagnostic.'
            'Runtime diagnostic.'
            'Runtime diagnostic.'
        )


        self._kb_known_values: Dict[str, Set[str]] = {}
        if hasattr(knowledge_db, 'rules_metadata'):
            import re as _re
            for rule in knowledge_db.rules_metadata.values():
                for cond in rule.get('conditions', []):
                    functor = cond.get('functor', '')
                    for arg in cond.get('args', []):
                        if arg.get('value_type') == 'concept':
                            v = str(arg.get('value', '')).strip().lower()

                            v = _re.sub(r'\s+\(.*?\)\s*$', '', v).strip()
                            if v and v != '?':
                                self._kb_known_values.setdefault(functor, set()).add(v)
        logger.info(
            'Runtime diagnostic.'
            'Runtime diagnostic.'
            'Runtime diagnostic.'
        )

        self.proxy_resolver = ProxyFunctorResolver(embedding_model, kb_functors=kb_functors)

        self.sym = SymmetricMatcher(
            knowledge_db, self.proxy_resolver, embedding_model,
            knowledge_registry=self.knowledge_registry,
            rule_effect_instance=self._rule_effect_instance,
            rule_cond_instance=self._rule_cond_instance
        )
        self.stage3_kb = stage3_knowledge_db or knowledge_db
        self.stage3_proxy_resolver = self.proxy_resolver
        self.stage3_sym = self.sym
        if stage3_knowledge_db is not None and stage3_knowledge_db is not knowledge_db:
            stage3_embedding_model = getattr(stage3_knowledge_db, "model", embedding_model)
            stage3_functors: List[str] = []
            if hasattr(stage3_knowledge_db, 'get_all_functors'):
                stage3_functors.extend(stage3_knowledge_db.get_all_functors())
            else:
                if hasattr(stage3_knowledge_db, 'get_all_eff_functors'):
                    stage3_functors.extend(stage3_knowledge_db.get_all_eff_functors())
                if hasattr(stage3_knowledge_db, 'get_all_cond_functors'):
                    stage3_functors.extend(stage3_knowledge_db.get_all_cond_functors())
            stage3_functors = sorted(set(stage3_functors))

            stage3_registry = PredicateSchemaRegistry()
            stage3_rule_effect_instance: Dict[str, Any] = {}
            stage3_rule_cond_instance: Dict[str, Any] = {}
            if hasattr(stage3_knowledge_db, 'rules_metadata'):
                for rule in stage3_knowledge_db.rules_metadata.values():
                    effect_pred = _dict_to_predicate(rule.get('effect'))
                    if effect_pred:
                        inst = stage3_registry.create_instance(effect_pred)
                        if inst:
                            stage3_rule_effect_instance[rule['rule_id']] = inst
                    conds = rule.get('conditions') or []
                    if conds:
                        cond_pred = _dict_to_predicate(conds[0])
                        if cond_pred:
                            inst = stage3_registry.create_instance(cond_pred)
                            if inst:
                                stage3_rule_cond_instance[rule['rule_id']] = inst

            self.stage3_proxy_resolver = ProxyFunctorResolver(
                stage3_embedding_model, kb_functors=stage3_functors
            )
            self.stage3_sym = SymmetricMatcher(
                stage3_knowledge_db, self.stage3_proxy_resolver, stage3_embedding_model,
                knowledge_registry=stage3_registry,
                rule_effect_instance=stage3_rule_effect_instance,
                rule_cond_instance=stage3_rule_cond_instance,
            )
            logger.info(
                'Runtime diagnostic.'
                'Runtime diagnostic.'
                'Runtime diagnostic.'
            )
        self.stage1_prior_checker = Stage1PriorChecker()
        self.vaxjo_knowledge = VaxjoKnowledge()
        self.vaxjo_mode = os.getenv("SAVANT_VAXJO_MODE", "evidence").strip().lower()
        self.stage2_endpoint_ontology = Stage2EndpointOntology()
        self.stage2_policy = os.getenv("SAVANT_STAGE2_POLICY", "broad").strip().lower()
        if self.stage2_policy not in {"broad", "strict"}:
            self.stage2_policy = "broad"
        self.stage2_endpoint_mode = os.getenv("SAVANT_STAGE2_ENDPOINT_MODE", "guarded").strip().lower()
        if self.stage2_endpoint_mode not in {"off", "evidence", "score", "guarded"}:
            self.stage2_endpoint_mode = "guarded"
        self.enable_derived_paths = os.getenv("SAVANT_ENABLE_DERIVED_PATHS", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        self.stage3_support_threshold = float(os.getenv("SAVANT_STAGE3_SUPPORT_THRESHOLD", "0.35"))
        self.stage3_confidence_floor = float(os.getenv("SAVANT_STAGE3_CONFIDENCE_FLOOR", "0.60"))
        self.stage3_confidence_weight = float(os.getenv("SAVANT_STAGE3_CONFIDENCE_WEIGHT", "0.40"))
        self.stage3_confidence_floor = min(max(self.stage3_confidence_floor, 0.0), 1.0)
        self.stage3_confidence_weight = min(max(self.stage3_confidence_weight, 0.0), 1.0)
        self._derived_path_rules: List[dict] = []
        self._derived_by_eff_functor: Dict[str, List[dict]] = {}
        self._derived_by_cond_functor: Dict[str, List[dict]] = {}
        if self.enable_derived_paths:
            derived_path = os.getenv(
                "SAVANT_DERIVED_PATH_INDEX",
                str(Path(_project_root) / "data" / "horn_clauses" / "immunology_round2_derived_path_index.json"),
            )
            self._load_derived_path_index(derived_path)
        logger.info('Runtime diagnostic.')

    def _load_derived_path_index(self, path_text: str) -> None:

        path = Path(path_text)
        if not path.exists():
            logger.warning(f"Derived path index not found: {path}")
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Derived path index load failed: {path}: {e}")
            return

        default_plugin = str(data.get("metadata", {}).get("plugin") or "immunology_round2")
        default_paper_id = f"{default_plugin}_derived_path_index"
        for item in data.get("derived_path_index", []):
            parts = _split_dsl_chain(item.get("path_dsl", ""))
            if len(parts) < 3:
                continue
            cond = _node_text_to_pred_dict(parts[0])
            eff = _node_text_to_pred_dict(parts[-1])
            if not cond or not eff:
                continue
            hops = int(item.get("hops") or (len(parts) - 1))
            rule = {
                "rule_id": item.get("path_id") or f"DERIVED_PATH_{len(self._derived_path_rules) + 1}",
                "experiment_id": f"{str(item.get('plugin') or default_plugin)}_plugin",
                "chain_id": item.get("chain_id", ""),
                "step_in_chain": 0,
                "arrow_type": ">>",
                "conditions": [cond],
                "effect": eff,
                "paper_id": item.get("source_record_file") or default_paper_id,
                "confidence": float(item.get("path_confidence") or 0.75),
                "priority": "derived",
                "source_type": "derived_path_index",
                "evidence_text": item.get("path_dsl", ""),
                "plugin": str(item.get("plugin") or default_plugin),
                "path_hops": hops,
                "storage_role": "derived_index_only",
                "store_as_atomic_theorem": False,
                "path_dsl": item.get("path_dsl", ""),
                "source_record_file": item.get("source_record_file", ""),
                "source_record_line": item.get("source_record_line"),
            }
            self._derived_path_rules.append(rule)
            self._derived_by_eff_functor.setdefault(eff.get("functor", ""), []).append(rule)
            self._derived_by_cond_functor.setdefault(cond.get("functor", ""), []).append(rule)
        logger.info(f"Derived path sidecar loaded: {len(self._derived_path_rules)} paths from {path}")


    def _score_property(self, prop_name: str, prop_val: str,
                         source_type: str = 'construction') -> Tuple[float, Optional[dict]]:

        import re as _re
        val_str = _re.sub(r'\s+\(.*?\)\s*$', '', str(prop_val)).strip()
        if not val_str:
            val_str = str(prop_val).strip()


        candidate_rules = self.kb.search_by_cond_value(prop_name, val_str)
        candidate_rules = [r for r in candidate_rules if r.get('source_type') == source_type]


        if not candidate_rules:
            candidate_rules = [
                r for r in self.kb.get_rules_by_cond_functor(prop_name)
                if r.get('source_type') == source_type
            ]

        if not candidate_rules:
            return 0.0, None

        best_score, best_rule = 0.0, None
        for r in candidate_rules:
            conds = r.get('conditions') or []
            cond_dict = conds[0] if conds else {}
            cond_functor = cond_dict.get('functor', '')

            f_score = self.sym._functor_score(prop_name, cond_functor)
            if f_score == 0.0:
                continue


            v_score = self.sym._positional_match(
                query_pred={'functor': prop_name, 'args': [
                    {'role': 'agent', 'value': '?', 'value_type': 'variable'},
                    {'role': 'value', 'value': val_str, 'value_type': 'concept'}
                ]},
                rule_pred=cond_dict,
            )
            if v_score == 0.0:
                continue


            score = f_score * v_score


            if prop_name in self._ONTOLOGY_LITERAL_VALUES:
                known_vals = self._kb_known_values.get(prop_name, set())
                literal_vals = self._ONTOLOGY_LITERAL_VALUES[prop_name]
                val_lower = val_str.lower()
                if val_lower not in known_vals and val_lower not in literal_vals:
                    score = score * 0.25

            if score > best_score:
                best_score, best_rule = score, r
        return best_score, best_rule

    @staticmethod
    def _stage1_has_value(value: Any) -> bool:
        if value is None or value == "":
            return False
        if isinstance(value, list):
            return any(LogicMatcher._stage1_has_value(v) for v in value)
        if isinstance(value, dict):
            return any(LogicMatcher._stage1_has_value(v) for v in value.values())
        return True

    @staticmethod
    def _stage1_optional_support(scores: List[float]) -> float:

        if not scores:
            return 0.50
        positives = [s for s in scores if s >= 0.30]
        if not positives:
            return 0.35
        return 0.65 * (sum(positives) / len(positives)) + 0.35 * (len(positives) / len(scores))

    @staticmethod
    def _stage1_prior_severity(prior_result: Any) -> Tuple[str, float]:
        cap = int(getattr(prior_result, "cap_5level", 5) or 5)
        warnings = " ".join(getattr(prior_result, "warnings", []) or []).lower()
        severe_tokens = ("incompatible", "exceeds acceptable", "below acceptable", "unusually high")
        moderate_tokens = ("not typical", "missing preparation method", "outside the ontology")

        if cap <= 2 or any(tok in warnings for tok in severe_tokens):
            return "severe", 0.18
        if cap <= 3 or any(tok in warnings for tok in moderate_tokens):
            return "moderate", 0.08
        if cap <= 4 or bool(warnings):
            return "mild", 0.03
        return "none", 0.0

    def _aggregate_stage1_v9(self,
                             adjuvant_properties: Dict[str, Any],
                             antigen_properties: Optional[Dict[str, Any]],
                             adj_prop_scores: Dict[str, float],
                             adj_prop_components: Dict[str, List[float]],
                             ant_prop_scores: Dict[str, float],
                             ant_prop_components: Dict[str, List[float]],
                             prior_result: Any,
                             kb_score: float) -> Tuple[float, Dict[str, Any]]:

        core_specs = (("Type", 0.40), ("Method", 0.35), ("Size", 0.25))
        core_debug: Dict[str, Dict[str, Any]] = {}
        core_score = 0.0
        unknown_core_count = 0
        weak_core_count = 0
        proven_core_count = 0
        missing_size = False

        for prop_name, weight in core_specs:
            present = self._stage1_has_value(adjuvant_properties.get(prop_name))
            if prop_name == "Size" and not present:


                score = 0.45
                missing_size = True
                level = "missing"
            else:
                score = float(adj_prop_scores.get(prop_name, 0.0))
                if score >= 0.65:
                    level = "proven"
                    proven_core_count += 1
                elif score >= 0.30:
                    level = "weak"
                    weak_core_count += 1
                else:
                    level = "unknown"
                    unknown_core_count += 1
            core_score += weight * score
            core_debug[prop_name] = {
                "score": round(score, 6),
                "present": bool(present),
                "level": level,
            }

        contains_components = adj_prop_components.get("Contains", [])
        contains_present = self._stage1_has_value(adjuvant_properties.get("Contains"))
        contains_support = self._stage1_optional_support(contains_components)
        ant_scores_flat = [s for vals in ant_prop_components.values() for s in vals]
        antigen_support = self._stage1_optional_support(ant_scores_flat) if antigen_properties else 0.50

        proof_support = (
            0.86 * core_score
            + 0.10 * contains_support
            + 0.04 * antigen_support
        )
        prior_score = float(getattr(prior_result, "score", 0.0) or 0.0)
        severity, severity_penalty = self._stage1_prior_severity(prior_result)

        legacy_score = 0.60 * float(kb_score) + 0.40 * prior_score
        raw_score = 0.68 * legacy_score + 0.32 * proof_support
        raw_score -= min(0.22, 0.11 * unknown_core_count)
        if weak_core_count >= 2:
            raw_score -= 0.06
        if missing_size:
            raw_score -= 0.05
        raw_score -= severity_penalty

        cap_to_raw = {1: 0.299, 2: 0.499, 3: 0.699, 4: 0.899, 5: 1.0}
        prior_cap = int(getattr(prior_result, "cap_5level", 5) or 5)
        band_cap = cap_to_raw.get(prior_cap, 1.0)
        cap_reasons = []
        uncertainty_reasons = []
        if prior_cap < 5:
            cap_reasons.append(f"prior_cap={prior_cap}")
        if unknown_core_count >= 2:
            uncertainty_reasons.append("multiple_unknown_core")
        elif unknown_core_count == 1 or weak_core_count >= 2:
            uncertainty_reasons.append("weak_or_unknown_core")
        if proven_core_count == 0:
            uncertainty_reasons.append("no_proven_core")
        if severity == "severe":
            uncertainty_reasons.append("severe_prior_guardrail")
        elif severity == "moderate":
            uncertainty_reasons.append("moderate_prior_guardrail")

        high_confidence_ready = (
            core_score >= 0.86
            and proof_support >= 0.82
            and prior_score >= 0.88
            and severity == "none"
            and unknown_core_count == 0
            and weak_core_count == 0
            and not missing_size
            and (not contains_present or contains_support >= 0.55)
        )
        allow_max5 = os.getenv("SAVANT_STAGE1_ALLOW_MAX5", "0").strip().lower() in {
            "1", "true", "yes"
        }
        if not (allow_max5 and high_confidence_ready):
            band_cap = min(band_cap, 0.899)
            cap_reasons.append("high_confidence_gate")


        final_score = max(0.0, min(legacy_score, band_cap, 1.0))

        if severity in {"moderate", "severe"}:
            score_band = "prior_violation"
        elif high_confidence_ready:
            score_band = "proven_core"
        elif unknown_core_count >= 2 or core_score < 0.45:
            score_band = "unknown_blindspot"
        else:
            score_band = "weak_but_plausible"

        debug = {
            "stage1_mode": "v9_minimal_subtheorem_guarded",
            "core_proof_score": round(core_score, 6),
            "proof_support": round(proof_support, 6),
            "prior_guard_score": round(prior_score, 6),
            "legacy_stage1_score": round(float(legacy_score), 6),
            "kb_score_legacy": round(float(kb_score), 6),
            "contains_support": round(contains_support, 6),
            "antigen_support": round(antigen_support, 6),
            "unknown_core_count": int(unknown_core_count),
            "weak_core_count": int(weak_core_count),
            "proven_core_count": int(proven_core_count),
            "prior_severity": severity,
            "high_confidence_ready": bool(high_confidence_ready),
            "allow_max5": bool(allow_max5),
            "guarded_raw_candidate": round(float(raw_score), 6),
            "score_band": score_band,
            "band_cap": round(band_cap, 6),
            "cap_reason": ";".join(cap_reasons),
            "uncertainty_reason": ";".join(uncertainty_reasons),
            "core": core_debug,
            "adj_prop_scores": {k: round(float(v), 6) for k, v in sorted(adj_prop_scores.items())},
            "ant_prop_scores": {k: round(float(v), 6) for k, v in sorted(ant_prop_scores.items())},
        }
        return final_score, debug

    def prove_design_sub_propositions(self,
                                      adjuvant_properties: Dict[str, Any],
                                      antigen_properties: Dict[str, Any] = None,
                                      source_type: str = 'construction',
                                      top_k: int = 5) -> StageResult:

        if not adjuvant_properties and not antigen_properties:
            return StageResult(stage=1, score=0.0, status="unverified",
                               detail="No design properties provided")

        all_evidence: List[Evidence] = []


        adj_prop_scores: Dict[str, float] = {}
        adj_prop_components: Dict[str, List[float]] = {}
        kb_prop_names = {"Type", "Size", "Method", "Contains"}
        for prop_name, prop_val in adjuvant_properties.items():


            if prop_name not in kb_prop_names:
                continue
            if not prop_val:
                continue
            vals = prop_val if isinstance(prop_val, list) else [prop_val]
            comp_scores = []
            for v in vals:
                score, rule = self._score_property(prop_name, str(v), source_type)
                comp_scores.append(score)
                if rule:
                    ev = _rule_to_evidence(rule, match_type="symmetric",
                                           matched_content=f"{prop_name}={v}")
                    ev.similarity_score = score
                    all_evidence.append(ev)
                else:

                    blind_ev = Evidence(
                        rule_id="—",
                        source_paper='Runtime diagnostic.',
                        confidence=0.0,
                        similarity_score=0.0,
                        match_type="blind",
                        matched_content=f"{prop_name}={v}",
                        conditions_repr=f"{prop_name}(value={v})",
                        effect_repr="HasVerifiedProperty(—)",
                        source_type="construction",
                    )
                    all_evidence.append(blind_ev)
            adj_prop_scores[prop_name] = sum(comp_scores) / len(comp_scores) if comp_scores else 0.0
            adj_prop_components[prop_name] = comp_scores

        adj_avg = sum(adj_prop_scores.values()) / len(adj_prop_scores) if adj_prop_scores else 0.0


        ant_avg = 0.0
        ant_prop_scores: Dict[str, float] = {}
        ant_prop_components: Dict[str, List[float]] = {}
        if antigen_properties:
            for prop_name, prop_val in antigen_properties.items():
                if not prop_val:
                    continue
                vals = prop_val if isinstance(prop_val, list) else [prop_val]
                comp_scores = []
                for v in vals:
                    if not v:
                        continue
                    score, rule = self._score_property(prop_name, str(v),
                                                        'antigen_construction')
                    comp_scores.append(score)
                    if rule:
                        ev = _rule_to_evidence(rule, match_type="semantic",
                                               matched_content=f"{prop_name}={v}")
                        ev.similarity_score = score
                        all_evidence.append(ev)
                    else:
                        blind_ev = Evidence(
                            rule_id="—",
                            source_paper='Runtime diagnostic.',
                            confidence=0.0,
                            similarity_score=0.0,
                            match_type="blind",
                            matched_content=f"{prop_name}={v}",
                            conditions_repr=f"{prop_name}(value={v})",
                            effect_repr="HasVerifiedProperty(—)",
                            source_type="antigen_construction",
                        )
                        all_evidence.append(blind_ev)
                ant_prop_scores[prop_name] = sum(comp_scores) / len(comp_scores) if comp_scores else 0.0
                ant_prop_components[prop_name] = comp_scores
            ant_avg = sum(ant_prop_scores.values()) / len(ant_prop_scores) if ant_prop_scores else 0.0


        prior_result = self.stage1_prior_checker.check(adjuvant_properties or {})
        if self.vaxjo_mode == "off":
            vaxjo_result = {"score": 0.0, "matches": [], "detail": "Vaxjo disabled"}
        else:
            vaxjo_result = self.vaxjo_knowledge.score_stage1_design(adjuvant_properties or {})
        for hit in vaxjo_result.get("matches", [])[:3]:
            profiles = "; ".join(hit.get("immune_profiles", [])[:3]) or "—"
            receptors = "; ".join(hit.get("target_receptors", [])[:3]) or "—"
            refs = hit.get("definition_source") or "; ".join(hit.get("evidence_refs", [])[:4])
            all_evidence.append(Evidence(
                rule_id=hit.get("id", "VAXJO"),
                source_paper=refs or "Vaxjo/VO",
                confidence=0.90 if hit.get("source") == "vo" else 0.75,
                similarity_score=vaxjo_result.get("score", 0.0),
                match_type="vaxjo_ontology",
                matched_content=f"matched_alias={hit.get('matched_alias', '')}",
                conditions_repr=f"AdjuvantConcept({hit.get('label', '')})",
                effect_repr=f"ImmuneProfile({profiles}); TargetReceptor({receptors})",
                evidence_text=hit.get("definition", "")[:180],
                source_type="vaxjo_stage1_ontology",
            ))


        if adj_prop_scores and ant_prop_scores:

            kb_score = 0.75 * adj_avg + 0.25 * ant_avg
        elif ant_prop_scores:

            kb_score = ant_avg
        else:

            kb_score = adj_avg

        stage1_mode = os.getenv("SAVANT_STAGE1_MODE", "legacy").strip().lower()
        if stage1_mode in {"v9", "minimal_v9", "minimal_subtheorem_guarded"}:
            final_score, stage1_debug = self._aggregate_stage1_v9(
                adjuvant_properties=adjuvant_properties or {},
                antigen_properties=antigen_properties,
                adj_prop_scores=adj_prop_scores,
                adj_prop_components=adj_prop_components,
                ant_prop_scores=ant_prop_scores,
                ant_prop_components=ant_prop_components,
                prior_result=prior_result,
                kb_score=kb_score,
            )
        else:
            final_score = 0.60 * kb_score + 0.40 * prior_result.score


            cap_to_raw = {1: 0.299, 2: 0.499, 3: 0.699, 4: 0.899, 5: 1.0}
            final_score = min(final_score, cap_to_raw.get(prior_result.cap_5level, 1.0))
            stage1_debug = {
                "stage1_mode": "legacy_weighted_average",
                "kb_score": round(float(kb_score), 6),
                "prior_guard_score": round(float(prior_result.score), 6),
                "prior_cap": int(prior_result.cap_5level),
            }

        status = ("verified"  if final_score >= 0.65
                  else "partial"  if final_score >= 0.40
                  else "hypothesis")


        adj_parts = " | ".join(f"{k}={v:.3f}" for k, v in sorted(adj_prop_scores.items()))
        prior_parts = " | ".join(prior_result.details)
        warning_parts = " | ".join(prior_result.warnings[:3])
        if antigen_properties and ant_prop_scores:
            ant_parts = " | ".join(f"{k}={v:.3f}" for k, v in sorted(ant_prop_scores.items()))
            detail = ('Runtime diagnostic.'
                      f" | Ant({ant_avg:.3f})=[{ant_parts}]"
                      f" | Prior({prior_result.score:.3f})=[{prior_parts}]")
        else:
            detail = ('Runtime diagnostic.'
                      f" | Prior({prior_result.score:.3f})=[{prior_parts}]")
        if warning_parts:
            detail += f" | Warnings=[{warning_parts}]"
        if vaxjo_result.get("score", 0.0) > 0:
            detail += f" | Vaxjo({vaxjo_result['score']:.3f})=[{vaxjo_result.get('detail', '')}]"
        if stage1_mode in {"v9", "minimal_v9", "minimal_subtheorem_guarded"}:
            detail += (
                f" | Stage1V9(band={stage1_debug.get('score_band')}, "
                f"core={stage1_debug.get('core_proof_score')}, "
                f"unknown_core={stage1_debug.get('unknown_core_count')}, "
                f"severity={stage1_debug.get('prior_severity')}, "
                f"cap={stage1_debug.get('band_cap')})"
            )

        return StageResult(
            stage=1, score=final_score, evidence=all_evidence, status=status,
            detail=detail, debug=stage1_debug
        )


    _BROAD_STAGE2_EFFECT_FUNCTORS: Set[str] = {
        "Protect", "Reduce", "Neutralize", "Kill", "Control", "Suppress",
        "Inhibit", "Avoid", "Enhance", "Increase", "Establish", "Prolong",
        "Balance", "Equal", "Positive", "Prime", "Differentiate", "Polarize",
        "Express", "Secrete", "Present", "Mature", "Recruit", "Induce",
        "Sustain", "Activate", "Uptake", "Target", "Migrate", "Release",
    }

    _STAGE2_TERMINAL_EFFECT_FUNCTORS: Set[str] = {
        "Protect", "Reduce", "Neutralize", "Kill", "Avoid", "Establish",
        "Sustain", "Balance", "Prolong",
    }

    _STAGE2_BROAD_EFFECT_FUNCTORS: Set[str] = {
        "Enhance", "Increase", "Reduce", "Inhibit", "Suppress", "Protect",
        "Avoid", "Control", "Balance", "Equal", "Positive", "Establish",
        "Tune", "Sustain", "Prolong", "Neutralize", "Kill",
    }

    _STAGE2_IMMUNE_EVENT_FUNCTORS: Set[str] = {
        "Activate", "Prime", "Differentiate", "Polarize", "Express",
        "Secrete", "Present", "Mature", "Recruit", "Induce",
    }

    _STAGE2_UPSTREAM_CLAIM_FUNCTORS: Set[str] = {
        "Route", "Target", "Release", "Uptake", "Migrate",
    }

    _STAGE2_GENERIC_OUTCOME_PHRASES: Tuple[str, ...] = (
        "immune response", "immune activation", "immunity", "protection",
        "efficacy", "effect", "therapeutic effect", "antigen response",
        "cellular response", "humoral response", "inflammation",
    )

    _STAGE2_SPECIFIC_OUTCOME_HINTS: Tuple[str, ...] = (
        "antibody", "titer", "neutralizing", "cytokine", "interferon",
        "cd8", "cd4", "t cell", "b cell", "dendritic", "memory",
        "survival", "viral load", "tumor", "pathogen", "antigen presentation",
        "mhc", "igg", "iga", "ctl", "th1", "th2", "th17",
    )

    _STAGE2_UPSTREAM_OUTCOME_HINTS: Tuple[str, ...] = (
        "size", "particle", "route", "dose", "formulation", "encapsulation",
        "release", "delivery", "uptake", "migration", "targeting", "drainage",
        "loading", "stability", "degradation",
    )

    @staticmethod
    def _stage2_claim_value(pred: Predicate) -> str:
        values = [
            str(arg.value)
            for arg in getattr(pred, "args", []) or []
            if getattr(arg, "value", None) not in ("?", "", None)
            and not arg.is_variable()
        ]
        return re.sub(r"\s+", " ", " ".join(values)).strip()

    @staticmethod
    def _stage2_norm_claim_value(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()

    @classmethod
    def _stage2_claim_quality_score(cls, claims: List[Predicate]) -> Tuple[float, Dict[str, float]]:

        if not claims:
            return 0.0, {
                "claim_quality": 0.0,
                "generic_frac": 0.0,
                "upstream_frac": 0.0,
                "specific_frac": 0.0,
                "duplicate_frac": 0.0,
            }

        values = [cls._stage2_norm_claim_value(cls._stage2_claim_value(pred)) for pred in claims]
        value_counts = {value: values.count(value) for value in set(values)}
        pair_values = [(pred.functor, value) for pred, value in zip(claims, values)]
        pair_counts = {pair: pair_values.count(pair) for pair in set(pair_values)}

        qualities: List[float] = []
        generic_count = 0
        upstream_count = 0
        specific_count = 0
        duplicate_count = 0

        for pred, value in zip(claims, values):
            if not value:
                qualities.append(0.10)
                continue

            score = 0.55
            cap = 1.0

            if pred.functor in cls._STAGE2_TERMINAL_EFFECT_FUNCTORS:
                score += 0.25
            elif pred.functor in cls._STAGE2_BROAD_EFFECT_FUNCTORS:
                score += 0.20
            elif pred.functor in cls._STAGE2_IMMUNE_EVENT_FUNCTORS:
                score += 0.15
            elif pred.functor in cls._STAGE2_UPSTREAM_CLAIM_FUNCTORS:
                score += 0.05
                cap = min(cap, 0.70)
            else:
                cap = min(cap, 0.65)

            is_generic = any(phrase == value or phrase in value for phrase in cls._STAGE2_GENERIC_OUTCOME_PHRASES)
            is_upstream = any(hint in value for hint in cls._STAGE2_UPSTREAM_OUTCOME_HINTS)
            is_specific = any(hint in value for hint in cls._STAGE2_SPECIFIC_OUTCOME_HINTS)
            is_duplicate = value_counts.get(value, 0) > 1 or pair_counts.get((pred.functor, value), 0) > 1

            if is_generic:
                generic_count += 1
                score -= 0.15
                cap = min(cap, 0.80)
            if is_upstream:
                upstream_count += 1
                score -= 0.20
                cap = min(cap, 0.65)
            if is_specific:
                specific_count += 1
                score += 0.10
            if is_duplicate:
                duplicate_count += 1
                score -= 0.15

            qualities.append(max(0.0, min(score, cap)))

        qualities_sorted = sorted(qualities)
        n = len(qualities_sorted)
        median_quality = (
            qualities_sorted[n // 2]
            if n % 2 == 1
            else 0.5 * (qualities_sorted[n // 2 - 1] + qualities_sorted[n // 2])
        )
        mean_quality = sum(qualities) / len(qualities)
        claim_quality = 0.70 * median_quality + 0.30 * mean_quality

        denom = float(len(claims))
        return max(0.0, min(claim_quality, 1.0)), {
            "claim_quality": claim_quality,
            "claim_quality_median": median_quality,
            "claim_quality_mean": mean_quality,
            "generic_frac": generic_count / denom,
            "upstream_frac": upstream_count / denom,
            "specific_frac": specific_count / denom,
            "duplicate_frac": duplicate_count / denom,
        }

    @classmethod
    def _stage2_specificity_score(cls, claims: List[Predicate], raw_score: float) -> Tuple[float, Dict[str, float]]:

        value_texts: List[str] = []
        all_tokens: List[str] = []
        for pred in claims:
            values = [
                str(arg.value)
                for arg in getattr(pred, "args", []) or []
                if getattr(arg, "value", None) not in ("?", "", None)
                and not arg.is_variable()
            ]
            text = " ".join(values).strip()
            if not text:
                continue
            value_texts.append(text)
            all_tokens.extend(re.findall(r"[A-Za-z0-9]+", text.lower()))

        token_diversity = (
            len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0
        )
        min_tokens = (
            min(
                len(re.findall(r"[A-Za-z0-9]+", text.lower()))
                for text in value_texts
            ) / 5.0
            if value_texts
            else 0.0
        )
        min_tokens = max(0.0, min(min_tokens, 1.0))
        terminal_functor_frac = (
            sum(1 for pred in claims if pred.functor in cls._STAGE2_TERMINAL_EFFECT_FUNCTORS)
            / len(claims)
            if claims
            else 0.0
        )
        dominant_functor_frac = (
            max(
                sum(1 for pred in claims if pred.functor == functor)
                for functor in {pred.functor for pred in claims}
            )
            / len(claims)
            if claims
            else 0.0
        )
        compact_claim_set = (
            max(0.0, min((5.0 - float(len(claims))) / 4.0, 1.0))
            if claims
            else 0.0
        )
        raw_concept_score = max(0.0, min(float(raw_score), 1.0))
        claim_quality_score, claim_quality_debug = cls._stage2_claim_quality_score(claims)

        formula = os.getenv(
            "SAVANT_STAGE2_SPECIFICITY_FORMULA", "intuitive"
        ).strip().lower()
        if formula == "legacy":
            specificity = (
                0.22 * token_diversity
                + 0.56 * min_tokens
                + 0.11 * terminal_functor_frac
                + 0.11 * raw_concept_score
            )
            center_default = "0.57"
            scale_default = "6.0"
        elif formula in {"balanced", "expert_v2_calibrated"}:
            formula = "expert_v2_calibrated"


            specificity = (
                0.50 * raw_concept_score
                + 0.25 * token_diversity
                + 0.10 * dominant_functor_frac
                + 0.10 * terminal_functor_frac
                + 0.05 * compact_claim_set
            )
            center_default = "0.65"
            scale_default = "4.0"
        elif formula == "claim_quality":


            specificity = (
                0.65 * raw_concept_score
                + 0.35 * claim_quality_score
            )
            center_default = "0.70"
            scale_default = "4.0"
        else:
            formula = "intuitive"

            specificity = (
                0.70 * raw_concept_score
                + 0.20 * terminal_functor_frac
                + 0.10 * token_diversity
            )
            center_default = "0.70"
            scale_default = "4.0"

        center = float(os.getenv("SAVANT_STAGE2_SPECIFICITY_CENTER", center_default))
        scale = float(os.getenv("SAVANT_STAGE2_SPECIFICITY_SCALE", scale_default))
        score_1to5 = 4.0 + scale * (specificity - center)
        score_1to5 = max(1.0, min(score_1to5, 5.0))
        return score_1to5, {
            "specificity": specificity,
            "formula": formula,
            "token_diversity": token_diversity,
            "min_tokens": min_tokens,
            "terminal_functor_frac": terminal_functor_frac,
            "dominant_functor_frac": dominant_functor_frac,
            "compact_claim_set": compact_claim_set,
            "raw_score": raw_concept_score,
            **claim_quality_debug,
        }

    @classmethod
    def _stage2_broad_floor(cls,
                            pred: Predicate,
                            endpoint: Dict[str, Any],
                            endpoint_score: float,
                            kb_score: float,
                            vaxjo_score: float) -> float:

        level = str(endpoint.get("best_level", ""))
        if endpoint_score > 0.0:
            if level == "strong_endpoint":
                return 0.90
            if level == "intermediate_endpoint":
                return 0.72
            if level == "delivery_endpoint":
                return 0.60
            if level == "early_mechanism":
                return 0.55
            return 0.55

        if max(kb_score, vaxjo_score) >= 0.55:
            return 0.72

        if pred.functor in cls._BROAD_STAGE2_EFFECT_FUNCTORS:
            has_value = any(
                getattr(arg, "value", None) not in ("?", "", None)
                and not arg.is_variable()
                for arg in pred.args
            )
            if has_value:
                return 0.55
        return 0.0

    def prove_predicate_sub_proposition(self,
                                        pred: Predicate,
                                        top_k: int = 5,
                                        adjuvant_properties: Optional[Dict[str, Any]] = None) -> StageResult:

        resolved_functor, proxy_score, proxy_type = self.proxy_resolver.resolve(pred.functor)


        candidates = self.sym._get_candidates_by_eff(
            resolved_functor, ['immune_effect', 'causal_chain']
        )


        stage3_only_plugins = {"immunology_round2", "adjuvant_bridge_round1"}
        candidates = [r for r in candidates if r.get("plugin") not in stage3_only_plugins]


        q_eff_roles: Dict[str, str] = {
            a.role: str(a.value)
            for a in pred.args
            if a.role and a.value not in ('?', '') and not a.is_variable()
        }
        q_eff_values = list(q_eff_roles.values()) if q_eff_roles else [
            str(a.value) for a in pred.args
            if a.value not in ('?', '') and not a.is_variable()
        ]

        ranked = []
        for r in candidates[:top_k * 3]:
            ef_f, ef_roles = _extract_functor_and_roles(r.get('effect', {}))
            ef_s_raw = self.sym._functor_score(resolved_functor, ef_f)
            if ef_s_raw == 0.0:
                continue
            ef_s = ef_s_raw * max(proxy_score, 0.5)
            ef_s = max(ef_s, self.sym._functor_score(pred.functor, ef_f))
            if q_eff_roles:
                ev_s = self.sym._value_sim_by_role(q_eff_roles, ef_roles)
            else:
                ev_s = self.sym._value_sim(q_eff_values, _roles_to_flat(ef_roles))


            s = ef_s * 0.30 + ev_s * 0.70
            if ev_s < 0.25:
                s *= 0.50
            ranked.append((s, r))
        ranked.sort(key=lambda x: x[0], reverse=True)

        best_score = ranked[0][0] if ranked else 0.0

        evidence = []
        for score, r in ranked[:3]:
            eff = r.get('effect', {})
            ev = _rule_to_evidence(
                r,
                match_type=proxy_type if proxy_type != "exact" else "exact",
                matched_content=(
                    f"functor={pred.functor}→proxy={resolved_functor}({proxy_score:.2f})"
                    f" | eff={eff.get('functor','')} [{r.get('source_type','')}]"
                ),
            )
            ev.similarity_score = score
            evidence.append(ev)

        if self.vaxjo_mode == "off":
            vaxjo = {"score": 0.0, "matched_categories": [], "source_refs": []}
        else:
            vaxjo = self.vaxjo_knowledge.score_terminal_effect(
                pred, props=adjuvant_properties or {}
            )
        vaxjo_score = float(vaxjo.get("score", 0.0))
        if vaxjo_score > 0.0:
            refs = vaxjo.get("source_refs") or []
            source_paper = "PMID:" + ",".join(refs[:4]) if refs else "Vaxjo/VO"
            vaxjo_ev = Evidence(
                rule_id="VAXJO-STAGE2",
                source_paper=source_paper,
                confidence=vaxjo_score,
                similarity_score=vaxjo_score,
                match_type="vaxjo_ontology",
                matched_content=(
                    f"categories={','.join(vaxjo.get('matched_categories', []))}"
                    f" | source={vaxjo.get('source_label', '')}"
                ),
                conditions_repr=f"AdjuvantContext({vaxjo.get('source_label', '')})",
                effect_repr=f"{pred.functor}({', '.join(q_eff_values)})",
                evidence_text=vaxjo.get("detail", ""),
                source_type="vaxjo_effect_ontology",
            )
            evidence.append(vaxjo_ev)
            evidence.sort(key=lambda ev: ev.similarity_score, reverse=True)

        if self.stage2_endpoint_mode == "off":
            endpoint = {"score": 0.0, "matched_categories": []}
        else:
            endpoint = self.stage2_endpoint_ontology.score(pred)
        endpoint_score = float(endpoint.get("score", 0.0))
        if endpoint_score > 0.0:
            endpoint_ev = Evidence(
                rule_id="STAGE2-ENDPOINT-ONTOLOGY",
                source_paper="stage2_effect_endpoint_ontology",
                confidence=endpoint_score,
                similarity_score=endpoint_score,
                match_type="stage2_endpoint_ontology",
                matched_content=(
                    f"categories={','.join(endpoint.get('matched_categories', []))}"
                    f" | mode={self.stage2_endpoint_mode}"
                ),
                conditions_repr="TerminalEffectConcept(?)",
                effect_repr=f"{pred.functor}({', '.join(q_eff_values)})",
                evidence_text=endpoint.get("detail", ""),
                source_type="stage2_endpoint_ontology",
            )
            evidence.append(endpoint_ev)
            evidence.sort(key=lambda ev: ev.similarity_score, reverse=True)

        final_score = best_score
        endpoint_fusion_score = 0.0
        broad_floor = 0.0
        if self.stage2_policy == "broad":
            broad_floor = self._stage2_broad_floor(
                pred=pred,
                endpoint=endpoint,
                endpoint_score=endpoint_score,
                kb_score=best_score,
                vaxjo_score=vaxjo_score,
            )
            final_score = max(final_score, endpoint_score, vaxjo_score, broad_floor)
        else:
            if endpoint_score > 0.0 and self.stage2_endpoint_mode in {"score", "guarded"}:
                if self.stage2_endpoint_mode == "score":
                    endpoint_fusion_score = endpoint_score
                else:


                    level = str(endpoint.get("best_level", ""))
                    if level == "strong_endpoint":
                        endpoint_fusion_score = endpoint_score
                    elif level == "intermediate_endpoint":
                        endpoint_fusion_score = min(endpoint_score, 0.49)
                    elif level == "delivery_endpoint":
                        endpoint_fusion_score = min(endpoint_score, 0.44)
                    else:
                        endpoint_fusion_score = min(endpoint_score, 0.39)
                final_score = max(final_score, endpoint_fusion_score)
            if self.vaxjo_mode == "score":
                final_score = max(final_score, vaxjo_score)

        status = ("verified" if final_score >= 0.55
                  else "partial"   if final_score >= 0.30
                  else "hypothesis")

        if not candidates and vaxjo_score <= 0.0 and endpoint_score <= 0.0:
            detail = f"No matching rules for {pred.functor}; no Vaxjo or endpoint ontology support"
        else:
            detail = (
                'Runtime diagnostic.'
                f"({proxy_type},{proxy_score:.2f}), kb_best={best_score:.3f}, "
                f"vaxjo={vaxjo_score:.3f}, endpoint={endpoint_score:.3f}, "
                f"endpoint_fused={endpoint_fusion_score:.3f}, "
                f"broad_floor={broad_floor:.3f}, "
                f"final={final_score:.3f}, mode={self.vaxjo_mode}, "
                f"stage2_policy={self.stage2_policy}, "
                f"endpoint_mode={self.stage2_endpoint_mode}, "
                f"policy=D_only(effect-end concept), candidates={len(candidates)}"
            )
            if vaxjo_score > 0.0:
                detail += f" | {vaxjo.get('detail', '')}"
            if endpoint_score > 0.0:
                detail += f" | {endpoint.get('detail', '')}"

        return StageResult(
            stage=2, score=min(final_score, 1.0), evidence=evidence, status=status,
            detail=detail
        )

    def _case_insensitive_lookup(self, table: Dict[str, List[dict]], functor: str) -> List[dict]:
        if functor in table:
            return table[functor]
        key_map = {k.lower(): k for k in table}
        actual = key_map.get(functor.lower())
        return table.get(actual, []) if actual else []

    def _match_derived_path_step(self, cond_text: str, eff_text: str) -> Tuple[float, Optional[dict]]:

        if not self.enable_derived_paths or not self._derived_path_rules:
            return 0.0, None

        stage3_sym = getattr(self, "stage3_sym", self.sym)
        stage3_proxy = getattr(self, "stage3_proxy_resolver", self.proxy_resolver)
        q_cond_functor, q_cond_values = stage3_sym._parse_node(cond_text)
        q_eff_functor, q_eff_values = stage3_sym._parse_node(eff_text)
        resolved_eff, eff_proxy_score, _ = stage3_proxy.resolve(q_eff_functor)
        resolved_cond, cond_proxy_score, _ = stage3_proxy.resolve(q_cond_functor)

        candidates = []
        candidates.extend(self._case_insensitive_lookup(self._derived_by_eff_functor, resolved_eff))
        candidates.extend(self._case_insensitive_lookup(self._derived_by_cond_functor, resolved_cond))
        if not candidates:
            return 0.0, None

        seen = set()
        unique_candidates = []
        for r in candidates:
            rid = r.get("rule_id")
            if rid in seen:
                continue
            seen.add(rid)
            unique_candidates.append(r)

        scored = []
        for rule in unique_candidates:
            semantic = stage3_sym._score_rule(
                rule,
                resolved_eff, eff_proxy_score, q_eff_values,
                resolved_cond, cond_proxy_score, q_cond_values,
            )
            if semantic <= 0:
                continue
            hops = max(2, int(rule.get("path_hops") or 2))
            path_conf = float(rule.get("confidence", 0.75) or 0.75)
            hop_penalty = max(0.65, 1.0 - 0.08 * (hops - 1))
            score = semantic * path_conf * hop_penalty
            scored.append((score, rule))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0] if scored else (0.0, None)


    def prove_inference_chain_sub_proposition(self,
                                              chain_dsl: str,
                                              design_facts: List[Predicate] = None
                                              ) -> StageResult:

        if self.verifier and design_facts:
            return self._prove_with_verifier(chain_dsl, design_facts)
        else:
            return self._prove_with_faiss_chain(chain_dsl)

    def _prove_with_verifier(self, chain_dsl: str,
                              design_facts: List[Predicate]) -> StageResult:

        try:

            goal_pred = self._parse_chain_goal(chain_dsl)
            if goal_pred is None:
                return StageResult(stage=3, score=0.0, status="unverified",
                                   detail="Cannot parse chain goal")

            result = self.verifier.prove(
                goal=goal_pred,
                known_facts=design_facts,
                depth_limit=6
            )

            score = getattr(result, 'confidence', 0.0)
            status_map = {
                'VERIFIED': 'verified',
                'PARTIAL': 'partial',
                'HYPOTHESIS': 'hypothesis',
                'REFUTED': 'unverified',
            }
            status_str = getattr(result, 'status', 'HYPOTHESIS')
            status = status_map.get(str(status_str), 'hypothesis')

            return StageResult(
                stage=3, score=score, status=status,
                detail=f"BackwardChain goal={goal_pred.functor}, status={status_str}"
            )
        except Exception as e:
            logger.warning('Runtime diagnostic.')
            return self._prove_with_faiss_chain(chain_dsl)

    def _prove_with_faiss_chain(self, chain_dsl: str) -> StageResult:


        chain_dsl = chain_dsl.replace('→', '>>').replace('-->', '>>')
        chain_dsl = re.sub(r'(?<!-)->', '>>', chain_dsl)
        parts = [p.strip() for p in chain_dsl.split('>>') if p.strip()]

        if len(parts) < 2:
            return StageResult(stage=3, score=0.0, status="unverified",
                               detail="Chain too short to evaluate",
                               debug={
                                   "chain": chain_dsl,
                                   "nodes": parts,
                                   "steps": [],
                                   "failure_reason": "chain_too_short",
                               })

        step_scores: List[float] = []
        step_rules:  List[Optional[dict]] = []
        all_evidence: List[Evidence] = []
        debug_steps: List[Dict[str, Any]] = []
        support_threshold = self.stage3_support_threshold

        for i in range(len(parts) - 1):
            cond_raw = parts[i]
            eff_raw  = parts[i + 1]


            stage3_sym = getattr(self, "stage3_sym", self.sym)
            step_score, best_rule = stage3_sym.match_step(
                cond_text    = cond_raw,
                eff_text     = eff_raw,
                source_types = ['causal_chain', 'immune_effect'],
            )
            match_type = "symmetric"

            derived_score, derived_rule = self._match_derived_path_step(cond_raw, eff_raw)
            if derived_rule is not None and derived_score > step_score:
                step_score = derived_score
                best_rule = derived_rule
                match_type = "derived_multihop"

            if best_rule is not None and step_score > 0:
                step_scores.append(step_score)
                step_rules.append(best_rule)
                ev = _rule_to_evidence(
                    best_rule,
                    match_type=match_type,
                    matched_content=(
                        'Runtime diagnostic.'
                        + (
                            f" | multi-hop sidecar, hops={best_rule.get('path_hops')}, not_atomic_theorem"
                            if match_type == "derived_multihop" else ""
                        )
                    ),
                )
                ev.step_in_chain = i + 1
                ev.similarity_score = step_score
                all_evidence.append(ev)
                edge_status = "supported" if step_score >= support_threshold else "weak"
                debug_steps.append({
                    "step": i + 1,
                    "condition": cond_raw,
                    "effect": eff_raw,
                    "score": round(float(step_score), 6),
                    "status": edge_status,
                    "match_type": match_type,
                    "rule_id": best_rule.get("rule_id") or best_rule.get("id", ""),
                    "paper_id": best_rule.get("paper_id", ""),
                    "source_type": best_rule.get("source_type", ""),
                    "plugin": best_rule.get("plugin", ""),
                    "chain_id": best_rule.get("chain_id", ""),
                    "step_in_rule_chain": best_rule.get("step_in_chain", 0),
                    "path_hops": best_rule.get("path_hops"),
                    "conditions_repr": ev.conditions_repr,
                    "effect_repr": ev.effect_repr,
                    "evidence_text": ev.evidence_text,
                    "matched_content": ev.matched_content,
                    "similarity_score": round(float(ev.similarity_score), 6),
                    "rule_confidence": round(float(ev.confidence or 0.0), 6),
                })
            else:
                step_scores.append(0.0)
                step_rules.append(None)
                debug_steps.append({
                    "step": i + 1,
                    "condition": cond_raw,
                    "effect": eff_raw,
                    "score": 0.0,
                    "status": "missing",
                    "match_type": "none",
                    "rule_id": "",
                    "paper_id": "",
                    "source_type": "",
                    "plugin": "",
                    "chain_id": "",
                    "step_in_rule_chain": 0,
                    "path_hops": None,
                    "conditions_repr": "",
                    "effect_repr": "",
                    "evidence_text": "",
                    "matched_content": "",
                    "similarity_score": 0.0,
                    "rule_confidence": 0.0,
                })

        if not step_scores:
            return StageResult(stage=3, score=0.0, status="unverified",
                               detail="No causal chain rules found",
                               debug={
                                   "chain": chain_dsl,
                                   "nodes": parts,
                                   "steps": debug_steps,
                                   "failure_reason": "no_steps",
                               })


        supported_scores = [s for s in step_scores if s >= support_threshold]
        weak_scores = [s for s in step_scores if 0 < s < support_threshold]
        if supported_scores:
            coverage = len(supported_scores) / len(step_scores)
            mean_supported = sum(supported_scores) / len(supported_scores)


            combo_bonus = 0.0
            confidence_factor = min(
                self.stage3_confidence_floor
                + self.stage3_confidence_weight * min(mean_supported + combo_bonus, 1.0),
                1.0,
            )
            final_score = coverage * confidence_factor
        else:
            coverage = 0.0
            mean_supported = 0.0
            combo_bonus = 0.0
            confidence_factor = 0.0
            final_score = 0.0

        status = ("verified"   if coverage >= 0.90 and final_score >= 0.60
                  else "partial"    if coverage >= 0.50 and final_score >= 0.35
                  else "hypothesis")

        return StageResult(
            stage=3, score=final_score, evidence=all_evidence, status=status,
            detail=(
                'Runtime diagnostic.'
                f"supported={len(supported_scores)}/{len(step_scores)}, "
                f"weak={len(weak_scores)}, coverage={coverage:.3f}, "
                f"mean_supported={mean_supported:.3f}, "
                f"confidence_factor={confidence_factor:.3f}, "
                f"combo={combo_bonus:.3f}, score={final_score:.3f}, "
                f"derived_paths={'on' if self.enable_derived_paths else 'off'}"
            ),
            debug={
                "chain": chain_dsl,
                "nodes": parts,
                "steps": debug_steps,
                "support_threshold": support_threshold,
                "coverage": round(float(coverage), 6),
                "mean_supported": round(float(mean_supported), 6),
                "confidence_factor": round(float(confidence_factor), 6),
                "confidence_floor": round(float(self.stage3_confidence_floor), 6),
                "confidence_weight": round(float(self.stage3_confidence_weight), 6),
                "weak_count": len(weak_scores),
                "supported_count": len(supported_scores),
                "total_steps": len(step_scores),
                "score": round(float(final_score), 6),
                "derived_paths_enabled": bool(self.enable_derived_paths),
            }
        )

    @staticmethod
    def _stage3_pred_values(pred: Predicate) -> List[str]:
        return [
            str(arg.value)
            for arg in getattr(pred, "args", []) or []
            if getattr(arg, "value", None) not in ("?", "", None)
            and not arg.is_variable()
        ]

    @staticmethod
    def _stage3_norm_text(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()

    @staticmethod
    def _stage3_flatten_values(value: Any) -> List[str]:
        if value is None or value == "":
            return []
        if isinstance(value, dict):
            out: List[str] = []
            for v in value.values():
                out.extend(LogicMatcher._stage3_flatten_values(v))
            return out
        if isinstance(value, list):
            out = []
            for v in value:
                out.extend(LogicMatcher._stage3_flatten_values(v))
            return out
        return [str(value)]

    def _stage3_endpoint_alignment(self, endpoint_text: str, claims: List[Predicate]) -> float:

        if not claims:
            return 0.50

        stage3_sym = getattr(self, "stage3_sym", self.sym)
        stage3_proxy = getattr(self, "stage3_proxy_resolver", self.proxy_resolver)
        endpoint_functor, endpoint_values = stage3_sym._parse_node(endpoint_text)
        if not endpoint_functor and not endpoint_values:
            return 0.0
        if not endpoint_values:
            endpoint_values = [endpoint_text]

        best = 0.0
        resolved_endpoint = endpoint_functor
        endpoint_proxy = 0.5
        if endpoint_functor:
            resolved_endpoint, endpoint_proxy, _ = stage3_proxy.resolve(endpoint_functor)

        for claim in claims:
            claim_values = self._stage3_pred_values(claim)
            if not claim_values:
                claim_values = [claim.functor]

            functor_score = 0.25
            if endpoint_functor:
                resolved_claim, claim_proxy, _ = stage3_proxy.resolve(claim.functor)
                functor_score = max(
                    stage3_sym._functor_score(endpoint_functor, claim.functor),
                    stage3_sym._functor_score(resolved_endpoint, resolved_claim)
                    * max(endpoint_proxy, claim_proxy, 0.5),
                )
            value_score = stage3_sym._value_sim(endpoint_values, claim_values)
            score = 0.45 * functor_score + 0.55 * value_score
            best = max(best, score)
        return max(0.0, min(best, 1.0))

    def _stage3_start_anchor_score(self,
                                   start_text: str,
                                   adjuvant_properties: Optional[Dict[str, Any]],
                                   antigen_properties: Optional[Dict[str, Any]]) -> float:

        prop_values: List[str] = []
        for props in (adjuvant_properties or {}, antigen_properties or {}):
            if isinstance(props, dict):
                for value in props.values():
                    prop_values.extend(self._stage3_flatten_values(value))
        prop_values = [v for v in prop_values if str(v).strip()]
        if not prop_values:
            return 0.50

        stage3_sym = getattr(self, "stage3_sym", self.sym)
        start_functor, start_values = stage3_sym._parse_node(start_text)
        query_values = start_values or [start_text]
        q_norm = self._stage3_norm_text(" ".join(query_values + ([start_functor] if start_functor else [])))
        prop_norm = self._stage3_norm_text(" ".join(prop_values))
        if q_norm and q_norm in prop_norm:
            return 1.0

        try:
            sim = stage3_sym._value_sim(query_values, prop_values)
        except Exception:
            sim = 0.0

        return max(0.45, min(1.0, 0.45 + 0.55 * sim))

    def _aggregate_stage3_closure_bridge(self,
                                         mechanisms_dsl: List[str],
                                         claims: List[Predicate],
                                         adjuvant_properties: Dict[str, Any],
                                         antigen_properties: Optional[Dict[str, Any]] = None
                                         ) -> StageResult:

        if not mechanisms_dsl:
            return StageResult(stage=3, score=0.0, status="hypothesis",
                               detail="No mechanism chains provided",
                               debug={"stage3_mode": "closure_bridge", "chains": []})

        chain_infos: List[Dict[str, Any]] = []
        all_evidence: List[Evidence] = []
        total_supported = 0
        total_weak = 0
        total_steps = 0

        for chain in mechanisms_dsl:
            sr = self._prove_with_faiss_chain(chain)
            all_evidence.extend(sr.evidence)
            nodes = sr.debug.get("nodes") or _split_dsl_chain(chain)
            steps = sr.debug.get("steps", []) or []
            scores = [float(step.get("score", 0.0) or 0.0) for step in steps]
            n_steps = len(scores)
            if n_steps == 0:
                endpoint_alignment = self._stage3_endpoint_alignment(nodes[-1] if nodes else "", claims)
                chain_infos.append({
                    "chain": chain,
                    "nodes": nodes,
                    "score": 0.0,
                    "mean_edge": 0.0,
                    "bottleneck": 0.0,
                    "closure": 0.0,
                    "endpoint_alignment": round(endpoint_alignment, 6),
                    "start_anchor": 0.0,
                    "supported_count": 0,
                    "weak_count": 0,
                    "missing_count": 0,
                    "total_steps": 0,
                    "failure_profile": "no_steps",
                })
                continue

            supported_count = sum(1 for s in scores if s >= self.stage3_support_threshold)
            weak_count = sum(1 for s in scores if 0.0 < s < self.stage3_support_threshold)
            missing_count = sum(1 for s in scores if s <= 0.0)
            total_supported += supported_count
            total_weak += weak_count
            total_steps += n_steps

            mean_edge = sum(scores) / n_steps
            sorted_scores = sorted(scores)
            if len(sorted_scores) >= 2:
                bottleneck = 0.65 * sorted_scores[0] + 0.35 * sorted_scores[1]
            else:
                bottleneck = sorted_scores[0]

            weak_coverage = (supported_count + 0.5 * weak_count) / n_steps
            structural = 1.0 if len(nodes) >= 3 else 0.70 if len(nodes) == 2 else 0.0
            start_anchor = self._stage3_start_anchor_score(
                nodes[0] if nodes else "",
                adjuvant_properties,
                antigen_properties,
            )
            endpoint_alignment = self._stage3_endpoint_alignment(nodes[-1] if nodes else "", claims)
            gap_penalty = max(0.50, 1.0 - 0.20 * (missing_count / n_steps) - 0.08 * (weak_count / n_steps))
            closure = (
                0.45 * structural
                + 0.35 * weak_coverage
                + 0.20 * start_anchor
            ) * gap_penalty
            closure = max(0.0, min(closure, 1.0))

            chain_score = (
                0.45 * mean_edge
                + 0.25 * bottleneck
                + 0.20 * closure
                + 0.10 * endpoint_alignment
            )
            chain_score = max(0.0, min(chain_score, 1.0))

            if missing_count >= max(1, math.ceil(n_steps / 2)):
                failure_profile = "broken"
            elif missing_count > 0:
                failure_profile = "missing_bridge"
            elif weak_count > 0:
                failure_profile = "weak_bridge"
            else:
                failure_profile = "closed_supported"

            chain_infos.append({
                "chain": chain,
                "nodes": nodes,
                "score": round(float(chain_score), 6),
                "mean_edge": round(float(mean_edge), 6),
                "bottleneck": round(float(bottleneck), 6),
                "closure": round(float(closure), 6),
                "endpoint_alignment": round(float(endpoint_alignment), 6),
                "start_anchor": round(float(start_anchor), 6),
                "supported_count": supported_count,
                "weak_count": weak_count,
                "missing_count": missing_count,
                "total_steps": n_steps,
                "failure_profile": failure_profile,
            })

        valid_chain_infos = [c for c in chain_infos if c.get("total_steps", 0) > 0]
        if not valid_chain_infos:
            final_score = 0.0
            top_chain_mean = 0.0
            global_coverage = 0.0
            diversity = 0.0
            broken_ratio = 1.0
        else:
            ranked = sorted(valid_chain_infos, key=lambda c: float(c["score"]), reverse=True)
            if len(ranked) == 1:
                top_chain_mean = float(ranked[0]["score"])
            else:
                top_chain_mean = 0.65 * float(ranked[0]["score"]) + 0.35 * float(ranked[1]["score"])

            global_coverage = (
                (total_supported + 0.5 * total_weak) / total_steps
                if total_steps else 0.0
            )
            endpoint_signatures = {
                self._stage3_norm_text((c.get("nodes") or [""])[-1])
                for c in valid_chain_infos
            }
            endpoint_signatures.discard("")
            diversity = (
                len(endpoint_signatures) / len(valid_chain_infos)
                if valid_chain_infos else 0.0
            )
            broken_count = sum(
                1 for c in valid_chain_infos
                if c.get("failure_profile") in {"broken", "missing_bridge"}
            )
            broken_ratio = broken_count / len(valid_chain_infos)

            final_score = (
                0.70 * top_chain_mean
                + 0.20 * global_coverage
                + 0.10 * diversity
            )
            if broken_ratio > 0.50:
                final_score *= 0.85
            best_endpoint = max(float(c.get("endpoint_alignment", 0.0)) for c in valid_chain_infos)
            best_closure = max(float(c.get("closure", 0.0)) for c in valid_chain_infos)
            if claims and best_endpoint < 0.20:
                final_score = min(final_score, 0.499)
            if best_closure < 0.45:
                final_score = min(final_score, 0.499)
            final_score = max(0.0, min(final_score, 1.0))

        status = ("verified" if final_score >= 0.70
                  else "partial" if final_score >= 0.50
                  else "hypothesis")
        return StageResult(
            stage=3,
            score=final_score,
            evidence=all_evidence,
            status=status,
            detail=(
                f"Stage3ClosureBridge chains={len(valid_chain_infos)}, "
                f"top_chain={top_chain_mean:.3f}, global_coverage={global_coverage:.3f}, "
                f"diversity={diversity:.3f}, broken_ratio={broken_ratio:.3f}, "
                f"score={final_score:.3f}"
            ),
            debug={
                "stage3_mode": "closure_bridge",
                "chains": chain_infos,
                "top_chain_mean": round(float(top_chain_mean), 6),
                "global_coverage": round(float(global_coverage), 6),
                "diversity": round(float(diversity), 6),
                "broken_ratio": round(float(broken_ratio), 6),
                "score": round(float(final_score), 6),
            }
        )


    @staticmethod
    def _coverage_to_5(coverage: float) -> int:

        if coverage >= 0.90:
            return 5
        elif coverage >= 0.70:
            return 4
        elif coverage >= 0.50:
            return 3
        elif coverage >= 0.30:
            return 2
        else:
            return 1


    def verify(self,
               adjuvant_properties: Dict[str, Any],
               claims: List[Predicate],
               mechanisms_dsl: List[str],
               antigen_properties: Dict[str, Any] = None,
               design_facts: List[Predicate] = None) -> MatchResult:


        s1_soft = self.prove_design_sub_propositions(
            adjuvant_properties=adjuvant_properties,
            antigen_properties=antigen_properties,
        )
        s1_score = self._coverage_to_5(s1_soft.score)
        s1_status = ("verified" if s1_score >= 4
                     else "partial" if s1_score >= 3
                     else "hypothesis")
        s1 = StageResult(
            stage=1, score=float(s1_score), evidence=s1_soft.evidence,
            status=s1_status, detail=s1_soft.detail, debug=s1_soft.debug
        )


        claim_scores = []
        for pred in claims:
            sr = self.prove_predicate_sub_proposition(
                pred, adjuvant_properties=adjuvant_properties
            )
            claim_scores.append(sr.score)

        if claim_scores:
            s2_avg = sum(claim_scores) / len(claim_scores)
        else:
            s2_avg = 0.0

        stage2_score_mode = os.getenv(
            "SAVANT_STAGE2_SCORE_MODE", "specificity"
        ).strip().lower()
        if stage2_score_mode == "discrete":
            s2_score = float(self._coverage_to_5(s2_avg))
            s2_specificity_debug: Dict[str, float] = {}
        elif stage2_score_mode == "continuous":

            s2_score = 1.0 + 4.0 * max(0.0, min(float(s2_avg), 1.0))
            s2_specificity_debug = {}
        else:


            s2_score, s2_specificity_debug = self._stage2_specificity_score(
                claims, s2_avg
            )
        s2_status = ("verified" if s2_score >= 4
                     else "partial" if s2_score >= 3
                     else "hypothesis")
        s2 = StageResult(
            stage=2, score=float(s2_score), status=s2_status,
            detail=(
                f"Avg raw concept score: {s2_avg:.3f} over {len(claim_scores)} "
                f"claims; score_mode={stage2_score_mode}; "
                f"specificity={s2_specificity_debug}"
            ),
            debug=s2_specificity_debug,
        )


        stage3_mode = os.getenv("SAVANT_STAGE3_MODE", "coverage").strip().lower()
        if stage3_mode in {"closure", "closure_bridge", "v9_closure_bridge"}:
            s3_soft = self._aggregate_stage3_closure_bridge(
                mechanisms_dsl=mechanisms_dsl,
                claims=claims,
                adjuvant_properties=adjuvant_properties or {},
                antigen_properties=antigen_properties,
            )
            avg_score = s3_soft.score
            chain_scores = [
                float(c.get("score", 0.0))
                for c in s3_soft.debug.get("chains", [])
                if c.get("total_steps", 0) > 0
            ]
        else:
            chain_scores = []
            total_supported = 0
            total_weak = 0
            total_steps = 0
            for chain in mechanisms_dsl:
                sr = self._prove_with_faiss_chain(chain)
                chain_scores.append(sr.score)
                debug = sr.debug or {}
                total_supported += int(debug.get("supported_count", 0) or 0)
                total_weak += int(debug.get("weak_count", 0) or 0)
                total_steps += int(debug.get("total_steps", 0) or 0)

            if chain_scores:
                chain_mean = sum(chain_scores) / len(chain_scores)
            else:
                chain_mean = 0.0
            global_partial_coverage = (
                (total_supported + 0.5 * total_weak) / total_steps
                if total_steps else 0.0
            )
            if stage3_mode in {"robust", "robust_coverage", "v9_robust_coverage"}:


                avg_score = 0.70 * chain_mean + 0.30 * global_partial_coverage
            else:
                avg_score = chain_mean
            stage3_debug_mode = (
                "v7_1_current"
                if stage3_mode in {"v7_1", "v7_1_current", "v7.1", "v7.1_current"}
                else "robust_coverage"
                if stage3_mode in {"robust", "robust_coverage", "v9_robust_coverage"}
                else "coverage"
            )
            s3_soft = StageResult(
                stage=3,
                score=avg_score,
                status="partial" if avg_score >= 0.50 else "hypothesis",
                detail=(
                    f"Chain aggregation ({stage3_debug_mode}): "
                    f"{len(chain_scores)} chains, avg={avg_score:.3f}"
                ),
                debug={
                    "stage3_mode": stage3_debug_mode,
                    "aggregation": (
                        "0.70_chain_mean_plus_0.30_global_partial_edge_coverage"
                        if stage3_debug_mode == "robust_coverage"
                        else "mean_chain_scores_then_coverage_to_5"
                    ),
                    "chain_scores": [round(float(s), 6) for s in chain_scores],
                    "chain_mean": round(float(chain_mean), 6),
                    "global_partial_coverage": round(float(global_partial_coverage), 6),
                    "total_supported": int(total_supported),
                    "total_weak": int(total_weak),
                    "total_steps": int(total_steps),
                    "score": round(float(avg_score), 6),
                },
            )

        s3_score = self._coverage_to_5(avg_score)
        s3_status = ("verified" if s3_score >= 4
                     else "partial" if s3_score >= 3
                     else "hypothesis")
        s3 = StageResult(
            stage=3, score=float(s3_score), status=s3_status,
            detail=s3_soft.detail,
            debug=s3_soft.debug,
        )

        result = MatchResult(stage1=s1, stage2=s2, stage3=s3)
        result.compute_overall()
        return result


    def _parse_chain_goal(self, chain_dsl: str) -> Optional[Predicate]:

        chain_dsl = chain_dsl.replace('→', '>>').replace('-->', '>>')
        parts = [p.strip() for p in chain_dsl.split('>>') if p.strip()]
        if not parts:
            return None
        last = self._clean_dsl_part(parts[-1])

        m = re.match(r'^(\w[\w\-/]*)\s*\((.*)?\)\s*$', last)
        if m:
            functor = m.group(1)
            arg_str = m.group(2) or ''
            return Predicate(
                functor=functor,
                args=[
                    create_argument("agent", "?", "variable"),
                    create_argument("target", arg_str, "concept"),
                ] if arg_str else [create_argument("agent", "?", "variable")]
            )

        m2 = re.match(r'^(\w+)\s*=\s*(.+)$', last)
        if m2:
            return Predicate(
                functor=m2.group(1),
                args=[
                    create_argument("agent", "?", "variable"),
                    create_argument("value", m2.group(2).strip(), "concept"),
                ]
            )
        return None

    def _clean_dsl_part(self, text: str) -> str:


        text = re.sub(r'\s*\([^)]*\)', lambda m: (
            '' if re.match(r'^\s*[\d\.\-\s]+\s*\w*\s*$', m.group()) else m.group()
        ), text)

        text = re.sub(r'^\w+\s*=\s*', '', text.strip())
        return text.strip()[:80]
