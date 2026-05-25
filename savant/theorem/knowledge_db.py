"""Theorem database and vector retrieval utilities for SAVANT."""

import os
import sys
import json
import pickle
import logging
import re
import numpy as np
import faiss
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer


_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from savant.grounding.predicate_core import Predicate, create_argument


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PredicateRule:
    rule_id: str
    condition_predicates: List[Predicate]
    effect_predicate: Predicate
    paper_id: str
    confidence: float = 1.0


class TheoremDatabase:


    def __init__(self,
                 model_name_or_path: Optional[str] = None,
                 index_dir: Optional[str] = None):
        if model_name_or_path is None:
            model_name_or_path = os.getenv(
                "SAVANT_EMBEDDING_MODEL",
                "FremyCompany/BioLORD-2023-M",
            )
        if index_dir is None:
            index_dir = str(
                Path(__file__).resolve().parents[2]
                / "theorem_library"
                / "index_stage12_runtime"
            )
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.index_dir / "rules.index"
        self.metadata_path = self.index_dir / "rules_metadata.pkl"


        if os.path.exists(model_name_or_path):
            logger.info('Runtime diagnostic.')
        else:
            logger.warning('Runtime diagnostic.')


        self.model = SentenceTransformer(model_name_or_path)


        self.index = None
        self.rules_metadata: Dict[int, dict] = {}

        self._eff_functor_index: Dict[str, List[int]] = {}

        self._cond_functor_index: Dict[str, List[int]] = {}

        self._cond_value_index: Dict[str, Dict[str, List[int]]] = {}
        self._eff_value_index: Dict[str, Dict[str, List[int]]] = {}
        self._load_index()

    def _load_index(self):

        if self.index_path.exists() and self.metadata_path.exists():
            logger.info('Runtime diagnostic.')
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.metadata_path, 'rb') as f:
                    self.rules_metadata = pickle.load(f)
                logger.info('Runtime diagnostic.')
                self._build_functor_index()
                self._load_value_indices()
            except Exception as e:
                logger.error('Runtime diagnostic.')
                self._init_empty_index()
        else:
            logger.info('Runtime diagnostic.')
            self._init_empty_index()

    def _init_empty_index(self):


        dummy_vec = self.model.encode(["test"])
        dim = dummy_vec.shape[1]
        logger.info('Runtime diagnostic.')


        self.index = faiss.IndexFlatIP(dim)
        self.rules_metadata = {}

    def _save_index(self):

        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.rules_metadata, f)
        logger.info('Runtime diagnostic.')

    def add_rules_from_json(self, json_dir: str = "data/extracted_rules"):

        json_path = Path(json_dir)
        if not json_path.exists():
            logger.warning('Runtime diagnostic.')
            return

        new_rules = []

        files = list(json_path.glob("*.json"))
        logger.info('Runtime diagnostic.')

        for file in files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    rules_list = json.load(f)

                    new_rules.extend(rules_list)
            except Exception as e:
                logger.error('Runtime diagnostic.')

        if not new_rules:
            logger.info('Runtime diagnostic.')
            return


        if len(new_rules) <= self.index.ntotal:
            logger.info('Runtime diagnostic.')
            return

        logger.info('Runtime diagnostic.')


        texts_to_embed = [r['effect'] for r in new_rules]

        logger.info('Runtime diagnostic.')


        embeddings = self.model.encode(texts_to_embed, normalize_embeddings=True, show_progress_bar=False)


        if self.index.ntotal == 0:
            self.index.add(embeddings)

            for i, rule in enumerate(new_rules):
                self.rules_metadata[i] = rule
        else:


            logger.info('Runtime diagnostic.')
            self._init_empty_index()
            self.index.add(embeddings)
            for i, rule in enumerate(new_rules):
                self.rules_metadata[i] = rule


        self._save_index()
        logger.info('Runtime diagnostic.')

    def get_rule_count(self) -> int:

        if self.index is None:
            return 0
        return self.index.ntotal

    def search_relevant_rules(self, query_claim: str, top_k: int = 5, threshold: float = 0.4) -> List[Dict]:

        if self.index is None or self.index.ntotal == 0:
            logger.warning('Runtime diagnostic.')
            return []


        query_vec = self.model.encode([query_claim], normalize_embeddings=True, show_progress_bar=False)


        D, I = self.index.search(query_vec, top_k)

        results = []


        for score, idx in zip(D[0], I[0]):
            if idx == -1: continue


            if score < threshold:
                continue

            rule = self.rules_metadata.get(idx)
            if rule:


                rule_copy = rule.copy()
                rule_copy['_similarity_score'] = float(score)
                results.append(rule_copy)

        logger.info('Runtime diagnostic.')
        return results

    def _convert_rule_to_predicates(self, rule: Dict) -> Optional['PredicateRule']:

        try:
            effect_dict = rule.get('effect')
            if not effect_dict or 'functor' not in effect_dict:
                logger.debug('Runtime diagnostic.')
                return None


            conditions_raw = rule.get('conditions')
            if not conditions_raw:
                single = rule.get('condition')
                conditions_raw = [single] if single else []

            if not conditions_raw:
                logger.debug('Runtime diagnostic.')
                return None


            condition_preds = []
            for cond_dict in conditions_raw:
                if not cond_dict or 'functor' not in cond_dict:
                    continue
                pred = self._dict_to_predicate(cond_dict)
                if pred:
                    condition_preds.append(pred)

            if not condition_preds:
                logger.debug('Runtime diagnostic.')
                return None

            effect_pred = self._dict_to_predicate(effect_dict)
            if not effect_pred:
                logger.debug('Runtime diagnostic.')
                return None

            pred_rule = PredicateRule(
                rule_id=rule.get('rule_id', 'unknown'),
                condition_predicates=condition_preds,
                effect_predicate=effect_pred,
                paper_id=rule.get('paper_id', 'unknown'),
                confidence=rule.get('confidence', 0.9)
            )
            return pred_rule

        except Exception as e:
            logger.error('Runtime diagnostic.', exc_info=True)
            return None

    def search_by_predicate(self, target_pred: 'Predicate', top_k: int = 5) -> List['PredicateRule']:


        query_str = str(target_pred)


        raw_rules = self.search_relevant_rules(query_str, top_k=top_k, threshold=0.3)


        predicate_rules = []
        for raw_rule in raw_rules:
            pred_rule = self._convert_rule_to_predicates(raw_rule)
            if pred_rule:
                predicate_rules.append(pred_rule)

        logger.info('Runtime diagnostic.')
        return predicate_rules

    def load_predicate_rules(self,
                             predicate_rules_path: str = "data/horn_clauses/predicate_rules_v3.json",
                             force_rebuild: bool = False):

        rules_path = Path(predicate_rules_path)
        if not rules_path.exists():
            logger.warning('Runtime diagnostic.')
            return

        logger.info('Runtime diagnostic.')
        with open(rules_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        rules = data.get('rules', [])
        logger.info('Runtime diagnostic.')

        if not rules:
            return


        if not force_rebuild and self.index.ntotal >= len(rules):
            logger.info('Runtime diagnostic.'
                        'Runtime diagnostic.')
            return


        if force_rebuild or self.index.ntotal > 0:
            logger.info('Runtime diagnostic.')
            self._init_empty_index()
            self.rules_metadata = {}


        rule_texts = []
        for rule in rules:
            conditions = rule.get('conditions') or []
            if not conditions:
                single = rule.get('condition')
                conditions = [single] if single else []

            cond_parts = []
            for cond in conditions:
                if not cond:
                    continue
                functor = cond.get('functor', '')
                args = cond.get('args', [])
                val = args[1]['value'] if len(args) > 1 else ''
                cond_parts.append(f"{functor}({val})" if val else functor)

            eff = rule.get('effect', {})
            eff_functor = eff.get('functor', '')
            eff_args = eff.get('args', [])
            eff_val = eff_args[1]['value'] if len(eff_args) > 1 else ''
            eff_str = f"{eff_functor}({eff_val})" if eff_val else eff_functor

            cond_str = ' AND '.join(cond_parts) if cond_parts else 'unknown'
            src = rule.get('source_type', '')
            evidence = rule.get('evidence_text', '')[:60]

            text = f"{cond_str} -> {eff_str}. [{src}] {evidence}"
            rule_texts.append(text)

        logger.info('Runtime diagnostic.')
        embeddings = self.model.encode(
            rule_texts, normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True
        )

        self.index.add(embeddings)


        for i, rule in enumerate(rules):
            self.rules_metadata[i] = rule

        self._save_index()
        self._build_functor_index()
        logger.info('Runtime diagnostic.')

    def _build_functor_index(self):

        self._eff_functor_index  = {}
        self._cond_functor_index = {}
        for idx, rule in self.rules_metadata.items():

            eff_f = rule.get('effect', {}).get('functor', '')
            if eff_f:
                self._eff_functor_index.setdefault(eff_f, []).append(idx)

            for cond in (rule.get('conditions') or []):
                cond_f = cond.get('functor', '') if cond else ''
                if cond_f:
                    self._cond_functor_index.setdefault(cond_f, []).append(idx)
        logger.info('Runtime diagnostic.'
                    'Runtime diagnostic.'
                    'Runtime diagnostic.')

    def _load_value_indices(self):

        cond_path = self.index_dir / "cond_value_index.json"
        eff_path = self.index_dir / "eff_value_index.json"

        if cond_path.exists():
            try:
                with open(cond_path, 'r', encoding='utf-8') as f:
                    self._cond_value_index = json.load(f)
                total_entries = sum(len(v) for v in self._cond_value_index.values())
                logger.info('Runtime diagnostic.')
            except Exception as e:
                logger.warning('Runtime diagnostic.')
                self._cond_value_index = {}
        else:
            logger.warning('Runtime diagnostic.')
            self._cond_value_index = {}

        if eff_path.exists():
            try:
                with open(eff_path, 'r', encoding='utf-8') as f:
                    self._eff_value_index = json.load(f)
                total_entries = sum(len(v) for v in self._eff_value_index.values())
                logger.info('Runtime diagnostic.')
            except Exception as e:
                logger.warning('Runtime diagnostic.')
                self._eff_value_index = {}
        else:
            logger.warning('Runtime diagnostic.')
            self._eff_value_index = {}

    def search_by_cond_value(self, functor: str, value: str) -> List[dict]:

        idxs = self._cond_value_index.get(functor, {}).get(value, [])
        if not idxs:

            normalized = value.replace('_', ' ')
            if normalized != value:
                idxs = self._cond_value_index.get(functor, {}).get(normalized, [])
        return [self.rules_metadata[i] for i in idxs if i in self.rules_metadata]

    def search_by_eff_value(self, functor: str, value: str) -> List[dict]:

        idxs = self._eff_value_index.get(functor, {}).get(value, [])
        if not idxs:

            normalized = value.replace('_', ' ')
            if normalized != value:
                idxs = self._eff_value_index.get(functor, {}).get(normalized, [])
        return [self.rules_metadata[i] for i in idxs if i in self.rules_metadata]

    def get_rules_by_eff_functor(self, functor: str) -> List[dict]:

        idxs = self._eff_functor_index.get(functor, [])
        if not idxs:

            key_map = {k.lower(): k for k in self._eff_functor_index.keys()}
            actual = key_map.get(functor.lower())
            if actual:
                idxs = self._eff_functor_index.get(actual, [])
        return [self.rules_metadata[i] for i in idxs if i in self.rules_metadata]

    def get_rules_by_cond_functor(self, functor: str) -> List[dict]:

        idxs = self._cond_functor_index.get(functor, [])
        if not idxs:
            key_map = {k.lower(): k for k in self._cond_functor_index.keys()}
            actual = key_map.get(functor.lower())
            if actual:
                idxs = self._cond_functor_index.get(actual, [])
        return [self.rules_metadata[i] for i in idxs if i in self.rules_metadata]

    def get_all_eff_functors(self) -> List[str]:

        return list(self._eff_functor_index.keys())

    def get_all_cond_functors(self) -> List[str]:

        return list(self._cond_functor_index.keys())

    def get_all_functors(self) -> List[str]:

        return sorted(set(self._eff_functor_index.keys()) | set(self._cond_functor_index.keys()))

    def _dict_to_predicate(self, pred_dict: Dict) -> Optional[Predicate]:

        if not pred_dict or 'functor' not in pred_dict:
            return None

        functor = pred_dict['functor']
        args_data = pred_dict.get('args', [])

        args = []
        for arg_data in args_data:
            role = arg_data.get('role', 'unknown')
            value_str = arg_data.get('value', '')
            value_type = arg_data.get('value_type', 'concept')


            if value_type == 'quantity' and isinstance(value_str, str):
                from savant.grounding.unit_utils import UnitParser


                if 'Quantity(' in value_str:

                    inner = value_str[value_str.find('(')+1:value_str.rfind(')')]
                else:
                    inner = value_str


                try:

                    if any(unit in inner for unit in ['nm', 'μm', 'mm', 'cm', 'm']):
                        value = UnitParser.parse_length(inner)

                    elif '%' in inner:
                        value = UnitParser.parse_percentage(inner)

                    elif 'fold' in inner.lower() or 'x' in inner.lower():
                        value = UnitParser.parse_fold_change(inner)

                    elif inner.startswith('P<') or inner.startswith('p<'):
                        value = inner

                    else:
                        value = UnitParser.parse_general(inner)
                except Exception as e:
                    logger.debug('Runtime diagnostic.')
                    value = value_str
            elif value_type == 'variable':
                value = value_str
            else:
                value = value_str

            args.append(create_argument(role, value, value_type))

        return Predicate(functor=functor, args=args, source="knowledge_base")
