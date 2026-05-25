"""Optional auxiliary ontology support for adjuvant verification."""


from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "vaxjo_data" / "Vaxjo_LLM_VO" / "Vaxjo_LLM_VO"
VO_ADJUVANT_CSV = RAW_ROOT / "VO" / "src" / "templates" / "vaccine_adjuvant.csv"
VAXJO_MECHANISM_JSONL = (
    RAW_ROOT
    / "Vaxjo-LLM"
    / "v1"
    / "Outputs"
    / "Vaxjo_PMIDs_mechanism_summary_raw_outputs_llama3.2.jsonl"
)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "vaxjo"


SOURCE_CONFIDENCE = {
    "vo_profile": 0.72,
    "vo_receptor": 0.66,
    "vaxjo_mechanism": 0.56,
    "design_text": 0.28,
    "global_effect": 0.18,
}


CATEGORY_LABELS = {
    "th1": "Th1-biased/cellular response",
    "th2": "Th2-biased/humoral skew",
    "th17": "Th17 response",
    "treg": "Treg/regulatory response",
    "antibody": "antibody or B-cell response",
    "cd8_ctl": "CD8/CTL cytotoxic response",
    "dc_apc": "dendritic cell/APC activation",
    "cytokine": "cytokine or interferon production",
    "type_i_ifn": "type I interferon response",
    "tlr2": "TLR2 recognition",
    "tlr3": "TLR3 recognition",
    "tlr4": "TLR4 recognition",
    "tlr5": "TLR5 recognition",
    "tlr7": "TLR7 recognition",
    "tlr8": "TLR8 recognition",
    "tlr9": "TLR9 recognition",
    "tlr7_8": "TLR7/8 recognition",
    "nod2": "NOD2 recognition",
    "nlrp3": "NLRP3 inflammasome",
    "clr": "C-type lectin receptor recognition",
    "sting": "cGAS/STING pathway",
    "rig_i": "RIG-I/MDA5 pathway",
    "prr": "pattern-recognition receptor sensing",
    "inflammasome": "inflammasome activation",
    "lymph_node": "lymph-node targeting or dLN response",
    "mucosal": "mucosal immune response",
    "uptake": "antigen/adjuvant uptake",
    "depot_release": "depot or sustained release",
    "cross_presentation": "cross-presentation",
    "antitumor": "anti-tumor immunity",
    "protection": "protective efficacy / pathogen reduction",
}


SPECIFIC_EFFECT_CATEGORIES = {
    "th1",
    "th2",
    "th17",
    "treg",
    "cd8_ctl",
    "type_i_ifn",
    "tlr2",
    "tlr3",
    "tlr4",
    "tlr5",
    "tlr7",
    "tlr8",
    "tlr9",
    "tlr7_8",
    "nod2",
    "nlrp3",
    "clr",
    "sting",
    "rig_i",
    "inflammasome",
    "mucosal",
    "cross_presentation",
    "antitumor",
    "protection",
}


STOP_ALIASES = {
    "",
    "adjuvant",
    "vaccine adjuvant",
    "synthetic",
    "combination",
    "microbial derivative",
    "emulsion",
    "mineral salt",
    "particulate antigen delivery system",
}


def _norm_text(value: Any) -> str:
    text = str(value or "").lower()
    text = (
        text.replace("µ", "u")
        .replace("μ", "u")
        .replace("γ", "gamma")
        .replace("α", "alpha")
        .replace("β", "beta")
        .replace("κ", "kappa")
        .replace("–", "-")
        .replace("—", "-")
    )
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def _norm_alias(value: Any) -> str:
    text = _norm_text(value)
    text = re.sub(r"\bvaccine\s+adjuvant\b", "", text)
    text = re.sub(r"[®™]", "", text)
    text = re.sub(r"[^a-z0-9/+.-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -")
    return text


def _flatten(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(_flatten(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(_flatten(v) for v in value)
    return str(value)


def _split_pipe(value: Any) -> List[str]:
    if value is None:
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def _alias_variants(value: Any) -> Set[str]:
    raw = str(value or "").strip()
    if not raw:
        return set()

    variants = {raw}
    stripped = re.sub(r"\bvaccine\s+adjuvant\b", "", raw, flags=re.I).strip()
    variants.add(stripped)


    for inner in re.findall(r"\(([^)]+)\)", raw):
        variants.add(inner)
    outside = re.sub(r"\([^)]*\)", "", raw).strip()
    if outside:
        variants.add(outside)

    cleaned: Set[str] = set()
    for item in variants:
        norm = _norm_alias(item)
        if len(norm) >= 3 and norm not in STOP_ALIASES:
            cleaned.add(norm)
    return cleaned


def _parse_embedded_json(raw: str) -> Optional[Dict[str, Any]]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def effect_categories(text: Any) -> Set[str]:


    t = _norm_text(text)
    if not t:
        return set()

    cats: Set[str] = set()
    compact = t.replace("-", "").replace("/", "")

    def has(*patterns: str) -> bool:
        return any(re.search(p, t) for p in patterns)

    if has(r"\bth\s*1\b", r"\bth1\b", r"ifn\s*-?\s*gamma", r"ifng", r"igg2a", r"igg2c", r"cellular immun", r"cell mediated"):
        cats.add("th1")
    if has(r"\bth\s*2\b", r"\bth2\b", r"igg1", r"il\s*-?\s*4", r"th2-skew", r"th2 dominant"):
        cats.add("th2")
    if has(r"\bth\s*17\b", r"\bth17\b", r"il\s*-?\s*17"):
        cats.add("th17")
    if has(r"\btreg\b", r"regulatory t", r"il\s*-?\s*10", r"regulatory response"):
        cats.add("treg")
    if has(r"antibod", r"\bigg\b", r"immunoglobulin", r"humoral", r"neutralizing", r"\bb cell", r"serum igg"):
        cats.add("antibody")
    if has(r"\bcd8\b", r"\bctl\b", r"cytotoxic", r"killing", r"cross[- ]?prim", r"effector t"):
        cats.add("cd8_ctl")
    if has(r"dendritic", r"\bdc\b", r"\bapc\b", r"antigen[- ]?present", r"maturation", r"\bpdc\b"):
        cats.add("dc_apc")
    if has(r"cytokine", r"interferon", r"\bifn\b", r"\btnf\b", r"il\s*-?\s*6", r"il\s*-?\s*12", r"pro[- ]?inflammatory"):
        cats.add("cytokine")
    if has(r"type i ifn", r"type i interferon", r"ifn[- ]?(alpha|beta)"):
        cats.add("type_i_ifn")

    if "tlr7/8" in t or "tlr-7/8" in t or "toll-like receptor 7 | toll-like receptor 8" in t:
        cats.update({"tlr7", "tlr8", "tlr7_8", "prr"})
    if has(r"\btlr\s*-?\s*2\b", r"toll[- ]like receptor 2"):
        cats.update({"tlr2", "prr"})
    if has(r"\btlr\s*-?\s*3\b", r"toll[- ]like receptor 3"):
        cats.update({"tlr3", "prr"})
    if has(r"\btlr\s*-?\s*4\b", r"toll[- ]like receptor 4"):
        cats.update({"tlr4", "prr"})
    if has(r"\btlr\s*-?\s*5\b", r"toll[- ]like receptor 5"):
        cats.update({"tlr5", "prr"})
    if has(r"\btlr\s*-?\s*7\b", r"toll[- ]like receptor 7"):
        cats.update({"tlr7", "prr"})
    if has(r"\btlr\s*-?\s*8\b", r"toll[- ]like receptor 8"):
        cats.update({"tlr8", "prr"})
    if has(r"\btlr\s*-?\s*9\b", r"toll[- ]like receptor 9"):
        cats.update({"tlr9", "prr"})
    if "tlr78" in compact:
        cats.update({"tlr7", "tlr8", "tlr7_8", "prr"})

    if has(r"\bnod2\b"):
        cats.update({"nod2", "prr"})
    if has(r"\bnlrp3\b"):
        cats.update({"nlrp3", "inflammasome"})
    if has(r"inflammasome"):
        cats.add("inflammasome")
    if has(r"dectin", r"mincle", r"c[- ]type lectin", r"clec"):
        cats.update({"clr", "prr"})
    if has(r"\bsting\b", r"\bcgas\b", r"mb21d1"):
        cats.update({"sting", "prr"})
    if has(r"rig[- ]?i", r"mda5", r"ifih1", r"helicase c"):
        cats.update({"rig_i", "prr"})
    if has(r"\bprr\b", r"pattern recognition"):
        cats.add("prr")

    if has(r"lymph node", r"\bdln\b", r"draining lymph", r"lymphoid organ"):
        cats.add("lymph_node")
    if has(r"mucosal", r"siga", r"\biga\b", r"peyer", r"intestinal", r"oral immun"):
        cats.add("mucosal")
    if has(r"uptake", r"endocyt", r"phagocyt", r"internalization"):
        cats.add("uptake")
    if has(r"depot", r"sustained release", r"slow release", r"controlled release"):
        cats.add("depot_release")
    if has(r"cross[- ]?present", r"mhc i"):
        cats.add("cross_presentation")
    if has(r"anti[- ]?tumou?r", r"tumou?r inhibition", r"tumou?r cell", r"melanoma"):
        cats.add("antitumor")
    if has(r"protect", r"efficacy", r"reduce infection", r"viral shedding", r"diarrhea", r"bacterial colonization", r"pathogen"):
        cats.add("protection")

    return cats


class VaxjoKnowledge:


    def __init__(
        self,
        vo_csv: Path = VO_ADJUVANT_CSV,
        mechanism_jsonl: Path = VAXJO_MECHANISM_JSONL,
    ):
        self.vo_csv = Path(vo_csv)
        self.mechanism_jsonl = Path(mechanism_jsonl)
        self.records: List[Dict[str, Any]] = []
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self.alias_index: Dict[str, List[str]] = defaultdict(list)
        self.global_category_counts: Counter[str] = Counter()
        self.available = False

        if self.vo_csv.exists():
            self._load_vo()
            if self.mechanism_jsonl.exists():
                self._load_mechanisms()
            self._add_manual_aliases()
            self._refresh_global_counts()
            self.available = bool(self.records)


    def _new_record(
        self,
        record_id: str,
        label: str,
        source: str,
        aliases: Iterable[str],
        parent: str = "",
        definition: str = "",
        definition_source: str = "",
        comment: str = "",
    ) -> Dict[str, Any]:
        record = {
            "id": record_id,
            "label": label,
            "source": source,
            "aliases": sorted(set(aliases)),
            "parent": parent,
            "definition": definition,
            "definition_source": definition_source,
            "comment": comment,
            "target_receptors": [],
            "immune_profiles": [],
            "mechanism_subtypes": [],
            "mechanism_summaries": [],
            "evidence_refs": [],
            "categories": defaultdict(set),
        }
        self.records.append(record)
        self._by_id[record_id] = record
        for alias in record["aliases"]:
            self._index_alias(alias, record_id)
        return record

    def _index_alias(self, alias: str, record_id: str) -> None:
        alias = _norm_alias(alias)
        if len(alias) < 3 or alias in STOP_ALIASES:
            return
        if record_id not in self.alias_index[alias]:
            self.alias_index[alias].append(record_id)

    def _load_vo(self) -> None:
        with open(self.vo_csv, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("ID") == "ID":
                    continue
                record_id = (row.get("ID") or "").strip()
                label = (row.get("LABEL") or "").strip()
                if not record_id or not label:
                    continue

                aliases = set(_alias_variants(label))
                for alt in _split_pipe(row.get("alternative label")):
                    aliases.update(_alias_variants(alt))

                record = self._new_record(
                    record_id=record_id,
                    label=label,
                    source="vo",
                    aliases=aliases,
                    parent=row.get("Parent", ""),
                    definition=row.get("definition", ""),
                    definition_source=row.get("definition source", ""),
                    comment=row.get("Comment", ""),
                )

                receptors = _split_pipe(row.get("target receptor"))
                profiles = _split_pipe(row.get("induces immune profile"))
                record["target_receptors"] = receptors
                record["immune_profiles"] = profiles
                self._add_categories(record, effect_categories(" ".join(receptors)), "vo_receptor")
                self._add_categories(record, effect_categories(" ".join(profiles)), "vo_profile")

    def _load_mechanisms(self) -> None:
        with open(self.mechanism_jsonl, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    outer = json.loads(line)
                except json.JSONDecodeError:
                    continue
                parsed = _parse_embedded_json(outer.get("raw", ""))
                if not parsed:
                    continue
                adjuvant_name = parsed.get("adjuvant") or outer.get("adjuvant") or ""
                if not adjuvant_name:
                    continue

                record = self._find_or_create_mechanism_record(
                    adjuvant_name, outer.get("row_index")
                )
                summary = str(parsed.get("summary") or "")
                if summary:
                    record["mechanism_summaries"].append(summary)
                    self._add_categories(record, effect_categories(summary), "vaxjo_mechanism")

                for item in parsed.get("mechanism_subtypes") or []:
                    subtype = (
                        item.get("mechanism subtype")
                        or item.get("mechanism_subtype")
                        or item.get("subtype")
                        or ""
                    )
                    if subtype:
                        record["mechanism_subtypes"].append(subtype)
                        self._add_categories(record, effect_categories(subtype), "vaxjo_mechanism")
                    refs = item.get("evidence_refs") or []
                    for ref in refs:
                        ref_text = str(ref).strip()
                        if ref_text and ref_text not in record["evidence_refs"]:
                            record["evidence_refs"].append(ref_text)

    def _find_or_create_mechanism_record(self, adjuvant_name: str, row_index: Any) -> Dict[str, Any]:
        variants = _alias_variants(adjuvant_name)
        for alias in variants:
            ids = self.alias_index.get(alias, [])
            if ids:
                record = self._by_id[ids[0]]
                for v in variants:
                    if v not in record["aliases"]:
                        record["aliases"].append(v)
                        self._index_alias(v, record["id"])
                return record

        record_id = f"VAXJO-LLM:{row_index}"
        return self._new_record(
            record_id=record_id,
            label=adjuvant_name,
            source="vaxjo_llm",
            aliases=variants,
        )

    def _add_manual_aliases(self) -> None:
        manual_pairs = [
            ("r848", "resiquimod"),
            ("r-848", "resiquimod"),
            ("tlr7/8 agonist", "resiquimod"),
            ("tlr7 8 agonist", "resiquimod"),
            ("mpla", "monophosphoryl lipid a"),
            ("mpl-a", "monophosphoryl lipid a"),
            ("cpg", "cpg dna"),
            ("cpg odn", "cpg dna"),
            ("cpg oligodeoxynucleotide", "cpg dna"),
            ("alum", "aluminum hydroxide"),
        ]
        for alias, target in manual_pairs:
            target_ids = self.alias_index.get(_norm_alias(target), [])
            for record_id in target_ids[:1]:
                record = self._by_id[record_id]
                norm_alias = _norm_alias(alias)
                if norm_alias not in record["aliases"]:
                    record["aliases"].append(norm_alias)
                self._index_alias(norm_alias, record_id)

    def _add_categories(self, record: Dict[str, Any], categories: Iterable[str], source: str) -> None:
        for cat in categories:
            record["categories"][cat].add(source)

    def _refresh_global_counts(self) -> None:
        self.global_category_counts.clear()
        for record in self.records:
            for cat in record["categories"]:
                self.global_category_counts[cat] += 1


    def design_text(self, props: Dict[str, Any]) -> str:
        return _flatten(props)

    def find_design_matches(self, props: Dict[str, Any], limit: int = 8) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        text = _norm_alias(self.design_text(props))
        if not text:
            return []

        hits: List[Tuple[int, Dict[str, Any]]] = []
        seen: Set[str] = set()
        aliases = sorted(self.alias_index.keys(), key=lambda x: (-len(x), x))
        for alias in aliases:
            if len(alias) < 3:
                continue

            if len(alias) <= 4:
                matched = re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text)
            else:
                matched = alias in text
            if not matched:
                continue
            for record_id in self.alias_index[alias]:
                if record_id in seen:
                    continue
                record = self._by_id.get(record_id)
                if not record:
                    continue
                seen.add(record_id)
                hit = self._public_record(record)
                hit["matched_alias"] = alias
                hits.append((len(alias), hit))
                if len(hits) >= limit:
                    break
            if len(hits) >= limit:
                break

        hits.sort(key=lambda item: item[0], reverse=True)
        return [hit for _, hit in hits[:limit]]

    def score_stage1_design(self, props: Dict[str, Any]) -> Dict[str, Any]:
        matches = self.find_design_matches(props)
        design_categories = effect_categories(self.design_text(props))
        if matches:
            score = min(1.0, 0.72 + 0.04 * min(len(matches), 5))
            detail = "matched Vaxjo adjuvant concepts: " + ", ".join(
                m["label"] for m in matches[:4]
            )
        elif design_categories:
            score = 0.55
            detail = "matched immune/adjuvant tokens in design text: " + ", ".join(
                sorted(design_categories)[:6]
            )
        else:
            score = 0.0
            detail = "no Vaxjo adjuvant concept matched"
        return {
            "score": score,
            "matches": matches,
            "categories": sorted(design_categories),
            "detail": detail,
        }

    def score_terminal_effect(self, pred: Any, props: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        pred_text = self._predicate_text(pred)
        query_categories = effect_categories(pred_text)
        if not self.available or not query_categories:
            return {
                "score": 0.0,
                "query_categories": sorted(query_categories),
                "matched_categories": [],
                "matches": [],
                "detail": "no Vaxjo effect category matched",
                "source_refs": [],
            }

        props = props or {}
        matches = self.find_design_matches(props)
        design_categories = effect_categories(self.design_text(props))

        best_score = 0.0
        best_overlap: Set[str] = set()
        best_sources: Set[str] = set()
        best_refs: List[str] = []
        best_label = ""

        for hit in matches:
            rec_categories = set(hit.get("categories", {}).keys())
            overlap = query_categories & rec_categories
            if not overlap:
                continue
            source_score = 0.0
            sources: Set[str] = set()
            for cat in overlap:
                cat_sources = set(hit["categories"].get(cat, []))
                sources.update(cat_sources)
                for source in cat_sources:
                    source_score = max(source_score, SOURCE_CONFIDENCE.get(source, 0.65))

            specific_overlap = bool(overlap & SPECIFIC_EFFECT_CATEGORIES)
            if "vaxjo_mechanism" in sources and "vo_profile" not in sources and "vo_receptor" not in sources:


                if not specific_overlap:
                    source_score = min(source_score, 0.44)
                elif len(hit.get("evidence_refs", [])) >= 2:
                    source_score = min(0.62, source_score + 0.04)
            if source_score > best_score:
                best_score = source_score
                best_overlap = overlap
                best_sources = sources
                best_refs = hit.get("evidence_refs", [])[:6]
                best_label = hit.get("label", "")

        direct_overlap = query_categories & design_categories
        if direct_overlap and SOURCE_CONFIDENCE["design_text"] > best_score:
            best_score = SOURCE_CONFIDENCE["design_text"]
            best_overlap = direct_overlap
            best_sources = {"design_text"}
            best_refs = []
            best_label = "design text"

        global_overlap = {
            cat for cat in query_categories if self.global_category_counts.get(cat, 0) >= 3
        }
        if global_overlap and SOURCE_CONFIDENCE["global_effect"] > best_score:
            best_score = SOURCE_CONFIDENCE["global_effect"]
            best_overlap = global_overlap
            best_sources = {"global_effect"}
            best_refs = []
            best_label = "Vaxjo global effect ontology"

        labels = [CATEGORY_LABELS.get(cat, cat) for cat in sorted(best_overlap)]
        detail = (
            f"Vaxjo support via {best_label}: "
            f"{', '.join(labels) if labels else 'no overlap'}"
        )
        if best_sources:
            detail += f" [{', '.join(sorted(best_sources))}]"

        return {
            "score": best_score,
            "query_categories": sorted(query_categories),
            "matched_categories": sorted(best_overlap),
            "matches": matches,
            "detail": detail,
            "source_refs": best_refs,
            "source_label": best_label,
            "source_types": sorted(best_sources),
        }

    def _predicate_text(self, pred: Any) -> str:
        functor = getattr(pred, "functor", "")
        values = []
        for arg in getattr(pred, "args", []) or []:
            value = getattr(arg, "value", "")
            if value not in ("?", "", None):
                values.append(str(value))
        return " ".join([str(functor)] + values)


    def export_processed(self, out_dir: Path = PROCESSED_DIR) -> Dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        records = [self._public_record(r) for r in self.records]
        stats = {
            "n_records": len(records),
            "n_aliases": len(self.alias_index),
            "n_records_with_receptors": sum(1 for r in records if r["target_receptors"]),
            "n_records_with_profiles": sum(1 for r in records if r["immune_profiles"]),
            "n_records_with_mechanisms": sum(1 for r in records if r["mechanism_subtypes"]),
            "category_counts": dict(self.global_category_counts.most_common()),
            "source_files": {
                "vo_adjuvant_csv": str(self.vo_csv),
                "vaxjo_mechanism_jsonl": str(self.mechanism_jsonl),
            },
        }

        with open(out_dir / "vaxjo_adjuvant_ontology.json", "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        with open(out_dir / "vaxjo_effect_ontology.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "category_labels": CATEGORY_LABELS,
                    "category_counts": stats["category_counts"],
                    "source_confidence": SOURCE_CONFIDENCE,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        with open(out_dir / "vaxjo_profile_report.md", "w", encoding="utf-8") as f:
            f.write(self._profile_markdown(stats))
        return stats

    def _public_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": record["id"],
            "label": record["label"],
            "source": record["source"],
            "aliases": sorted(set(record.get("aliases", []))),
            "parent": record.get("parent", ""),
            "definition": record.get("definition", ""),
            "definition_source": record.get("definition_source", ""),
            "comment": record.get("comment", ""),
            "target_receptors": list(record.get("target_receptors", [])),
            "immune_profiles": list(record.get("immune_profiles", [])),
            "mechanism_subtypes": list(dict.fromkeys(record.get("mechanism_subtypes", []))),
            "mechanism_summaries": list(dict.fromkeys(record.get("mechanism_summaries", []))),
            "evidence_refs": list(dict.fromkeys(record.get("evidence_refs", []))),
            "categories": {
                cat: sorted(sources)
                for cat, sources in record.get("categories", {}).items()
            },
        }

    def _profile_markdown(self, stats: Dict[str, Any]) -> str:
        lines = [
            "# Vaxjo Processed Knowledge Profile",
            "",
            f"- Records: {stats['n_records']}",
            f"- Aliases: {stats['n_aliases']}",
            f"- Records with VO target receptors: {stats['n_records_with_receptors']}",
            f"- Records with VO immune profiles: {stats['n_records_with_profiles']}",
            f"- Records with Vaxjo mechanism subtypes: {stats['n_records_with_mechanisms']}",
            "",
            "## Effect Category Counts",
            "",
        ]
        for cat, count in Counter(stats["category_counts"]).most_common(30):
            lines.append(f"- {cat}: {count}")
        lines.extend(
            [
                "",
                "## Integration Notes",
                "",
                "- VO target receptor and immune profile fields are used as high-confidence ontology support.",
                "- Vaxjo LLM-mined mechanism summaries are used as medium-confidence candidate evidence.",
                "- Global effect concepts are deliberately weak and cannot by themselves provide strong proof.",
            ]
        )
        return "\n".join(lines) + "\n"


def main() -> None:
    kb = VaxjoKnowledge()
    stats = kb.export_processed()
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
