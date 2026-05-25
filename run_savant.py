#!/usr/bin/env python
"""Run SAVANT generation and verification."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from savant.grounding.predicate_core import Predicate, create_argument


DEFAULT_MODEL = os.getenv("SAVANT_EMBEDDING_MODEL", "FremyCompany/BioLORD-2023-M")
DEFAULT_GENERATOR_MODEL = os.getenv("SAVANT_GENERATOR_MODEL", "gpt-4-turbo")
DEFAULT_STAGE12_INDEX = PROJECT_ROOT / "theorem_library" / "index_stage12_runtime"
DEFAULT_STAGE3_INDEX = PROJECT_ROOT / "theorem_library" / "index_stage3_effective"

CLAIM_FUNCTOR_ALIASES = {
    "Improve": "Enhance",
    "Amplify": "Enhance",
    "Maximize": "Enhance",
    "Generate": "Induce",
    "Prevent": "Avoid",
    "Minimize": "Reduce",
    "Eradicate": "Reduce",
    "Bias": "Tune",
    "Synergize": "Enhance",
}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def _unwrap_design(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        if "structured_design" in obj:
            return _unwrap_design(obj.get("structured_design") or {})
        if "formalized_design" in obj:
            return _unwrap_design(obj.get("formalized_design") or {})
        if "vaccine" in obj:
            return obj
    raise ValueError("Cannot find a structured design with key 'vaccine'.")


def load_design_input(input_path: str, qid: str = "", row_index: int = 0) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    path = Path(input_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"No JSONL rows found in {path}")
        if qid:
            selected = next((row for row in rows if str(row.get("id", "")) == qid), None)
            if selected is None:
                raise ValueError(f"qid={qid} not found in {path}")
        else:
            if row_index < 0 or row_index >= len(rows):
                raise IndexError(f"row_index={row_index} out of range for {path} ({len(rows)} rows)")
            selected = rows[row_index]
        return _unwrap_design(selected), {
            "input_path": _display_path(path),
            "input_format": "jsonl",
            "selected_id": selected.get("id", ""),
            "row_index": rows.index(selected),
            "total_rows": len(rows),
        }

    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        if qid:
            selected = next((row for row in obj if str(row.get("id", "")) == qid), None)
            if selected is None:
                raise ValueError(f"qid={qid} not found in list JSON {path}")
        else:
            if row_index < 0 or row_index >= len(obj):
                raise IndexError(f"row_index={row_index} out of range for {path} ({len(obj)} rows)")
            selected = obj[row_index]
        return _unwrap_design(selected), {
            "input_path": _display_path(path),
            "input_format": "json-list",
            "selected_id": selected.get("id", ""),
            "row_index": obj.index(selected),
            "total_rows": len(obj),
        }

    return _unwrap_design(obj), {
        "input_path": _display_path(path),
        "input_format": "json",
        "selected_id": obj.get("id", ""),
    }


def _component_names(components: List[Dict[str, Any]]) -> List[str]:
    return [str(c.get("name", "")) for c in (components or []) if c.get("name")]


def _component_types(components: List[Dict[str, Any]]) -> List[str]:
    return [str(c.get("component_type", "")) for c in (components or []) if c.get("component_type")]


def extract_design_props(design: Dict[str, Any]) -> Dict[str, Any]:
    adjuvant = design["vaccine"]["adjuvant"]
    preparation = adjuvant.get("preparation") or {}
    components = adjuvant.get("components") or []
    return {
        "Type": adjuvant.get("type", ""),
        "Size": adjuvant.get("particle_size") or adjuvant.get("size", ""),
        "Method": preparation.get("composition_type", ""),
        "Contains": _component_names(components),
        "ParticleSize": adjuvant.get("particle_size") or adjuvant.get("size", ""),
        "Shape": adjuvant.get("shape", ""),
        "ZetaPotential": adjuvant.get("zeta_potential", ""),
        "PreparationName": preparation.get("name", ""),
        "PreparationDetails": {
            "preparation_details": preparation.get("preparation_details"),
            "mixing_conditions": preparation.get("mixing_conditions"),
            "conjugation_chemistry": preparation.get("conjugation_chemistry"),
            "assembly_method": preparation.get("assembly_method"),
            "encapsulation_details": preparation.get("encapsulation_details"),
            "emulsification_method": preparation.get("emulsification_method"),
            "encapsulation_technique": preparation.get("encapsulation_technique"),
            "crosslinking_agent": preparation.get("crosslinking_agent"),
            "cargo": preparation.get("cargo"),
            "carrier": preparation.get("carrier"),
            "complexation": preparation.get("complexation"),
        },
        "ComponentTypes": _component_types(components),
        "ComponentDetails": components,
        "OilPhase": adjuvant.get("oil_phase"),
        "WaterPhase": adjuvant.get("water_phase"),
        "Target": adjuvant.get("target"),
        "FunctionalGroups": adjuvant.get("functional_groups"),
        "Salt": adjuvant.get("salt"),
        "Concentration": adjuvant.get("concentration"),
        "Polymer": adjuvant.get("polymer"),
        "Crosslinker": adjuvant.get("crosslinker"),
        "NeedleLength": adjuvant.get("needle_length"),
        "NeedleMaterial": adjuvant.get("needle_material"),
        "Properties": adjuvant.get("properties"),
    }


def extract_antigen_props(design: Dict[str, Any]) -> Dict[str, Any]:
    antigen = design.get("vaccine", {}).get("antigen", {}) or {}
    preparation = antigen.get("preparation") or {}
    components = antigen.get("components") or []
    props: Dict[str, Any] = {}
    antigen_type = antigen.get("type") or antigen.get("antigen_type")
    if antigen_type:
        props["AgType"] = antigen_type
    if antigen.get("size"):
        props["AgSize"] = antigen.get("size")
    if preparation.get("composition_type"):
        props["AgMethod"] = preparation.get("composition_type")
    if os.getenv("SAVANT_STAGE1_ANTIGEN_CONTAINS", "0").strip().lower() in {"1", "true", "yes"}:
        contains = _component_types(components) or _component_names(components)
        if contains:
            props["AgContains"] = contains
    return props


def parse_claims_dsl(claims_dsl: List[str]) -> List[Tuple[str, Predicate]]:
    parsed = []
    for raw in claims_dsl or []:
        match = re.search(r"(?:^|=)\s*(\w[\w-]*)\(([^)]*)\)\s*$", str(raw))
        if match:
            functor = CLAIM_FUNCTOR_ALIASES.get(match.group(1), match.group(1))
            arg_parts = [item.strip() for item in match.group(2).split(",") if item.strip()]
            target = arg_parts[0] if arg_parts else "?"
            pred = Predicate(
                functor=functor,
                args=[
                    create_argument("agent", "?", "variable"),
                    create_argument("target", target, "concept"),
                ],
                source="claims_dsl",
            )
        else:
            pred = Predicate(
                functor="UnknownClaim",
                args=[create_argument("agent", "?", "variable")],
                source="claims_dsl",
            )
        parsed.append((str(raw), pred))
    return parsed


def parse_mechanism_terminals(mechanisms_dsl: List[str]) -> List[Tuple[str, str, Predicate]]:
    parsed = []
    for chain in mechanisms_dsl or []:
        norm = str(chain).replace("→", ">>").replace("-->", ">>")
        norm = re.sub(r"(?<!-)->", ">>", norm)
        parts = [part.strip() for part in norm.split(">>") if part.strip()]
        if not parts:
            continue
        terminal = parts[-1]
        match = re.match(r"^(\w[\w\-/]*)\s*\((.*)?\)\s*$", terminal)
        if match:
            arg_parts = [item.strip() for item in (match.group(2) or "").split(",") if item.strip()]
            pred = Predicate(
                functor=match.group(1),
                args=[
                    create_argument("agent", "?", "variable"),
                    create_argument("target", arg_parts[0] if arg_parts else "?", "concept"),
                ],
                source="mechanisms_dsl_terminal",
            )
        else:
            pred = Predicate(
                functor=terminal.split("=")[0].strip() if "=" in terminal else terminal,
                args=[create_argument("agent", "?", "variable")],
                source="mechanisms_dsl_terminal",
            )
        parsed.append((str(chain), terminal, pred))
    return parsed


def predicate_to_dict(pred: Predicate) -> Dict[str, Any]:
    return {
        "functor": pred.functor,
        "source": pred.source,
        "args": [
            {
                "role": arg.role,
                "value": _json_safe(arg.value),
                "value_type": arg.value_type,
            }
            for arg in pred.args
        ],
    }


def stage_result_to_dict(stage_result: Any) -> Dict[str, Any]:
    return {
        "stage": stage_result.stage,
        "score": stage_result.score,
        "status": stage_result.status,
        "detail": stage_result.detail,
        "debug": _json_safe(getattr(stage_result, "debug", {})),
        "evidence": [_json_safe(asdict(ev)) for ev in stage_result.evidence],
    }


def collect_cited_papers(*stage_groups: Any) -> List[str]:
    papers: Dict[str, None] = {}
    for group in stage_groups:
        if not group:
            continue
        items = group if isinstance(group, list) else [group]
        for stage_result in items:
            for evidence in getattr(stage_result, "evidence", []):
                paper = (evidence.source_paper or "").strip()
                if paper:
                    papers.setdefault(paper, None)
    return list(papers.keys())


def evaluate_design(
    design: Dict[str, Any],
    input_meta: Dict[str, Any],
    model_path: str,
    stage12_index: str,
    stage3_index: str,
) -> Dict[str, Any]:
    from savant.theorem.knowledge_db import TheoremDatabase
    from savant.verification.logic_matcher import LogicMatcher

    stage12_db = TheoremDatabase(model_name_or_path=model_path, index_dir=stage12_index)
    stage3_db = None
    if Path(stage3_index).resolve() != Path(stage12_index).resolve():
        stage3_db = TheoremDatabase(model_name_or_path=model_path, index_dir=stage3_index)
    matcher = LogicMatcher(knowledge_db=stage12_db, stage3_knowledge_db=stage3_db)

    design_props = extract_design_props(design)
    antigen_props = extract_antigen_props(design)
    parsed_claims = parse_claims_dsl(design.get("claims_dsl", []))
    mechanism_terminals = parse_mechanism_terminals(design.get("mechanisms_dsl", []))

    stage2_source = os.getenv("SAVANT_STAGE2_SOURCE", "claims").strip().lower()
    if stage2_source == "claims" and parsed_claims:
        stage2_targets = [(raw, raw, pred, "claims_dsl") for raw, pred in parsed_claims]
    else:
        stage2_targets = [
            (chain, terminal, pred, "mechanism_terminal")
            for chain, terminal, pred in mechanism_terminals
        ]

    stage1_result = matcher.prove_design_sub_propositions(
        design_props,
        antigen_properties=antigen_props or None,
    )
    stage2_results = [
        matcher.prove_predicate_sub_proposition(pred, adjuvant_properties=design_props)
        for _, _, pred, _ in stage2_targets
    ]
    stage3_results = [
        matcher.prove_inference_chain_sub_proposition(chain)
        for chain in design.get("mechanisms_dsl", [])
    ]
    final = matcher.verify(
        adjuvant_properties=design_props,
        claims=[pred for _, _, pred, _ in stage2_targets],
        mechanisms_dsl=design.get("mechanisms_dsl", []),
        antigen_properties=antigen_props or None,
    )

    vaccine = design.get("vaccine", {})
    adjuvant = vaccine.get("adjuvant", {})
    antigen = vaccine.get("antigen", {})
    return {
        "input": _json_safe(input_meta),
        "runtime": {
            "stage12_rule_count": stage12_db.get_rule_count(),
            "stage3_rule_count": (stage3_db or stage12_db).get_rule_count(),
            "stage12_index": _display_path(Path(stage12_index)),
            "stage3_index": _display_path(Path(stage3_index)),
            "model": model_path if not Path(model_path).exists() else Path(model_path).name,
        },
        "design_summary": {
            "vaccine_name": vaccine.get("name", ""),
            "target_disease": vaccine.get("target_disease", ""),
            "adjuvant_name": adjuvant.get("name", ""),
            "adjuvant_type": adjuvant.get("type", ""),
            "antigen_name": antigen.get("name", ""),
        },
        "grounding": {
            "adjuvant_properties": _json_safe(design_props),
            "antigen_properties": _json_safe(antigen_props),
            "claims_dsl": [
                {"raw": raw, "predicate": predicate_to_dict(pred)}
                for raw, pred in parsed_claims
            ],
            "mechanism_terminals": [
                {
                    "chain": chain,
                    "terminal_text": terminal,
                    "terminal_predicate": predicate_to_dict(pred),
                }
                for chain, terminal, pred in mechanism_terminals
            ],
        },
        "stage1": {
            "soft_result": stage_result_to_dict(stage1_result),
            "final_1_to_5": stage_result_to_dict(final.stage1),
        },
        "stage2": {
            "per_target_soft_results": [
                {
                    "source": source,
                    "chain": chain,
                    "target_text": target,
                    "target_predicate": predicate_to_dict(pred),
                    "result": stage_result_to_dict(result),
                }
                for (chain, target, pred, source), result in zip(stage2_targets, stage2_results)
            ],
            "final_1_to_5": stage_result_to_dict(final.stage2),
        },
        "stage3": {
            "per_chain_soft_results": [
                {"chain": chain, "result": stage_result_to_dict(result)}
                for chain, result in zip(design.get("mechanisms_dsl", []), stage3_results)
            ],
            "final_1_to_5": stage_result_to_dict(final.stage3),
        },
        "final": {
            "weights": {"stage1": 0.30, "stage2": 0.30, "stage3": 0.40},
            "overall_score": final.overall_score,
            "verdict": final.verdict,
        },
        "cited_papers": collect_cited_papers(stage1_result, stage2_results, stage3_results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SAVANT generation and evidence-grounded verification.")
    parser.add_argument("--input", default="", help="Structured design JSON or JSONL file.")
    parser.add_argument("--query", default="", help="Natural-language design request. If provided, SAVANT generates a structured design first.")
    parser.add_argument("--qid", default="", help="Select a JSONL/list item by id.")
    parser.add_argument("--row-index", type=int, default=0, help="Select a JSONL/list item by row index.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model path or model id.")
    parser.add_argument("--generator-mode", default="api", choices=["api", "local", "mock"], help="Generator backend.")
    parser.add_argument("--generator-model", default=DEFAULT_GENERATOR_MODEL, help="Generator model name or local model path.")
    parser.add_argument("--api-key", default="", help="Optional API key for generator API mode. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--base-url", default="", help="Optional API base URL for generator API mode. Defaults to OPENAI_BASE_URL.")
    parser.add_argument("--device", default=None, help='Local generator device, e.g. "cuda:0", "cuda", or "cpu".')
    parser.add_argument("--device-map", default=None, choices=["auto", "balanced", "balanced_low_0"], help="Optional local generator device_map.")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True, help="Enable model-specific thinking traces for local generation.")
    parser.add_argument("--max-retries", type=int, default=2, help="Maximum generator retries.")
    parser.add_argument("--stage12-index", default=str(DEFAULT_STAGE12_INDEX), help="Stage 1/2 theorem index.")
    parser.add_argument("--stage3-index", default=str(DEFAULT_STAGE3_INDEX), help="Stage 3 theorem index.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "savant_results.json"))
    parser.add_argument("--no-output", action="store_true", help="Print the result instead of saving it.")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")
    os.environ["SAVANT_EMBEDDING_MODEL"] = args.model
    os.environ["SAVANT_TOKENIZER_MODEL"] = args.model
    generated_payload = None
    if args.query:
        from savant.generation.generator import LLMGenerator

        generator = LLMGenerator(
            mode=args.generator_mode,
            model_path=args.generator_model,
            api_key=args.api_key or None,
            base_url=args.base_url or None,
            device=args.device,
            device_map=args.device_map,
            enable_thinking=args.enable_thinking,
            max_retries=args.max_retries,
        )
        generated = generator.generate(args.query)
        design = generated.formalized_design.model_dump(mode="json")
        generated_payload = json.loads(generated.model_dump_json())
        input_meta = {
            "input_format": "generated",
            "selected_id": "",
            "query": args.query,
            "generator_mode": args.generator_mode,
            "generator_model": args.generator_model if not Path(args.generator_model).exists() else Path(args.generator_model).name,
        }
    else:
        if not args.input:
            parser.error("Either --input or --query is required.")
        design, input_meta = load_design_input(args.input, qid=args.qid, row_index=args.row_index)
    result = evaluate_design(
        design=design,
        input_meta=input_meta,
        model_path=args.model,
        stage12_index=args.stage12_index,
        stage3_index=args.stage3_index,
    )
    if generated_payload is not None:
        result["generation"] = generated_payload

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.no_output:
        print(payload)
    else:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
        print(f"SAVANT verdict: {result['final']['verdict']} ({result['final']['overall_score']:.4f})")
        print(f"Saved result: {_display_path(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
