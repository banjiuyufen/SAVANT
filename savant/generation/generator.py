"""LLM-based structured design generator for SAVANT."""

import os
import sys
import json
import logging
import torch
import re
from typing import Optional, Dict, Union, Tuple, Any, List
from pydantic import ValidationError

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None

try:
    import outlines
    from outlines import from_transformers, Generator
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    outlines = None
    from_transformers = None
    Generator = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    import instructor
    INSTRUCTOR_AVAILABLE = True
except ImportError:
    instructor = None
    INSTRUCTOR_AVAILABLE = False


_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


from savant.generation.schemas import CompleteResponse, ExperimentDesignData


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMGenerator:


    def __init__(self,
                 mode: str = 'api',
                 model_path: str = "gpt-4-turbo",
                 api_key: str = None,
                 base_url: str = None,
                 device: str = None,
                 device_map: Union[str, Dict, None] = None,
                 enable_thinking: bool = True,
                 max_retries: int = 2):

        self.mode = mode
        self.model_path = model_path
        self.enable_thinking = enable_thinking


        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device


        self.device_map = device_map

        self.max_retries = max_retries


        self.outlines_model = None
        self.outlines_text_generator = None
        self.outlines_json_generator = None
        self.hf_tokenizer = None

        logger.info('Runtime diagnostic.')


        if self.mode == 'api':
            from openai import OpenAI
            key = api_key or os.getenv("OPENAI_API_KEY")
            url = base_url or os.getenv("OPENAI_BASE_URL")
            if not key: raise ValueError("API mode requires OPENAI_API_KEY")


            self._raw_client = OpenAI(api_key=key, base_url=url)


            if INSTRUCTOR_AVAILABLE:
                self.client = instructor.from_openai(OpenAI(api_key=key, base_url=url))
                logger.info('Runtime diagnostic.')
            else:
                self.client = self._raw_client
                logger.warning('Runtime diagnostic.')
                logger.warning('Runtime diagnostic.')

        elif self.mode == 'local':
            if outlines is None or from_transformers is None or Generator is None:
                raise ImportError("Local mode requires 'outlines' library.")

            logger.info('Runtime diagnostic.')


            if self.device_map is not None:

                final_device_map = self.device_map
                logger.info('Runtime diagnostic.')
            elif self.device == "cpu":

                final_device_map = None
                logger.info('Runtime diagnostic.')
            else:

                if ":" in self.device:

                    final_device_map = self.device
                else:

                    final_device_map = "auto"
                logger.info('Runtime diagnostic.')


            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info('Runtime diagnostic.')
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")


            logger.info('Runtime diagnostic.')
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map=final_device_map
            )
            hf_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )


            self.hf_tokenizer = hf_tokenizer


            logger.info('Runtime diagnostic.')
            self.outlines_model = from_transformers(
                hf_model,
                hf_tokenizer,
                device_dtype=torch.float16
            )

            logger.info('Runtime diagnostic.')


            logger.info('Runtime diagnostic.')


            self.outlines_text_generator = Generator(self.outlines_model, output_type=None)


            logger.info('Runtime diagnostic.')
            self.outlines_json_generator = Generator(self.outlines_model, output_type=ExperimentDesignData)

            logger.info('Runtime diagnostic.')


        elif self.mode == 'mock':
            logger.info("Mock mode active.")

    def generate(self, user_query: str) -> CompleteResponse:

        try:
            if self.mode == 'local':
                return self._generate_local_outlines(user_query)
            elif self.mode == 'api':
                return self._generate_api_robust(user_query)
            elif self.mode == 'mock':
                return self._generate_mock(user_query)
        except Exception as e:
            logger.error(f"Generate process failed: {e}")
            raise e


    def _generate_local_outlines(self, query: str) -> CompleteResponse:


        logger.info(">>> [Local Phase 1] Generating Rationale (Free-form)...")
        logger.info(f"Enable Thinking Mode: {self.enable_thinking}")


        system_message = """You are an expert scientist in immunology.
Goal: Design a novel vaccine adjuvant system based on the user's request.
Instruction: Provide a detailed scientific rationale explaining the design principles, mechanisms of action, delivery strategy, and expected immune outcomes.
Write in natural scientific language. Do NOT output JSON or structured formats yet.If your design relies on specific physical properties (e.g., size, charge, molecular weight) for its mechanism of action, you MUST explicitly state their exact values in the rationale paragraph."""

        user_message = f"{query}"


        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]


        prompt_cot = self.hf_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )

        logger.info('Runtime diagnostic.')


        rationale_text = self.outlines_text_generator(
            prompt_cot,
            max_new_tokens=4096,
            temperature=0.8,
            top_p=1.0,
            top_k=50
        )


        if self.enable_thinking:

            stop_strings = ["<|im_end|>", "```json", "Formalized Design:"]
        else:

            stop_strings = ["<|im_end|>", "Formalized Design:", "```json"]

        for stop_str in stop_strings:
            if stop_str in rationale_text:
                rationale_text = rationale_text.split(stop_str)[0]
                break

        logger.info(f"Rationale generated ({len(rationale_text)} chars).")
        if self.enable_thinking and "<think>" in rationale_text:
            logger.info("✅ Thinking mode enabled, <think> tag detected in output.")
        elif self.enable_thinking:
            logger.warning("⚠️ Thinking mode enabled but NO <think> tag found in output.")


        logger.info(">>> [Local Phase 2] Generating Strict JSON (FSM Guided)...")
        logger.info('Runtime diagnostic.')


        system_message_json = """You are a strictly constrained data extraction assistant.
Task: Convert the provided Scientific Rationale into a strict structural design (JSON).

CRITICAL RULES:
1. **ZERO-HALLUCINATION (STRICT EXTRACTION):** You must ONLY extract entities, properties, and claims that are explicitly stated in the provided Scientific Rationale. DO NOT invent, hallucinate, or fabricate new components, values (like size or dose), pathways, or mechanisms. If a specific parameter is not mentioned in the rationale, you MUST leave it as null or omit it. Do not guess.

2. **MANDATORY COMPONENTS**: Vaccine antigen and adjuvant MUST have a "components" array listing all constituent components. Every component must have a "component_type" field with one of: "Protein", "Antibody", "NucleicAcid", "Lipid", "InorganicSalt", "Polymer", "General".

3. Adjuvant must have a "type" field with one of: "Nanoparticle", "Emulsion", "Molecule", "Inorganic_salt", "Hydrogel", "Microneedling".

4. **MANDATORY PREPARATION METHOD**: Vaccine, antigen, and adjuvant MUST have a "preparation" object. You must choose EXACTLY ONE preparation method from: "mix", "assemble", "conjugate", "emulsify", "encapsulate", "crosslink", "new_prepare". Each preparation method has its own specific fields:
   - mix: requires "composition_type": "mix", optional "mixing_conditions"
   - assemble: requires "composition_type": "assemble", optional "assembly_method"
   - conjugate: requires "composition_type": "conjugate", optional "conjugation_chemistry"
   - emulsify: requires "composition_type": "emulsify", optional "emulsification_method"
   - encapsulate: requires "composition_type": "encapsulate", optional "encapsulation_technique"
   - crosslink: requires "composition_type": "crosslink", optional "crosslinking_agent"
   - new_prepare: requires "composition_type": "new_prepare", optional "preparation_details"

5. **MANDATORY DSL OUTPUTS**: You MUST generate `claim_statements` and `mechanism_chains`. These are the authoritative FSM-masked outputs. The legacy `claims_dsl` and `mechanisms_dsl` strings will be serialized automatically; do not generate or rely on free-form DSL strings as the primary output.

6. **STRUCTURED CLAIM STATEMENTS (FSM-MASKED FUNCTORS)**: Generate `claim_statements` as an array of compact concept claim objects. Each object MUST have exactly the Stage 2 claim surface:
   - `functor`: one exact controlled claim functor from the schema enum.
   - `outcome`: compact outcome/concept phrase, NOT a sentence.

   Claim rules:
     * Do NOT generate `claims_dsl` directly; it is serialized from `claim_statements`.
     * Do NOT include treatment, control, comparator, magnitude, dose, group labels, or evidence fields in `claim_statements`.
     * Use only the schema enum for `functor`; do not invent verbs such as "Improve", "Generate", or "Prevent".
     * Keep `outcome` short and canonical. Use spaces, not snake_case.
     * Generate 2-5 high-confidence claim statements. Prefer fewer precise claims over many generic/redundant claims.

7. **STRUCTURED MECHANISM CHAINS (FSM-MASKED FUNCTORS)**: Generate `mechanism_chains` as an array of chain objects. Each object MUST have `nodes`, and each node MUST have:
   - `functor`: one exact controlled functor from the schema.
   - `value`: short canonical event phrase, NOT a sentence. Keep it under 8 words when possible. Put mechanisms, explanations, evidence, and qualifiers in the chain `rationale`, not inside node values. Use spaces, not snake_case. Good values: "draining lymph node", "DC endocytosis", "MHC I presentation", "TLR3 signaling", "type I IFN production".

   Example object:
   `{"nodes": [{"functor": "Size", "value": "150 nm"}, {"functor": "Target", "value": "draining lymph node"}, {"functor": "Activate", "value": "dendritic cell activation"}], "rationale": "Particle size supports lymph node drainage and APC activation."}`

   Mechanism rules:
     * `mechanism_chains` describes BIOLOGICAL CAUSAL MECHANISMS only: how adjuvant properties or first immune events lead to immune effects.
     * Each adjacent node must be a cause-to-effect relationship.
     * The final node of each chain is used as the Stage 2 D endpoint. Therefore, do not end a chain with pure upstream bookkeeping nodes such as Size, Shape, Type, Method, Route, Contains, Form, Display, Mix, Conjugate, Assemble, Encapsulate, Crosslink, New_Prepare, Release, Bind, Process, Trigger, Enable, or Target unless no downstream immune event is stated.
     * Do not end with internal receptor/pathway activation such as Activate(TLR signaling), Activate(STING pathway), Activate(TRIF pathway), or Activate(NF-kB pathway). Continue to a biological endpoint when supported, e.g. Mature(dendritic cells), Express(CD80 CD86), Secrete(type I IFN), Polarize(Th1 response), Prime(CD8 T cells), Enhance(antibody response), Neutralize(pathogen), Reduce(viral load), or Protect(survival).
     * A chain with only one node is invalid and will be rejected.
     * Generate 3-5 non-overlapping chains total; prefer 3 high-confidence chains over many weak/redundant chains.
     * Use 2-5 nodes per chain; prefer 3 nodes when the rationale supports a complete path.
     * Do not create a separate chain for every design attribute. Merge overlapping mechanisms into the main causal axes: delivery/uptake, PRR signaling/DC activation, and sustained release/immune persistence. Add a fourth or fifth chain only when it captures a clearly distinct mechanism.
     * Node values must be compact mechanism nodes, not explanatory sentences. Bad: `Size(150 nm nanoparticle size enables passive lymphatic drainage...)`. Good: `Size(150 nm)` then put the explanation in `rationale`.
     * Do NOT emit standalone material listings as mechanisms, such as `Contains(PLGA)` alone.
     * Do NOT emit standalone endpoint declarations as mechanisms, such as `Enhance(IgG production)` alone.
     * Functional formulation states are valid only when they causally connect to biological effects, e.g. Form(sustained release) -> Sustain(antigen exposure) -> Enhance(immune response).
     * Match functor semantics to the biological event: use Uptake only for internalization/endocytosis; Present for MHC antigen presentation; Express for CD80/CD86/MHC upregulation; Secrete or Increase for cytokine/antibody production; Migrate only for cell trafficking; Bind only for receptor/ligand engagement; Activate for named signaling pathways such as TRIF or NF-kB.
     * Use Contains sparingly in mechanisms. It is allowed only as a formulation-to-immune bridge when the component directly explains the next immune event. Prefer Release, Uptake, or Bind when the chain is already describing delivery or receptor engagement.

   Preferred mechanism functor families:
   - Property/formulation entry: Size, Shape, Type, Method, Route, Target, Contains, Form, Display, Mix, Conjugate, Assemble, Encapsulate, Crosslink, New_Prepare, Release, Sustain
   - Immune events: Bind, Uptake, Process, Present, Activate, Mature, Migrate, Recruit, Prime, Differentiate, Polarize, Express, Secrete, Trigger, Enable, Induce
   - Effects/endpoints/constraints: Enhance, Increase, Reduce, Inhibit, Suppress, Protect, Avoid, Control, Balance, Equal, Positive, Establish, Tune, Neutralize, Kill, Disrupt, Irradiate, Prolong
   - For size: always use "Size" (NOT "Particle_size", "Nanoparticle_size", etc.)
   - For adjuvant type: always use "Type" (NOT "Adjuvant_type", etc.)
   - For composition: always use "Contains" (NOT "Include", "Composed_of", etc.)
   - For immune activation: use "Activate", "Enhance"
   - For delivery: use "Target", "Route"
   - Every node has ONE open `value`; agent is implicit. Use spaces rather than underscores in `value`. Do not put multiple comma-separated arguments in `value` unless the source explicitly states a coupled concept.

8. **FIELD PLACEMENT GUIDANCE**: Place standard physicochemical properties at their DEDICATED top-level fields when possible. Complex or multi-part values MAY be placed in `properties`:
   - `type` (e.g., "Nanoparticle", "Microparticle", "Emulsion") → `adjuvant.type`
   - `size` (e.g., "200 nm", "1–10 µm") → `adjuvant.size` / `antigen.size`
   - `shape` (e.g., "spherical", "rod-like") → `adjuvant.shape` / `antigen.shape`
   - `zeta_potential` (e.g., "-10 mV") → `adjuvant.zeta_potential` / `antigen.zeta_potential`
   - `dose` → `antigen.dose`
   - `target_disease` → `vaccine.target_disease`
   - `properties` can hold additional attributes (e.g., crystallinity, encapsulation_efficiency, S_M_E_N, loading_capacity, surface_modification_details, stability_parameters) when they are too complex for a single top-level string.

9. **CRITICAL COMPONENTS GUIDANCE**: The `components` array is ESSENTIAL for structural validation. You MUST list ALL constituent materials explicitly mentioned in the rationale:
   - **Adjuvant components** include: polymer scaffolds (PLGA → Polymer), lipids (MPLA → Lipid), small molecules (R848 → SmallMolecule), inorganic salts (alum → InorganicSalt), nucleic acids (CpG → NucleicAcid), proteins/peptides.
   - **Antigen components** include: the antigen itself (OVA → Protein, spike protein → Protein), carrier proteins, peptide epitopes.
   - **Component type selection**: Protein (enzymes, antigens, antibodies), Antibody (mAb, IgG), NucleicAcid (DNA, RNA, CpG), Lipid (phospholipids, MPLA, cholesterol), InorganicSalt (alum, calcium phosphate), Polymer (PLGA, PLG, chitosan, PEG), SmallMolecule (R848, TLR agonists), General (fallback for unclear cases).
   - **DO NOT** emit empty arrays `[]` or omit components if materials are mentioned in the rationale. Extract them from the preparation method description if needed.
   - Example: For "PLGA microspheres prepared by emulsion-solvent evaporation", components should include at minimum `[{"name": "PLGA", "component_type": "Polymer"}, {"name": "PVA", "component_type": "Polymer"}]`.

10. **CRITICAL CONSISTENCY CONSTRAINT**: If a mechanism chain starts from a design/property node such as Size, Shape, Type, Method, Route, Contains, Form, Encapsulate, or Release, that value MUST be explicitly supported by a populated field or statement in the structured vaccine design above. Do not invent upstream properties that are absent from the rationale.

11. **CRITICAL FOR rationale FIELD**: Each string in the rationale array MUST be:
   - A complete, independent, self-contained point extracted directly from Phase 1.
   - STRICTLY no more than 200 words per item.
   - Must end with proper punctuation (. ! or ?).
   - When generating the last item, STOP immediately after completing a sentence within 200 words.
   - DO NOT continue generating beyond 200 words for any rationale item."""

        user_message_json = f"""Scientific Rationale:
{rationale_text}

Extract the structured vaccine design from above."""

        messages_json = [
            {"role": "system", "content": system_message_json},
            {"role": "user", "content": user_message_json}
        ]


        prompt_json = self.hf_tokenizer.apply_chat_template(
            messages_json,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )


        logger.info('Runtime diagnostic.')


        prompt_tokens = len(self.hf_tokenizer.encode(prompt_json))
        logger.info(f"Phase 2 Prompt tokens: {prompt_tokens}")


        max_model_length = 32768
        safety_buffer = 512
        available_tokens = max_model_length - prompt_tokens - safety_buffer


        max_new_tokens_for_json = min(available_tokens, 8192)

        logger.info('Runtime diagnostic.')


        json_str = self.outlines_json_generator(
            prompt_json,
            max_new_tokens=max_new_tokens_for_json,
            temperature=0.7,
            top_p=0.8,
            top_k=20
        )

        logger.info(f"FSM Generation complete. Generated {len(json_str)} characters.")


        logger.info("=" * 80)
        logger.info('Runtime diagnostic.')
        logger.info("-" * 80)
        try:

            formatted_json = json.dumps(json.loads(json_str), indent=2, ensure_ascii=False)
            logger.info(formatted_json)
        except json.JSONDecodeError:

            logger.info(json_str)
        logger.info("=" * 80)


        left_braces = json_str.count('{')
        right_braces = json_str.count('}')

        if left_braces != right_braces:
            logger.warning('Runtime diagnostic.')
            logger.info('Runtime diagnostic.')
            json_str = self._repair_truncated_json(json_str)


        try:
            design_obj = ExperimentDesignData.model_validate_json(json_str)
            logger.info('Runtime diagnostic.')

        except ValidationError as e:
            logger.error('Runtime diagnostic.')
            logger.error('Runtime diagnostic.')
            logger.error('Runtime diagnostic.')


            logger.error("=" * 80)
            logger.error('Runtime diagnostic.')
            for error in e.errors():
                error_loc = '.'.join(map(str, error['loc']))
                logger.error('Runtime diagnostic.')
                logger.error('Runtime diagnostic.')
                logger.error('Runtime diagnostic.')
                logger.error('Runtime diagnostic.')
                logger.error("-" * 40)
            logger.error("=" * 80)


            logger.warning('Runtime diagnostic.')
            try:

                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    if repair_json:
                        logger.info('Runtime diagnostic.')
                        data = json.loads(repair_json(json_str))
                    else:
                        raise


                for field in ("vaccine", "disease_model"):
                    val = data.get(field)
                    if isinstance(val, str) and val.strip().startswith("{"):
                        try:
                            data[field] = json.loads(val)
                            logger.info('Runtime diagnostic.')
                        except json.JSONDecodeError:
                            pass

                if isinstance(data.get("disease_model"), str):
                    data["disease_model"] = {"name": data["disease_model"], "host_animal": "not specified"}
                    logger.info('Runtime diagnostic.')


                if "vaccine" in data and isinstance(data.get("vaccine"), dict) and "adjuvant" in data["vaccine"]:
                    adj = data["vaccine"]["adjuvant"]
                    if "type" not in adj:

                        inferred_type = "Nanoparticle"
                        query_lower = query.lower()

                        if "emulsion" in query_lower:
                            inferred_type = "Emulsion"
                        elif "hydrogel" in query_lower:
                            inferred_type = "Hydrogel"
                        elif "microneedl" in query_lower:
                            inferred_type = "Microneedling"
                        elif "salt" in query_lower or "alum" in query_lower:
                            inferred_type = "Inorganic_salt"
                        elif "molecule" in query_lower or "chemical" in query_lower:
                            inferred_type = "Molecule"


                        logger.info('Runtime diagnostic.')
                        adj["type"] = inferred_type


                if "vaccine" in data and "antigen" in data["vaccine"]:
                    antigen = data["vaccine"]["antigen"]
                    if "components" in antigen and isinstance(antigen["components"], list):
                        for comp in antigen["components"]:
                            if isinstance(comp, dict) and "component_type" not in comp:

                                comp_name = comp.get("name", "").lower()
                                if any(kw in comp_name for kw in ["poly i:c", "cpg", "mrna", "dna", "rna", "oligonucleotide"]):
                                    comp["component_type"] = "NucleicAcid"
                                elif any(kw in comp_name for kw in ["peptide", "protein", "antigen"]):
                                    comp["component_type"] = "Protein"
                                elif any(kw in comp_name for kw in ["antibody", "mab", "igg"]):
                                    comp["component_type"] = "Antibody"
                                elif any(kw in comp_name for kw in ["mpla", "monophosphoryl"]) or ("lipid" in comp_name and "lactide" not in comp_name):
                                    comp["component_type"] = "Lipid"
                                elif any(kw in comp_name for kw in ["plg", "plga", "lactide", "glycolide", "polymer", "chitosan", "peg"]):
                                    comp["component_type"] = "Polymer"
                                elif any(kw in comp_name for kw in ["mgco3", "magnesium", "calcium", "aluminum", "alum", "salt", "carbonate", "phosphate"]):
                                    comp["component_type"] = "InorganicSalt"
                                else:
                                    comp["component_type"] = "General"
                                logger.info('Runtime diagnostic.')


                if "vaccine" in data and "adjuvant" in data["vaccine"]:
                    adj = data["vaccine"]["adjuvant"]
                    if "components" in adj and isinstance(adj["components"], list):
                        for comp in adj["components"]:
                            if isinstance(comp, dict) and "component_type" not in comp:
                                comp_name = comp.get("name", "").lower()
                                if any(kw in comp_name for kw in ["poly i:c", "cpg", "mrna", "dna", "rna"]):
                                    comp["component_type"] = "NucleicAcid"
                                elif any(kw in comp_name for kw in ["peptide", "protein"]):
                                    comp["component_type"] = "Protein"
                                elif any(kw in comp_name for kw in ["antibody", "mab"]):
                                    comp["component_type"] = "Antibody"
                                elif any(kw in comp_name for kw in ["mpla", "monophosphoryl"]) or ("lipid" in comp_name and "lactide" not in comp_name):
                                    comp["component_type"] = "Lipid"
                                elif any(kw in comp_name for kw in ["plg", "plga", "lactide", "glycolide", "polymer", "chitosan", "peg"]):
                                    comp["component_type"] = "Polymer"
                                elif any(kw in comp_name for kw in ["mgco3", "magnesium", "calcium", "aluminum", "alum", "salt", "carbonate", "phosphate"]):
                                    comp["component_type"] = "InorganicSalt"
                                else:
                                    comp["component_type"] = "General"
                                logger.info('Runtime diagnostic.')


                if isinstance(data.get("mechanism_chains"), list):
                    valid_chains = []
                    dropped = 0
                    for chain in data["mechanism_chains"]:
                        nodes = chain.get("nodes") if isinstance(chain, dict) else None
                        if isinstance(nodes, list) and len(nodes) >= 2:
                            valid_chains.append(chain)
                        else:
                            dropped += 1
                    if dropped:
                        logger.warning('Runtime diagnostic.')
                    data["mechanism_chains"] = valid_chains


                design_obj = ExperimentDesignData.model_validate(data)
                logger.info('Runtime diagnostic.')

            except Exception as repair_error:
                logger.error('Runtime diagnostic.')


                if repair_json:
                    logger.info('Runtime diagnostic.')
                    try:
                        repaired_json = repair_json(json_str)
                        design_obj = ExperimentDesignData.model_validate_json(repaired_json)
                        logger.info('Runtime diagnostic.')
                    except Exception as final_error:
                        logger.error('Runtime diagnostic.')
                        raise e
                else:
                    logger.error('Runtime diagnostic.')
                    raise e

        return CompleteResponse(
            scientific_rationale=rationale_text,
            formalized_design=design_obj
        )


    def _generate_api_robust(self, query: str) -> CompleteResponse:


        logger.info(">>> [API Phase 1] Generating Rationale...")
        sys_prompt_1 = """You are an expert scientist in immunology.
Goal: Design a novel vaccine adjuvant system based on the user's request.
Instruction: Provide a detailed scientific rationale explaining the design principles, mechanisms of action, delivery strategy, and expected immune outcomes. Structure your response around three aspects:
(1) Adjuvant construction and physical/chemical properties (type, size, formulation, preparation method).
(2) Expected immune effects and outcomes (antibody responses, T cell responses, cytokine profiles, protection).
(3) Step-by-step causal mechanism from adjuvant design to immune response.
Write in natural scientific language. Do NOT output JSON or structured formats yet.
If your design relies on specific physical properties (e.g., size, charge, molecular weight) for its mechanism of action, you MUST explicitly state their exact values in the rationale paragraph."""

        rationale_text = self._call_openai(sys_prompt_1, query, json_mode=False)
        logger.info(f"Rationale generated ({len(rationale_text)} chars).")


        logger.info(">>> [API Phase 2] Extracting Structured Design...")

        if INSTRUCTOR_AVAILABLE and hasattr(self.client, 'chat'):

            return self._generate_api_instructor(query, rationale_text)
        else:

            return self._generate_api_legacy(query, rationale_text)

    def _generate_api_instructor(self, query: str, rationale_text: str) -> CompleteResponse:

        logger.info('Runtime diagnostic.')

        sys_prompt = """You are a strictly constrained data extraction assistant.
Task: Convert the provided Scientific Rationale into a strict structural design (JSON).

CRITICAL RULES:
1. **ZERO-HALLUCINATION (STRICT EXTRACTION):** You must ONLY extract entities, properties, and claims that are explicitly stated in the provided Scientific Rationale. DO NOT invent, hallucinate, or fabricate new components, values (like size or dose), pathways, or mechanisms. If a specific parameter is not mentioned in the rationale, you MUST leave it as null or omit it. Do not guess.

2. **MANDATORY COMPONENTS**: Vaccine antigen and adjuvant MUST have a "components" array listing all constituent components. Every component must have a "component_type" field with one of: "Protein", "Antibody", "NucleicAcid", "Lipid", "InorganicSalt", "Polymer", "General".

3. Adjuvant must have a "type" field with one of: "Nanoparticle", "Emulsion", "Molecule", "Inorganic_salt", "Hydrogel", "Microneedling".

4. **MANDATORY PREPARATION METHOD**: Vaccine, antigen, and adjuvant MUST have a "preparation" object. You must choose EXACTLY ONE preparation method from: "mix", "assemble", "conjugate", "emulsify", "encapsulate", "crosslink", "new_prepare". Each preparation method has its own specific fields:
   - mix: requires "composition_type": "mix", optional "mixing_conditions"
   - assemble: requires "composition_type": "assemble", optional "assembly_method"
   - conjugate: requires "composition_type": "conjugate", optional "conjugation_chemistry"
   - emulsify: requires "composition_type": "emulsify", optional "emulsification_method"
   - encapsulate: requires "composition_type": "encapsulate", optional "encapsulation_technique"
   - crosslink: requires "composition_type": "crosslink", optional "crosslinking_agent"
   - new_prepare: requires "composition_type": "new_prepare", optional "preparation_details"

5. **MANDATORY DSL OUTPUTS**: You MUST generate `claim_statements` and `mechanism_chains`. These are the authoritative FSM-masked outputs. The legacy `claims_dsl` and `mechanisms_dsl` strings will be serialized automatically; do not generate or rely on free-form DSL strings as the primary output.

6. **STRUCTURED CLAIM STATEMENTS (FSM-MASKED FUNCTORS)**: Generate `claim_statements` as an array of compact concept claim objects. Each object MUST have exactly the Stage 2 claim surface:
   - `functor`: one exact controlled claim functor from the schema enum.
   - `outcome`: compact outcome/concept phrase, NOT a sentence.

   Claim rules:
     * Do NOT generate `claims_dsl` directly; it is serialized from `claim_statements`.
     * Do NOT include treatment, control, comparator, magnitude, dose, group labels, or evidence fields in `claim_statements`.
     * Use only the schema enum for `functor`; do not invent verbs such as "Improve", "Generate", or "Prevent".
     * Keep `outcome` short and canonical. Use spaces, not snake_case.
     * Generate 2-5 high-confidence claim statements. Prefer fewer precise claims over many generic/redundant claims.

7. **STRUCTURED MECHANISM CHAINS (FSM-MASKED FUNCTORS)**: Generate `mechanism_chains` as an array of chain objects. Each object MUST have `nodes`, and each node MUST have:
   - `functor`: one exact controlled functor from the schema.
   - `value`: short canonical event phrase, NOT a sentence. Keep it under 8 words when possible. Put mechanisms, explanations, evidence, and qualifiers in the chain `rationale`, not inside node values. Use spaces, not snake_case. Good values: "draining lymph node", "DC endocytosis", "MHC I presentation", "TLR3 signaling", "type I IFN production".

   Example object:
   `{"nodes": [{"functor": "Size", "value": "150 nm"}, {"functor": "Target", "value": "draining lymph node"}, {"functor": "Activate", "value": "dendritic cell activation"}], "rationale": "Particle size supports lymph node drainage and APC activation."}`

   Mechanism rules:
     * `mechanism_chains` describes BIOLOGICAL CAUSAL MECHANISMS only: how adjuvant properties or first immune events lead to immune effects.
     * Each adjacent node must be a cause-to-effect relationship.
     * The final node of each chain is used as the Stage 2 D endpoint. Therefore, do not end a chain with pure upstream bookkeeping nodes such as Size, Shape, Type, Method, Route, Contains, Form, Display, Mix, Conjugate, Assemble, Encapsulate, Crosslink, New_Prepare, Release, Bind, Process, Trigger, Enable, or Target unless no downstream immune event is stated.
     * Do not end with internal receptor/pathway activation such as Activate(TLR signaling), Activate(STING pathway), Activate(TRIF pathway), or Activate(NF-kB pathway). Continue to a biological endpoint when supported, e.g. Mature(dendritic cells), Express(CD80 CD86), Secrete(type I IFN), Polarize(Th1 response), Prime(CD8 T cells), Enhance(antibody response), Neutralize(pathogen), Reduce(viral load), or Protect(survival).
     * A chain with only one node is invalid and will be rejected.
     * Generate 3-5 non-overlapping chains total; prefer 3 high-confidence chains over many weak/redundant chains.
     * Use 2-5 nodes per chain; prefer 3 nodes when the rationale supports a complete path.
     * Do not create a separate chain for every design attribute. Merge overlapping mechanisms into the main causal axes: delivery/uptake, PRR signaling/DC activation, and sustained release/immune persistence. Add a fourth or fifth chain only when it captures a clearly distinct mechanism.
     * Node values must be compact mechanism nodes, not explanatory sentences. Bad: `Size(150 nm nanoparticle size enables passive lymphatic drainage...)`. Good: `Size(150 nm)` then put the explanation in `rationale`.
     * Do NOT emit standalone material listings as mechanisms, such as `Contains(PLGA)` alone.
     * Do NOT emit standalone endpoint declarations as mechanisms, such as `Enhance(IgG production)` alone.
     * Functional formulation states are valid only when they causally connect to biological effects, e.g. Form(sustained release) -> Sustain(antigen exposure) -> Enhance(immune response).
     * Match functor semantics to the biological event: use Uptake only for internalization/endocytosis; Present for MHC antigen presentation; Express for CD80/CD86/MHC upregulation; Secrete or Increase for cytokine/antibody production; Migrate only for cell trafficking; Bind only for receptor/ligand engagement; Activate for named signaling pathways such as TRIF or NF-kB.
     * Use Contains sparingly in mechanisms. It is allowed only as a formulation-to-immune bridge when the component directly explains the next immune event. Prefer Release, Uptake, or Bind when the chain is already describing delivery or receptor engagement.

   Preferred mechanism functor families:
   - Property/formulation entry: Size, Shape, Type, Method, Route, Target, Contains, Form, Display, Mix, Conjugate, Assemble, Encapsulate, Crosslink, New_Prepare, Release, Sustain
   - Immune events: Bind, Uptake, Process, Present, Activate, Mature, Migrate, Recruit, Prime, Differentiate, Polarize, Express, Secrete, Trigger, Enable, Induce
   - Effects/endpoints/constraints: Enhance, Increase, Reduce, Inhibit, Suppress, Protect, Avoid, Control, Balance, Equal, Positive, Establish, Tune, Neutralize, Kill, Disrupt, Irradiate, Prolong
   - For size: always use "Size" (NOT "Particle_size", "Nanoparticle_size", etc.)
   - For adjuvant type: always use "Type" (NOT "Adjuvant_type", etc.)
   - For composition: always use "Contains" (NOT "Include", "Composed_of", etc.)
   - For immune activation: use "Activate", "Enhance"
   - For delivery: use "Target", "Route"
   - Every node has ONE open `value`; agent is implicit. Use spaces rather than underscores in `value`. Do not put multiple comma-separated arguments in `value` unless the source explicitly states a coupled concept.

8. **FIELD PLACEMENT GUIDANCE**: Place standard physicochemical properties at their DEDICATED top-level fields when possible. Complex or multi-part values MAY be placed in `properties`:
   - `type` (e.g., "Nanoparticle", "Microparticle", "Emulsion") → `adjuvant.type`
   - `size` (e.g., "200 nm", "1–10 µm") → `adjuvant.size` / `antigen.size`
   - `shape` (e.g., "spherical", "rod-like") → `adjuvant.shape` / `antigen.shape`
   - `zeta_potential` (e.g., "-10 mV") → `adjuvant.zeta_potential` / `antigen.zeta_potential`
   - `dose` → `antigen.dose`
   - `target_disease` → `vaccine.target_disease`
   - `properties` can hold additional attributes (e.g., crystallinity, encapsulation_efficiency, S_M_E_N, loading_capacity, surface_modification_details, stability_parameters) when they are too complex for a single top-level string.

9. **CRITICAL COMPONENTS GUIDANCE**: The `components` array is ESSENTIAL for structural validation. You MUST list ALL constituent materials explicitly mentioned in the rationale:
   - **Adjuvant components** include: polymer scaffolds (PLGA → Polymer), lipids (MPLA → Lipid), small molecules (R848 → SmallMolecule), inorganic salts (alum → InorganicSalt), nucleic acids (CpG → NucleicAcid), proteins/peptides.
   - **Antigen components** include: the antigen itself (OVA → Protein, spike protein → Protein), carrier proteins, peptide epitopes.
   - **Component type selection**: Protein (enzymes, antigens, antibodies), Antibody (mAb, IgG), NucleicAcid (DNA, RNA, CpG), Lipid (phospholipids, MPLA, cholesterol), InorganicSalt (alum, calcium phosphate), Polymer (PLGA, PLG, chitosan, PEG), SmallMolecule (R848, TLR agonists), General (fallback for unclear cases).
   - **DO NOT** emit empty arrays `[]` or omit components if materials are mentioned in the rationale. Extract them from the preparation method description if needed.
   - Example: For "PLGA microspheres prepared by emulsion-solvent evaporation", components should include at minimum `[{"name": "PLGA", "component_type": "Polymer"}, {"name": "PVA", "component_type": "Polymer"}]`.

10. **CRITICAL CONSISTENCY CONSTRAINT**: If a mechanism chain starts from a design/property node such as Size, Shape, Type, Method, Route, Contains, Form, Encapsulate, or Release, that value MUST be explicitly supported by a populated field or statement in the structured vaccine design above. Do not invent upstream properties that are absent from the rationale.

11. **CRITICAL FOR rationale FIELD**: Each string in the rationale array MUST be:
   - A complete, independent, self-contained point extracted directly from Phase 1.
   - STRICTLY no more than 200 words per item.
   - Must end with proper punctuation (. ! or ?).
   - When generating the last item, STOP immediately after completing a sentence within 200 words.
   - DO NOT continue generating beyond 200 words for any rationale item."""

        user_content = f"""Scientific Rationale:
{rationale_text}

Extract the structured vaccine design from above."""

        try:

            _model_lower = self.model_path.lower()
            _no_think_kwargs = {}
            if "deepseek" in _model_lower:
                _no_think_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            elif "glm" in _model_lower or "chatglm" in _model_lower:
                _no_think_kwargs["extra_body"] = {"enable_thinking": False}
            elif "gpt" in _model_lower:

                pass
            else:
                _no_think_kwargs["extra_body"] = {"thinking": {"budget_tokens": 0}}


            design_obj = self.client.chat.completions.create(
                model=self.model_path,
                response_model=ExperimentDesignData,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_retries=self.max_retries,
                temperature=0.7,
                **_no_think_kwargs,
            )

            logger.info('Runtime diagnostic.')

            return CompleteResponse(
                scientific_rationale=rationale_text,
                formalized_design=design_obj
            )

        except Exception as e:
            logger.error('Runtime diagnostic.')

            logger.warning('Runtime diagnostic.')
            return self._generate_api_legacy(query, rationale_text)

    def _generate_api_legacy(self, query: str, rationale_text: str) -> CompleteResponse:

        logger.info('Runtime diagnostic.')

        sys_prompt_2 = """Extract the design into JSON.
Ensure 'adjuvant' is strictly nested inside 'vaccine'.
Adhere to the Strict Schema logic (e.g., if Nanoparticle, do not include oil_phase).
Generate claim_statements as structured objects:
{"functor": "Enhance", "outcome": "target concept"}.
Claim functor is controlled by schema; outcome is open compact text. Do not include treatment, control, comparator, or magnitude fields. Do not generate free-form claims_dsl as the primary output.
Generate mechanism_chains as structured objects:
{"nodes": [{"functor": "Size", "value": "150 nm"}, {"functor": "Target", "value": "draining lymph node"}]}.
Functor is controlled by schema; value is open scientific text. Do not generate free-form mechanism strings as the primary output.
Mechanism chain rules:
- Generate 3-5 non-overlapping chains total; prefer 3 high-confidence chains.
- Each chain must have at least 2 nodes and at most 5 nodes.
- Node value must be a short canonical event phrase, not a full sentence, preferably under 8 words.
- Use spaces rather than underscores in values.
- Put explanations and evidence in chain rationale, not inside node values.
- The final node of each chain is used as the Stage 2 D endpoint. Do not end a chain with pure upstream nodes such as Size, Type, Route, Target, Contains, Form, Release, Bind, or Process when a downstream immune effect is available.
- Do not end with internal pathway activation such as Activate(TLR signaling), Activate(STING pathway), Activate(TRIF pathway), or Activate(NF-kB pathway). Continue to Mature(dendritic cells), Express(CD80 CD86), Secrete(type I IFN), Polarize(Th1 response), Prime(CD8 T cells), Enhance(antibody response), Neutralize(pathogen), Reduce(viral load), or Protect(survival) when supported.
- Match functor semantics: Uptake=endocytosis/internalization; Present=MHC antigen presentation; Express=CD80/CD86/MHC upregulation; Secrete/Increase=cytokine or antibody production; Activate=named signaling or cell activation.
- Use Contains sparingly, only as a formulation-to-immune bridge."""

        user_content = f"Request: {query}\n\nRationale:\n{rationale_text}"

        for attempt in range(self.max_retries + 1):
            try:
                raw_json = self._call_openai(sys_prompt_2, user_content, json_mode=True, disable_thinking=True)


                json_str = self._clean_json_string(raw_json)
                raw_data = json.loads(json_str)


                struct_data = self._restructure_root_json(raw_data)


                clean_data = self._normalize_json_data(struct_data)


                fd_data = clean_data.get("formalized_design", clean_data)
                design_obj = ExperimentDesignData.model_validate(fd_data)

                return CompleteResponse(
                    scientific_rationale=rationale_text,
                    formalized_design=design_obj
                )

            except ValidationError as e:
                logger.warning(f"Validation failed (Attempt {attempt}): {e}")
                user_content += f"\n\nPrevious attempt failed validation:\n{str(e)}\nPlease correct the JSON structure."
            except Exception as e:
                logger.error(f"Extraction error: {e}")

        raise RuntimeError("Max retries reached in API mode.")


    def _repair_truncated_json(self, json_str: str) -> str:

        logger.info('Runtime diagnostic.')


        json_str = json_str.rstrip()


        quote_count = json_str.count('"')

        escaped_quote_count = json_str.count('\\"')
        actual_quote_count = quote_count - escaped_quote_count

        if actual_quote_count % 2 != 0:
            logger.info('Runtime diagnostic.')
            json_str += '"'


        left_braces = json_str.count('{')
        right_braces = json_str.count('}')
        missing_braces = left_braces - right_braces

        if missing_braces > 0:
            logger.info('Runtime diagnostic.')
            json_str += '}' * missing_braces


        left_brackets = json_str.count('[')
        right_brackets = json_str.count(']')
        missing_brackets = left_brackets - right_brackets

        if missing_brackets > 0:
            logger.info('Runtime diagnostic.')
            json_str += ']' * missing_brackets

        return json_str


    def _call_openai(self, sys_p, user_msg, json_mode=False, disable_thinking=False):

        raw = getattr(self, '_raw_client', self.client)
        kwargs = dict(
            model=self.model_path,
            messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": user_msg}],
            temperature=0.7,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}


        if disable_thinking:
            model_lower = self.model_path.lower()
            if "deepseek" in model_lower:
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            elif "glm" in model_lower or "chatglm" in model_lower:
                kwargs["extra_body"] = {"enable_thinking": False}
            else:

                kwargs["extra_body"] = {"thinking": {"budget_tokens": 0}}
        resp = raw.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    def _clean_json_string(self, raw_str: str) -> str:
        if not raw_str: return "{}"
        text = re.sub(r'```json\s*', '', raw_str)
        text = re.sub(r'```', '', text).strip()
        start = text.find('{')
        if start == -1: return text
        json_cand = text[start:]
        if repair_json:
            try: return repair_json(json_cand)
            except: pass
        return json_cand.rstrip().rstrip(',') + '}' * (json_cand.count('{') - json_cand.count('}'))

    def _restructure_root_json(self, data: Dict) -> Dict:

        if "formalized_design" in data and "vaccine" in data["formalized_design"]:
            return data

        fd = {
            "vaccine": self._deep_search(data, ["vaccine", "vaccine_design"]) or {},
            "disease_model": self._deep_search(data, ["disease_model"]) or {},
            "claim_statements": self._deep_search(data, ["claim_statements", "claims_structured"]) or [],
            "claims_dsl": self._deep_search(data, ["claims_dsl", "claims"]) or [],
            "mechanism_chains": self._deep_search(data, ["mechanism_chains", "mechanisms_structured", "causal_chains"]) or [],
            "mechanisms_dsl": self._deep_search(data, ["mechanisms_dsl", "mechanisms"]) or [],
            "rationale": self._deep_search(data, ["rationale", "reasoning"]) or ["Extracted."]
        }
        if "adjuvant" in data and not fd["vaccine"].get("adjuvant"):
            fd["vaccine"]["adjuvant"] = data["adjuvant"]
        return {"formalized_design": fd}

    def _deep_search(self, data: Any, keys: List[str]) -> Any:
        if isinstance(data, dict):
            for k in keys:
                if k in data and data[k]: return data[k]
            for v in data.values():
                res = self._deep_search(v, keys)
                if res: return res
        return None

    def _normalize_json_data(self, data: Any) -> Any:

        if isinstance(data, dict):
            new_data = {}
            for k, v in data.items():
                new_k, new_v = k, v


                if k == 'type' and isinstance(v, str):
                    if any(x in v.lower() for x in ['protein', 'peptide', 'lipid', 'nucleic']):
                        new_k = 'component_type'


                if new_k == 'component_type' and isinstance(v, str):
                    v_low = v.lower()
                    if 'protein' in v_low: new_v = 'Protein'
                    elif 'antibody' in v_low: new_v = 'Antibody'
                    elif 'nucleic' in v_low or 'dna' in v_low: new_v = 'NucleicAcid'
                    elif 'lipid' in v_low: new_v = 'Lipid'
                    elif 'polymer' in v_low: new_v = 'Polymer'
                    elif 'salt' in v_low: new_v = 'InorganicSalt'
                    elif 'chemical' in v_low: new_v = 'Chemical'
                    else: new_v = 'General'
                elif new_k == 'type' and isinstance(v, str):
                    v_low = v.lower()
                    if 'nanoparticle' in v_low: new_v = 'Nanoparticle'
                    elif 'emulsion' in v_low: new_v = 'Emulsion'
                    elif 'molecule' in v_low: new_v = 'Molecule'
                    elif 'hydrogel' in v_low: new_v = 'Hydrogel'
                    elif 'microneedling' in v_low: new_v = 'Microneedling'
                    elif 'inorganic' in v_low: new_v = 'Inorganic_salt'
                elif new_k == 'phase_type' and isinstance(v, str):
                    new_v = v.capitalize()


                if new_k in ['size', 'particle_size', 'shape', 'newfeature', 'composition_type'] and isinstance(v, (int, float)):
                    new_v = str(v)


                new_data[new_k] = self._normalize_json_data(new_v)
            return new_data
        elif isinstance(data, list):
            return [self._normalize_json_data(item) for item in data]
        return data

    def _generate_mock(self, query):
        logger.info("Returning Mock Data...")
        return CompleteResponse(
            scientific_rationale="Mock Rationale for testing.",
            formalized_design=ExperimentDesignData(
                vaccine={
                    "name": "MockVac", "target_disease": "Cancer",
                    "adjuvant": {
                        "type": "Nanoparticle", "name": "PLGA NP", "shape": "sphere",
                        "particle_size": "200 nm",
                        "components": [{"name": "PLGA", "component_type": "Polymer"}],
                        "preparation": {"name": "Self-Assembly", "composition_type": "assemble",
                                        "assembly_method": "Nanoprecipitation"}
                    },
                    "antigen": {
                        "name": "OVA",
                        "components": [{"name": "Ovalbumin", "component_type": "Protein"}],
                        "preparation": {"name": "Mix", "composition_type": "mix",
                                        "mixing_conditions": "Room temperature, 30 min"}
                    },
                    "preparation": {"name": "Final Mix", "composition_type": "mix"}
                },
                disease_model={
                    "name": "B16-OVA Tumor Model",
                    "type": "Subcutaneous tumor",
                    "host_animal": "C57BL/6 mouse",
                    "challenge_route": "Subcutaneous injection"
                },
                claim_statements=[
                    {
                        "functor": "Enhance",
                        "outcome": "IgG titer"
                    }
                ],
                mechanism_chains=[
                    {
                        "nodes": [
                            {"functor": "Size", "value": "200 nm"},
                            {"functor": "Target", "value": "draining lymph node"},
                            {"functor": "Activate", "value": "dendritic cell activation"}
                        ],
                        "rationale": "Mock nanoparticle size supports lymph node drainage and APC activation."
                    }
                ],
                rationale=["Mock rationale for testing purposes only."]
            )
        )


