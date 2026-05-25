"""Structured design schemas used by the SAVANT generator."""

import re
from typing import List, Literal, Union, Optional, Annotated, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator


class StrictBaseModel(BaseModel):

    model_config = ConfigDict(extra='forbid')

MechanismFunctor = Literal[

    'Size', 'Shape', 'Type', 'Method', 'Route', 'Target',
    'Contains', 'Form', 'Display', 'Mix', 'Conjugate', 'Assemble',
    'Encapsulate', 'Crosslink', 'New_Prepare', 'Release', 'Sustain',

    'Bind', 'Uptake', 'Process', 'Present', 'Activate', 'Mature',
    'Migrate', 'Recruit', 'Prime', 'Differentiate', 'Polarize',
    'Express', 'Secrete', 'Trigger', 'Enable', 'Induce',

    'Enhance', 'Increase', 'Reduce', 'Inhibit', 'Suppress', 'Protect',
    'Avoid', 'Control', 'Balance', 'Equal', 'Positive', 'Establish',
    'Tune', 'Neutralize', 'Kill', 'Disrupt', 'Irradiate', 'Prolong',
]


class MechanismNodeSchema(StrictBaseModel):

    functor: MechanismFunctor = Field(
        ...,
        description="Controlled DSL functor. Use exactly one value from the enum."
    )
    value: str = Field(
        ...,
        min_length=1,
        max_length=96,
        description=(
            "Open but compact scientific target/value phrase, not a full sentence; "
            "e.g. 'draining lymph node' or 'MHC I presentation'."
        )
    )

    @field_validator("value", mode="before")
    @classmethod
    def normalize_value_text(cls, value):
        if value is None:
            return value
        text = str(value).replace("_", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text.strip(" .;:")

    def to_dsl(self) -> str:
        clean_value = re.sub(r"\s+", " ", str(self.value)).strip()
        return f"{self.functor}({clean_value})"


class MechanismChainSchema(StrictBaseModel):

    nodes: List[MechanismNodeSchema] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Ordered causal chain nodes. Prefer 3 nodes when the mechanism supports it."
    )
    rationale: Optional[str] = Field(
        None,
        description="Brief evidence note for this chain, extracted from the rationale."
    )

    def to_dsl(self) -> str:
        return " >> ".join(node.to_dsl() for node in self.nodes)

    @model_validator(mode="after")
    def validate_causal_chain_length(self):
        if len(self.nodes) < 2:
            raise ValueError("mechanism chain must contain at least two causal nodes")
        return self


ClaimFunctor = Literal[

    'Enhance', 'Increase', 'Reduce', 'Inhibit', 'Suppress', 'Protect',
    'Avoid', 'Control', 'Balance', 'Equal', 'Positive', 'Establish',
    'Tune', 'Sustain', 'Prolong',

    'Activate', 'Prime', 'Differentiate', 'Polarize', 'Express', 'Secrete',
    'Present', 'Mature', 'Recruit', 'Induce', 'Neutralize', 'Kill', 'Disrupt',
    'Irradiate', 'Route', 'Target', 'Release', 'Uptake', 'Migrate',
]


_CLAIM_FUNCTOR_ALIASES = {
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


class ClaimStatementSchema(StrictBaseModel):

    functor: ClaimFunctor = Field(
        ...,
        description="Controlled Stage 2 claim functor. Use exactly one value from the enum."
    )
    outcome: str = Field(
        ...,
        min_length=1,
        max_length=96,
        description="Compact outcome/concept phrase, not a full sentence."
    )

    @model_validator(mode="before")
    @classmethod
    def drop_legacy_claim_fields(cls, data):

        if isinstance(data, dict):
            return {
                "functor": data.get("functor"),
                "outcome": data.get("outcome"),
            }
        return data

    @field_validator("outcome", mode="before")
    @classmethod
    def normalize_claim_text(cls, value):
        if value is None:
            return value
        text = str(value).replace("_", " ").replace(",", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text.strip(" .;:")

    @field_validator("functor", mode="before")
    @classmethod
    def normalize_claim_functor(cls, value):
        if value is None:
            return value
        text = str(value).strip()
        return _CLAIM_FUNCTOR_ALIASES.get(text, text)

    def to_dsl(self) -> str:
        outcome = re.sub(r"\s+", " ", str(self.outcome)).strip()
        return f"Claim = {self.functor}({outcome})"


class BaseComponentSchema(StrictBaseModel):
    name: str = Field(..., description="Component name")


class ProteinSchema(BaseComponentSchema):

    component_type: Literal['Protein', 'protein']
    molecular_weight: Optional[float] = None
    amino_acid_sequence: Optional[str] = None
    isoelectric_point: Optional[float] = None
    structure: Optional[str] = None
    function: Optional[str] = None
    modifications: Optional[str] = None
    source: Optional[str] = None
    stability: Optional[str] = None

class AntibodySchema(BaseComponentSchema):

    component_type: Literal['Antibody', 'antibody']
    isotype: Optional[str] = None
    affinity: Optional[float] = None
    molecular_weight: Optional[float] = None
    epitope: Optional[str] = None
    species: Optional[str] = None
    target_disease: Optional[str] = None

class NucleicAcidSchema(BaseComponentSchema):

    component_type: Literal['NucleicAcid', 'nucleicacid', 'Nucleic_Acid', 'nucleic_acid', 'DNA', 'dna', 'RNA', 'rna']
    sequence: Optional[str] = None
    molecular_weight: Optional[float] = None
    type: Optional[str] = None
    length: Optional[int] = None
    modifications: Optional[str] = None

class SmallMoleculeSchema(BaseComponentSchema):

    component_type: Literal['SmallMolecule', 'small_molecule', 'Small_molecule']
    formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    cas_number: Optional[str] = None
    description: Optional[str] = None
    properties: Optional[str] = None
    source: Optional[str] = None

class PhotosensitizerSchema(BaseComponentSchema):

    component_type: Literal['Photosensitizer', 'photosensitizer']
    formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    activation_wavelength: Optional[str] = None
    description: Optional[str] = None
    properties: Optional[str] = None
    source: Optional[str] = None

class LipidSchema(BaseComponentSchema):

    component_type: Literal['Lipid', 'lipid']
    molecular_weight: Optional[float] = None
    formula: Optional[str] = None
    lipid_type: Optional[str] = None
    hydrophobic_tail: Optional[str] = None
    hydrophilic_head: Optional[str] = None
    modifications: Optional[str] = None
    properties: Optional[str] = None
    source: Optional[str] = None

class InorganicSaltSchema(BaseComponentSchema):

    component_type: Literal['InorganicSalt', 'inorganicsalt', 'Salt', 'salt']
    formula: str
    properties: Optional[str] = None
    solubility: Optional[str] = None

class PolymerSchema(BaseComponentSchema):

    component_type: Literal['Polymer', 'polymer']
    formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    properties: Optional[str] = None

class GeneralComponentSchema(BaseComponentSchema):


    component_type: Literal[
        'General', 'general',
        'Complex', 'complex',
        'Peptide', 'peptide'
    ]
    formula: Optional[str] = None
    description: Optional[str] = None
    properties: Optional[str] = None
    source: Optional[str] = None


ComponentUnion = Annotated[
    Union[
        ProteinSchema,
        AntibodySchema,
        NucleicAcidSchema,
        SmallMoleculeSchema,
        PhotosensitizerSchema,
        LipidSchema,
        InorganicSaltSchema,
        PolymerSchema,
        GeneralComponentSchema
    ],
    Field(discriminator='component_type')
]


ComponentsFieldType = Optional[List[ComponentUnion]]

class TargetSchema(StrictBaseModel):
    name: str = Field(..., description="Target name")
    function: Optional[str] = None
    pathway: Optional[str] = None

class AqueousPhaseSchema(StrictBaseModel):
    name: str
    phase_type: Literal['Aqueous'] = 'Aqueous'
    temperature: Optional[float] = None
    pH: Optional[float] = None
    ionic_strength: Optional[float] = None
    viscosity: Optional[float] = None
    density: Optional[float] = None

class OilPhaseSchema(StrictBaseModel):
    name: str
    phase_type: Literal['Organic'] = 'Organic'
    temperature: Optional[float] = None
    viscosity: Optional[float] = None
    density: Optional[float] = None

class FunctionalGroupSchema(StrictBaseModel):
    name: str
    formula: str
    properties: Optional[str] = None
    role: Optional[str] = None


class BasePreparationSchema(StrictBaseModel):

    name: str = Field(..., description="Preparation method name")
    preparation_details: Optional[str] = Field(None, description="Optional detailed description of the preparation method")


class MixPreparationSchema(BasePreparationSchema):
    composition_type: Literal['mix']
    mixing_conditions: Optional[str] = Field(None, description="Mixing conditions (time, temperature, speed)")


class AssemblePreparationSchema(BasePreparationSchema):
    composition_type: Literal['assemble']
    assembly_method: Optional[str] = Field(None, description="Assembly method (self-assembly, directed assembly)")


class ConjugatePreparationSchema(BasePreparationSchema):
    composition_type: Literal['conjugate']
    conjugation_chemistry: Optional[str] = Field(None, description="Conjugation chemistry (EDC/NHS, click chemistry)")


class EmulsifyPreparationSchema(BasePreparationSchema):
    composition_type: Literal['emulsify']
    emulsification_method: Optional[str] = Field(None, description="Emulsification method (high-speed homogenization, microfluidics)")


class EncapsulatePreparationSchema(BasePreparationSchema):
    composition_type: Literal['encapsulate']
    encapsulation_technique: Optional[str] = Field(None, description="Encapsulation technique (spray drying, coacervation)")


class CrosslinkPreparationSchema(BasePreparationSchema):
    composition_type: Literal['crosslink']
    crosslinking_agent: Optional[str] = Field(None, description="Crosslinking agent (glutaraldehyde, EDC)")


class NewPreparePreparationSchema(BasePreparationSchema):
    composition_type: Literal['new_prepare']


PreparationUnion = Union[
    MixPreparationSchema,
    AssemblePreparationSchema,
    ConjugatePreparationSchema,
    EmulsifyPreparationSchema,
    EncapsulatePreparationSchema,
    CrosslinkPreparationSchema,
    NewPreparePreparationSchema
]


class BaseAdjuvantSchema(StrictBaseModel):
    name: str = Field(..., description="Adjuvant name")
    size: Optional[str] = Field(None, description="Size description (e.g., diameter, length)")
    shape: Optional[str] = Field(None, description="Shape description (e.g., spherical, rod-like)")
    zeta_potential: Optional[str] = Field(None, description="Zeta potential description")
    newfeature: Optional[str] = Field(None, description="New feature description")
    properties: Optional[Dict[str, Any]] = Field(None, description="Additional physicochemical properties (S_M_E_N, crystallinity, encapsulation_efficiency, etc.)")
    components: Optional[List[ComponentUnion]] = Field(None, description="Constituent components (e.g., PLGA Polymer, MPLA Lipid, OVA Protein). List ALL materials explicitly mentioned in the rationale.")

    preparation: PreparationUnion = Field(..., description="Preparation method details")


class NanoparticleAdjuvantSchema(BaseAdjuvantSchema):

    type: Literal['Nanoparticle', 'nanoparticle']

    particle_size: Optional[str] = Field(None, description="Particle size description (optional, base class size is preferred)")


class MicroparticleAdjuvantSchema(BaseAdjuvantSchema):
    type: Literal['Microparticle', 'microparticle']
    particle_size: Optional[str] = Field(None, description="Microparticle size description (e.g., 1-10 µm)")


class EmulsionAdjuvantSchema(BaseAdjuvantSchema):

    type: Literal['Emulsion', 'emulsion']
    oil_phase: Optional[OilPhaseSchema] = None
    water_phase: Optional[AqueousPhaseSchema] = None


class MoleculeAdjuvantSchema(BaseAdjuvantSchema):

    type: Literal['Molecule', 'molecule']

    target: Optional[List[TargetSchema]] = None
    functional_groups: Optional[List[FunctionalGroupSchema]] = None


class InorganicSaltAdjuvantSchema(BaseAdjuvantSchema):

    type: Literal['Inorganic_salt', 'inorganic_salt']
    salt: Optional[InorganicSaltSchema] = None
    concentration: Optional[float] = None


class HydrogelAdjuvantSchema(BaseAdjuvantSchema):

    type: Literal['Hydrogel', 'hydrogel']
    polymer: Optional[PolymerSchema] = None
    crosslinker: Optional[str] = None


class MicroneedlingAdjuvantSchema(BaseAdjuvantSchema):

    type: Literal['Microneedling', 'microneedling']
    needle_length: Optional[float] = None
    needle_material: Optional[str] = None


AdjuvantUnion = Annotated[
    Union[
        NanoparticleAdjuvantSchema,
        MicroparticleAdjuvantSchema,
        EmulsionAdjuvantSchema,
        MoleculeAdjuvantSchema,
        InorganicSaltAdjuvantSchema,
        HydrogelAdjuvantSchema,
        MicroneedlingAdjuvantSchema
    ],
    Field(discriminator='type')
]


class AntigenSchema(StrictBaseModel):
    name: str
    dose: Optional[float] = None
    size: Optional[str] = Field(None, description="Size description")
    shape: Optional[str] = Field(None, description="Shape description")
    zeta_potential: Optional[str] = Field(None, description="Zeta potential description")
    properties: Optional[Dict[str, Any]] = Field(None, description="Additional physicochemical properties")
    components: Optional[List[ComponentUnion]] = Field(None, description="Constituent components (e.g., OVA Protein, peptide epitopes). List ALL materials explicitly mentioned in the rationale.")

    preparation: PreparationUnion = Field(..., description="Preparation method details")

class VaccineSchema(StrictBaseModel):
    name: str
    target_disease: str
    structure: Optional[str] = None
    properties: Optional[Dict[str, Any]] = Field(None, description="Additional vaccine properties")

    antigen: AntigenSchema


    adjuvant: AdjuvantUnion


    preparation: PreparationUnion = Field(..., description="Vaccine preparation method details")

class DiseaseModelSchema(StrictBaseModel):

    name: str = Field(..., description="Disease model name")
    type: Optional[str] = Field(None, description="Model type")
    host_animal: str = Field(..., description="Host animal")
    challenge_route: Optional[str] = Field(None, description="Challenge route")
    rationale: Optional[str] = Field(None, description="Rationale for selecting this model. Must be extremely concise, no more than 200 words per point.")

    reasoning: Optional[str] = Field(None, description="Reasoning explanation about this disease model")

class ExperimentDesignData(StrictBaseModel):

    vaccine: VaccineSchema

    disease_model: DiseaseModelSchema

    claim_statements: List[ClaimStatementSchema] = Field(
        ...,
        min_length=1,
        max_length=8,
        description=(
            "FSM-constrained Stage 2 concept claims with only functor and outcome. "
            "Generate this field instead of free-form claim strings; claims_dsl "
            "is serialized from it."
        )
    )
    claims_dsl: List[str] = Field(
        default_factory=list,
        description="Legacy downstream claim DSL strings, automatically serialized from claim_statements."
    )
    mechanism_chains: List[MechanismChainSchema] = Field(
        ...,
        min_length=1,
        max_length=5,
        description=(
            "FSM-constrained mechanism chains. Generate this field instead of free-form "
            "mechanism strings; mechanisms_dsl is serialized from it."
        )
    )
    mechanisms_dsl: List[str] = Field(
        default_factory=list,
        description="Legacy downstream DSL strings, automatically serialized from mechanism_chains."
    )

    rationale: List[str] = Field(
        ...,
        description="Scientific rationale and analysis of the experimental design. CRITICAL: Each item in this list must be a complete, independent, and self-contained point. Each point MUST be extremely concise - STRICTLY no more than 200 words. Each point MUST end with proper punctuation (. ! or ?). When generating the last item, stop immediately after completing a sentence within the 200-word limit. Do NOT continue beyond 200 words for any item."
    )

    @staticmethod
    def _parse_legacy_mechanism_node(node_text: str) -> Dict[str, str]:
        node_text = node_text.strip()
        match = re.match(r"^([A-Za-z][A-Za-z0-9_]*)\((.*)\)$", node_text)
        if not match:
            raise ValueError(f"Invalid mechanism node syntax: {node_text}")
        value = re.sub(r"\s+", " ", match.group(2).replace("_", " ")).strip()
        if not value:
            raise ValueError(f"Empty mechanism node value: {node_text}")
        return {"functor": match.group(1), "value": value}

    @classmethod
    def _parse_legacy_mechanism_chain(cls, chain_text: str) -> Dict[str, Any]:
        parts = [p.strip() for p in chain_text.split(">>") if p.strip()]
        if len(parts) < 2:
            raise ValueError(f"Mechanism chain must contain at least two nodes: {chain_text}")
        return {"nodes": [cls._parse_legacy_mechanism_node(part) for part in parts]}

    @staticmethod
    def _parse_legacy_claim(claim_text: str) -> Dict[str, Any]:
        claim_text = str(claim_text).strip()
        m = re.search(
            r"Compare\(([^,)]*),\s*([^)]+)\)\s*=\s*([A-Za-z][A-Za-z0-9_]*)\((.*)\)\s*$",
            claim_text,
        )
        if m:
            _, _, functor, args_text = m.groups()
        else:
            rhs = re.search(r"(?:^|=)\s*([A-Za-z][A-Za-z0-9_]*)\((.*)\)\s*$", claim_text)
            if not rhs:
                raise ValueError(f"Invalid claim DSL syntax: {claim_text}")
            functor, args_text = rhs.groups()

        arg_parts = [p.strip() for p in args_text.split(",") if p.strip()]
        if not arg_parts:
            raise ValueError(f"Claim DSL must contain an outcome: {claim_text}")
        return {
            "functor": _CLAIM_FUNCTOR_ALIASES.get(functor, functor),
            "outcome": arg_parts[0],
        }

    @model_validator(mode="before")
    @classmethod
    def derive_mechanism_chains_from_legacy(cls, data):

        if not isinstance(data, dict):
            return data

        data = dict(data)
        if data.get("claim_statements"):
            claims = data.get("claim_statements") or []
            data["claim_statements"] = [
                cls._parse_legacy_claim(claim)
                if isinstance(claim, str) else claim
                for claim in claims
            ]
        elif data.get("claims_dsl"):
            legacy_claims = data.get("claims_dsl") or []
            data["claim_statements"] = [
                cls._parse_legacy_claim(claim)
                if isinstance(claim, str) else claim
                for claim in legacy_claims
            ]

        if not data.get("mechanism_chains") and data.get("mechanisms_dsl"):
            legacy_chains = data.get("mechanisms_dsl") or []
            data["mechanism_chains"] = [
                cls._parse_legacy_mechanism_chain(chain)
                if isinstance(chain, str) else chain
                for chain in legacy_chains
            ]

        if isinstance(data.get("rationale"), str):
            data["rationale"] = [data["rationale"]]

        return data

    @model_validator(mode="after")
    def serialize_legacy_dsl(self):
        self.claims_dsl = [claim.to_dsl() for claim in self.claim_statements]
        self.mechanisms_dsl = [chain.to_dsl() for chain in self.mechanism_chains]
        return self

class CompleteResponse(StrictBaseModel):
    scientific_rationale: str = Field(..., description="Scientific rationale.")
    formalized_design: ExperimentDesignData
