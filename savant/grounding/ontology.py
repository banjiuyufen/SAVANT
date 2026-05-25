"""Ontology dataclasses used for schema-level constraints."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Literal, Union, Any


@dataclass
class Preparation:

    name: str
    composition_type: Optional[Literal['express', 'mix', 'assemble', 'conjugate', 'emulsify', 'encapsulate', 'crosslink', 'new_prepare']] = None
    preparation_details: Optional[str] = None
    mixing_conditions: Optional[str] = None
    conjugation_chemistry: Optional[str] = None
    assembly_method: Optional[str] = None
    encapsulation_details: Optional[str] = None
    cargo: Optional[str] = None
    carrier: Optional[str] = None
    complexation: Optional[str] = None


@dataclass
class BiomoleculeComponent:

    name: str
    features: Optional[Dict[str, Union[str, float]]] = field(default_factory=dict)

@dataclass
class Protein(BiomoleculeComponent):

    molecular_weight: Optional[float] = None
    amino_acid_sequence: Optional[str] = None
    isoelectric_point: Optional[float] = None
    structure: Optional[str] = None
    function: Optional[str] = None
    modifications: Optional[str] = None
    source: Optional[str] = None
    stability: Optional[str] = None

    def __post_init__(self):
        self.features.update({
            'molecular_weight': self.molecular_weight,
            'structure': self.structure,
            'function': self.function
        })

@dataclass
class Antibody(BiomoleculeComponent):

    isotype: Optional[str] = None
    affinity: Optional[float] = None
    molecular_weight: Optional[float] = None
    structure: Optional[str] = None
    epitope: Optional[str] = None
    species: Optional[str] = None
    target_disease: Optional[str] = None

@dataclass
class NucleicAcid(BiomoleculeComponent):

    sequence: Optional[str] = None
    molecular_weight: Optional[float] = None
    type: Optional[str] = None
    length: Optional[int] = None
    modifications: Optional[str] = None

@dataclass
class Photosensitizer(BiomoleculeComponent):

    formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    activation_wavelength: Optional[str] = None
    description: Optional[str] = None

@dataclass
class Lipid(BiomoleculeComponent):

    molecular_weight: Optional[float] = None
    formula: Optional[str] = None
    lipid_type: Optional[str] = None
    hydrophobic_tail: Optional[str] = None
    hydrophilic_head: Optional[str] = None
    modifications: Optional[str] = None

@dataclass
class SmallMolecule(BiomoleculeComponent):

    molecular_weight: Optional[float] = None
    formula: Optional[str] = None
    smiles: Optional[str] = None
    solubility: Optional[str] = None
    bioactivity: Optional[str] = None


@dataclass
class Phase:
    name: str
    phase_type: Literal['Aqueous', 'Organic']
    temperature: Optional[float] = None
    pH: Optional[float] = None
    ionic_strength: Optional[float] = None
    viscosity: Optional[float] = None
    density: Optional[float] = None

@dataclass
class AqueousPhase(Phase):
    def __post_init__(self):
        self.phase_type = 'Aqueous'

@dataclass
class OilPhase(Phase):
    def __post_init__(self):
        self.phase_type = 'Organic'

@dataclass
class FunctionalGroup:
    name: str
    formula: str
    properties: Optional[str] = None
    role: Optional[str] = None

@dataclass
class InorganicSalt:
    name: str
    formula: str
    properties: Optional[str] = None
    solubility: Optional[str] = None

@dataclass
class Polymer:
    name: str
    formula: Optional[str] = None
    properties: Optional[str] = None

@dataclass
class Target:
    name: str
    function: Optional[str] = None
    pathway: Optional[str] = None


@dataclass
class Adjuvant:
    name: str
    type: Optional[Literal['Particle', 'Nanoparticle', 'Microparticle', 'Molecule', 'Inorganic_salt', 'Hydrogel', 'Microneedling', 'Emulsion', 'Unknown']] = 'Unknown'
    preparation: Optional[Preparation] = None
    form: Optional[Any] = None
    components: Optional[List[Any]] = None
    size: Optional[Any] = None
    shape: Optional[str] = None
    zeta_potential: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)


    @staticmethod
    def express(name: str, components: Any, adjuvant_type: str = 'Unknown', size: Optional[Any] = None, shape: Optional[str] = None, zeta_potential: Optional[str] = None, properties: Optional[Dict[str, Any]] = None, preparation: Optional[Preparation] = None) -> 'Adjuvant':
        return Adjuvant(name=name, type=adjuvant_type, preparation=preparation, size=size, shape=shape, zeta_potential=zeta_potential, properties=properties or {}, form=components)

    @staticmethod
    def mix(name: str, components: List[Any], adjuvant_type: str = 'Unknown', size: Optional[Any] = None, shape: Optional[str] = None, zeta_potential: Optional[str] = None, properties: Optional[Dict[str, Any]] = None, preparation: Optional[Preparation] = None) -> 'Adjuvant':
        return Adjuvant(name=name, type=adjuvant_type, preparation=preparation, size=size, shape=shape, zeta_potential=zeta_potential, properties=properties or {}, form=components)

    @staticmethod
    def assemble(name: str, components: Any, adjuvant_type: str = 'Unknown', size: Optional[Any] = None, shape: Optional[str] = None, zeta_potential: Optional[str] = None, properties: Optional[Dict[str, Any]] = None, preparation: Optional[Preparation] = None) -> 'Adjuvant':
        return Adjuvant(name=name, type=adjuvant_type, preparation=preparation, size=size, shape=shape, zeta_potential=zeta_potential, properties=properties or {}, form=components)

    @staticmethod
    def new_prepare(name: str, components: Any, adjuvant_type: str = 'Unknown', size: Optional[Any] = None, shape: Optional[str] = None, zeta_potential: Optional[str] = None, properties: Optional[Dict[str, Any]] = None, preparation: Optional[Preparation] = None) -> 'Adjuvant':
        return Adjuvant(name=name, type=adjuvant_type, preparation=preparation, size=size, shape=shape, zeta_potential=zeta_potential, properties=properties or {}, form=components)

    @staticmethod
    def emulsify(name: str, components: Any, adjuvant_type: str = 'Unknown', size: Optional[Any] = None, shape: Optional[str] = None, zeta_potential: Optional[str] = None, properties: Optional[Dict[str, Any]] = None, preparation: Optional[Preparation] = None) -> 'Adjuvant':
        return Adjuvant(name=name, type=adjuvant_type, preparation=preparation, size=size, shape=shape, zeta_potential=zeta_potential, properties=properties or {}, form=components)

    @staticmethod
    def encapsulate(name: str, components: Any, adjuvant_type: str = 'Unknown', size: Optional[Any] = None, shape: Optional[str] = None, zeta_potential: Optional[str] = None, properties: Optional[Dict[str, Any]] = None, preparation: Optional[Preparation] = None) -> 'Adjuvant':
        return Adjuvant(name=name, type=adjuvant_type, preparation=preparation, size=size, shape=shape, zeta_potential=zeta_potential, properties=properties or {}, form=components)

    @staticmethod
    def crosslink(name: str, components: Any, adjuvant_type: str = 'Unknown', size: Optional[Any] = None, shape: Optional[str] = None, zeta_potential: Optional[str] = None, properties: Optional[Dict[str, Any]] = None, preparation: Optional[Preparation] = None) -> 'Adjuvant':
        return Adjuvant(name=name, type=adjuvant_type, preparation=preparation, size=size, shape=shape, zeta_potential=zeta_potential, properties=properties or {}, form=components)

@dataclass
class NanoparticleAdjuvant(Adjuvant):
    newfeature: Optional[str] = None
    particle_size: Optional[Literal['Nano','Micro','5-100nm','67nm']] = None
    def __post_init__(self): self.type = 'Nanoparticle'

@dataclass
class MicroparticleAdjuvant(Adjuvant):

    newfeature: Optional[str] = None
    particle_size: Optional[Literal['Micro','1-10um','5um','10-100um']] = None
    def __post_init__(self): self.type = 'Microparticle'

@dataclass
class EmulsionAdjuvant(Adjuvant):
    oil_phase: Optional[OilPhase] = None
    water_phase: Optional[AqueousPhase] = None
    newfeature: Optional[str] = None
    def __post_init__(self): self.type = 'Emulsion'

@dataclass
class MoleculeAdjuvant(Adjuvant):
    functional_groups: Optional[List[FunctionalGroup]] = None
    target: Optional[List[Target]] = None
    newfeature: Optional[str] = None
    def __post_init__(self): self.type = 'Molecule'

@dataclass
class InorganicSaltAdjuvant(Adjuvant):
    salt: Optional[InorganicSalt] = None
    concentration: Optional[float] = None
    newfeature: Optional[str] = None
    def __post_init__(self): self.type = 'Inorganic_salt'

@dataclass
class HydrogelAdjuvant(Adjuvant):
    polymer: Optional[Polymer] = None
    crosslinker: Optional[str] = None
    newfeature: Optional[str] = None
    def __post_init__(self): self.type = 'Hydrogel'

@dataclass
class MicroneedlingAdjuvant(Adjuvant):
    needle_length: Optional[float] = None
    needle_material: Optional[str] = None
    newfeature: Optional[str] = None
    def __post_init__(self): self.type = 'Microneedling'


@dataclass
class Antigen:
    name: str
    preparation: Optional[Preparation] = None
    dose: Optional[float] = None
    form: Optional[Union[BiomoleculeComponent, List, Dict]] = None
    components: Optional[List[Any]] = None
    size: Optional[Any] = None
    shape: Optional[str] = None
    zeta_potential: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def express(name: str, components: Any, dose: Optional[Any] = None, size: Optional[Any] = None, shape: Optional[str] = None, zeta_potential: Optional[str] = None, properties: Optional[Dict[str, Any]] = None, preparation: Optional[Preparation] = None) -> 'Antigen':
        return Antigen(name=name, preparation=preparation, form=components, dose=dose, size=size, shape=shape, zeta_potential=zeta_potential, properties=properties or {})
    @staticmethod
    def mix(name: str, components: List[Any], dose: Optional[Any] = None, size: Optional[Any] = None, shape: Optional[str] = None, zeta_potential: Optional[str] = None, properties: Optional[Dict[str, Any]] = None, preparation: Optional[Preparation] = None) -> 'Antigen':
        return Antigen(name=name, preparation=preparation, form=components, dose=dose, size=size, shape=shape, zeta_potential=zeta_potential, properties=properties or {})
    @staticmethod
    def conjugate(name: str, object_: Any, instance_: Any, dose: Optional[Any] = None, size: Optional[Any] = None, shape: Optional[str] = None, zeta_potential: Optional[str] = None, properties: Optional[Dict[str, Any]] = None, preparation: Optional[Preparation] = None) -> 'Antigen':
        return Antigen(name=name, preparation=preparation, form={'object': object_, 'instance': instance_}, dose=dose, size=size, shape=shape, zeta_potential=zeta_potential, properties=properties or {})
    @staticmethod
    def new_prepare(name: str, components: Any, dose: Optional[Any] = None, size: Optional[Any] = None, shape: Optional[str] = None, zeta_potential: Optional[str] = None, properties: Optional[Dict[str, Any]] = None, preparation: Optional[Preparation] = None) -> 'Antigen':
        return Antigen(name=name, preparation=preparation, form=components, dose=dose, size=size, shape=shape, zeta_potential=zeta_potential, properties=properties or {})

    @staticmethod
    def assemble(name: str, components: Any, dose: Optional[Any] = None, size: Optional[Any] = None, shape: Optional[str] = None, zeta_potential: Optional[str] = None, properties: Optional[Dict[str, Any]] = None, preparation: Optional[Preparation] = None) -> 'Antigen':
        return Antigen(name=name, preparation=preparation, form=components, dose=dose, size=size, shape=shape, zeta_potential=zeta_potential, properties=properties or {})

    @staticmethod
    def emulsify(name: str, components: Any, dose: Optional[Any] = None, size: Optional[Any] = None, shape: Optional[str] = None, zeta_potential: Optional[str] = None, properties: Optional[Dict[str, Any]] = None, preparation: Optional[Preparation] = None) -> 'Antigen':
        return Antigen(name=name, preparation=preparation, form=components, dose=dose, size=size, shape=shape, zeta_potential=zeta_potential, properties=properties or {})

    @staticmethod
    def encapsulate(name: str, components: Any, dose: Optional[Any] = None, size: Optional[Any] = None, shape: Optional[str] = None, zeta_potential: Optional[str] = None, properties: Optional[Dict[str, Any]] = None, preparation: Optional[Preparation] = None) -> 'Antigen':
        return Antigen(name=name, preparation=preparation, form=components, dose=dose, size=size, shape=shape, zeta_potential=zeta_potential, properties=properties or {})

    @staticmethod
    def crosslink(name: str, components: Any, dose: Optional[Any] = None, size: Optional[Any] = None, shape: Optional[str] = None, zeta_potential: Optional[str] = None, properties: Optional[Dict[str, Any]] = None, preparation: Optional[Preparation] = None) -> 'Antigen':
        return Antigen(name=name, preparation=preparation, form=components, dose=dose, size=size, shape=shape, zeta_potential=zeta_potential, properties=properties or {})


@dataclass
class DiseaseModel:

    name: str
    type: Literal['solid_tumor', 'metastasis', 'virus', 'autoimmune', 'infection']
    host_animal: str
    challenge_route: Optional[str] = None
    rationale: Optional[str] = None


@dataclass
class Vaccine:
    name: str
    target_disease: str
    antigen: Optional[Antigen] = None
    adjuvant: Optional[Adjuvant] = None
    preparation: Optional[Preparation] = None
    structure: Optional[str] = None
    form: Dict[str, Any] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.antigen is not None:
            self.form['antigen'] = self.antigen
        if self.adjuvant is not None:
            self.form['adjuvant'] = self.adjuvant

    @staticmethod
    def express(name: str, target_disease: str, components: Any, structure: Optional[str] = None, properties: Optional[Dict[str, Any]] = None, preparation: Optional[Preparation] = None) -> 'Vaccine':
        return Vaccine(name=name, target_disease=target_disease, preparation=preparation,
                       antigen=(components if isinstance(components, Antigen) else None),
                       structure=structure, properties=properties or {})

    @staticmethod
    def mix(name: str, target_disease: str, antigen: Antigen, adjuvant: Adjuvant, structure: Optional[str] = None, properties: Optional[Dict[str, Any]] = None, preparation: Optional[Preparation] = None) -> 'Vaccine':
        return Vaccine(name=name, target_disease=target_disease, preparation=preparation,
                       antigen=antigen, adjuvant=adjuvant, structure=structure, properties=properties or {})

    @staticmethod
    def conjugate(name: str, target_disease: str, object_: Any, instance_: Any, structure: Optional[str] = None, properties: Optional[Dict[str, Any]] = None, preparation: Optional[Preparation] = None) -> 'Vaccine':
        return Vaccine(name=name, target_disease=target_disease, preparation=preparation,
                       antigen=(object_ if isinstance(object_, Antigen) else instance_),
                       adjuvant=(instance_ if isinstance(instance_, Adjuvant) else object_),
                       structure=structure, properties=properties or {})

    @staticmethod
    def assemble(name: str, target_disease: str, components: Any, structure: Optional[str] = None, properties: Optional[Dict[str, Any]] = None, preparation: Optional[Preparation] = None) -> 'Vaccine':

        ag = components.get('antigen') if isinstance(components, dict) else None
        adj = components.get('adjuvant') if isinstance(components, dict) else None
        return Vaccine(name=name, target_disease=target_disease, preparation=preparation,
                       antigen=ag, adjuvant=adj, structure=structure, properties=properties or {})

    @staticmethod
    def emulsify(name: str, target_disease: str, antigen: Antigen, adjuvant: Adjuvant, structure: Optional[str] = None, properties: Optional[Dict[str, Any]] = None, preparation: Optional[Preparation] = None) -> 'Vaccine':
        return Vaccine(name=name, target_disease=target_disease, preparation=preparation,
                       antigen=antigen, adjuvant=adjuvant, structure=structure, properties=properties or {})

    @staticmethod
    def encapsulate(name: str, target_disease: str, antigen: Antigen, adjuvant: Adjuvant, structure: Optional[str] = None, properties: Optional[Dict[str, Any]] = None, preparation: Optional[Preparation] = None) -> 'Vaccine':
        return Vaccine(name=name, target_disease=target_disease, preparation=preparation,
                       antigen=antigen, adjuvant=adjuvant, structure=structure, properties=properties or {})

    @staticmethod
    def crosslink(name: str, target_disease: str, antigen: Antigen, adjuvant: Adjuvant, structure: Optional[str] = None, properties: Optional[Dict[str, Any]] = None, preparation: Optional[Preparation] = None) -> 'Vaccine':
        return Vaccine(name=name, target_disease=target_disease, preparation=preparation,
                       antigen=antigen, adjuvant=adjuvant, structure=structure, properties=properties or {})

    @staticmethod
    def new_prepare(name: str, target_disease: str, antigen: Antigen, adjuvant: Adjuvant, structure: Optional[str] = None, properties: Optional[Dict[str, Any]] = None, preparation: Optional[Preparation] = None) -> 'Vaccine':
        return Vaccine(name=name, target_disease=target_disease, preparation=preparation,
                       antigen=antigen, adjuvant=adjuvant, structure=structure, properties=properties or {})
