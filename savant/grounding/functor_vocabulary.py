"""Canonical predicate functor vocabulary and aliases."""


from typing import Dict, List, Set
from dataclasses import dataclass


@dataclass
class FunctorDefinition:

    name: str
    category: str
    description: str
    arg_roles: List[str]
    examples: List[str]


STANDARD_FUNCTORS = {

    'Size': FunctorDefinition(
        name='Size',
        category='property',
        description='Runtime diagnostic.',
        arg_roles=['agent', 'value'],
        examples=['Size(?, 10-20 nm)', 'Size(nanoparticle, 100 nm)']
    ),

    'Shape': FunctorDefinition(
        name='Shape',
        category='property',
        description='Runtime diagnostic.',
        arg_roles=['agent', 'type'],
        examples=['Shape(?, sphere)', 'Shape(particle, rod)']
    ),

    'Type': FunctorDefinition(
        name='Type',
        category='property',
        description='Runtime diagnostic.',
        arg_roles=['agent', 'category'],
        examples=['Type(adjuvant, nanoparticle)', 'Type(?, emulsion)']
    ),

    'Contains': FunctorDefinition(
        name='Contains',
        category='property',
        description='Runtime diagnostic.',
        arg_roles=['agent', 'component'],
        examples=['Contains(vaccine, CpG)', 'Contains(?, antigen)']
    ),

    'Route': FunctorDefinition(
        name='Route',
        category='property',
        description='Runtime diagnostic.',
        arg_roles=['agent', 'type'],
        examples=['Route(vaccine, subcutaneous)', 'Route(?, intravenous)']
    ),


    'Target': FunctorDefinition(
        name='Target',
        category='process',
        description='Runtime diagnostic.',
        arg_roles=['agent', 'target'],
        examples=['Target(?, lymph_nodes)', 'Target(nanoparticle, tumor)']
    ),

    'Activate': FunctorDefinition(
        name='Activate',
        category='process',
        description='Runtime diagnostic.',
        arg_roles=['agent', 'target'],
        examples=['Activate(?, APCs)', 'Activate(adjuvant, T_cells)']
    ),

    'Enhance': FunctorDefinition(
        name='Enhance',
        category='process',
        description='Runtime diagnostic.',
        arg_roles=['agent', 'target', 'magnitude'],
        examples=['Enhance(?, immune_response)', 'Enhance(vaccine, IFN-γ, 10-fold)']
    ),

    'Inhibit': FunctorDefinition(
        name='Inhibit',
        category='process',
        description='Runtime diagnostic.',
        arg_roles=['agent', 'target'],
        examples=['Inhibit(?, tumor_growth)', 'Inhibit(drug, enzyme)']
    ),

    'Reduce': FunctorDefinition(
        name='Reduce',
        category='process',
        description='Runtime diagnostic.',
        arg_roles=['agent', 'target', 'magnitude'],
        examples=['Reduce(?, toxicity)', 'Reduce(treatment, side_effects)']
    ),

    'Bind': FunctorDefinition(
        name='Bind',
        category='process',
        description='Runtime diagnostic.',
        arg_roles=['agent', 'target'],
        examples=['Bind(?, TLR9)', 'Bind(ligand, receptor)']
    ),

    'Trigger': FunctorDefinition(
        name='Trigger',
        category='process',
        description='Runtime diagnostic.',
        arg_roles=['agent', 'process'],
        examples=['Trigger(?, immune_response)', 'Trigger(signal, cascade)']
    ),

    'Release': FunctorDefinition(
        name='Release',
        category='process',
        description='Runtime diagnostic.',
        arg_roles=['agent', 'substance'],
        examples=['Release(?, antigen)', 'Release(cell, cytokine)']
    ),

    'Uptake': FunctorDefinition(
        name='Uptake',
        category='process',
        description='Runtime diagnostic.',
        arg_roles=['agent', 'substance'],
        examples=['Uptake(APCs, antigen)', 'Uptake(?, nanoparticle)']
    ),

    'Express': FunctorDefinition(
        name='Express',
        category='process',
        description='Runtime diagnostic.',
        arg_roles=['agent', 'molecule'],
        examples=['Express(?, CD80)', 'Express(cell, MHC)']
    ),


    'Compare': FunctorDefinition(
        name='Compare',
        category='relation',
        description='Runtime diagnostic.',
        arg_roles=['subject', 'comparator', 'metric'],
        examples=['Compare(vaccine, PBS, efficacy)', 'Compare(?, control, response)']
    ),

    'Equal': FunctorDefinition(
        name='Equal',
        category='relation',
        description='Runtime diagnostic.',
        arg_roles=['subject', 'comparator'],
        examples=['Equal(treatment_A, treatment_B)']
    ),


    'mix': FunctorDefinition(
        name='mix',
        category='construction',
        description='Runtime diagnostic.',
        arg_roles=['components'],
        examples=['mix([antigen, adjuvant])']
    ),

    'assemble': FunctorDefinition(
        name='assemble',
        category='construction',
        description='Runtime diagnostic.',
        arg_roles=['components'],
        examples=['assemble({core: lipid, shell: protein})']
    ),

    'conjugate': FunctorDefinition(
        name='conjugate',
        category='construction',
        description='Runtime diagnostic.',
        arg_roles=['object', 'instance'],
        examples=['conjugate(antigen, carrier)']
    ),
}


FUNCTOR_SYNONYMS: Dict[str, str] = {

    'particle_size':          'Size',
    'nanoparticle_size':      'Size',
    'particle_diameter':      'Size',
    'diameter':               'Size',
    'hydrodynamic_diameter':  'Size',
    'hydrodynamic_size':      'Size',
    'size_range':             'Size',
    'bead_size':              'Size',
    'mean_diameter':          'Size',

    'encapsulate':            'Contains',
    'encapsulation':          'Contains',
    'incorporate':            'Contains',
    'load':                   'Contains',
    'embed':                  'Contains',
    'composition':            'Contains',
    'component':              'Contains',

    'targeting':              'Target',
    'localization':           'Target',
    'accumulation':           'Target',
    'delivery':               'Target',
    'homing':                 'Target',
    'colocalize':             'Target',
    'co_localize':            'Target',
    'co-localize':            'Target',
    'co-deliver':             'Target',
    'migrate':                'Target',
    'traffic':                'Target',
    'direct':                 'Target',

    'activation':             'Activate',
    'stimulate':              'Activate',
    'stimulation':            'Activate',
    'prime':                  'Activate',
    'priming':                'Activate',
    'induce':                 'Activate',
    'initiate':               'Activate',
    'enable':                 'Activate',
    'differentiate':          'Activate',

    'increase':               'Enhance',
    'boost':                  'Enhance',
    'amplify':                'Enhance',
    'elevate':                'Enhance',
    'improve':                'Enhance',
    'strengthen':             'Enhance',
    'augment':                'Enhance',
    'promote':                'Enhance',
    'facilitate':             'Enhance',
    'potentiate':             'Enhance',
    'sensitize':              'Enhance',
    'skew':                   'Enhance',
    'drive':                  'Enhance',
    'support':                'Enhance',
    'synergize':              'Enhance',
    'elicit':                 'Enhance',
    'evoke':                  'Enhance',

    'upregulate':             'Enhance',
    'upregulation':           'Enhance',
    'up-regulate':            'Enhance',
    'up_regulate':            'Enhance',

    'downregulate':           'Reduce',
    'downregulation':         'Reduce',
    'down-regulate':          'Reduce',

    'suppress':               'Reduce',
    'inhibit':                'Reduce',
    'block':                  'Inhibit',
    'prevent':                'Reduce',
    'silence':                'Reduce',
    'abrogate':               'Reduce',
    'abolish':                'Reduce',

    'decrease':               'Reduce',
    'diminish':               'Reduce',
    'lower':                  'Reduce',
    'mitigate':               'Reduce',
    'dampen':                 'Reduce',
    'alleviate':              'Reduce',
    'attenuate':              'Reduce',

    'sustain':                'Sustain',
    'prolong':                'Prolong',
    'maintain':               'Sustain',
    'preserve':               'Sustain',
    'persist':                'Sustain',
    'extend':                 'Prolong',
    'elongate':               'Prolong',
    'delay':                  'Prolong',

    'trigger':                'Activate',

    'binding':                'Activate',
    'attach':                 'Activate',
    'interact':               'Activate',
    'recognize':              'Activate',
    'ligate':                 'Activate',
    'bind':                   'Activate',

    'mediate':                'Cause',
    'generate':               'Cause',
    'produce':                'Cause',
    'result in':              'Cause',
    'lead to':                'Cause',
    'cause':                  'Cause',

    'secrete':                'Enhance',
    'secretion':              'Enhance',
    'discharge':              'Enhance',
    'shed':                   'Enhance',
    'release':                'Enhance',

    'internalization':        'Enhance',
    'endocytosis':            'Enhance',
    'phagocytosis':           'Enhance',
    'pinocytosis':            'Enhance',
    'macropinocytosis':       'Enhance',
    'uptake':                 'Enhance',
}


CONSTRUCTION_METHOD_VALUES: Set[str] = {
    'express', 'mix', 'assemble', 'conjugate', 'emulsify',
    'encapsulate', 'crosslink', 'new_prepare', 'formulate',
    'lyophilize', 'precipitate',
}


def is_construction_method_value(word: str) -> bool:

    return word.lower() in CONSTRUCTION_METHOD_VALUES


EXTENDED_FUNCTORS: Set[str] = {

    'Transcriptional_upregulation',
    'Endosomal_escape',
    'Cross-presentation',
    'Permeabilization_of_membranes',
    'Immunogenic_cell_death',
    'Receptor_binding',
    'DC_recruitment_and_maturement',
    'CTL_effectiveness_in_tumors',


    'IsEffectiveAdjuvant',
    'IsVaccine',
}


def normalize_functor(functor: str) -> str:


    functor_normalized = functor.strip()


    if functor_normalized in STANDARD_FUNCTORS:
        return functor_normalized


    functor_lower = functor_normalized.lower()
    if functor_lower in FUNCTOR_SYNONYMS:
        return FUNCTOR_SYNONYMS[functor_lower]


    return functor_normalized


def is_standard_functor(functor: str) -> bool:

    return functor in STANDARD_FUNCTORS


def is_extended_functor(functor: str) -> bool:

    return functor in EXTENDED_FUNCTORS


def get_functor_definition(functor: str) -> FunctorDefinition:

    normalized = normalize_functor(functor)
    return STANDARD_FUNCTORS.get(normalized)


def get_all_standard_functors() -> List[str]:

    return list(STANDARD_FUNCTORS.keys())


def get_functors_by_category(category: str) -> List[str]:

    return [
        name for name, defn in STANDARD_FUNCTORS.items()
        if defn.category == category
    ]


if __name__ == "__main__":
    print("="*80)
    print('Runtime diagnostic.')
    print("="*80)

    print('Runtime diagnostic.')
    for category in ['property', 'process', 'relation', 'construction']:
        functors = get_functors_by_category(category)
        print(f"\n  [{category.upper()}]")
        for f in functors:
            defn = STANDARD_FUNCTORS[f]
            print(f"    - {f}: {defn.description}")

    print('Runtime diagnostic.')
    test_cases = ['targeting', 'increase', 'activation', 'suppress']
    for word in test_cases:
        normalized = normalize_functor(word)
        print(f"    '{word}' → '{normalized}'")

    print('Runtime diagnostic.')
    for f in sorted(EXTENDED_FUNCTORS):
        print(f"    - {f}")

    print("\n" + "="*80)
