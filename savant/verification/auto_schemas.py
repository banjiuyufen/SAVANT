"""Generated predicate schemas for the theorem library."""


from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class Conjugate:

    agent: Optional[Any] = None
    target: Optional[Any] = None
    magnitude: Optional[Any] = None
    arg0: Optional[Any] = None
    arg1: Optional[Any] = None
    arg2: Optional[Any] = None

@dataclass
class Reduce:

    agent: Optional[Any] = None
    target: Optional[Any] = None
    magnitude: Optional[Any] = None

@dataclass
class VaccineProperty:

    agent: Optional[Any] = None
    property: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class AdjuvantProperty:

    agent: Optional[Any] = None
    property: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class AntigenProperty:

    agent: Optional[Any] = None
    property: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class Equal:

    agent: Optional[Any] = None
    target: Optional[Any] = None
    magnitude: Optional[Any] = None

@dataclass
class Positive:

    agent: Optional[Any] = None
    target: Optional[Any] = None
    magnitude: Optional[Any] = None

@dataclass
class Sustain:

    agent: Optional[Any] = None
    target: Optional[Any] = None
    magnitude: Optional[Any] = None

@dataclass
class Route:

    agent: Optional[Any] = None
    target: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class DiseaseProperty:

    agent: Optional[Any] = None
    property: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class Size:

    agent: Optional[Any] = None
    value: Optional[Any] = None
    target: Optional[Any] = None

@dataclass
class Mix:

    agent: Optional[Any] = None
    target: Optional[Any] = None
    magnitude: Optional[Any] = None

@dataclass
class Encapsulate:

    agent: Optional[Any] = None
    target: Optional[Any] = None
    magnitude: Optional[Any] = None

@dataclass
class Express:

    agent: Optional[Any] = None
    target: Optional[Any] = None
    magnitude: Optional[Any] = None

@dataclass
class Control:

    agent: Optional[Any] = None
    target: Optional[Any] = None
    magnitude: Optional[Any] = None

@dataclass
class Tune:

    agent: Optional[Any] = None
    target: Optional[Any] = None
    magnitude: Optional[Any] = None

@dataclass
class Protect:

    agent: Optional[Any] = None
    target: Optional[Any] = None
    magnitude: Optional[Any] = None

@dataclass
class Storage:

    agent: Optional[Any] = None
    target: Optional[Any] = None
    magnitude: Optional[Any] = None

@dataclass
class Enhance:

    agent: Optional[Any] = None
    target: Optional[Any] = None

@dataclass
class Method:

    agent: Optional[Any] = None
    method: Optional[Any] = None

@dataclass
class Contains:

    agent: Optional[Any] = None
    component: Optional[Any] = None

@dataclass
class Type:

    agent: Optional[Any] = None
    type: Optional[Any] = None

@dataclass
class Name:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class Increase:

    agent: Optional[Any] = None
    target: Optional[Any] = None

@dataclass
class Activate:

    agent: Optional[Any] = None
    target: Optional[Any] = None

@dataclass
class AgMethod:

    agent: Optional[Any] = None
    method: Optional[Any] = None

@dataclass
class Enable:

    agent: Optional[Any] = None
    target: Optional[Any] = None

@dataclass
class Disease:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class Target:

    agent: Optional[Any] = None
    target: Optional[Any] = None

@dataclass
class Agtype:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class Avoid:

    agent: Optional[Any] = None
    target: Optional[Any] = None

@dataclass
class description:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class New_Prepare:

    agent: Optional[Any] = None
    target: Optional[Any] = None

@dataclass
class Model:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class shape:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class Description:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class Display:

    agent: Optional[Any] = None
    target: Optional[Any] = None

@dataclass
class Induce:

    agent: Optional[Any] = None
    target: Optional[Any] = None

@dataclass
class Irradiate:

    agent: Optional[Any] = None
    target: Optional[Any] = None

@dataclass
class structure:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class Establish:

    agent: Optional[Any] = None
    target: Optional[Any] = None

@dataclass
class zeta_potential:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class Assemble:

    agent: Optional[Any] = None
    target: Optional[Any] = None

@dataclass
class Form:

    agent: Optional[Any] = None
    target: Optional[Any] = None

@dataclass
class Regimen:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class geometry:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class AgContains:

    agent: Optional[Any] = None
    component: Optional[Any] = None

@dataclass
class Balance:

    agent: Optional[Any] = None
    target: Optional[Any] = None

@dataclass
class net_charge:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class thermal_stability:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class Agcell_count:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class Agcomponents:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class Agoutput:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class Agsize:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class CpG_dose:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class Formation:

    agent: Optional[Any] = None
    target: Optional[Any] = None

@dataclass
class MPL_dose:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class Multivalent:

    agent: Optional[Any] = None
    target: Optional[Any] = None

@dataclass
class Zeta_potential:

    agent: Optional[Any] = None
    target: Optional[Any] = None

@dataclass
class alum_dose:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class base_diameter:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class biodegradability:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class cell_wall:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class charge:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class components:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class conjugation_stability:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class core_material:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class crystallinity:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class diameter:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class encapsulation_efficiency:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class height:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class length:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class lengths:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class mannose_sugar_chains:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class mass_ratio:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class peptide_loading:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class pores:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class shell_material:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class size_range:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class spacing:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class stability:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class storage_temperature:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class surface:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class thermostability:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class tip_angle:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class tip_diameter:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class total_dose:

    agent: Optional[Any] = None
    value: Optional[Any] = None

@dataclass
class volume_decrease:

    agent: Optional[Any] = None
    value: Optional[Any] = None


SCHEMAS = {
    "Conjugate": Conjugate,
    "Reduce": Reduce,
    "VaccineProperty": VaccineProperty,
    "AdjuvantProperty": AdjuvantProperty,
    "AntigenProperty": AntigenProperty,
    "Equal": Equal,
    "Positive": Positive,
    "Sustain": Sustain,
    "Route": Route,
    "DiseaseProperty": DiseaseProperty,
    "Size": Size,
    "Mix": Mix,
    "Encapsulate": Encapsulate,
    "Express": Express,
    "Control": Control,
    "Tune": Tune,
    "Protect": Protect,
    "Storage": Storage,
    "Enhance": Enhance,
    "Method": Method,
    "Contains": Contains,
    "Type": Type,
    "Name": Name,
    "Increase": Increase,
    "Activate": Activate,
    "AgMethod": AgMethod,
    "Enable": Enable,
    "Disease": Disease,
    "Target": Target,
    "Agtype": Agtype,
    "Avoid": Avoid,
    "description": description,
    "New_Prepare": New_Prepare,
    "Model": Model,
    "shape": shape,
    "Description": Description,
    "Display": Display,
    "Induce": Induce,
    "Irradiate": Irradiate,
    "structure": structure,
    "Establish": Establish,
    "zeta_potential": zeta_potential,
    "Assemble": Assemble,
    "Form": Form,
    "Regimen": Regimen,
    "geometry": geometry,
    "AgContains": AgContains,
    "Balance": Balance,
    "net_charge": net_charge,
    "thermal_stability": thermal_stability,
    "Agcell_count": Agcell_count,
    "Agcomponents": Agcomponents,
    "Agoutput": Agoutput,
    "Agsize": Agsize,
    "CpG_dose": CpG_dose,
    "Formation": Formation,
    "MPL_dose": MPL_dose,
    "Multivalent": Multivalent,
    "Zeta_potential": Zeta_potential,
    "alum_dose": alum_dose,
    "base_diameter": base_diameter,
    "biodegradability": biodegradability,
    "cell_wall": cell_wall,
    "charge": charge,
    "components": components,
    "conjugation_stability": conjugation_stability,
    "core_material": core_material,
    "crystallinity": crystallinity,
    "diameter": diameter,
    "encapsulation_efficiency": encapsulation_efficiency,
    "height": height,
    "length": length,
    "lengths": lengths,
    "mannose_sugar_chains": mannose_sugar_chains,
    "mass_ratio": mass_ratio,
    "peptide_loading": peptide_loading,
    "pores": pores,
    "shell_material": shell_material,
    "size_range": size_range,
    "spacing": spacing,
    "stability": stability,
    "storage_temperature": storage_temperature,
    "surface": surface,
    "thermostability": thermostability,
    "tip_angle": tip_angle,
    "tip_diameter": tip_diameter,
    "total_dose": total_dose,
    "volume_decrease": volume_decrease,
}

def get_schema(functor: str):

    return SCHEMAS.get(functor)

FUNCTOR_META = {
    "Conjugate": {"roles": ['agent', 'target', 'magnitude', 'arg0', 'arg1', 'arg2'], "max_freq": 7 },
    "Reduce": {"roles": ['agent', 'target', 'magnitude'], "max_freq": 469 },
    "VaccineProperty": {"roles": ['agent', 'property', 'value'], "max_freq": 286 },
    "AdjuvantProperty": {"roles": ['agent', 'property', 'value'], "max_freq": 273 },
    "AntigenProperty": {"roles": ['agent', 'property', 'value'], "max_freq": 172 },
    "Equal": {"roles": ['agent', 'target', 'magnitude'], "max_freq": 148 },
    "Positive": {"roles": ['agent', 'target', 'magnitude'], "max_freq": 76 },
    "Sustain": {"roles": ['agent', 'target', 'magnitude'], "max_freq": 75 },
    "Route": {"roles": ['agent', 'target', 'value'], "max_freq": 62 },
    "DiseaseProperty": {"roles": ['agent', 'property', 'value'], "max_freq": 53 },
    "Size": {"roles": ['agent', 'value', 'target'], "max_freq": 51 },
    "Mix": {"roles": ['agent', 'target', 'magnitude'], "max_freq": 30 },
    "Encapsulate": {"roles": ['agent', 'target', 'magnitude'], "max_freq": 9 },
    "Express": {"roles": ['agent', 'target', 'magnitude'], "max_freq": 8 },
    "Control": {"roles": ['agent', 'target', 'magnitude'], "max_freq": 7 },
    "Tune": {"roles": ['agent', 'target', 'magnitude'], "max_freq": 7 },
    "Protect": {"roles": ['agent', 'target', 'magnitude'], "max_freq": 2 },
    "Storage": {"roles": ['agent', 'target', 'magnitude'], "max_freq": 1 },
    "Enhance": {"roles": ['agent', 'target'], "max_freq": 1625 },
    "Method": {"roles": ['agent', 'method'], "max_freq": 1242 },
    "Contains": {"roles": ['agent', 'component'], "max_freq": 835 },
    "Type": {"roles": ['agent', 'type'], "max_freq": 435 },
    "Name": {"roles": ['agent', 'value'], "max_freq": 264 },
    "Increase": {"roles": ['agent', 'target'], "max_freq": 141 },
    "Activate": {"roles": ['agent', 'target'], "max_freq": 101 },
    "AgMethod": {"roles": ['agent', 'method'], "max_freq": 84 },
    "Enable": {"roles": ['agent', 'target'], "max_freq": 35 },
    "Disease": {"roles": ['agent', 'value'], "max_freq": 29 },
    "Target": {"roles": ['agent', 'target'], "max_freq": 26 },
    "Agtype": {"roles": ['agent', 'value'], "max_freq": 18 },
    "Avoid": {"roles": ['agent', 'target'], "max_freq": 15 },
    "description": {"roles": ['agent', 'value'], "max_freq": 13 },
    "New_Prepare": {"roles": ['agent', 'target'], "max_freq": 12 },
    "Model": {"roles": ['agent', 'value'], "max_freq": 10 },
    "shape": {"roles": ['agent', 'value'], "max_freq": 8 },
    "Description": {"roles": ['agent', 'value'], "max_freq": 7 },
    "Display": {"roles": ['agent', 'target'], "max_freq": 7 },
    "Induce": {"roles": ['agent', 'target'], "max_freq": 7 },
    "Irradiate": {"roles": ['agent', 'target'], "max_freq": 6 },
    "structure": {"roles": ['agent', 'value'], "max_freq": 6 },
    "Establish": {"roles": ['agent', 'target'], "max_freq": 5 },
    "zeta_potential": {"roles": ['agent', 'value'], "max_freq": 5 },
    "Assemble": {"roles": ['agent', 'target'], "max_freq": 3 },
    "Form": {"roles": ['agent', 'target'], "max_freq": 3 },
    "Regimen": {"roles": ['agent', 'value'], "max_freq": 3 },
    "geometry": {"roles": ['agent', 'value'], "max_freq": 3 },
    "AgContains": {"roles": ['agent', 'component'], "max_freq": 2 },
    "Balance": {"roles": ['agent', 'target'], "max_freq": 2 },
    "net_charge": {"roles": ['agent', 'value'], "max_freq": 2 },
    "thermal_stability": {"roles": ['agent', 'value'], "max_freq": 2 },
    "Agcell_count": {"roles": ['agent', 'value'], "max_freq": 1 },
    "Agcomponents": {"roles": ['agent', 'value'], "max_freq": 1 },
    "Agoutput": {"roles": ['agent', 'value'], "max_freq": 1 },
    "Agsize": {"roles": ['agent', 'value'], "max_freq": 1 },
    "CpG_dose": {"roles": ['agent', 'value'], "max_freq": 1 },
    "Formation": {"roles": ['agent', 'target'], "max_freq": 1 },
    "MPL_dose": {"roles": ['agent', 'value'], "max_freq": 1 },
    "Multivalent": {"roles": ['agent', 'target'], "max_freq": 1 },
    "Zeta_potential": {"roles": ['agent', 'target'], "max_freq": 1 },
    "alum_dose": {"roles": ['agent', 'value'], "max_freq": 1 },
    "base_diameter": {"roles": ['agent', 'value'], "max_freq": 1 },
    "biodegradability": {"roles": ['agent', 'value'], "max_freq": 1 },
    "cell_wall": {"roles": ['agent', 'value'], "max_freq": 1 },
    "charge": {"roles": ['agent', 'value'], "max_freq": 1 },
    "components": {"roles": ['agent', 'value'], "max_freq": 1 },
    "conjugation_stability": {"roles": ['agent', 'value'], "max_freq": 1 },
    "core_material": {"roles": ['agent', 'value'], "max_freq": 1 },
    "crystallinity": {"roles": ['agent', 'value'], "max_freq": 1 },
    "diameter": {"roles": ['agent', 'value'], "max_freq": 1 },
    "encapsulation_efficiency": {"roles": ['agent', 'value'], "max_freq": 1 },
    "height": {"roles": ['agent', 'value'], "max_freq": 1 },
    "length": {"roles": ['agent', 'value'], "max_freq": 1 },
    "lengths": {"roles": ['agent', 'value'], "max_freq": 1 },
    "mannose_sugar_chains": {"roles": ['agent', 'value'], "max_freq": 1 },
    "mass_ratio": {"roles": ['agent', 'value'], "max_freq": 1 },
    "peptide_loading": {"roles": ['agent', 'value'], "max_freq": 1 },
    "pores": {"roles": ['agent', 'value'], "max_freq": 1 },
    "shell_material": {"roles": ['agent', 'value'], "max_freq": 1 },
    "size_range": {"roles": ['agent', 'value'], "max_freq": 1 },
    "spacing": {"roles": ['agent', 'value'], "max_freq": 1 },
    "stability": {"roles": ['agent', 'value'], "max_freq": 1 },
    "storage_temperature": {"roles": ['agent', 'value'], "max_freq": 1 },
    "surface": {"roles": ['agent', 'value'], "max_freq": 1 },
    "thermostability": {"roles": ['agent', 'value'], "max_freq": 1 },
    "tip_angle": {"roles": ['agent', 'value'], "max_freq": 1 },
    "tip_diameter": {"roles": ['agent', 'value'], "max_freq": 1 },
    "total_dose": {"roles": ['agent', 'value'], "max_freq": 1 },
    "volume_decrease": {"roles": ['agent', 'value'], "max_freq": 1 },
}
