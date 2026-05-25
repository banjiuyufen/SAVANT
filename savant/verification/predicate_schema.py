"""Predicate schema registry used by the matcher."""


import sys
import os
from dataclasses import dataclass, make_dataclass, fields
from typing import Dict, List, Any, Optional, Union, Type
import logging

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from savant.grounding.predicate_core import Predicate, Argument


from savant.verification.auto_schemas import SCHEMAS, get_schema

logger = logging.getLogger(__name__)


PREDEFINED_SCHEMAS = SCHEMAS


class PredicateSchemaRegistry:


    def __init__(self):
        self.schemas: Dict[str, Type] = dict(SCHEMAS)
        self.instances: Dict[str, List[Any]] = {k: [] for k in self.schemas}

    def get_schema(self, functor: str) -> Optional[Type]:

        return self.schemas.get(functor)

    def ensure_schema(self, predicate: Predicate) -> Type:

        functor = predicate.functor

        if functor in self.schemas:
            return self.schemas[functor]


        role_names = []
        for arg in predicate.args:
            if isinstance(arg, Argument):
                role = arg.role
            else:
                role = "value"
            if role not in role_names:
                role_names.append(role)


        if "agent" in role_names:
            role_names.remove("agent")
            role_names.insert(0, "agent")

        fields_def = [(r, Optional[Any], None) for r in role_names]
        dynamic_class = make_dataclass(
            functor,
            fields_def,
            namespace={'__module__': __name__}
        )
        self.schemas[functor] = dynamic_class
        if functor not in self.instances:
            self.instances[functor] = []
        logger.info('Runtime diagnostic.')
        return dynamic_class

    def ensure_schema_from_dict(self, pred_dict: Dict[str, Any]) -> Optional[Type]:

        if not pred_dict or not pred_dict.get("functor"):
            return None

        functor = pred_dict["functor"]
        if functor in self.schemas:
            return self.schemas[functor]

        role_names = []
        for arg in pred_dict.get("args", []):
            role = arg.get("role", "value") if isinstance(arg, dict) else "value"
            if role not in role_names:
                role_names.append(role)

        if "agent" in role_names:
            role_names.remove("agent")
            role_names.insert(0, "agent")
        if not role_names:
            role_names = ["agent", "target"]

        fields_def = [(r, Optional[Any], None) for r in role_names]
        dynamic_class = make_dataclass(
            functor,
            fields_def,
            namespace={'__module__': __name__}
        )
        self.schemas[functor] = dynamic_class
        if functor not in self.instances:
            self.instances[functor] = []
        logger.info('Runtime diagnostic.')
        return dynamic_class

    def create_instance(self, predicate: Predicate) -> Any:

        schema_cls = self.ensure_schema(predicate)

        kwargs = {}
        for arg in predicate.args:
            if isinstance(arg, Argument):
                kwargs[arg.role] = arg.value
            else:
                kwargs["value"] = arg

        try:
            instance = schema_cls(**kwargs)
            self.instances[predicate.functor].append(instance)
            return instance
        except Exception as e:
            logger.error('Runtime diagnostic.')
            return None

    def get_instances_by_functor(self, functor_name: str) -> List[Any]:
        return self.instances.get(functor_name, [])


    get_instances_by_class = get_instances_by_functor

    def get_all_functors(self) -> List[str]:
        return list(self.schemas.keys())


    get_all_class_names = get_all_functors

    def get_instance_count(self) -> int:
        return sum(len(insts) for insts in self.instances.values())

    def clear(self):
        self.instances.clear()
        self.schemas = dict(SCHEMAS)


    register_predicate = create_instance

    def summary(self) -> str:
        lines = [
            'Runtime diagnostic.',
            'Runtime diagnostic.',
            'Runtime diagnostic.',
            'Runtime diagnostic.' if len(self.schemas) > 20 else 'Runtime diagnostic.',
        ]
        return "\n".join(lines)


class ProxyFunctorMapper:


    def __init__(self):
        self.functor_mappings = {
            'Runtime diagnostic.': "Enhance",
            'Runtime diagnostic.': "Activate",
            'Runtime diagnostic.': "Prolong",
            'Runtime diagnostic.': "Increase",
            'Runtime diagnostic.': "Reduce",
            'Runtime diagnostic.': "Contains",
            'Runtime diagnostic.': "AssignProperty",
            "boost": "Enhance",
            "amplify": "Enhance",
            "strengthen": "Enhance",
            "activate": "Activate",
            "stimulate": "Activate",
            "extend": "Prolong",
            "maintain": "Sustain",
            "decrease": "Reduce",
            "lower": "Reduce",
            "contain": "Contains",
            "include": "Contains",
        }

    def map_functor(self, original_functor: str) -> str:
        if original_functor in self.functor_mappings:
            return self.functor_mappings[original_functor]
        return original_functor

    def add_mapping(self, original: str, proxy: str):
        self.functor_mappings[original] = proxy
