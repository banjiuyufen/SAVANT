"""Predicate data structures used by SAVANT."""


from dataclasses import dataclass, field
from typing import List, Optional, Any, Union
import sys
import os


_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from savant.grounding.unit_utils import PhysicalQuantity


@dataclass
class Argument:

    role: str
    value: Any
    value_type: str
    embedding: Optional[List[float]] = None

    def __repr__(self):
        if self.value_type == "quantity" and isinstance(self.value, PhysicalQuantity):
            return f"Arg({self.role}={self.value})"
        elif self.value_type == "object":

            obj_type = type(self.value).__name__
            return f"Arg({self.role}={obj_type})"
        else:
            return f"Arg({self.role}={self.value})"

    def is_variable(self) -> bool:

        return self.value_type == "variable" or self.value == "?"

    def get_comparable_value(self):

        if isinstance(self.value, PhysicalQuantity):
            return self.value
        elif isinstance(self.value, str):
            return self.value.lower().strip()
        else:
            return str(self.value)


@dataclass
class Predicate:

    functor: str
    args: List[Argument]
    source: Optional[str] = None

    def __repr__(self):
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.functor}({args_str})"

    def __hash__(self):

        return hash((self.functor, tuple(arg.role for arg in self.args)))

    def matches_signature(self, other: 'Predicate') -> bool:

        return (self.functor == other.functor and
                len(self.args) == len(other.args))

    def get_arg_by_role(self, role: str) -> Optional[Argument]:

        for arg in self.args:
            if arg.role == role:
                return arg
        return None

    def has_variable(self) -> bool:

        return any(arg.is_variable() for arg in self.args)


@dataclass
class AtomicInference:

    premise: Predicate
    conclusion: Predicate
    source: Optional[str] = None
    confidence: float = 1.0

    def __repr__(self):
        return f"{self.premise} → {self.conclusion}"

    def bind_variable(self, var_name: str, value: Any):


        for arg in self.premise.args + self.conclusion.args:
            if arg.value == var_name or arg.value == "?":
                arg.value = value
                arg.value_type = "object" if hasattr(value, '__dict__') else "concept"

@dataclass
class InferenceChain:

    steps: List[AtomicInference]
    name: Optional[str] = None

    def __repr__(self):
        if not self.steps:
            return "InferenceChain(empty)"
        chain_str = " >> ".join(str(step.conclusion.functor) for step in self.steps)
        return f"InferenceChain({chain_str})"

    def __len__(self):
        return len(self.steps)

    def get_start_premise(self) -> Optional[Predicate]:

        return self.steps[0].premise if self.steps else None

    def get_final_conclusion(self) -> Optional[Predicate]:

        return self.steps[-1].conclusion if self.steps else None

    def to_atomic_inferences(self) -> List[AtomicInference]:

        return self.steps


def create_predicate(functor: str, *args, source: str = None) -> Predicate:

    arguments = []
    for arg in args:
        if isinstance(arg, tuple) and len(arg) >= 3:
            role, value, value_type = arg[0], arg[1], arg[2]
            arguments.append(Argument(role=role, value=value, value_type=value_type))
        elif isinstance(arg, Argument):
            arguments.append(arg)

    return Predicate(functor=functor, args=arguments, source=source)

def create_argument(role: str, value: Any, value_type: str = None) -> Argument:

    if value_type is None:

        if isinstance(value, PhysicalQuantity):
            value_type = "quantity"
        elif hasattr(value, '__dict__'):
            value_type = "object"
        elif value == "?" or value == "?x":
            value_type = "variable"
        elif isinstance(value, str):
            value_type = "concept"
        else:
            value_type = "category"

    return Argument(role=role, value=value, value_type=value_type)


if __name__ == "__main__":
    print("="*60)
    print('Runtime diagnostic.')
    print("="*60)


    print('Runtime diagnostic.')
    pred1 = Predicate(
        functor="Type",
        args=[
            Argument(role="agent", value="adjuvant_obj", value_type="object"),
            Argument(role="category", value="nanoparticle", value_type="category")
        ],
        source="design"
    )
    print('Runtime diagnostic.')


    print('Runtime diagnostic.')
    from savant.grounding.unit_utils import UnitParser
    size_qty = UnitParser.parse_length("10-20 nm")
    pred2 = Predicate(
        functor="Size",
        args=[
            Argument(role="agent", value="adjuvant_obj", value_type="object"),
            Argument(role="value", value=size_qty, value_type="quantity")
        ],
        source="design"
    )
    print('Runtime diagnostic.')


    print('Runtime diagnostic.')
    premise = Predicate(
        functor="Size",
        args=[
            Argument(role="agent", value="?", value_type="variable"),
            Argument(role="value", value=size_qty, value_type="quantity")
        ]
    )
    conclusion = Predicate(
        functor="Target",
        args=[
            Argument(role="agent", value="?", value_type="variable"),
            Argument(role="target", value="lymph nodes", value_type="concept")
        ]
    )
    inference = AtomicInference(premise=premise, conclusion=conclusion)
    print('Runtime diagnostic.')


    print('Runtime diagnostic.')
    step2 = AtomicInference(
        premise=conclusion,
        conclusion=Predicate(
            functor="Activate",
            args=[
                Argument(role="agent", value="?", value_type="variable"),
                Argument(role="target", value="APCs", value_type="concept")
            ]
        )
    )
    chain = InferenceChain(steps=[inference, step2])
    print('Runtime diagnostic.')
    print('Runtime diagnostic.')
    print('Runtime diagnostic.')

    print("\n" + "="*60)
