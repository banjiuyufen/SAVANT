"""Unit parsing helpers used during predicate matching."""

import re
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class PhysicalQuantity:

    original_text: str
    min_val: float
    max_val: float
    avg_val: float
    unit: str
    category: str

    def __repr__(self):
        if self.min_val == self.max_val:
            return f"Quantity({self.avg_val} {self.unit})"
        return f"Quantity([{self.min_val}-{self.max_val}] {self.unit})"


SEMI_QUANTITATIVE_MAP = {

    "dense": {"ordinal": 3, "level": "high", "numeric_range": (70, 100)},
    "sparse": {"ordinal": 1, "level": "low", "numeric_range": (0, 30)},
    "moderate": {"ordinal": 2, "level": "medium", "numeric_range": (30, 70)},


    "strong": {"ordinal": 3, "level": "high", "numeric_range": (70, 100)},
    "weak": {"ordinal": 1, "level": "low", "numeric_range": (0, 30)},
    "mild": {"ordinal": 1, "level": "low", "numeric_range": (0, 30)},


    "high": {"ordinal": 3, "level": "high", "numeric_range": (70, 100)},
    "low": {"ordinal": 1, "level": "low", "numeric_range": (0, 30)},
    "medium": {"ordinal": 2, "level": "medium", "numeric_range": (30, 70)},


    "abundant": {"ordinal": 3, "level": "high", "numeric_range": (70, 100)},
    "rare": {"ordinal": 1, "level": "low", "numeric_range": (0, 30)},
    "common": {"ordinal": 2, "level": "medium", "numeric_range": (30, 70)},


    "present": {"ordinal": 2, "level": "medium", "numeric_range": (1, 100)},
    "absent": {"ordinal": 0, "level": "none", "numeric_range": (0, 0)},
    "positive": {"ordinal": 2, "level": "medium", "numeric_range": (50, 100)},
    "negative": {"ordinal": 0, "level": "none", "numeric_range": (0, 0)},
}


LENGTH_CONVERSION = {
    'nm': 1.0,
    'nanometer': 1.0,
    'nanometers': 1.0,
    'um': 1000.0,
    'µm': 1000.0,
    'μm': 1000.0,
    'micrometer': 1000.0,
    'mm': 1e6,
    'cm': 1e7,
    'm': 1e9
}


SEQUENCE_LENGTH_CONVERSION = {
    'nt': 1.0,
    'bp': 1.0,
    'kbp': 1000.0,
    'kb': 1000.0,
    'knt': 1000.0,
    'mer': 1.0,
}


MASS_CONVERSION = {
    'ug': 1.0,
    'µg': 1.0,
    'μg': 1.0,
    'microgram': 1.0,
    'mg': 1000.0,
    'milligram': 1000.0,
    'g': 1e6,
    'ng': 0.001
}


CONC_CONVERSION = {
    'ug/ml': 1.0,
    'µg/ml': 1.0,
    'mg/ml': 1000.0,
    'g/l': 1000.0,
    'g/ml': 1e6,
    'ng/ml': 0.001,
    'μm': 1.0,
    'mm': 1.0,
    '%': 10000.0,
    'w/v': 10000.0
}


TEMPERATURE_CONVERSION = {
    'c': 1.0,
    '°c': 1.0,
    'celsius': 1.0,
    'k': lambda x: x - 273.15,
    'kelvin': lambda x: x - 273.15,
    'f': lambda x: (x - 32) * 5/9,
    'fahrenheit': lambda x: (x - 32) * 5/9
}


TIME_CONVERSION = {
    'h': 1.0,
    'hr': 1.0,
    'hrs': 1.0,
    'hour': 1.0,
    'hours': 1.0,
    'min': 1/60,
    'minute': 1/60,
    'minutes': 1/60,
    's': 1/3600,
    'sec': 1/3600,
    'second': 1/3600,
    'seconds': 1/3600,
    'd': 24.0,
    'day': 24.0,
    'days': 24.0,
    'week': 168.0,
    'weeks': 168.0
}


VOLUME_CONVERSION = {
    'ml': 1.0,
    'milliliter': 1.0,
    'milliliters': 1.0,
    'ul': 0.001,
    'µl': 0.001,
    'microliter': 0.001,
    'microliters': 0.001,
    'l': 1000.0,
    'liter': 1000.0,
    'liters': 1000.0,
    'nl': 1e-6,
    'nanoliter': 1e-6
}


MOLARITY_CONVERSION = {
    'mm': 1.0,
    'millimolar': 1.0,
    'μm': 0.001,
    'µm': 0.001,
    'micromolar': 0.001,
    'm': 1000.0,
    'molar': 1000.0,
    'nm': 1e-6,
    'nanomolar': 1e-6
}


PERCENTAGE_CONVERSION = {
    '%': 1.0,
    'percent': 1.0,
    'percentage': 1.0
}


VOLTAGE_CONVERSION = {
    'mv': 1.0,
    'millivolt': 1.0,
    'millivolts': 1.0,
    'v': 1000.0,
    'volt': 1000.0,
    'volts': 1000.0
}


class UnitParser:


    @staticmethod
    def parse_voltage(text: str) -> Optional[PhysicalQuantity]:

        return UnitParser._parse_generic(text, VOLTAGE_CONVERSION, 'Voltage', 'mV')

    @staticmethod
    def parse_length(text: str) -> Optional[PhysicalQuantity]:

        return UnitParser._parse_generic(text, LENGTH_CONVERSION, 'Length', 'nm')

    @staticmethod
    def parse_sequence_length(text: str) -> Optional[PhysicalQuantity]:

        return UnitParser._parse_generic(text, SEQUENCE_LENGTH_CONVERSION, 'SequenceLength', 'nt')

    @staticmethod
    def parse_mass(text: str) -> Optional[PhysicalQuantity]:

        return UnitParser._parse_generic(text, MASS_CONVERSION, 'Mass', 'ug')

    @staticmethod
    def parse_concentration(text: str) -> Optional[PhysicalQuantity]:

        return UnitParser._parse_generic(text, CONC_CONVERSION, 'Concentration', 'ug/ml')

    @staticmethod
    def parse_fold_change(text: str) -> Optional[PhysicalQuantity]:


        clean_txt = text.lower().replace(" ", "")


        match = re.search(r"(\d+(?:\.\d+)?)", clean_txt)
        if match:
            val = float(match.group(1))
            return PhysicalQuantity(
                original_text=text,
                min_val=val, max_val=val, avg_val=val,
                unit='fold', category='Ratio'
            )
        return None

    @staticmethod
    def parse_temperature(text: str) -> Optional[PhysicalQuantity]:

        if not text:
            return None

        clean_txt = text.lower().strip()


        if re.search(r'(\d+)\s*-\s*(\d+)', clean_txt):

            numbers = re.findall(r'(\d+(?:\.\d+)?)', clean_txt)
        else:

            numbers = re.findall(r'(-?\d+(?:\.\d+)?)', clean_txt)

        if not numbers:
            return None

        vals = [float(n) for n in numbers]


        found_unit = None
        converter = None

        for unit, conv in TEMPERATURE_CONVERSION.items():
            if unit in clean_txt:
                found_unit = unit
                converter = conv
                break

        if not found_unit:

            converter = 1.0
            found_unit = 'celsius'


        if callable(converter):

            converted_vals = [converter(v) for v in vals]
        else:

            converted_vals = [v * converter for v in vals]

        min_v = min(converted_vals)
        max_v = max(converted_vals)


        if '<' in clean_txt:
            max_v = min_v
            min_v = -273.15
        elif '>' in clean_txt:
            min_v = max_v
            max_v = float('inf')

        return PhysicalQuantity(
            original_text=text,
            min_val=min_v,
            max_val=max_v,
            avg_val=(min_v + max_v) / 2 if max_v != float('inf') else min_v,
            unit='°C',
            category='Temperature'
        )

    @staticmethod
    def parse_time(text: str) -> Optional[PhysicalQuantity]:

        return UnitParser._parse_generic(text, TIME_CONVERSION, 'Time', 'h')

    @staticmethod
    def parse_volume(text: str) -> Optional[PhysicalQuantity]:

        return UnitParser._parse_generic(text, VOLUME_CONVERSION, 'Volume', 'ml')

    @staticmethod
    def parse_pH(text: str) -> Optional[PhysicalQuantity]:

        if not text:
            return None

        clean_txt = text.lower().replace("ph", "").strip()


        numbers = re.findall(r"(\d+(?:\.\d+)?)", clean_txt)
        if not numbers:
            return None

        vals = [float(n) for n in numbers]


        valid_vals = [v for v in vals if 0 <= v <= 14]
        if not valid_vals:
            logger.warning('Runtime diagnostic.')
            return None

        min_v = min(valid_vals)
        max_v = max(valid_vals)


        if '<' in clean_txt:
            max_v = min_v
            min_v = 0.0
        elif '>' in clean_txt:
            min_v = max_v
            max_v = 14.0

        return PhysicalQuantity(
            original_text=text,
            min_val=min_v,
            max_val=max_v,
            avg_val=(min_v + max_v) / 2,
            unit='pH',
            category='pH'
        )

    @staticmethod
    def parse_molarity(text: str) -> Optional[PhysicalQuantity]:

        return UnitParser._parse_generic(text, MOLARITY_CONVERSION, 'Molarity', 'mM')

    @staticmethod
    def parse_percentage(text: str) -> Optional[PhysicalQuantity]:

        return UnitParser._parse_generic(text, PERCENTAGE_CONVERSION, 'Percentage', '%')

    @staticmethod
    def parse(text: str) -> Optional[PhysicalQuantity]:

        if not text:
            return None
        import re
        clean = text.strip().lower()


        letter_seqs = re.findall(r'[a-z]+', clean)
        known_units = {'nm', 'um', 'µm', 'mm', 'cm', 'm',
                       'ng', 'ug', 'µg', 'mg', 'g', 'kg',
                       'min', 'hour', 'h', 'hr', 'day', 'days', 'd', 'week', 'month', 'year', 'sec',
                       'fold', 'times', 'x', 'ml', 'l', 'ul', 'µl', 'ph',
                       'celsius', 'fahrenheit', 'kelvin', 'percent',
                       'nanometer', 'micrometer', 'millimeter', 'centimeter', 'meter',
                       'microgram', 'milligram', 'nanogram', 'gram', 'liter', 'molar',
}
        for seq in letter_seqs:
            if seq not in known_units:
                return None


        if '%' in clean:
            r = UnitParser.parse_percentage(text)
            if r: return r

        if 'fold' in clean or clean.endswith('x'):
            r = UnitParser.parse_fold_change(text)
            if r: return r
        def _has_unit(text, units):

            for u in units:
                pattern = r'(?<![a-zA-Z])' + re.escape(u) + r'(?![a-zA-Z])'
                if re.search(pattern, text):
                    return True
            return False


        length_units = {'nm', 'µm', 'um', 'mm', 'cm', 'm'}
        if _has_unit(clean, length_units):
            r = UnitParser.parse_length(text)
            if r: return r

        mass_units = {'ng', 'µg', 'ug', 'mg', 'g', 'kg'}
        if _has_unit(clean, mass_units):
            r = UnitParser.parse_mass(text)
            if r: return r

        time_units = {'min', 'hour', 'h', 'day', 'd', 'week', 'month', 'year'}
        if _has_unit(clean, time_units):
            r = UnitParser.parse_time(text)
            if r: return r

        conc_units = {'mg/ml', 'ug/ml', 'ng/ml', 'molar', 'm'}
        if _has_unit(clean, conc_units):
            r = UnitParser.parse_concentration(text)
            if r: return r

        vol_units = {'µl', 'ul', 'ml', 'l'}
        if _has_unit(clean, vol_units):
            r = UnitParser.parse_volume(text)
            if r: return r

        molar_units = {'mm', 'µm', 'um', 'nm', 'pm', 'm'}
        if _has_unit(clean, molar_units):
            r = UnitParser.parse_molarity(text)
            if r: return r

        if re.search(r'(?<![a-zA-Z])ph(?![a-zA-Z])', clean):
            r = UnitParser.parse_pH(text)
            if r: return r

        temp_units = {'°c', '°f', 'k', 'celsius', 'fahrenheit'}
        if _has_unit(clean, temp_units):
            r = UnitParser.parse_temperature(text)
            if r: return r

        if re.search(r'\d', text):
            nums = re.findall(r'(-?\d+(?:\.\d+)?)', text)
            if nums:
                vals = [float(n) for n in nums]
                return PhysicalQuantity(
                    original_text=text,
                    min_val=min(vals), max_val=max(vals),
                    avg_val=sum(vals)/len(vals),
                    unit='dimensionless', category='Dimensionless'
                )
        return None

    @staticmethod
    def parse_semi_quantitative(text: str) -> Optional[PhysicalQuantity]:

        if not text:
            return None

        clean_txt = text.lower().strip()

        if clean_txt not in SEMI_QUANTITATIVE_MAP:
            return None

        mapping = SEMI_QUANTITATIVE_MAP[clean_txt]
        min_v, max_v = mapping["numeric_range"]

        return PhysicalQuantity(
            original_text=text,
            min_val=float(min_v),
            max_val=float(max_v),
            avg_val=(min_v + max_v) / 2,
            unit='ordinal',
            category=f'SemiQuantitative[{mapping["level"]}]'
        )

    @staticmethod
    def _preprocess_range(text: str) -> str:

        import re

        text = re.sub(r'(-?\d+(?:\.\d+)?)\s*to\s*(-?\d+(?:\.\d+)?)', r'\1-\2', text, flags=re.IGNORECASE)

        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        return text

    @staticmethod
    def parse_auto(text: str) -> Optional[PhysicalQuantity]:

        if not text:
            return None

        import re
        text = UnitParser._preprocess_range(text)
        clean_txt = text.lower().strip()


        semi_quant = UnitParser.parse_semi_quantitative(text)
        if semi_quant:
            return semi_quant


        if 'ph' in clean_txt and re.search(r'\bph\b', clean_txt):
            return UnitParser.parse_pH(text)


        if any(u in clean_txt for u in ['°c', 'celsius', 'kelvin', 'fahrenheit', '°f']):
            return UnitParser.parse_temperature(text)


        if any(u in clean_txt for u in ['fold', 'times']) or re.search(r'\d+\s*[x×]\b', clean_txt):
            return UnitParser.parse_fold_change(text)


        if any(u in clean_txt for u in ['hour', 'hr', 'min', 'day', 'week', 'sec']):
            return UnitParser.parse_time(text)


        if re.search(r'(-?\d+(?:\.\d+)?(?:\s*-\s*-?\d+(?:\.\d+)?)?)\s*(mv|millivolt|v|volt)\b', clean_txt):
            return UnitParser.parse_voltage(text)


        if re.search(r'(\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\s*(kbp|knt|kb|bp|nt)\b', clean_txt) or\
           re.search(r'\d+-mer\b', clean_txt):
            return UnitParser.parse_sequence_length(text)


        if re.search(r'(\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\s*(nm|nanometer|um|µm|μm|micrometer|mm|millimeter|cm|centimeter|m|meter)\b', clean_txt) and '/' not in clean_txt:
            return UnitParser.parse_length(text)


        if re.search(r'(\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\s*(ug|µg|μg|microgram|mg|milligram|ng|nanogram|g|gram)\b', clean_txt) and '/' not in clean_txt:
            return UnitParser.parse_mass(text)


        if any(u in clean_txt for u in ['ml', 'µl', 'ul', 'liter']):
            return UnitParser.parse_volume(text)


        if '/' in clean_txt:
            if any(u in clean_txt for u in ['ug', 'µg', 'mg', 'g']):
                return UnitParser.parse_concentration(text)
            elif any(u in clean_txt for u in ['mm', 'µm', 'μm', 'molar']):
                return UnitParser.parse_molarity(text)


        if '%' in clean_txt or 'percent' in clean_txt:
            return UnitParser.parse_percentage(text)


        logger.debug('Runtime diagnostic.')
        return None

    @staticmethod
    def _parse_generic(text: str, conversion_map: dict, category: str, base_unit_name: str) -> Optional[PhysicalQuantity]:
        if not text:
            return None

        text = UnitParser._preprocess_range(text)
        clean_txt = text.lower().strip()


        found_unit = None
        multiplier = 1.0


        sorted_units = sorted(conversion_map.keys(), key=len, reverse=True)
        for unit in sorted_units:
            if unit in clean_txt:
                found_unit = unit
                multiplier = conversion_map[unit]
                break

        if not found_unit:

            mer_m = re.match(r'^(\d+(?:\.\d+)?)-mer\b', clean_txt)
            if mer_m and category == 'SequenceLength':
                val = float(mer_m.group(1))
                return PhysicalQuantity(
                    original_text=text,
                    min_val=val, max_val=val, avg_val=val,
                    unit='nt', category='SequenceLength'
                )
            logger.warning('Runtime diagnostic.')
            multiplier = 1.0
            found_unit = base_unit_name


        if re.search(r'(\d+)\s*-\s*(\d+)', clean_txt):

            matches = re.findall(r'(\d+(?:\.\d+)?)', clean_txt)
            numbers = matches
        else:

            numbers = re.findall(r'(-?\d+(?:\.\d+)?)', clean_txt)

        if not numbers:
            return None

        vals = [float(n) for n in numbers]


        if callable(multiplier):
            norm_vals = [multiplier(v) for v in vals]
        else:
            norm_vals = [v * multiplier for v in vals]

        min_v = min(norm_vals)
        max_v = max(norm_vals)


        if '<' in clean_txt:
            max_v = min_v
            min_v = 0.0 if category != 'Temperature' else -273.15
        elif '>' in clean_txt:
            min_v = max_v
            max_v = float('inf')

        return PhysicalQuantity(
            original_text=text,
            min_val=min_v,
            max_val=max_v,
            avg_val=(min_v + max_v) / 2 if max_v != float('inf') else min_v,
            unit=base_unit_name,
            category=category
        )


def check_value_match(design_val: Optional[str], target_condition: str, parser_func) -> bool:

    if not design_val or not target_condition:
        return False


    q_design = parser_func(design_val)
    q_target = parser_func(target_condition)

    if not q_design or not q_target:
        return False


    overlap_min = max(q_design.min_val, q_target.min_val)
    overlap_max = min(q_design.max_val, q_target.max_val)

    if overlap_min <= overlap_max:
        return True

    return False


if __name__ == "__main__":
    print("="*80)
    print('Runtime diagnostic.')
    print("="*80)


    test_cases = {
        'Runtime diagnostic.': [
            ("100 nm", UnitParser.parse_length),
            ("10-20 nm", UnitParser.parse_length),
            ("0.5 um", UnitParser.parse_length),
            ("< 50 nm", UnitParser.parse_length),
        ],
        'Runtime diagnostic.': [
            ("5 mg", UnitParser.parse_mass),
            ("100 ug", UnitParser.parse_mass),
            ("1-2 g", UnitParser.parse_mass),
        ],
        'Runtime diagnostic.': [
            ("8-fold", UnitParser.parse_fold_change),
            ("12x", UnitParser.parse_fold_change),
            ("3 times", UnitParser.parse_fold_change),
        ],
        'Runtime diagnostic.': [
            ("37°C", UnitParser.parse_temperature),
            ("310K", UnitParser.parse_temperature),
            ("98.6F", UnitParser.parse_temperature),
            ("4 celsius", UnitParser.parse_temperature),
        ],
        'Runtime diagnostic.': [
            ("24 hours", UnitParser.parse_time),
            ("7 days", UnitParser.parse_time),
            ("30 min", UnitParser.parse_time),
            ("1 week", UnitParser.parse_time),
        ],
        'Runtime diagnostic.': [
            ("100 ml", UnitParser.parse_volume),
            ("1 L", UnitParser.parse_volume),
            ("50 µl", UnitParser.parse_volume),
        ],
        'Runtime diagnostic.': [
            ("pH 7.4", UnitParser.parse_pH),
            ("6.5-7.5", UnitParser.parse_pH),
            ("< 8", UnitParser.parse_pH),
        ],
        'Runtime diagnostic.': [
            ("10 mM", UnitParser.parse_molarity),
            ("1 µM", UnitParser.parse_molarity),
            ("0.5 M", UnitParser.parse_molarity),
        ],
        'Runtime diagnostic.': [
            ("50%", UnitParser.parse_percentage),
            ("10-20%", UnitParser.parse_percentage),
        ],
    }


    for category, tests in test_cases.items():
        print('Runtime diagnostic.')
        for text, parser_func in tests:
            result = parser_func(text)
            status = "✅" if result else "❌"
            print(f"  {status} '{text}' -> {result}")


    print('Runtime diagnostic.')
    auto_tests = [
        "100 nm", "37°C", "pH 7.4", "24 hours",
        "8-fold", "50%", "10 mM", "100 ml"
    ]
    for text in auto_tests:
        result = UnitParser.parse_auto(text)
        print(f"  '{text}' -> {result}")


    print('Runtime diagnostic.')
    comparison_tests = [
        ("150 nm", "< 200 nm", UnitParser.parse_length, True),
        ("500 nm", "100-200 nm", UnitParser.parse_length, False),
        ("0.15 um", "100-200 nm", UnitParser.parse_length, True),
        ("37°C", "30-40°C", UnitParser.parse_temperature, True),
        ("pH 7.4", "6.5-7.5", UnitParser.parse_pH, True),
    ]

    for design_val, target, parser, expected in comparison_tests:
        result = check_value_match(design_val, target, parser)
        status = "✅" if result == expected else "❌"
        print('Runtime diagnostic.')

    print("\n" + "="*80)
