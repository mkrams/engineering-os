"""
SysMLv2 Textual Notation Validator
-----------------------------------
A grammar-aware validator for SysMLv2 textual notation.
Checks syntax, balanced braces/brackets, keyword usage,
reference resolution, and constraint consistency.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ValidationError:
    line: int
    col: int
    message: str
    severity: str = "error"  # "error" | "warning"

    def to_dict(self):
        return {
            "line": self.line,
            "col": self.col,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass
class Symbol:
    name: str
    kind: str  # "part_def", "port_def", "attribute", "requirement_def", etc.
    line: int
    parent: Optional[str] = None
    type_ref: Optional[str] = None
    children: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------

DEFINITION_KEYWORDS = {
    "part", "port", "attribute", "requirement", "constraint",
    "action", "state", "enum", "item", "connection",
    "interface", "allocation", "flow", "use", "case",
    "analysis", "verification", "concern", "viewpoint", "view",
    "rendering", "metadata", "package", "library",
}

DEFINITION_SUFFIXES = {"def"}

MODIFIER_KEYWORDS = {
    "abstract", "private", "public", "protected", "readonly",
    "derived", "ordered", "nonunique", "in", "out", "inout",
}

RELATIONSHIP_KEYWORDS = {
    "specializes", "subsets", "redefines", "references",
    "chains", "inverses", "conjugates",
}

BEHAVIOUR_KEYWORDS = {
    "entry", "do", "exit", "then", "first", "parallel",
    "perform", "accept", "send", "assign", "if", "else",
    "while", "for", "loop", "merge", "decide",
}

CONSTRAINT_KEYWORDS = {"require", "assume", "assert"}

IMPORT_KEYWORDS = {"import", "alias"}

OPERATORS = {":>", ":>>", "=", "==", "!=", "<=", ">=", "<", ">", "&&", "||"}


class SysMLv2Validator:
    """Validate SysMLv2 textual notation source code."""

    def __init__(self):
        self.errors: list[ValidationError] = []
        self.symbols: dict[str, Symbol] = {}
        self.lines: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, source: str) -> list[dict]:
        """Run all validation passes and return error dicts."""
        self.errors = []
        self.symbols = {}
        self.lines = source.splitlines()

        self._check_balanced(source)
        self._check_keywords(source)
        self._collect_symbols(source)
        self._check_type_references(source)
        self._check_multiplicity(source)
        self._check_constraint_blocks(source)
        self._check_semicolons(source)

        # Sort by line number
        self.errors.sort(key=lambda e: (e.line, e.col))
        return [e.to_dict() for e in self.errors]

    # ------------------------------------------------------------------
    # Pass 1: Balanced delimiters
    # ------------------------------------------------------------------

    def _check_balanced(self, source: str):
        stack = []
        pairs = {"{": "}", "(": ")", "[": "]"}
        closing = set(pairs.values())
        in_string = False
        in_line_comment = False
        in_block_comment = False
        prev_char = ""

        for lineno, line in enumerate(self.lines, 1):
            in_line_comment = False
            for col, ch in enumerate(line, 1):
                # Track comments
                if not in_string and not in_block_comment and prev_char == "/" and ch == "/":
                    in_line_comment = True
                if not in_string and not in_line_comment and prev_char == "/" and ch == "*":
                    in_block_comment = True
                if in_block_comment and prev_char == "*" and ch == "/":
                    in_block_comment = False
                    prev_char = ch
                    continue
                if in_line_comment or in_block_comment:
                    prev_char = ch
                    continue

                # Track strings
                if ch == '"' and prev_char != "\\":
                    in_string = not in_string
                if in_string:
                    prev_char = ch
                    continue

                if ch in pairs:
                    stack.append((ch, lineno, col))
                elif ch in closing:
                    if not stack:
                        self.errors.append(ValidationError(lineno, col, f"Unmatched closing '{ch}'"))
                    else:
                        open_ch, open_line, open_col = stack.pop()
                        expected = pairs[open_ch]
                        if ch != expected:
                            self.errors.append(
                                ValidationError(
                                    lineno, col,
                                    f"Mismatched delimiter: expected '{expected}' (opened at line {open_line}:{open_col}) but found '{ch}'"
                                )
                            )
                prev_char = ch

        for open_ch, open_line, open_col in stack:
            expected = pairs[open_ch]
            self.errors.append(
                ValidationError(open_line, open_col, f"Unclosed '{open_ch}' — expected '{expected}'")
            )

    # ------------------------------------------------------------------
    # Pass 2: Keyword validation
    # ------------------------------------------------------------------

    def _check_keywords(self, source: str):
        # Check for common misspellings / SysMLv1 holdovers
        v1_to_v2 = {
            "block": "part def",
            "value type": "attribute def",
            "flowPort": "port",
            "SysML::": "SysML v2 uses package imports, not SysML:: prefix",
        }
        for lineno, line in enumerate(self.lines, 1):
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("/*"):
                continue
            for v1_kw, suggestion in v1_to_v2.items():
                if v1_kw in line:
                    col = line.index(v1_kw) + 1
                    self.errors.append(
                        ValidationError(
                            lineno, col,
                            f"'{v1_kw}' is SysML v1 syntax. In SysML v2, use '{suggestion}' instead.",
                            severity="warning",
                        )
                    )

    # ------------------------------------------------------------------
    # Pass 3: Symbol collection
    # ------------------------------------------------------------------

    def _collect_symbols(self, source: str):
        # Match definitions: <keyword> def <Name> { ... }
        def_pattern = re.compile(
            r"(part|port|requirement|constraint|action|state|enum|item|interface|connection|attribute)\s+def\s+(\w[\w']*)"
        )
        # Match package declarations
        pkg_pattern = re.compile(r"package\s+(\w[\w':]*)")
        # Match usages: <keyword> <name> : <TypeRef>
        usage_pattern = re.compile(
            r"(part|port|attribute|requirement|constraint|action|state|item)\s+(\w[\w']*)\s*:\s*(\w[\w':]*)"
        )

        for lineno, line in enumerate(self.lines, 1):
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("/*"):
                continue

            for m in def_pattern.finditer(line):
                kind = m.group(1) + "_def"
                name = m.group(2)
                self.symbols[name] = Symbol(name=name, kind=kind, line=lineno)

            for m in pkg_pattern.finditer(line):
                name = m.group(1)
                self.symbols[name] = Symbol(name=name, kind="package", line=lineno)

            for m in usage_pattern.finditer(line):
                name = m.group(2)
                type_ref = m.group(3)
                self.symbols[name] = Symbol(
                    name=name, kind=m.group(1), line=lineno, type_ref=type_ref
                )

    # ------------------------------------------------------------------
    # Pass 4: Type reference checking
    # ------------------------------------------------------------------

    def _check_type_references(self, source: str):
        # Built-in / primitive types that don't need to be defined
        builtins = {
            "Real", "Integer", "Boolean", "String", "Natural",
            "Positive", "ScalarValues", "NumericalValue",
            "MassValue", "LengthValue", "TimeValue", "ForceValue",
            "VoltageValue", "CurrentValue", "PowerValue", "TemperatureValue",
            "PressureValue", "FrequencyValue", "SpeedValue", "AccelerationValue",
            "AngularVelocityValue", "TorqueValue", "EnergyValue",
        }
        # Collect all type references from usages
        type_ref_pattern = re.compile(
            r"(?:part|port|attribute|item)\s+\w[\w']*\s*:\s*(\w[\w']*)"
        )
        specialization_pattern = re.compile(r":>\s*(\w[\w']*)")

        for lineno, line in enumerate(self.lines, 1):
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("import"):
                continue

            for m in type_ref_pattern.finditer(line):
                ref = m.group(1)
                if ref not in self.symbols and ref not in builtins:
                    self.errors.append(
                        ValidationError(
                            lineno, line.index(ref) + 1,
                            f"Unresolved type reference '{ref}'. Define it with '{ref}' def or add an import.",
                            severity="warning",
                        )
                    )

            for m in specialization_pattern.finditer(line):
                ref = m.group(1)
                if ref not in self.symbols and ref not in builtins:
                    self.errors.append(
                        ValidationError(
                            lineno, line.index(ref) + 1,
                            f"Unresolved specialization target '{ref}'.",
                            severity="warning",
                        )
                    )

    # ------------------------------------------------------------------
    # Pass 5: Multiplicity syntax
    # ------------------------------------------------------------------

    def _check_multiplicity(self, source: str):
        mult_pattern = re.compile(r"\[([^\]]*)\]")
        valid_mult = re.compile(r"^(\d+|\*)(\.\.(\d+|\*))?$")

        for lineno, line in enumerate(self.lines, 1):
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("/*"):
                continue
            for m in mult_pattern.finditer(line):
                content = m.group(1).strip()
                # Skip if it looks like a unit (e.g., [kg])
                if content.isalpha():
                    continue
                if content and not valid_mult.match(content):
                    self.errors.append(
                        ValidationError(
                            lineno, m.start() + 1,
                            f"Invalid multiplicity '[{content}]'. Use forms like [1], [*], [0..1], [1..*].",
                        )
                    )

    # ------------------------------------------------------------------
    # Pass 6: Constraint block validation
    # ------------------------------------------------------------------

    def _check_constraint_blocks(self, source: str):
        constraint_pattern = re.compile(r"(require|assume|assert)\s+constraint\b")
        for lineno, line in enumerate(self.lines, 1):
            m = constraint_pattern.search(line)
            if m:
                # Check that there's a { somewhere after
                rest = line[m.end():]
                if "{" not in rest:
                    # Look at next non-empty line
                    found_brace = False
                    for next_line in self.lines[lineno:]:
                        if next_line.strip():
                            if "{" in next_line:
                                found_brace = True
                            break
                    if not found_brace:
                        self.errors.append(
                            ValidationError(
                                lineno, m.start() + 1,
                                f"'{m.group(1)} constraint' should be followed by a block '{{ expression }}'.",
                                severity="warning",
                            )
                        )

    # ------------------------------------------------------------------
    # Pass 7: Statement termination
    # ------------------------------------------------------------------

    def _check_semicolons(self, source: str):
        """Check for statements that look like they need semicolons."""
        needs_semi = re.compile(
            r"^\s*(import|alias)\s+.*[^;{}\s]\s*$"
        )
        for lineno, line in enumerate(self.lines, 1):
            if needs_semi.match(line):
                self.errors.append(
                    ValidationError(
                        lineno, len(line),
                        "Import/alias statements should end with ';'.",
                        severity="warning",
                    )
                )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def validate_sysmlv2(source: str) -> dict:
    """Validate SysMLv2 source and return a result dict."""
    validator = SysMLv2Validator()
    errors = validator.validate(source)
    error_count = sum(1 for e in errors if e["severity"] == "error")
    warning_count = sum(1 for e in errors if e["severity"] == "warning")
    return {
        "valid": error_count == 0,
        "error_count": error_count,
        "warning_count": warning_count,
        "diagnostics": errors,
        "symbols": {
            name: {"kind": sym.kind, "line": sym.line, "type_ref": sym.type_ref}
            for name, sym in validator.symbols.items()
        },
    }


if __name__ == "__main__":
    test_source = """
package Drone {
    import ISQ::*;

    part def Frame {
        attribute mass : Real;
        attribute armLength : Real;
    }

    part def Motor :> Actuator {
        attribute maxThrust : ForceValue;
        attribute kv : Integer;
        port powerIn : PowerPort;
    }

    part def Drone {
        part frame : Frame;
        part motors : Motor [4];
        attribute totalMass : Real;

        require constraint { totalMass <= 2.0 }
    }

    requirement def FlightRequirement {
        subject drone : Drone;
        require constraint {
            drone.totalMass <= 2.0
        }
    }
}
"""
    result = validate_sysmlv2(test_source)
    import json
    print(json.dumps(result, indent=2))
