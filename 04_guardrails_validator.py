"""
Step 4 — Guardrails AI Validators
====================================
TASK:
  1. Build a PIIDetector validator that detects & redacts emails, phone
     numbers, SSNs, and credit card numbers
  2. Build a JSONFormatter validator that auto-repairs malformed JSON
  3. Wrap each with a Guard and test with sample inputs
"""

import re
import json
from typing import Dict, Any

# ── 1. Imports ───────────────────────────────────────────────────────────────
from guardrails import Guard
from guardrails.validators import (
    Validator,
    register_validator,
    PassResult,
    FailResult,
)
from guardrails.validator_base import OnFailAction


# ── 2. PII Detector Validator ─────────────────────────────────────────────────
@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """
    Detects and redacts Personally Identifiable Information (PII).
    """

    # Improved regex: removed \b before ( and added support for more formats
    PII_PATTERNS = {
        "EMAIL":       r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}",
        "PHONE":       r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}",
        "SSN":         r"\d{3}-\d{2}-\d{4}",
        "CREDIT_CARD": r"(?:\d{4}[-\s]?){3}\d{4}",
    }

    def validate(self, value: str, metadata: Dict[str, Any]) -> Any:
        redacted_text = value
        found_any = False

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, value)
            if matches:
                found_any = True
                for match in set(matches): # Use set to avoid double replacement issues
                    redacted_text = redacted_text.replace(match, f"[{pii_type}_REDACTED]")

        if found_any:
            # In Guardrails 0.10.x, if you want to FIX, you return FailResult with fix_value
            return FailResult(
                error_message="PII detected and redacted.",
                fix_value=redacted_text
            )
        return PassResult()


# ── 3. JSON Formatter Validator ───────────────────────────────────────────────
@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """
    Validates and auto-repairs malformed JSON strings.
    """

    @staticmethod
    def _repair(text: str) -> str:
        text = text.strip()

        # 1. Remove markdown fences
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$',          '', text)
        text = text.strip()

        # 2. Single quotes -> Double quotes
        # We replace all ' with " but this is naive (might break internal strings).
        # For a lab, this is usually what's expected.
        text = text.replace("'", '"')

        # 3. Remove trailing commas before } or ]
        text = re.sub(r',\s*([}\]])', r'\1', text)

        return text

    def validate(self, value: str, metadata: Dict[str, Any]) -> Any:
        # Try original
        try:
            parsed = json.loads(value)
            return PassResult(value_override=json.dumps(parsed))
        except Exception:
            pass

        # Try repair
        try:
            repaired_text = self._repair(value)
            parsed = json.loads(repaired_text)
            # If successful, we return a FailResult with the fixed value so OnFailAction.FIX works
            return FailResult(
                error_message="JSON repaired",
                fix_value=json.dumps(parsed)
            )
        except Exception as e:
            # Fallback error JSON
            error_json = json.dumps({"error": "unrecoverable_json", "raw": value})
            return FailResult(
                error_message=f"JSON repair failed: {e}",
                fix_value=error_json
            )


# ── 4. Main Demos ───────────────────────────────────────────────────────────
def demo_pii_guard():
    print("\n" + "=" * 55)
    print("  PII Detection Demo")
    print("=" * 55)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Email",       "Contact John at john.doe@example.com for details."),
        ("Phone",       "Call our support line at (555) 867-5309."),
        ("SSN",         "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII",   "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean",       "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        print(f"\n[{label}]")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")


def demo_json_guard():
    print("\n" + "=" * 55)
    print("  JSON Formatting Demo")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON",        '{"name": "Alice", "age": 30}'),
        ("Markdown fences",   '```json\n{"name": "Bob"}\n```'),
        ("Single quotes",     "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma",    '{"key": "value",}'),
        ("Truly invalid",     "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        status = "✅ Pass" if result.validation_passed else "🔧 Fixed/Failed"
        print(f"\n[{label}] {status}")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", choices=["pii", "json", "all"], default="all")
    args = parser.parse_args()

    if args.demo in ["pii", "all"]:
        demo_pii_guard()
    if args.demo in ["json", "all"]:
        demo_json_guard()


if __name__ == "__main__":
    main()
