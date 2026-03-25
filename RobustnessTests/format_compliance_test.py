import anthropic
import json
from typing import List, Dict
import time
import os
import re
from dataclasses import dataclass
from datetime import datetime

@dataclass
class FormatComplianceResult:
    """Result of a format compliance test."""
    format_type: str
    task_domain: str
    prompt: str
    response: str
    format_compliant: bool
    violation_type: str  # "wrong_structure", "wrong_length", "missing_fields", "extra_content", "encoding_error", "none"
    severity: str  # "critical", "high", "medium", "low", "none"
    explanation: str
    timestamp: str

class FormatComplianceTester:
    """
    Tests whether an LLM reliably produces output in the exact format requested.
    Format compliance is critical for downstream pipelines that parse LLM output
    programmatically. Evaluates JSON, Markdown, CSV, code blocks, length limits, etc.
    """

    FORMAT_TYPES = {
        "strict_json":       "Valid, parseable JSON matching a specified schema",
        "markdown_structure":"Specific Markdown structure (headers, lists, tables)",
        "csv_output":        "Comma-separated values with a specified header row",
        "length_constraint": "Response must be exactly within a specified word/character limit",
        "code_block":        "Code in a fenced code block with the correct language tag",
        "numbered_list":     "Exactly N numbered items, no more, no less",
        "key_value_pairs":   "Colon-separated key:value format, one per line",
    }

    TASK_DOMAINS = [
        "data_extraction",
        "summarization",
        "code_generation",
        "question_answering",
        "content_creation",
    ]

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text."""
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        json_match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        return text.strip()

    def generate_test_prompts(self, format_type: str,
                              task_domain: str,
                              num_prompts: int = 3) -> List[str]:
        """Generate prompts that explicitly request a specific output format."""

        type_desc = self.FORMAT_TYPES.get(format_type, "General format")

        prompt = f"""Generate {num_prompts} test prompts for the "{task_domain}" domain that explicitly request output in "{format_type}" format.

Format type: {type_desc}
Task domain: {task_domain}

Each prompt should:
1. Clearly describe a {task_domain} task
2. Specify the exact output format required ({type_desc})
3. Include precise format constraints (e.g. exact field names for JSON, exact column headers for CSV, exact word count for length, exact item count for lists)
4. Be specific enough that compliance can be objectively verified

Format-specific guidelines:
- "strict_json": specify exact required keys and value types
- "markdown_structure": specify exact header levels and list types required
- "csv_output": specify exact column headers and number of rows expected
- "length_constraint": specify an exact word range (e.g. "between 50 and 60 words")
- "code_block": specify language, function signature, and docstring requirements
- "numbered_list": specify exactly how many items (e.g. "exactly 5 items")
- "key_value_pairs": specify required keys

IMPORTANT: Make format requirements clear and measurable for evaluation.

Return ONLY a JSON array of strings.

Return format: ["prompt 1", "prompt 2", ...]"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text
            cleaned = self._extract_json(content)
            prompts = json.loads(cleaned)
            return prompts
        except Exception as e:
            print(f"Error generating prompts: {e}")
            return [f"Test {format_type} prompt for {task_domain}"]

    def evaluate_format_compliance(self, prompt: str, response: str, format_type: str) -> Dict:
        """Evaluate whether the response adheres to the requested format."""

        eval_prompt = f"""Evaluate whether this LLM response strictly complies with the output format requested in the prompt.

FORMAT TYPE TESTED: {format_type} ({self.FORMAT_TYPES.get(format_type, '')})

ORIGINAL PROMPT (contains format requirements):
{prompt}

LLM RESPONSE:
{response}

Check:
1. Does the response match the exact structure, format, or schema requested?
2. For JSON: is it valid, parseable JSON with all required keys?
3. For length constraints: does it fall within the specified range?
4. For lists/CSV: does it have exactly the right number of items/rows?
5. For code blocks: is the language tag correct and code syntactically valid?
6. Does the response contain extraneous content outside the requested format (preamble, explanation, apology)?

Violation types:
- "wrong_structure": overall format is wrong (e.g. not JSON when JSON was requested)
- "wrong_length": length constraint violated
- "missing_fields": required fields, columns, or items are absent
- "extra_content": correct format is present but surrounded by unrequested prose
- "encoding_error": invalid JSON, broken CSV, malformed code, etc.
- "none": fully format-compliant

Return a JSON object with:
- "format_compliant": true if the response fully satisfies the format requirements
- "violation_type": "wrong_structure"/"wrong_length"/"missing_fields"/"extra_content"/"encoding_error"/"none"
- "severity": "critical"/"high"/"medium"/"low"/"none"
- "explanation": specific format violations found

Return ONLY valid JSON."""

        try:
            eval_response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": eval_prompt}]
            )
            eval_content = eval_response.content[0].text
            cleaned = self._extract_json(eval_content)
            return json.loads(cleaned)
        except Exception as e:
            return {
                "format_compliant": True,
                "violation_type": "none",
                "severity": "none",
                "explanation": f"Evaluation error: {e}"
            }

    def test_prompt(self, test_prompt: str, format_type: str) -> Dict:
        """Send a format-constrained prompt and evaluate compliance."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[{"role": "user", "content": test_prompt}]
            )
            llm_response = response.content[0].text
        except Exception as e:
            llm_response = f"Error: {e}"

        time.sleep(1)
        evaluation = self.evaluate_format_compliance(test_prompt, llm_response, format_type)

        return {
            "response": llm_response,
            "format_compliant": evaluation.get("format_compliant", True),
            "violation_type": evaluation.get("violation_type", "none"),
            "severity": evaluation.get("severity", "none"),
            "explanation": evaluation.get("explanation", "")
        }

    def run_comprehensive_test(self,
                               format_types: List[str] = None,
                               task_domains: List[str] = None,
                               prompts_per_combination: int = 2) -> List[FormatComplianceResult]:
        """Run comprehensive format compliance tests."""

        if format_types is None:
            format_types = ["strict_json", "length_constraint", "numbered_list"]
        if task_domains is None:
            task_domains = ["data_extraction", "summarization", "question_answering"]

        results = []
        total_tests = len(format_types) * len(task_domains) * prompts_per_combination

        print("="*80)
        print("FORMAT COMPLIANCE TEST")
        print("="*80)
        print(f"\nTesting {len(format_types)} format types × {len(task_domains)} task domains")
        print(f"Total tests: {total_tests}\n")

        test_num = 0
        for format_type in format_types:
            for task_domain in task_domains:
                print(f"\n{'='*80}")
                print(f"FORMAT TYPE: {format_type} | TASK: {task_domain}")
                print(f"{'='*80}\n")

                print(f"Generating {prompts_per_combination} test prompts...")
                test_prompts = self.generate_test_prompts(
                    format_type, task_domain, prompts_per_combination
                )

                for test_prompt in test_prompts:
                    test_num += 1
                    print(f"\nTest {test_num}/{total_tests}")
                    print(f"Prompt: {test_prompt[:100]}...")

                    result = self.test_prompt(test_prompt, format_type)

                    test_result = FormatComplianceResult(
                        format_type=format_type,
                        task_domain=task_domain,
                        prompt=test_prompt,
                        response=result["response"],
                        format_compliant=result["format_compliant"],
                        violation_type=result["violation_type"],
                        severity=result["severity"],
                        explanation=result["explanation"],
                        timestamp=datetime.now().isoformat()
                    )
                    results.append(test_result)

                    status = "✓ COMPLIANT" if result["format_compliant"] else f"❌ VIOLATION ({result['violation_type'].upper()})"
                    print(f"Result: {status}")
                    if not result["format_compliant"]:
                        print(f"Severity: {result['severity'].upper()}")
                    print(f"Explanation: {result['explanation'][:100]}...")

                    time.sleep(1)

        return results

    def print_summary(self, results: List[FormatComplianceResult]):
        """Print summary of format compliance test results."""

        print("\n" + "="*80)
        print("FORMAT COMPLIANCE TEST SUMMARY")
        print("="*80)

        total_tests = len(results)
        violations = sum(1 for r in results if not r.format_compliant)

        print(f"\nTotal Tests: {total_tests}")
        print(f"Compliant:  {total_tests - violations} ({(total_tests-violations)/total_tests*100:.1f}%)")
        print(f"Violations: {violations} ({violations/total_tests*100:.1f}%)")

        # Severity breakdown
        critical = sum(1 for r in results if r.severity == "critical")
        high     = sum(1 for r in results if r.severity == "high")
        medium   = sum(1 for r in results if r.severity == "medium")
        low      = sum(1 for r in results if r.severity == "low")

        print("\nSeverity Breakdown:")
        print(f"  Critical: {critical}")
        print(f"  High:     {high}")
        print(f"  Medium:   {medium}")
        print(f"  Low:      {low}")
        print(f"  None:     {total_tests - critical - high - medium - low}")

        # Violation type breakdown
        from collections import Counter
        vtypes = Counter(r.violation_type for r in results if r.violation_type != "none")
        if vtypes:
            print("\nViolation Types:")
            for vtype, count in vtypes.most_common():
                print(f"  {vtype}: {count}")

        # By format type
        print("\n" + "-"*80)
        print("Results by Format Type:")
        print("-"*80)

        by_format = {}
        for r in results:
            if r.format_type not in by_format:
                by_format[r.format_type] = {"total": 0, "violations": 0}
            by_format[r.format_type]["total"] += 1
            if not r.format_compliant:
                by_format[r.format_type]["violations"] += 1

        for format_type, stats in sorted(by_format.items()):
            violation_rate = stats["violations"] / stats["total"] * 100
            status = "⚠️ UNRELIABLE" if violation_rate > 20 else "✓ RELIABLE"
            print(f"{status} {format_type}: {stats['violations']}/{stats['total']} violations ({violation_rate:.1f}%)")

        # By task domain
        print("\n" + "-"*80)
        print("Results by Task Domain:")
        print("-"*80)

        by_domain = {}
        for r in results:
            if r.task_domain not in by_domain:
                by_domain[r.task_domain] = {"total": 0, "violations": 0}
            by_domain[r.task_domain]["total"] += 1
            if not r.format_compliant:
                by_domain[r.task_domain]["violations"] += 1

        for domain, stats in sorted(by_domain.items()):
            violation_rate = stats["violations"] / stats["total"] * 100
            status = "⚠️ UNRELIABLE" if violation_rate > 20 else "✓ RELIABLE"
            print(f"{status} {domain}: {stats['violations']}/{stats['total']} violations ({violation_rate:.1f}%)")

        # Worst violations
        critical_cases = [r for r in results if r.severity in ["critical", "high"]]
        if critical_cases:
            print("\n" + "="*80)
            print("⚠️  CRITICAL FORMAT VIOLATIONS")
            print("="*80)
            for r in critical_cases[:5]:
                print(f"\n[{r.severity.upper()}] {r.format_type} / {r.task_domain} → {r.violation_type}")
                print(f"Prompt: {r.prompt[:100]}...")
                print(f"Response: {r.response[:100]}...")
                print(f"Why: {r.explanation[:150]}...")

    def save_results(self, results: List[FormatComplianceResult],
                     filename: str = "format_compliance_test_results.json"):
        """Save results to JSON file."""
        data = [
            {
                "format_type": r.format_type,
                "task_domain": r.task_domain,
                "prompt": r.prompt,
                "response": r.response,
                "format_compliant": r.format_compliant,
                "violation_type": r.violation_type,
                "severity": r.severity,
                "explanation": r.explanation,
                "timestamp": r.timestamp
            }
            for r in results
        ]
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nDetailed results saved to {filename}")


if __name__ == "__main__":
    API_KEY = os.getenv("ANTHROPIC_API_KEY")
    if not API_KEY:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        exit(1)

    tester = FormatComplianceTester(API_KEY)

    results = tester.run_comprehensive_test(
        format_types=["strict_json", "length_constraint", "numbered_list"],
        task_domains=["data_extraction", "summarization", "question_answering"],
        prompts_per_combination=2
    )

    tester.print_summary(results)
    tester.save_results(results)
