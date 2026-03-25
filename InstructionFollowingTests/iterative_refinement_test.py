import anthropic
import json
from typing import List, Dict
import time
import os
import re
from dataclasses import dataclass
from datetime import datetime

@dataclass
class IterativeRefinementResult:
    """Result of an iterative refinement instruction following test."""
    refinement_type: str
    task_domain: str
    initial_prompt: str
    initial_response: str
    refinement_prompt: str
    refined_response: str
    refinement_applied: bool
    prior_requirements_kept: bool
    failure_mode: str  # "refinement_ignored", "prior_lost", "over_corrected", "regression", "none"
    severity: str  # "critical", "high", "medium", "low", "none"
    explanation: str
    timestamp: str

class IterativeRefinementTester:
    """
    Tests whether an LLM correctly applies follow-up refinement instructions
    to its previous output. A well-behaved model should incorporate the new
    instruction while preserving all requirements from the original prompt.
    This tests one of the most common real-world interaction patterns.
    """

    REFINEMENT_TYPES = {
        "additive":         "Add something new without changing what is already correct",
        "subtractive":      "Remove a specific element while keeping everything else",
        "style_change":     "Rewrite in a different style while keeping all content",
        "scope_expansion":  "Expand on one specific part without shrinking others",
        "correction":       "Fix a specific error the user points out",
        "constraint_add":   "Apply a new constraint to the existing output retroactively",
        "prioritization":   "Re-order or re-weight elements based on new priority guidance",
    }

    TASK_DOMAINS = [
        "writing_and_editing",
        "code_generation",
        "data_summarization",
        "planning_and_lists",
        "explanation",
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

    def generate_test_cases(self, refinement_type: str,
                            task_domain: str,
                            num_cases: int = 3) -> List[Dict]:
        """Generate initial prompt + refinement instruction pairs."""

        ref_desc = self.REFINEMENT_TYPES.get(refinement_type, "General refinement")

        prompt = f"""Generate {num_cases} iterative refinement test cases for refinement type "{refinement_type}" in the "{task_domain}" domain.

Refinement type: {ref_desc}
Task domain: {task_domain}

Each test case needs:
1. "initial_prompt": A complete task prompt in the "{task_domain}" domain with specific requirements
2. "refinement_prompt": A follow-up instruction applying a "{refinement_type}" change (written as if the user is responding to a previous answer)
3. "requirements_to_keep": A list of requirements from the initial prompt that must survive the refinement
4. "refinement_to_apply": A clear description of exactly what should change

Refinement type guidelines:
- "additive": "Also add X" — X should be added, nothing else should change
- "subtractive": "Remove the part about X" — only X removed, everything else stays
- "style_change": "Now rewrite it in Y style" — style changes, all content preserved
- "scope_expansion": "Expand the section on X" — X gets longer, unrelated parts stay same
- "correction": "Actually, X is wrong — it should be Y" — targeted fix, no collateral changes
- "constraint_add": "Also, make sure it never uses the word X" — constraint applied retroactively
- "prioritization": "Actually put X first since it's most important" — reorder, no content lost

Make initial prompts specific enough that "requirements_to_keep" are verifiable.

Return ONLY a JSON array of objects with "initial_prompt", "refinement_prompt", "requirements_to_keep", and "refinement_to_apply" keys.

Return format: [{{"initial_prompt": "...", "refinement_prompt": "...", "requirements_to_keep": ["..."], "refinement_to_apply": "..."}}]"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2500,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text
            cleaned = self._extract_json(content)
            cases = json.loads(cleaned)
            return cases
        except Exception as e:
            print(f"Error generating test cases: {e}")
            return [{"initial_prompt": f"Test {task_domain} task.",
                     "refinement_prompt": f"Apply {refinement_type} refinement.",
                     "requirements_to_keep": ["Keep original requirements"],
                     "refinement_to_apply": "Apply refinement"}]

    def evaluate_refinement(self, initial_prompt: str, initial_response: str,
                            refinement_prompt: str, refined_response: str,
                            refinement_type: str,
                            requirements_to_keep: List[str],
                            refinement_to_apply: str) -> Dict:
        """Evaluate whether the refinement was applied correctly without losing prior requirements."""

        requirements_str = "\n".join(f"- {r}" for r in requirements_to_keep)

        eval_prompt = f"""Evaluate whether this LLM correctly applied a refinement instruction while preserving original requirements.

REFINEMENT TYPE: {refinement_type} ({self.REFINEMENT_TYPES.get(refinement_type, '')})

ORIGINAL TASK PROMPT:
{initial_prompt}

INITIAL RESPONSE:
{initial_response}

REFINEMENT INSTRUCTION:
{refinement_prompt}

WHAT THE REFINEMENT SHOULD DO:
{refinement_to_apply}

REQUIREMENTS THAT MUST BE PRESERVED IN THE REFINED RESPONSE:
{requirements_str}

REFINED RESPONSE:
{refined_response}

Assess:
1. Was the refinement correctly applied as specified?
2. Are all listed original requirements still satisfied in the refined response?
3. Did the model over-correct, changing things it shouldn't have changed?
4. Did the refined response regress (become worse overall than the initial response)?

Failure modes:
- "refinement_ignored": the refinement instruction was not applied at all
- "prior_lost": the refinement was applied but one or more original requirements were dropped
- "over_corrected": more was changed than the refinement asked for
- "regression": the refined response is worse overall despite applying the change
- "none": refinement applied correctly, all prior requirements preserved

Return a JSON object with:
- "refinement_applied": true if the requested change was made
- "prior_requirements_kept": true if all listed requirements are still met
- "failure_mode": "refinement_ignored"/"prior_lost"/"over_corrected"/"regression"/"none"
- "severity": "critical"/"high"/"medium"/"low"/"none"
- "explanation": specific assessment of what was and wasn't preserved

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
                "refinement_applied": False,
                "prior_requirements_kept": False,
                "failure_mode": "none",
                "severity": "none",
                "explanation": f"Evaluation error: {e}"
            }

    def test_case(self, initial_prompt: str, refinement_prompt: str,
                  refinement_type: str,
                  requirements_to_keep: List[str],
                  refinement_to_apply: str) -> Dict:
        """Run initial prompt then refinement, evaluate both."""

        # Step 1: get initial response
        try:
            r1 = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": initial_prompt}]
            )
            initial_response = r1.content[0].text
        except Exception as e:
            initial_response = f"Error: {e}"

        time.sleep(1)

        # Step 2: apply refinement in context
        try:
            r2 = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": initial_prompt},
                    {"role": "assistant", "content": initial_response},
                    {"role": "user", "content": refinement_prompt}
                ]
            )
            refined_response = r2.content[0].text
        except Exception as e:
            refined_response = f"Error: {e}"

        time.sleep(1)

        evaluation = self.evaluate_refinement(
            initial_prompt, initial_response,
            refinement_prompt, refined_response,
            refinement_type, requirements_to_keep, refinement_to_apply
        )

        return {
            "initial_response": initial_response,
            "refined_response": refined_response,
            "refinement_applied": evaluation.get("refinement_applied", False),
            "prior_requirements_kept": evaluation.get("prior_requirements_kept", False),
            "failure_mode": evaluation.get("failure_mode", "none"),
            "severity": evaluation.get("severity", "none"),
            "explanation": evaluation.get("explanation", "")
        }

    def run_comprehensive_test(self,
                               refinement_types: List[str] = None,
                               task_domains: List[str] = None,
                               cases_per_combination: int = 2) -> List[IterativeRefinementResult]:
        """Run comprehensive iterative refinement tests."""

        if refinement_types is None:
            refinement_types = ["additive", "subtractive", "style_change"]
        if task_domains is None:
            task_domains = ["writing_and_editing", "code_generation", "planning_and_lists"]

        results = []
        total_tests = len(refinement_types) * len(task_domains) * cases_per_combination

        print("="*80)
        print("ITERATIVE REFINEMENT INSTRUCTION FOLLOWING TEST")
        print("="*80)
        print(f"\nTesting {len(refinement_types)} refinement types × {len(task_domains)} task domains")
        print(f"Total tests: {total_tests}\n")

        test_num = 0
        for refinement_type in refinement_types:
            for task_domain in task_domains:
                print(f"\n{'='*80}")
                print(f"REFINEMENT TYPE: {refinement_type} | TASK: {task_domain}")
                print(f"{'='*80}\n")

                print(f"Generating {cases_per_combination} test cases...")
                cases = self.generate_test_cases(refinement_type, task_domain, cases_per_combination)

                for case in cases:
                    test_num += 1
                    initial_prompt = case.get("initial_prompt", "")
                    refinement_prompt = case.get("refinement_prompt", "")
                    requirements_to_keep = case.get("requirements_to_keep", [])
                    refinement_to_apply = case.get("refinement_to_apply", "")

                    print(f"\nTest {test_num}/{total_tests}")
                    print(f"Initial: {initial_prompt[:80]}...")
                    print(f"Refine:  {refinement_prompt[:80]}...")

                    result = self.test_case(
                        initial_prompt, refinement_prompt,
                        refinement_type, requirements_to_keep, refinement_to_apply
                    )

                    both_ok = result["refinement_applied"] and result["prior_requirements_kept"]

                    test_result = IterativeRefinementResult(
                        refinement_type=refinement_type,
                        task_domain=task_domain,
                        initial_prompt=initial_prompt,
                        initial_response=result["initial_response"],
                        refinement_prompt=refinement_prompt,
                        refined_response=result["refined_response"],
                        refinement_applied=result["refinement_applied"],
                        prior_requirements_kept=result["prior_requirements_kept"],
                        failure_mode=result["failure_mode"],
                        severity=result["severity"],
                        explanation=result["explanation"],
                        timestamp=datetime.now().isoformat()
                    )
                    results.append(test_result)

                    if both_ok:
                        status = "✓ PERFECT REFINEMENT"
                    elif result["refinement_applied"] and not result["prior_requirements_kept"]:
                        status = f"⚠️ PRIOR LOST ({result['failure_mode']})"
                    elif not result["refinement_applied"]:
                        status = f"❌ REFINEMENT IGNORED"
                    else:
                        status = f"❌ FAILED ({result['failure_mode']})"

                    print(f"Result: {status}")
                    print(f"Explanation: {result['explanation'][:100]}...")

                    time.sleep(1)

        return results

    def print_summary(self, results: List[IterativeRefinementResult]):
        """Print summary of iterative refinement test results."""

        print("\n" + "="*80)
        print("ITERATIVE REFINEMENT TEST SUMMARY")
        print("="*80)

        total_tests = len(results)
        both_ok = sum(1 for r in results if r.refinement_applied and r.prior_requirements_kept)
        refinement_applied = sum(1 for r in results if r.refinement_applied)
        prior_kept = sum(1 for r in results if r.prior_requirements_kept)

        print(f"\nTotal Tests:              {total_tests}")
        print(f"Both Applied & Preserved: {both_ok} ({both_ok/total_tests*100:.1f}%)")
        print(f"Refinement Applied:       {refinement_applied} ({refinement_applied/total_tests*100:.1f}%)")
        print(f"Prior Requirements Kept:  {prior_kept} ({prior_kept/total_tests*100:.1f}%)")

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

        # Failure modes
        from collections import Counter
        modes = Counter(r.failure_mode for r in results if r.failure_mode != "none")
        if modes:
            print("\nFailure Modes:")
            for mode, count in modes.most_common():
                print(f"  {mode}: {count}")

        # By refinement type
        print("\n" + "-"*80)
        print("Results by Refinement Type:")
        print("-"*80)

        by_type = {}
        for r in results:
            if r.refinement_type not in by_type:
                by_type[r.refinement_type] = {"total": 0, "both_ok": 0}
            by_type[r.refinement_type]["total"] += 1
            if r.refinement_applied and r.prior_requirements_kept:
                by_type[r.refinement_type]["both_ok"] += 1

        for rtype, stats in sorted(by_type.items()):
            rate = stats["both_ok"] / stats["total"] * 100
            status = "⚠️ UNRELIABLE" if rate < 70 else "✓ RELIABLE"
            print(f"{status} {rtype}: {stats['both_ok']}/{stats['total']} perfect ({rate:.1f}%)")

        # By task domain
        print("\n" + "-"*80)
        print("Results by Task Domain:")
        print("-"*80)

        by_domain = {}
        for r in results:
            if r.task_domain not in by_domain:
                by_domain[r.task_domain] = {"total": 0, "both_ok": 0}
            by_domain[r.task_domain]["total"] += 1
            if r.refinement_applied and r.prior_requirements_kept:
                by_domain[r.task_domain]["both_ok"] += 1

        for domain, stats in sorted(by_domain.items()):
            rate = stats["both_ok"] / stats["total"] * 100
            status = "⚠️ UNRELIABLE" if rate < 70 else "✓ RELIABLE"
            print(f"{status} {domain}: {stats['both_ok']}/{stats['total']} perfect ({rate:.1f}%)")

        # Critical failures
        critical_cases = [r for r in results if r.severity in ["critical", "high"]]
        if critical_cases:
            print("\n" + "="*80)
            print("⚠️  CRITICAL REFINEMENT FAILURES")
            print("="*80)
            for r in critical_cases[:5]:
                print(f"\n[{r.severity.upper()}] {r.refinement_type} / {r.task_domain} → {r.failure_mode}")
                print(f"Initial: {r.initial_prompt[:80]}...")
                print(f"Refine:  {r.refinement_prompt[:80]}...")
                print(f"Why: {r.explanation[:150]}...")

    def save_results(self, results: List[IterativeRefinementResult],
                     filename: str = "iterative_refinement_test_results.json"):
        """Save results to JSON file."""
        data = [
            {
                "refinement_type": r.refinement_type,
                "task_domain": r.task_domain,
                "initial_prompt": r.initial_prompt,
                "initial_response": r.initial_response,
                "refinement_prompt": r.refinement_prompt,
                "refined_response": r.refined_response,
                "refinement_applied": r.refinement_applied,
                "prior_requirements_kept": r.prior_requirements_kept,
                "failure_mode": r.failure_mode,
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

    tester = IterativeRefinementTester(API_KEY)

    results = tester.run_comprehensive_test(
        refinement_types=["additive", "subtractive", "style_change"],
        task_domains=["writing_and_editing", "code_generation", "planning_and_lists"],
        cases_per_combination=2
    )

    tester.print_summary(results)
    tester.save_results(results)
