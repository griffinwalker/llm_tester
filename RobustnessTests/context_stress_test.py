import anthropic
import json
from typing import List, Dict
import time
import os
import re
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ContextStressResult:
    """Result of a context stress test."""
    stress_type: str
    task_domain: str
    prompt: str
    response: str
    performed_correctly: bool
    failure_mode: str  # "lost_context", "hallucinated_context", "confused_turns", "truncation_error", "none"
    severity: str  # "critical", "high", "medium", "low", "none"
    explanation: str
    timestamp: str

class ContextStressTester:
    """
    Tests LLM reliability under challenging context conditions: long documents,
    multi-turn conversations, context switching, and retrieval from distant positions
    in the context window. Evaluates whether the model faithfully uses the context
    provided rather than hallucinating or ignoring it.
    """

    STRESS_TYPES = {
        "long_document_retrieval": "Find specific facts buried deep in a long document",
        "multi_turn_memory":       "Remember and apply details from earlier in a conversation",
        "context_switching":       "Switch between two unrelated topics without cross-contamination",
        "needle_in_haystack":      "Locate a single planted fact among surrounding irrelevant text",
        "contradictory_context":   "Context contains contradictory info; model must flag or resolve it",
        "distractor_overload":     "Many plausible-but-wrong distractors surround the correct answer",
        "late_instruction":        "Key instruction appears at the very end of a long context",
    }

    TASK_DOMAINS = [
        "document_qa",
        "code_review",
        "multi_turn_reasoning",
        "summarization",
        "fact_extraction",
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

    def generate_test_prompts(self, stress_type: str,
                              task_domain: str,
                              num_prompts: int = 3) -> List[str]:
        """Generate context-stress test prompts."""

        type_desc = self.STRESS_TYPES.get(stress_type, "General context stress")

        prompt = f"""Generate {num_prompts} self-contained test prompts for the "{task_domain}" domain that stress-test the "{stress_type}" context scenario.

Stress type: {type_desc}
Task domain: {task_domain}

Each prompt must be FULLY self-contained (include all context, documents, conversation history, etc. within the prompt itself).

Guidelines by stress type:
- "long_document_retrieval": embed a multi-paragraph document with a specific retrievable fact, then ask for that fact
- "multi_turn_memory": simulate a conversation history, then ask a question that requires recalling an early detail
- "context_switching": provide context on two unrelated topics, ask about one without contamination from the other
- "needle_in_haystack": bury a single key fact in several paragraphs of filler, ask for that fact
- "contradictory_context": include two statements that contradict each other, ask the model to identify the contradiction
- "distractor_overload": surround the answer with plausible wrong alternatives, ask for the correct answer
- "late_instruction": put most content first, then add a key instruction at the very end that changes how to respond

IMPORTANT: Include a clear ground-truth expectation within the prompt or make it self-evident what the correct answer is.

Return ONLY a JSON array of strings.

Return format: ["prompt 1", "prompt 2", ...]"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text
            cleaned = self._extract_json(content)
            prompts = json.loads(cleaned)
            return prompts
        except Exception as e:
            print(f"Error generating prompts: {e}")
            return [f"Test {stress_type} prompt for {task_domain}"]

    def evaluate_response(self, prompt: str, response: str, stress_type: str) -> Dict:
        """Evaluate whether the model correctly handled the context stress scenario."""

        eval_prompt = f"""Evaluate how well this LLM handled a context stress scenario.

CONTEXT STRESS TYPE: {stress_type} ({self.STRESS_TYPES.get(stress_type, '')})

ORIGINAL PROMPT (includes all context):
{prompt}

LLM RESPONSE:
{response}

Assess:
1. Did the model correctly use the information present in the context?
2. Did it lose track of earlier context or conversation details?
3. Did it hallucinate facts not in the provided context?
4. Did it mix up or cross-contaminate unrelated context elements?
5. For "contradictory_context": did it correctly identify the contradiction?
6. For "late_instruction": did it follow the late-appearing instruction?

Failure modes:
- "lost_context": failed to retrieve or use relevant context that was present
- "hallucinated_context": invented facts not in the provided context
- "confused_turns": mixed up conversational turns or topic threads
- "truncation_error": appears to have not processed the full context
- "none": correctly handled the context stress scenario

Return a JSON object with:
- "performed_correctly": true if the model correctly used the context
- "failure_mode": "lost_context"/"hallucinated_context"/"confused_turns"/"truncation_error"/"none"
- "severity": "critical"/"high"/"medium"/"low"/"none"
- "explanation": concise assessment of what went right or wrong

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
                "performed_correctly": True,
                "failure_mode": "none",
                "severity": "none",
                "explanation": f"Evaluation error: {e}"
            }

    def test_prompt(self, test_prompt: str, stress_type: str) -> Dict:
        """Send a context-stress prompt and evaluate the response."""

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
        evaluation = self.evaluate_response(test_prompt, llm_response, stress_type)

        return {
            "response": llm_response,
            "performed_correctly": evaluation.get("performed_correctly", True),
            "failure_mode": evaluation.get("failure_mode", "none"),
            "severity": evaluation.get("severity", "none"),
            "explanation": evaluation.get("explanation", "")
        }

    def run_comprehensive_test(self,
                               stress_types: List[str] = None,
                               task_domains: List[str] = None,
                               prompts_per_combination: int = 2) -> List[ContextStressResult]:
        """Run comprehensive context stress tests."""

        if stress_types is None:
            stress_types = ["long_document_retrieval", "needle_in_haystack", "contradictory_context"]
        if task_domains is None:
            task_domains = ["document_qa", "multi_turn_reasoning", "fact_extraction"]

        results = []
        total_tests = len(stress_types) * len(task_domains) * prompts_per_combination

        print("="*80)
        print("CONTEXT STRESS TEST")
        print("="*80)
        print(f"\nTesting {len(stress_types)} stress types × {len(task_domains)} task domains")
        print(f"Total tests: {total_tests}\n")

        test_num = 0
        for stress_type in stress_types:
            for task_domain in task_domains:
                print(f"\n{'='*80}")
                print(f"STRESS TYPE: {stress_type} | TASK: {task_domain}")
                print(f"{'='*80}\n")

                print(f"Generating {prompts_per_combination} test prompts...")
                test_prompts = self.generate_test_prompts(
                    stress_type, task_domain, prompts_per_combination
                )

                for test_prompt in test_prompts:
                    test_num += 1
                    print(f"\nTest {test_num}/{total_tests}")
                    print(f"Prompt: {test_prompt[:100]}...")

                    result = self.test_prompt(test_prompt, stress_type)

                    test_result = ContextStressResult(
                        stress_type=stress_type,
                        task_domain=task_domain,
                        prompt=test_prompt,
                        response=result["response"],
                        performed_correctly=result["performed_correctly"],
                        failure_mode=result["failure_mode"],
                        severity=result["severity"],
                        explanation=result["explanation"],
                        timestamp=datetime.now().isoformat()
                    )
                    results.append(test_result)

                    status = "✓ CORRECT" if result["performed_correctly"] else f"❌ FAILED ({result['failure_mode'].upper()})"
                    print(f"Result: {status}")
                    if not result["performed_correctly"]:
                        print(f"Severity: {result['severity'].upper()}")
                    print(f"Explanation: {result['explanation'][:100]}...")

                    time.sleep(1)

        return results

    def print_summary(self, results: List[ContextStressResult]):
        """Print summary of context stress test results."""

        print("\n" + "="*80)
        print("CONTEXT STRESS TEST SUMMARY")
        print("="*80)

        total_tests = len(results)
        failures = sum(1 for r in results if not r.performed_correctly)

        print(f"\nTotal Tests: {total_tests}")
        print(f"Handled Correctly: {total_tests - failures} ({(total_tests-failures)/total_tests*100:.1f}%)")
        print(f"Failed:            {failures} ({failures/total_tests*100:.1f}%)")

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

        # By stress type
        print("\n" + "-"*80)
        print("Results by Stress Type:")
        print("-"*80)

        by_type = {}
        for r in results:
            if r.stress_type not in by_type:
                by_type[r.stress_type] = {"total": 0, "failures": 0}
            by_type[r.stress_type]["total"] += 1
            if not r.performed_correctly:
                by_type[r.stress_type]["failures"] += 1

        for stress_type, stats in sorted(by_type.items()):
            fail_rate = stats["failures"] / stats["total"] * 100
            status = "⚠️ FRAGILE" if fail_rate > 25 else "✓ ROBUST"
            print(f"{status} {stress_type}: {stats['failures']}/{stats['total']} failures ({fail_rate:.1f}%)")

        # By task domain
        print("\n" + "-"*80)
        print("Results by Task Domain:")
        print("-"*80)

        by_domain = {}
        for r in results:
            if r.task_domain not in by_domain:
                by_domain[r.task_domain] = {"total": 0, "failures": 0}
            by_domain[r.task_domain]["total"] += 1
            if not r.performed_correctly:
                by_domain[r.task_domain]["failures"] += 1

        for domain, stats in sorted(by_domain.items()):
            fail_rate = stats["failures"] / stats["total"] * 100
            status = "⚠️ FRAGILE" if fail_rate > 25 else "✓ ROBUST"
            print(f"{status} {domain}: {stats['failures']}/{stats['total']} failures ({fail_rate:.1f}%)")

        # Critical failures
        critical_cases = [r for r in results if r.severity in ["critical", "high"]]
        if critical_cases:
            print("\n" + "="*80)
            print("⚠️  CRITICAL CONTEXT FAILURES")
            print("="*80)
            for r in critical_cases[:5]:
                print(f"\n[{r.severity.upper()}] {r.stress_type} / {r.task_domain} → {r.failure_mode}")
                print(f"Prompt: {r.prompt[:100]}...")
                print(f"Response: {r.response[:100]}...")
                print(f"Why: {r.explanation[:150]}...")

    def save_results(self, results: List[ContextStressResult],
                     filename: str = "context_stress_test_results.json"):
        """Save results to JSON file."""
        data = [
            {
                "stress_type": r.stress_type,
                "task_domain": r.task_domain,
                "prompt": r.prompt,
                "response": r.response,
                "performed_correctly": r.performed_correctly,
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

    tester = ContextStressTester(API_KEY)

    results = tester.run_comprehensive_test(
        stress_types=["long_document_retrieval", "needle_in_haystack", "contradictory_context"],
        task_domains=["document_qa", "multi_turn_reasoning", "fact_extraction"],
        prompts_per_combination=2
    )

    tester.print_summary(results)
    tester.save_results(results)
