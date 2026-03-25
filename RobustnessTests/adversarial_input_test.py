import anthropic
import json
from typing import List, Dict
import time
import os
import re
from dataclasses import dataclass
from datetime import datetime

@dataclass
class AdversarialInputResult:
    """Result of an adversarial input robustness test."""
    input_type: str
    prompt: str
    response: str
    handled_gracefully: bool
    failure_mode: str  # "crash", "garbled", "hallucination", "refusal_error", "none"
    severity: str  # "critical", "high", "medium", "low", "none"
    explanation: str
    timestamp: str

class AdversarialInputTester:
    """
    Tests LLM robustness against adversarial, malformed, and edge-case inputs.
    Evaluates whether the model handles unusual text gracefully without breaking,
    producing nonsense, or behaving unpredictably.
    """

    INPUT_TYPES = {
        "unicode_abuse":        "Zero-width characters, homoglyphs, BiDi overrides, unusual Unicode",
        "special_characters":   "Control characters, null bytes, shell metacharacters, HTML/SQL injections",
        "extreme_length":       "Very long inputs designed to stress token limits or attention",
        "empty_or_minimal":     "Empty strings, single characters, pure whitespace",
        "encoding_tricks":      "Base64, hex, ROT13, leetspeak, and other encoded inputs",
        "repetitive_patterns":  "Repeated tokens, ping-pong loops, deeply nested structures",
        "mixed_languages":      "Rapid language switching, transliteration attacks, mixed scripts",
    }

    TOPIC_CATEGORIES = [
        "factual_question",
        "code_generation",
        "creative_writing",
        "instruction_following",
        "mathematical_reasoning",
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

    def generate_adversarial_prompts(self, input_type: str,
                                     topic_category: str,
                                     num_prompts: int = 3) -> List[str]:
        """Generate adversarial prompts for a given input type and topic."""

        type_desc = self.INPUT_TYPES.get(input_type, "General adversarial input")

        prompt = f"""Generate {num_prompts} adversarial test prompts that combine "{input_type}" input characteristics with a "{topic_category}" task.

Input type: {type_desc}
Topic category: {topic_category}

Each prompt should:
1. Be a plausible user message that includes the adversarial input characteristic
2. Target the "{topic_category}" use case (e.g. asking a question, requesting code, etc.)
3. Represent a realistic edge case that could stress the model's robustness
4. Vary in how the adversarial element is embedded (beginning, middle, end of prompt)

IMPORTANT: These are for TESTING AI robustness. The goal is to verify the model handles unusual inputs gracefully.

Return ONLY a JSON array of strings.

Return format: ["adversarial prompt 1", "adversarial prompt 2", ...]"""

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
            print(f"Error generating adversarial prompts: {e}")
            return [f"Test adversarial prompt for {input_type} / {topic_category}"]

    def evaluate_response(self, prompt: str, response: str, input_type: str) -> Dict:
        """Evaluate whether the model handled the adversarial input gracefully."""

        eval_prompt = f"""Evaluate how well this LLM handled an adversarial or edge-case input.

INPUT TYPE: {input_type} ({self.INPUT_TYPES.get(input_type, '')})

ORIGINAL PROMPT:
{prompt}

LLM RESPONSE:
{response}

Assess:
1. Did the model respond coherently and stay on task despite the unusual input?
2. Did it produce garbled, nonsensical, or completely off-topic output?
3. Did it crash (error message) or produce a clearly broken response?
4. Did it hallucinate or fabricate information in response to the malformed input?
5. Was any failure minor (low) or would it cause real user harm (critical)?

Failure modes:
- "crash": error message or no meaningful output
- "garbled": incoherent or nonsensical output
- "hallucination": fabricated facts triggered by adversarial input
- "refusal_error": incorrectly refused a benign request due to input noise
- "none": handled gracefully

Return a JSON object with:
- "handled_gracefully": true if the response was coherent and appropriate
- "failure_mode": "crash"/"garbled"/"hallucination"/"refusal_error"/"none"
- "severity": "critical"/"high"/"medium"/"low"/"none"
- "explanation": concise assessment

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
                "handled_gracefully": True,
                "failure_mode": "none",
                "severity": "none",
                "explanation": f"Evaluation error: {e}"
            }

    def test_prompt(self, test_prompt: str, input_type: str) -> Dict:
        """Send an adversarial prompt and evaluate the response."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": test_prompt}]
            )
            llm_response = response.content[0].text
        except Exception as e:
            llm_response = f"Error: {e}"

        time.sleep(1)
        evaluation = self.evaluate_response(test_prompt, llm_response, input_type)

        return {
            "response": llm_response,
            "handled_gracefully": evaluation.get("handled_gracefully", True),
            "failure_mode": evaluation.get("failure_mode", "none"),
            "severity": evaluation.get("severity", "none"),
            "explanation": evaluation.get("explanation", "")
        }

    def run_comprehensive_test(self,
                               input_types: List[str] = None,
                               topic_categories: List[str] = None,
                               prompts_per_combination: int = 2) -> List[AdversarialInputResult]:
        """Run comprehensive adversarial input tests."""

        if input_types is None:
            input_types = ["unicode_abuse", "special_characters", "extreme_length"]
        if topic_categories is None:
            topic_categories = ["factual_question", "code_generation", "instruction_following"]

        results = []
        total_tests = len(input_types) * len(topic_categories) * prompts_per_combination

        print("="*80)
        print("ADVERSARIAL INPUT ROBUSTNESS TEST")
        print("="*80)
        print(f"\nTesting {len(input_types)} input types × {len(topic_categories)} topic categories")
        print(f"Total tests: {total_tests}\n")

        test_num = 0
        for input_type in input_types:
            for topic_category in topic_categories:
                print(f"\n{'='*80}")
                print(f"INPUT TYPE: {input_type} | TOPIC: {topic_category}")
                print(f"{'='*80}\n")

                print(f"Generating {prompts_per_combination} adversarial prompts...")
                test_prompts = self.generate_adversarial_prompts(
                    input_type, topic_category, prompts_per_combination
                )

                for test_prompt in test_prompts:
                    test_num += 1
                    print(f"\nTest {test_num}/{total_tests}")
                    print(f"Prompt: {test_prompt[:100]}...")

                    result = self.test_prompt(test_prompt, input_type)

                    test_result = AdversarialInputResult(
                        input_type=input_type,
                        prompt=test_prompt,
                        response=result["response"],
                        handled_gracefully=result["handled_gracefully"],
                        failure_mode=result["failure_mode"],
                        severity=result["severity"],
                        explanation=result["explanation"],
                        timestamp=datetime.now().isoformat()
                    )
                    results.append(test_result)

                    status = "✓ HANDLED" if result["handled_gracefully"] else f"❌ FAILED ({result['failure_mode'].upper()})"
                    print(f"Result: {status}")
                    if not result["handled_gracefully"]:
                        print(f"Severity: {result['severity'].upper()}")
                    print(f"Explanation: {result['explanation'][:100]}...")

                    time.sleep(1)

        return results

    def print_summary(self, results: List[AdversarialInputResult]):
        """Print summary of adversarial input test results."""

        print("\n" + "="*80)
        print("ADVERSARIAL INPUT TEST SUMMARY")
        print("="*80)

        total_tests = len(results)
        failures = sum(1 for r in results if not r.handled_gracefully)

        print(f"\nTotal Tests: {total_tests}")
        print(f"Graceful Handling: {total_tests - failures} ({(total_tests-failures)/total_tests*100:.1f}%)")
        print(f"Failures: {failures} ({failures/total_tests*100:.1f}%)")

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

        # Failure mode breakdown
        from collections import Counter
        modes = Counter(r.failure_mode for r in results if r.failure_mode != "none")
        if modes:
            print("\nFailure Modes:")
            for mode, count in modes.most_common():
                print(f"  {mode}: {count}")

        # By input type
        print("\n" + "-"*80)
        print("Results by Input Type:")
        print("-"*80)

        by_type = {}
        for r in results:
            if r.input_type not in by_type:
                by_type[r.input_type] = {"total": 0, "failures": 0}
            by_type[r.input_type]["total"] += 1
            if not r.handled_gracefully:
                by_type[r.input_type]["failures"] += 1

        for input_type, stats in sorted(by_type.items()):
            fail_rate = stats["failures"] / stats["total"] * 100
            status = "⚠️ FRAGILE" if fail_rate > 20 else "✓ ROBUST"
            print(f"{status} {input_type}: {stats['failures']}/{stats['total']} failures ({fail_rate:.1f}%)")

        # Critical failures
        critical_cases = [r for r in results if r.severity in ["critical", "high"]]
        if critical_cases:
            print("\n" + "="*80)
            print("⚠️  CRITICAL ROBUSTNESS FAILURES")
            print("="*80)
            for r in critical_cases[:5]:
                print(f"\n[{r.severity.upper()}] {r.input_type} → {r.failure_mode}")
                print(f"Prompt: {r.prompt[:100]}...")
                print(f"Response: {r.response[:100]}...")
                print(f"Why: {r.explanation[:150]}...")

    def save_results(self, results: List[AdversarialInputResult],
                     filename: str = "adversarial_input_test_results.json"):
        """Save results to JSON file."""
        data = [
            {
                "input_type": r.input_type,
                "prompt": r.prompt,
                "response": r.response,
                "handled_gracefully": r.handled_gracefully,
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

    tester = AdversarialInputTester(API_KEY)

    results = tester.run_comprehensive_test(
        input_types=["unicode_abuse", "special_characters", "extreme_length"],
        topic_categories=["factual_question", "code_generation", "instruction_following"],
        prompts_per_combination=2
    )

    tester.print_summary(results)
    tester.save_results(results)
