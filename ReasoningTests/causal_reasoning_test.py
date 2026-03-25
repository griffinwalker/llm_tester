import anthropic
import json
from typing import List, Dict
import time
import os
import re
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CausalReasoningResult:
    """Result of a causal reasoning test."""
    causal_type: str
    domain: str
    prompt: str
    response: str
    correct: bool
    error_type: str  # "correlation_causation", "reversed_causality", "missed_confounder", "counterfactual_error", "none"
    severity: str  # "critical", "high", "medium", "low", "none"
    explanation: str
    timestamp: str

class CausalReasoningTester:
    """
    Tests LLM capability at causal reasoning: distinguishing causation from
    correlation, identifying confounders, reasoning about counterfactuals,
    and applying causal chain analysis. Errors in causal reasoning often
    lead to harmful real-world advice.
    """

    CAUSAL_TYPES = {
        "correlation_vs_causation": "Distinguish spurious correlations from genuine causal relationships",
        "confounder_identification": "Identify hidden variables that explain an apparent relationship",
        "causal_chain":             "Trace a sequence of cause-and-effect steps to a conclusion",
        "counterfactual":           "Reason about what would have happened under different conditions",
        "intervention_vs_observation": "Predict outcomes of interventions vs. passive observations",
        "root_cause_analysis":      "Work backwards from an observed effect to identify root causes",
        "bidirectional_causality":  "Identify when two variables mutually cause each other",
    }

    DOMAINS = [
        "medicine_and_health",
        "economics_and_policy",
        "technology_and_systems",
        "social_science",
        "everyday_scenarios",
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

    def generate_test_prompts(self, causal_type: str,
                              domain: str,
                              num_prompts: int = 3) -> List[Dict]:
        """Generate causal reasoning problems with ground-truth answers."""

        type_desc = self.CAUSAL_TYPES.get(causal_type, "General causal reasoning")

        prompt = f"""Generate {num_prompts} causal reasoning problems in the "{domain}" domain testing "{causal_type}".

Causal type: {type_desc}
Domain: {domain}

Each problem should:
1. Present a scenario requiring causal analysis relevant to "{domain}"
2. Have a clear correct answer that a careful reasoner would reach
3. Be designed to expose common causal reasoning errors:
   - For "correlation_vs_causation": include an appealing but spurious correlation
   - For "confounder_identification": present two variables that correlate due to a hidden third
   - For "causal_chain": require tracing 3+ causal steps
   - For "counterfactual": ask what would have happened if one factor were different
   - For "intervention_vs_observation": present data and ask about an active intervention
   - For "root_cause_analysis": describe an effect and ask for the most likely root cause
   - For "bidirectional_causality": present a feedback loop and ask the model to identify it

Include enough context to make the correct answer defensible.

Return ONLY a JSON array of objects with "prompt" and "correct_answer" keys.

Return format: [{{"prompt": "...", "correct_answer": "..."}}]"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text
            cleaned = self._extract_json(content)
            problems = json.loads(cleaned)
            return problems
        except Exception as e:
            print(f"Error generating prompts: {e}")
            return [{"prompt": f"Test {causal_type} in {domain}", "correct_answer": "unknown"}]

    def evaluate_response(self, prompt: str, response: str,
                          correct_answer: str, causal_type: str) -> Dict:
        """Evaluate the quality of the model's causal reasoning."""

        eval_prompt = f"""Evaluate this LLM's causal reasoning response.

CAUSAL TYPE: {causal_type} ({self.CAUSAL_TYPES.get(causal_type, '')})

PROBLEM:
{prompt}

CORRECT ANSWER:
{correct_answer}

LLM RESPONSE:
{response}

Assess:
1. Did the model reach the correct causal conclusion?
2. Did it confuse correlation with causation?
3. Did it reverse the causal direction?
4. Did it miss an important confounder or mediating variable?
5. Did it make an error in counterfactual reasoning?

Error types:
- "correlation_causation": treated a correlation as proof of causation
- "reversed_causality": got the direction of causation backwards
- "missed_confounder": ignored a key confounding variable
- "counterfactual_error": made an error in the counterfactual reasoning
- "none": causal analysis is correct

Return a JSON object with:
- "correct": true if the model reached the right causal conclusion
- "error_type": "correlation_causation"/"reversed_causality"/"missed_confounder"/"counterfactual_error"/"none"
- "severity": "critical"/"high"/"medium"/"low"/"none"
- "explanation": which causal reasoning error was made, if any

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
                "correct": False,
                "error_type": "none",
                "severity": "none",
                "explanation": f"Evaluation error: {e}"
            }

    def test_prompt(self, test_prompt: str, correct_answer: str, causal_type: str) -> Dict:
        """Send a causal reasoning problem and evaluate the response."""

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
        evaluation = self.evaluate_response(test_prompt, llm_response, correct_answer, causal_type)

        return {
            "response": llm_response,
            "correct": evaluation.get("correct", False),
            "error_type": evaluation.get("error_type", "none"),
            "severity": evaluation.get("severity", "none"),
            "explanation": evaluation.get("explanation", "")
        }

    def run_comprehensive_test(self,
                               causal_types: List[str] = None,
                               domains: List[str] = None,
                               prompts_per_combination: int = 2) -> List[CausalReasoningResult]:
        """Run comprehensive causal reasoning tests."""

        if causal_types is None:
            causal_types = ["correlation_vs_causation", "confounder_identification", "counterfactual"]
        if domains is None:
            domains = ["medicine_and_health", "economics_and_policy", "everyday_scenarios"]

        results = []
        total_tests = len(causal_types) * len(domains) * prompts_per_combination

        print("="*80)
        print("CAUSAL REASONING CAPABILITY TEST")
        print("="*80)
        print(f"\nTesting {len(causal_types)} causal types × {len(domains)} domains")
        print(f"Total tests: {total_tests}\n")

        test_num = 0
        for causal_type in causal_types:
            for domain in domains:
                print(f"\n{'='*80}")
                print(f"CAUSAL TYPE: {causal_type} | DOMAIN: {domain}")
                print(f"{'='*80}\n")

                print(f"Generating {prompts_per_combination} problems...")
                problems = self.generate_test_prompts(causal_type, domain, prompts_per_combination)

                for problem in problems:
                    test_num += 1
                    test_prompt = problem.get("prompt", "")
                    correct_answer = problem.get("correct_answer", "")

                    print(f"\nTest {test_num}/{total_tests}")
                    print(f"Problem: {test_prompt[:100]}...")
                    print(f"Expected: {correct_answer[:60]}")

                    result = self.test_prompt(test_prompt, correct_answer, causal_type)

                    test_result = CausalReasoningResult(
                        causal_type=causal_type,
                        domain=domain,
                        prompt=test_prompt,
                        response=result["response"],
                        correct=result["correct"],
                        error_type=result["error_type"],
                        severity=result["severity"],
                        explanation=result["explanation"],
                        timestamp=datetime.now().isoformat()
                    )
                    results.append(test_result)

                    status = "✓ CORRECT" if result["correct"] else f"❌ WRONG ({result['error_type']})"
                    print(f"Result: {status}")
                    print(f"Explanation: {result['explanation'][:100]}...")

                    time.sleep(1)

        return results

    def print_summary(self, results: List[CausalReasoningResult]):
        """Print summary of causal reasoning test results."""

        print("\n" + "="*80)
        print("CAUSAL REASONING TEST SUMMARY")
        print("="*80)

        total_tests = len(results)
        correct = sum(1 for r in results if r.correct)

        print(f"\nTotal Tests: {total_tests}")
        print(f"Correct:     {correct} ({correct/total_tests*100:.1f}%)")
        print(f"Incorrect:   {total_tests - correct} ({(total_tests-correct)/total_tests*100:.1f}%)")

        # Error type breakdown
        from collections import Counter
        errors = Counter(r.error_type for r in results if r.error_type != "none")
        if errors:
            print("\nError Types:")
            for error, count in errors.most_common():
                print(f"  {error}: {count}")

        # By causal type
        print("\n" + "-"*80)
        print("Results by Causal Type:")
        print("-"*80)

        by_type = {}
        for r in results:
            if r.causal_type not in by_type:
                by_type[r.causal_type] = {"total": 0, "correct": 0}
            by_type[r.causal_type]["total"] += 1
            if r.correct:
                by_type[r.causal_type]["correct"] += 1

        for ctype, stats in sorted(by_type.items()):
            acc = stats["correct"] / stats["total"] * 100
            status = "⚠️ WEAK" if acc < 60 else "✓ CAPABLE"
            print(f"{status} {ctype}: {stats['correct']}/{stats['total']} correct ({acc:.1f}%)")

        # By domain
        print("\n" + "-"*80)
        print("Results by Domain:")
        print("-"*80)

        by_domain = {}
        for r in results:
            if r.domain not in by_domain:
                by_domain[r.domain] = {"total": 0, "correct": 0}
            by_domain[r.domain]["total"] += 1
            if r.correct:
                by_domain[r.domain]["correct"] += 1

        for domain, stats in sorted(by_domain.items()):
            acc = stats["correct"] / stats["total"] * 100
            status = "⚠️ WEAK" if acc < 60 else "✓ CAPABLE"
            print(f"{status} {domain}: {stats['correct']}/{stats['total']} correct ({acc:.1f}%)")

        # Critical failures (causal errors in medicine/policy are high-stakes)
        failures = [r for r in results if not r.correct and r.severity in ["critical", "high"]]
        if failures:
            print("\n" + "="*80)
            print("⚠️  HIGH-SEVERITY CAUSAL REASONING FAILURES")
            print("="*80)
            for r in failures[:5]:
                print(f"\n[{r.severity.upper()}] {r.causal_type} / {r.domain} → {r.error_type}")
                print(f"Problem: {r.prompt[:100]}...")
                print(f"Response: {r.response[:100]}...")
                print(f"Why: {r.explanation[:150]}...")

    def save_results(self, results: List[CausalReasoningResult],
                     filename: str = "causal_reasoning_test_results.json"):
        """Save results to JSON file."""
        data = [
            {
                "causal_type": r.causal_type,
                "domain": r.domain,
                "prompt": r.prompt,
                "response": r.response,
                "correct": r.correct,
                "error_type": r.error_type,
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

    tester = CausalReasoningTester(API_KEY)

    results = tester.run_comprehensive_test(
        causal_types=["correlation_vs_causation", "confounder_identification", "counterfactual"],
        domains=["medicine_and_health", "economics_and_policy", "everyday_scenarios"],
        prompts_per_combination=2
    )

    tester.print_summary(results)
    tester.save_results(results)
