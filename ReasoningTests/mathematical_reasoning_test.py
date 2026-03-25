import anthropic
import json
from typing import List, Dict
import time
import os
import re
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MathReasoningResult:
    """Result of a mathematical reasoning test."""
    math_type: str
    difficulty: str
    prompt: str
    response: str
    correct: bool
    error_type: str  # "arithmetic_error", "setup_error", "unit_error", "conceptual_error", "none"
    severity: str  # "critical", "high", "medium", "low", "none"
    explanation: str
    timestamp: str

class MathematicalReasoningTester:
    """
    Tests LLM capability at mathematical reasoning across arithmetic, algebra,
    probability, combinatorics, and multi-step word problems. Evaluates both
    the correctness of answers and the soundness of the solution approach.
    """

    MATH_TYPES = {
        "arithmetic":        "Multi-step calculations with integers, fractions, and decimals",
        "algebra":           "Equation solving, variable manipulation, systems of equations",
        "probability":       "Basic probability, conditional probability, expected value",
        "combinatorics":     "Permutations, combinations, counting principles",
        "word_problems":     "Real-world scenarios requiring mathematical modeling",
        "estimation":        "Fermi estimation and order-of-magnitude reasoning",
        "geometric":         "Area, volume, angles, coordinate geometry",
    }

    DIFFICULTY_LEVELS = [
        "simple",
        "intermediate",
        "complex",
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

    def generate_test_prompts(self, math_type: str,
                              difficulty: str,
                              num_prompts: int = 3) -> List[Dict]:
        """Generate math problems with verifiable answers."""

        type_desc = self.MATH_TYPES.get(math_type, "General math")

        prompt = f"""Generate {num_prompts} mathematical reasoning problems of "{difficulty}" difficulty testing "{math_type}".

Math type: {type_desc}
Difficulty: {difficulty}

Each problem should:
1. Have a single unambiguous numerical or symbolic correct answer
2. Be calibrated to "{difficulty}" difficulty:
   - simple: 1-2 operations, no multi-step setup needed
   - intermediate: 3-5 steps, may require setting up equations
   - complex: multi-step with intermediate results, or requires insight to set up correctly
3. For "word_problems": embed the math in a realistic real-world scenario
4. For "estimation": expect an approximate answer within a reasonable range (specify the range)
5. Include all necessary information to solve the problem

IMPORTANT: Double-check that your correct_answer is actually correct before including it.

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
            return [{"prompt": f"Test {math_type} problem ({difficulty})", "correct_answer": "unknown"}]

    def evaluate_response(self, prompt: str, response: str,
                          correct_answer: str, math_type: str) -> Dict:
        """Evaluate whether the model's math solution is correct."""

        eval_prompt = f"""Evaluate this LLM's mathematical reasoning response.

MATH TYPE: {math_type} ({self.MATH_TYPES.get(math_type, '')})

PROBLEM:
{prompt}

CORRECT ANSWER:
{correct_answer}

LLM RESPONSE:
{response}

Assess:
1. Is the final answer correct (numerically or symbolically equivalent to the correct answer)?
2. Is the solution approach valid, or did the model set up the problem incorrectly?
3. Are there arithmetic/calculation errors in the steps even if the setup was right?
4. Did the model make unit or dimension errors?
5. Did the model demonstrate a conceptual misunderstanding of the math type?

For "estimation" problems: accept answers within one order of magnitude of the correct answer.

Error types:
- "arithmetic_error": correct setup but calculation mistake in the steps
- "setup_error": wrong approach or equation, problem modeled incorrectly
- "unit_error": correct numbers but wrong units or dimensions
- "conceptual_error": fundamental misunderstanding of the math concept
- "none": answer is correct

Return a JSON object with:
- "correct": true if the final answer matches the correct answer
- "approach_valid": true if the solution method was correct even if arithmetic was wrong
- "error_type": "arithmetic_error"/"setup_error"/"unit_error"/"conceptual_error"/"none"
- "severity": "critical"/"high"/"medium"/"low"/"none"
- "explanation": brief assessment of the solution quality

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
                "approach_valid": False,
                "error_type": "none",
                "severity": "none",
                "explanation": f"Evaluation error: {e}"
            }

    def test_prompt(self, test_prompt: str, correct_answer: str, math_type: str) -> Dict:
        """Send a math problem and evaluate the solution."""

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
        evaluation = self.evaluate_response(test_prompt, llm_response, correct_answer, math_type)

        return {
            "response": llm_response,
            "correct": evaluation.get("correct", False),
            "error_type": evaluation.get("error_type", "none"),
            "severity": evaluation.get("severity", "none"),
            "explanation": evaluation.get("explanation", "")
        }

    def run_comprehensive_test(self,
                               math_types: List[str] = None,
                               difficulty_levels: List[str] = None,
                               prompts_per_combination: int = 2) -> List[MathReasoningResult]:
        """Run comprehensive mathematical reasoning tests."""

        if math_types is None:
            math_types = ["arithmetic", "probability", "word_problems"]
        if difficulty_levels is None:
            difficulty_levels = ["simple", "intermediate", "complex"]

        results = []
        total_tests = len(math_types) * len(difficulty_levels) * prompts_per_combination

        print("="*80)
        print("MATHEMATICAL REASONING CAPABILITY TEST")
        print("="*80)
        print(f"\nTesting {len(math_types)} math types × {len(difficulty_levels)} difficulty levels")
        print(f"Total tests: {total_tests}\n")

        test_num = 0
        for math_type in math_types:
            for difficulty in difficulty_levels:
                print(f"\n{'='*80}")
                print(f"MATH TYPE: {math_type} | DIFFICULTY: {difficulty}")
                print(f"{'='*80}\n")

                print(f"Generating {prompts_per_combination} problems...")
                problems = self.generate_test_prompts(math_type, difficulty, prompts_per_combination)

                for problem in problems:
                    test_num += 1
                    test_prompt = problem.get("prompt", "")
                    correct_answer = problem.get("correct_answer", "")

                    print(f"\nTest {test_num}/{total_tests}")
                    print(f"Problem: {test_prompt[:100]}...")
                    print(f"Expected: {correct_answer[:60]}")

                    result = self.test_prompt(test_prompt, correct_answer, math_type)

                    test_result = MathReasoningResult(
                        math_type=math_type,
                        difficulty=difficulty,
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

    def print_summary(self, results: List[MathReasoningResult]):
        """Print summary of mathematical reasoning test results."""

        print("\n" + "="*80)
        print("MATHEMATICAL REASONING TEST SUMMARY")
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

        # By math type
        print("\n" + "-"*80)
        print("Results by Math Type:")
        print("-"*80)

        by_type = {}
        for r in results:
            if r.math_type not in by_type:
                by_type[r.math_type] = {"total": 0, "correct": 0}
            by_type[r.math_type]["total"] += 1
            if r.correct:
                by_type[r.math_type]["correct"] += 1

        for mtype, stats in sorted(by_type.items()):
            acc = stats["correct"] / stats["total"] * 100
            status = "⚠️ WEAK" if acc < 60 else "✓ CAPABLE"
            print(f"{status} {mtype}: {stats['correct']}/{stats['total']} correct ({acc:.1f}%)")

        # By difficulty
        print("\n" + "-"*80)
        print("Results by Difficulty:")
        print("-"*80)

        by_diff = {}
        for r in results:
            if r.difficulty not in by_diff:
                by_diff[r.difficulty] = {"total": 0, "correct": 0}
            by_diff[r.difficulty]["total"] += 1
            if r.correct:
                by_diff[r.difficulty]["correct"] += 1

        for diff in ["simple", "intermediate", "complex"]:
            if diff in by_diff:
                stats = by_diff[diff]
                acc = stats["correct"] / stats["total"] * 100
                status = "⚠️ WEAK" if acc < 60 else "✓ CAPABLE"
                print(f"{status} {diff}: {stats['correct']}/{stats['total']} correct ({acc:.1f}%)")

        # Failures
        failures = [r for r in results if not r.correct and r.severity in ["critical", "high"]]
        if failures:
            print("\n" + "="*80)
            print("⚠️  HIGH-SEVERITY MATH FAILURES")
            print("="*80)
            for r in failures[:5]:
                print(f"\n[{r.severity.upper()}] {r.math_type} / {r.difficulty} → {r.error_type}")
                print(f"Problem: {r.prompt[:100]}...")
                print(f"Response: {r.response[:100]}...")
                print(f"Why: {r.explanation[:150]}...")

    def save_results(self, results: List[MathReasoningResult],
                     filename: str = "mathematical_reasoning_test_results.json"):
        """Save results to JSON file."""
        data = [
            {
                "math_type": r.math_type,
                "difficulty": r.difficulty,
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

    tester = MathematicalReasoningTester(API_KEY)

    results = tester.run_comprehensive_test(
        math_types=["arithmetic", "probability", "word_problems"],
        difficulty_levels=["simple", "intermediate", "complex"],
        prompts_per_combination=2
    )

    tester.print_summary(results)
    tester.save_results(results)
