import anthropic
import json
from typing import List, Dict
import time
import os
import re
from dataclasses import dataclass
from datetime import datetime

@dataclass
class LogicalReasoningResult:
    """Result of a logical reasoning test."""
    reasoning_type: str
    difficulty: str
    prompt: str
    response: str
    correct: bool
    error_type: str  # "invalid_inference", "missed_premise", "fallacy_accepted", "correct_rejected", "none"
    severity: str  # "critical", "high", "medium", "low", "none"
    explanation: str
    timestamp: str

class LogicalReasoningTester:
    """
    Tests LLM capability at formal and informal logical reasoning.
    Evaluates deductive validity, inductive strength, fallacy detection,
    syllogisms, and multi-step logical chains.
    """

    REASONING_TYPES = {
        "deductive":          "Drawing valid conclusions from given premises (syllogisms, modus ponens)",
        "inductive":          "Generalizing from specific examples to probable conclusions",
        "abductive":          "Inferring the most likely explanation for observed facts",
        "fallacy_detection":  "Identifying named logical fallacies in an argument",
        "counterfactual":     "Reasoning about what would be true if some premise were different",
        "constraint_logic":   "Solving problems defined by a set of logical constraints",
        "argument_validity":  "Distinguishing valid argument structure from sound argument",
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

    def generate_test_prompts(self, reasoning_type: str,
                              difficulty: str,
                              num_prompts: int = 3) -> List[Dict]:
        """Generate logical reasoning test prompts with ground-truth answers."""

        type_desc = self.REASONING_TYPES.get(reasoning_type, "General logical reasoning")

        prompt = f"""Generate {num_prompts} logical reasoning problems of "{difficulty}" difficulty that test "{reasoning_type}".

Reasoning type: {type_desc}
Difficulty: {difficulty}

Each problem should:
1. Present a self-contained logical reasoning challenge
2. Have a single objectively correct answer or assessment
3. Be calibrated to "{difficulty}" difficulty:
   - simple: 1-2 step reasoning, everyday language
   - intermediate: 3-4 steps, may use abstract entities
   - complex: 5+ steps, nested conditionals or multi-premise chains
4. For "fallacy_detection": name a fallacy and ask the model to identify it in an argument
5. For "argument_validity": include arguments that are valid-but-unsound OR invalid-but-persuasive

Return ONLY a JSON array of objects with "prompt" and "correct_answer" keys.
The "correct_answer" should be a brief, unambiguous ground truth.

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
            return [{"prompt": f"Test {reasoning_type} problem ({difficulty})", "correct_answer": "unknown"}]

    def evaluate_response(self, prompt: str, response: str,
                          correct_answer: str, reasoning_type: str) -> Dict:
        """Evaluate whether the model's reasoning and answer are correct."""

        eval_prompt = f"""Evaluate this LLM's logical reasoning response.

REASONING TYPE: {reasoning_type} ({self.REASONING_TYPES.get(reasoning_type, '')})

PROBLEM:
{prompt}

CORRECT ANSWER / GROUND TRUTH:
{correct_answer}

LLM RESPONSE:
{response}

Assess:
1. Did the model reach the correct conclusion?
2. Was its reasoning valid, or did it make a logical error even if the answer was right?
3. Did it accept a fallacious argument it should have rejected?
4. Did it reject a valid argument it should have accepted?
5. Did it skip steps in the reasoning chain in a way that undermines reliability?

Error types:
- "invalid_inference": drew a conclusion not supported by the premises
- "missed_premise": failed to use a key premise, leading to wrong or incomplete answer
- "fallacy_accepted": treated a fallacious argument as valid
- "correct_rejected": correctly structured argument was rejected without justification
- "none": reasoning and conclusion are correct

Return a JSON object with:
- "correct": true if the final answer matches the ground truth
- "reasoning_valid": true if the logical steps are sound regardless of final answer
- "error_type": "invalid_inference"/"missed_premise"/"fallacy_accepted"/"correct_rejected"/"none"
- "severity": "critical"/"high"/"medium"/"low"/"none"
- "explanation": brief assessment of the reasoning quality

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
                "reasoning_valid": False,
                "error_type": "none",
                "severity": "none",
                "explanation": f"Evaluation error: {e}"
            }

    def test_prompt(self, test_prompt: str, correct_answer: str, reasoning_type: str) -> Dict:
        """Send a reasoning problem and evaluate the response."""

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
        evaluation = self.evaluate_response(test_prompt, llm_response, correct_answer, reasoning_type)

        return {
            "response": llm_response,
            "correct": evaluation.get("correct", False),
            "error_type": evaluation.get("error_type", "none"),
            "severity": evaluation.get("severity", "none"),
            "explanation": evaluation.get("explanation", "")
        }

    def run_comprehensive_test(self,
                               reasoning_types: List[str] = None,
                               difficulty_levels: List[str] = None,
                               prompts_per_combination: int = 2) -> List[LogicalReasoningResult]:
        """Run comprehensive logical reasoning tests."""

        if reasoning_types is None:
            reasoning_types = ["deductive", "fallacy_detection", "constraint_logic"]
        if difficulty_levels is None:
            difficulty_levels = ["simple", "intermediate", "complex"]

        results = []
        total_tests = len(reasoning_types) * len(difficulty_levels) * prompts_per_combination

        print("="*80)
        print("LOGICAL REASONING CAPABILITY TEST")
        print("="*80)
        print(f"\nTesting {len(reasoning_types)} reasoning types × {len(difficulty_levels)} difficulty levels")
        print(f"Total tests: {total_tests}\n")

        test_num = 0
        for reasoning_type in reasoning_types:
            for difficulty in difficulty_levels:
                print(f"\n{'='*80}")
                print(f"REASONING TYPE: {reasoning_type} | DIFFICULTY: {difficulty}")
                print(f"{'='*80}\n")

                print(f"Generating {prompts_per_combination} problems...")
                problems = self.generate_test_prompts(reasoning_type, difficulty, prompts_per_combination)

                for problem in problems:
                    test_num += 1
                    test_prompt = problem.get("prompt", "")
                    correct_answer = problem.get("correct_answer", "")

                    print(f"\nTest {test_num}/{total_tests}")
                    print(f"Problem: {test_prompt[:100]}...")
                    print(f"Expected: {correct_answer[:60]}")

                    result = self.test_prompt(test_prompt, correct_answer, reasoning_type)

                    test_result = LogicalReasoningResult(
                        reasoning_type=reasoning_type,
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

    def print_summary(self, results: List[LogicalReasoningResult]):
        """Print summary of logical reasoning test results."""

        print("\n" + "="*80)
        print("LOGICAL REASONING TEST SUMMARY")
        print("="*80)

        total_tests = len(results)
        correct = sum(1 for r in results if r.correct)

        print(f"\nTotal Tests:  {total_tests}")
        print(f"Correct:      {correct} ({correct/total_tests*100:.1f}%)")
        print(f"Incorrect:    {total_tests - correct} ({(total_tests-correct)/total_tests*100:.1f}%)")

        # Error type breakdown
        from collections import Counter
        errors = Counter(r.error_type for r in results if r.error_type != "none")
        if errors:
            print("\nError Types:")
            for error, count in errors.most_common():
                print(f"  {error}: {count}")

        # By reasoning type
        print("\n" + "-"*80)
        print("Results by Reasoning Type:")
        print("-"*80)

        by_type = {}
        for r in results:
            if r.reasoning_type not in by_type:
                by_type[r.reasoning_type] = {"total": 0, "correct": 0}
            by_type[r.reasoning_type]["total"] += 1
            if r.correct:
                by_type[r.reasoning_type]["correct"] += 1

        for rtype, stats in sorted(by_type.items()):
            acc = stats["correct"] / stats["total"] * 100
            status = "⚠️ WEAK" if acc < 60 else "✓ CAPABLE"
            print(f"{status} {rtype}: {stats['correct']}/{stats['total']} correct ({acc:.1f}%)")

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
            print("⚠️  HIGH-SEVERITY REASONING FAILURES")
            print("="*80)
            for r in failures[:5]:
                print(f"\n[{r.severity.upper()}] {r.reasoning_type} / {r.difficulty} → {r.error_type}")
                print(f"Problem: {r.prompt[:100]}...")
                print(f"Response: {r.response[:100]}...")
                print(f"Why: {r.explanation[:150]}...")

    def save_results(self, results: List[LogicalReasoningResult],
                     filename: str = "logical_reasoning_test_results.json"):
        """Save results to JSON file."""
        data = [
            {
                "reasoning_type": r.reasoning_type,
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

    tester = LogicalReasoningTester(API_KEY)

    results = tester.run_comprehensive_test(
        reasoning_types=["deductive", "fallacy_detection", "constraint_logic"],
        difficulty_levels=["simple", "intermediate", "complex"],
        prompts_per_combination=2
    )

    tester.print_summary(results)
    tester.save_results(results)
