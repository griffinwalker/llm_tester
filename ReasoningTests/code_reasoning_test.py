import anthropic
import json
from typing import List, Dict
import time
import os
import re
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CodeReasoningResult:
    """Result of a code reasoning test."""
    reasoning_type: str
    language: str
    prompt: str
    response: str
    correct: bool
    error_type: str  # "trace_error", "bug_missed", "complexity_wrong", "logic_error", "none"
    severity: str  # "critical", "high", "medium", "low", "none"
    explanation: str
    timestamp: str

class CodeReasoningTester:
    """
    Tests LLM capability at reasoning about code: tracing execution, identifying
    bugs, assessing algorithmic complexity, explaining what code does, predicting
    output, and detecting security vulnerabilities. Evaluates whether the model
    reasons about code correctly rather than just pattern-matching to familiar idioms.
    """

    REASONING_TYPES = {
        "execution_trace":      "Trace through code step-by-step to determine the exact output",
        "bug_identification":   "Find one or more bugs in code that produces wrong or no output",
        "complexity_analysis":  "Determine the time or space complexity of an algorithm",
        "code_explanation":     "Explain what a non-trivial piece of code does and why",
        "output_prediction":    "Predict the exact output of code given specific inputs",
        "security_audit":       "Identify security vulnerabilities (injection, overflow, race condition, etc.)",
        "refactor_reasoning":   "Determine whether a proposed refactor preserves correctness",
    }

    LANGUAGES = [
        "python",
        "javascript",
        "pseudocode",
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
                              language: str,
                              num_prompts: int = 3) -> List[Dict]:
        """Generate code reasoning problems with verifiable answers."""

        type_desc = self.REASONING_TYPES.get(reasoning_type, "General code reasoning")

        prompt = f"""Generate {num_prompts} code reasoning problems in "{language}" testing "{reasoning_type}".

Reasoning type: {type_desc}
Language: {language}

Each problem should:
1. Include a complete, self-contained code snippet in "{language}"
2. Have an unambiguous correct answer that can be objectively verified
3. Be designed to require actual reasoning rather than surface pattern matching:
   - "execution_trace": code with loops, conditionals, or recursion where the trace is non-obvious
   - "bug_identification": code with a subtle bug (off-by-one, wrong operator, edge case missed)
   - "complexity_analysis": algorithm whose complexity requires analysis (not trivially O(n))
   - "code_explanation": code that uses non-obvious logic or an unintuitive approach
   - "output_prediction": code with side effects, scope issues, or non-obvious evaluation order
   - "security_audit": code with a realistic security flaw (SQL injection, buffer issue, TOCTOU, etc.)
   - "refactor_reasoning": show original and refactored code, ask if they are equivalent
4. Keep code snippets concise (10-30 lines) but substantive enough to require careful reading

For "security_audit": include the vulnerable code and ask what the vulnerability is and how to fix it.

Return ONLY a JSON array of objects with "prompt" and "correct_answer" keys.

Return format: [{{"prompt": "...", "correct_answer": "..."}}]"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text
            cleaned = self._extract_json(content)
            problems = json.loads(cleaned)
            return problems
        except Exception as e:
            print(f"Error generating prompts: {e}")
            return [{"prompt": f"Test {reasoning_type} in {language}", "correct_answer": "unknown"}]

    def evaluate_response(self, prompt: str, response: str,
                          correct_answer: str, reasoning_type: str) -> Dict:
        """Evaluate the model's code reasoning accuracy."""

        eval_prompt = f"""Evaluate this LLM's code reasoning response.

REASONING TYPE: {reasoning_type} ({self.REASONING_TYPES.get(reasoning_type, '')})

PROBLEM (includes code):
{prompt}

CORRECT ANSWER:
{correct_answer}

LLM RESPONSE:
{response}

Assess:
1. Did the model reach the correct answer or identify all key points?
2. For "execution_trace" / "output_prediction": is the traced output exactly correct?
3. For "bug_identification": did it find the actual bug (not a false positive)?
4. For "complexity_analysis": is the Big-O correct and well-justified?
5. For "security_audit": did it identify the right vulnerability and a valid fix?
6. For "refactor_reasoning": correctly identified equivalence or non-equivalence?

Error types:
- "trace_error": incorrect execution trace or wrong predicted output
- "bug_missed": failed to find the actual bug, or reported wrong location
- "complexity_wrong": incorrect Big-O or wrong justification
- "logic_error": fundamental misunderstanding of what the code does
- "none": code reasoning is correct

Return a JSON object with:
- "correct": true if the answer matches the correct answer
- "error_type": "trace_error"/"bug_missed"/"complexity_wrong"/"logic_error"/"none"
- "severity": "critical"/"high"/"medium"/"low"/"none"
- "explanation": specific assessment of the code reasoning

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

    def test_prompt(self, test_prompt: str, correct_answer: str, reasoning_type: str) -> Dict:
        """Send a code reasoning problem and evaluate the response."""

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
                               languages: List[str] = None,
                               prompts_per_combination: int = 2) -> List[CodeReasoningResult]:
        """Run comprehensive code reasoning tests."""

        if reasoning_types is None:
            reasoning_types = ["execution_trace", "bug_identification", "complexity_analysis"]
        if languages is None:
            languages = ["python", "javascript", "pseudocode"]

        results = []
        total_tests = len(reasoning_types) * len(languages) * prompts_per_combination

        print("="*80)
        print("CODE REASONING CAPABILITY TEST")
        print("="*80)
        print(f"\nTesting {len(reasoning_types)} reasoning types × {len(languages)} languages")
        print(f"Total tests: {total_tests}\n")

        test_num = 0
        for reasoning_type in reasoning_types:
            for language in languages:
                print(f"\n{'='*80}")
                print(f"REASONING TYPE: {reasoning_type} | LANGUAGE: {language}")
                print(f"{'='*80}\n")

                print(f"Generating {prompts_per_combination} problems...")
                problems = self.generate_test_prompts(reasoning_type, language, prompts_per_combination)

                for problem in problems:
                    test_num += 1
                    test_prompt = problem.get("prompt", "")
                    correct_answer = problem.get("correct_answer", "")

                    print(f"\nTest {test_num}/{total_tests}")
                    print(f"Problem: {test_prompt[:100]}...")
                    print(f"Expected: {correct_answer[:60]}")

                    result = self.test_prompt(test_prompt, correct_answer, reasoning_type)

                    test_result = CodeReasoningResult(
                        reasoning_type=reasoning_type,
                        language=language,
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

    def print_summary(self, results: List[CodeReasoningResult]):
        """Print summary of code reasoning test results."""

        print("\n" + "="*80)
        print("CODE REASONING TEST SUMMARY")
        print("="*80)

        total_tests = len(results)
        correct = sum(1 for r in results if r.correct)

        print(f"\nTotal Tests: {total_tests}")
        print(f"Correct:     {correct} ({correct/total_tests*100:.1f}%)")
        print(f"Incorrect:   {total_tests - correct} ({(total_tests-correct)/total_tests*100:.1f}%)")

        # Severity breakdown
        critical = sum(1 for r in results if r.severity == "critical")
        high     = sum(1 for r in results if r.severity == "high")
        medium   = sum(1 for r in results if r.severity == "medium")
        low      = sum(1 for r in results if r.severity == "low")

        print("\nSeverity of Errors:")
        print(f"  Critical: {critical}")
        print(f"  High:     {high}")
        print(f"  Medium:   {medium}")
        print(f"  Low:      {low}")
        print(f"  None:     {total_tests - critical - high - medium - low}")

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

        # By language
        print("\n" + "-"*80)
        print("Results by Language:")
        print("-"*80)

        by_lang = {}
        for r in results:
            if r.language not in by_lang:
                by_lang[r.language] = {"total": 0, "correct": 0}
            by_lang[r.language]["total"] += 1
            if r.correct:
                by_lang[r.language]["correct"] += 1

        for lang, stats in sorted(by_lang.items()):
            acc = stats["correct"] / stats["total"] * 100
            status = "⚠️ WEAK" if acc < 60 else "✓ CAPABLE"
            print(f"{status} {lang}: {stats['correct']}/{stats['total']} correct ({acc:.1f}%)")

        # Critical failures
        failures = [r for r in results if not r.correct and r.severity in ["critical", "high"]]
        if failures:
            print("\n" + "="*80)
            print("⚠️  HIGH-SEVERITY CODE REASONING FAILURES")
            print("="*80)
            for r in failures[:5]:
                print(f"\n[{r.severity.upper()}] {r.reasoning_type} / {r.language} → {r.error_type}")
                print(f"Problem: {r.prompt[:100]}...")
                print(f"Response: {r.response[:100]}...")
                print(f"Why: {r.explanation[:150]}...")

    def save_results(self, results: List[CodeReasoningResult],
                     filename: str = "code_reasoning_test_results.json"):
        """Save results to JSON file."""
        data = [
            {
                "reasoning_type": r.reasoning_type,
                "language": r.language,
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

    tester = CodeReasoningTester(API_KEY)

    results = tester.run_comprehensive_test(
        reasoning_types=["execution_trace", "bug_identification", "complexity_analysis"],
        languages=["python", "javascript", "pseudocode"],
        prompts_per_combination=2
    )

    tester.print_summary(results)
    tester.save_results(results)
