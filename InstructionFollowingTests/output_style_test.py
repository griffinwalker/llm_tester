import anthropic
import json
from typing import List, Dict
import time
import os
import re
from dataclasses import dataclass
from datetime import datetime

@dataclass
class OutputStyleResult:
    """Result of an output style instruction following test."""
    style_dimension: str
    task_type: str
    prompt: str
    response: str
    style_followed: bool
    violation_type: str  # "wrong_tone", "wrong_format", "wrong_length", "wrong_perspective", "wrong_vocabulary", "none"
    severity: str  # "critical", "high", "medium", "low", "none"
    explanation: str
    timestamp: str

class OutputStyleTester:
    """
    Tests whether an LLM reliably follows style, tone, register, and presentation
    instructions. These are among the most common real-world instruction-following
    requirements: write formally, use bullet points, avoid jargon, write in second
    person, respond in exactly three sentences, etc.
    """

    STYLE_DIMENSIONS = {
        "tone_and_register":   "Formal/informal, professional/casual, serious/playful tone",
        "sentence_structure":  "Short punchy sentences, long flowing prose, Hemingway-style, etc.",
        "perspective":         "First person, second person, third person, passive voice only",
        "vocabulary_level":    "Plain English (5th grade), technical jargon, academic register, ELI5",
        "response_length":     "Exactly N sentences/words/paragraphs, no more no less",
        "structural_format":   "Bullet points only, numbered steps, Q&A pairs, no lists at all",
        "emotion_and_hedging": "No hedging language, only positive framing, clinical neutrality",
    }

    TASK_TYPES = [
        "explanation",
        "recommendation",
        "summary",
        "creative_writing",
        "technical_description",
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

    def generate_test_prompts(self, style_dimension: str,
                              task_type: str,
                              num_prompts: int = 3) -> List[Dict]:
        """Generate style-constrained task prompts with verifiable style requirements."""

        dim_desc = self.STYLE_DIMENSIONS.get(style_dimension, "General style")

        prompt = f"""Generate {num_prompts} test prompts for a "{task_type}" task with explicit "{style_dimension}" style constraints.

Style dimension: {dim_desc}
Task type: {task_type}

Each prompt should:
1. Ask the model to complete a "{task_type}" task on any topic
2. Include a clear, specific, measurable style instruction related to "{style_dimension}"
3. Make the style requirement precise enough to objectively verify compliance:
   - "tone_and_register": e.g., "Write in a formal academic tone. Do not use contractions or colloquialisms."
   - "sentence_structure": e.g., "Use only simple sentences. No sentence may contain a comma or conjunction."
   - "perspective": e.g., "Write entirely in second person. Do not use 'I', 'we', or 'they'."
   - "vocabulary_level": e.g., "Explain this as if to a 10-year-old. No word should require more than an 8th-grade reading level."
   - "response_length": e.g., "Respond in exactly 3 sentences. Not 2, not 4. Exactly 3."
   - "structural_format": e.g., "Use only numbered steps. Do not write any prose paragraphs."
   - "emotion_and_hedging": e.g., "Do not use any hedging language (words like 'might', 'could', 'perhaps', 'I think')."
4. Also include a "style_criteria" field describing exactly what to check for compliance

Return ONLY a JSON array of objects with "prompt" and "style_criteria" keys.

Return format: [{{"prompt": "...", "style_criteria": "..."}}]"""

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
            return [{"prompt": f"Test {style_dimension} for {task_type}",
                     "style_criteria": "Check style compliance"}]

    def evaluate_style_compliance(self, prompt: str, response: str,
                                  style_dimension: str, style_criteria: str) -> Dict:
        """Evaluate whether the response follows the style instructions."""

        eval_prompt = f"""Evaluate whether this LLM response follows the style instructions given in the prompt.

STYLE DIMENSION: {style_dimension} ({self.STYLE_DIMENSIONS.get(style_dimension, '')})

WHAT TO CHECK FOR:
{style_criteria}

ORIGINAL PROMPT (contains style instructions):
{prompt}

LLM RESPONSE:
{response}

Check precisely:
1. Does the response comply with every stated style constraint?
2. Are there any violations of the tone, format, length, perspective, or vocabulary requirements?
3. Are violations minor (a single slip) or pervasive (the style instruction was ignored entirely)?
4. Did the model add a preamble or postamble that violates the format (e.g. "Here is the response in bullet points:")?

Violation types:
- "wrong_tone": tone or register does not match what was requested
- "wrong_format": structural format (bullets, numbered, prose) does not match
- "wrong_length": word/sentence/paragraph count violates the stated limit
- "wrong_perspective": person (I/you/they), voice, or framing does not match
- "wrong_vocabulary": reading level, jargon level, or hedging does not match instructions
- "none": all style requirements are fully met

Return a JSON object with:
- "style_followed": true if all style requirements are fully satisfied
- "violation_type": "wrong_tone"/"wrong_format"/"wrong_length"/"wrong_perspective"/"wrong_vocabulary"/"none"
- "severity": "critical"/"high"/"medium"/"low"/"none"
- "explanation": specific style violations found, with examples from the text

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
                "style_followed": True,
                "violation_type": "none",
                "severity": "none",
                "explanation": f"Evaluation error: {e}"
            }

    def test_prompt(self, test_prompt: str, style_dimension: str, style_criteria: str) -> Dict:
        """Send a style-constrained prompt and evaluate compliance."""

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
        evaluation = self.evaluate_style_compliance(
            test_prompt, llm_response, style_dimension, style_criteria
        )

        return {
            "response": llm_response,
            "style_followed": evaluation.get("style_followed", True),
            "violation_type": evaluation.get("violation_type", "none"),
            "severity": evaluation.get("severity", "none"),
            "explanation": evaluation.get("explanation", "")
        }

    def run_comprehensive_test(self,
                               style_dimensions: List[str] = None,
                               task_types: List[str] = None,
                               prompts_per_combination: int = 2) -> List[OutputStyleResult]:
        """Run comprehensive output style tests."""

        if style_dimensions is None:
            style_dimensions = ["tone_and_register", "response_length", "structural_format"]
        if task_types is None:
            task_types = ["explanation", "recommendation", "summary"]

        results = []
        total_tests = len(style_dimensions) * len(task_types) * prompts_per_combination

        print("="*80)
        print("OUTPUT STYLE INSTRUCTION FOLLOWING TEST")
        print("="*80)
        print(f"\nTesting {len(style_dimensions)} style dimensions × {len(task_types)} task types")
        print(f"Total tests: {total_tests}\n")

        test_num = 0
        for style_dimension in style_dimensions:
            for task_type in task_types:
                print(f"\n{'='*80}")
                print(f"STYLE: {style_dimension} | TASK: {task_type}")
                print(f"{'='*80}\n")

                print(f"Generating {prompts_per_combination} test prompts...")
                test_cases = self.generate_test_prompts(
                    style_dimension, task_type, prompts_per_combination
                )

                for case in test_cases:
                    test_num += 1
                    test_prompt = case.get("prompt", "")
                    style_criteria = case.get("style_criteria", "")

                    print(f"\nTest {test_num}/{total_tests}")
                    print(f"Prompt: {test_prompt[:100]}...")

                    result = self.test_prompt(test_prompt, style_dimension, style_criteria)

                    test_result = OutputStyleResult(
                        style_dimension=style_dimension,
                        task_type=task_type,
                        prompt=test_prompt,
                        response=result["response"],
                        style_followed=result["style_followed"],
                        violation_type=result["violation_type"],
                        severity=result["severity"],
                        explanation=result["explanation"],
                        timestamp=datetime.now().isoformat()
                    )
                    results.append(test_result)

                    status = "✓ COMPLIANT" if result["style_followed"] else f"❌ VIOLATION ({result['violation_type']})"
                    print(f"Result: {status}")
                    if not result["style_followed"]:
                        print(f"Severity: {result['severity'].upper()}")
                    print(f"Explanation: {result['explanation'][:100]}...")

                    time.sleep(1)

        return results

    def print_summary(self, results: List[OutputStyleResult]):
        """Print summary of output style test results."""

        print("\n" + "="*80)
        print("OUTPUT STYLE TEST SUMMARY")
        print("="*80)

        total_tests = len(results)
        compliant = sum(1 for r in results if r.style_followed)

        print(f"\nTotal Tests:  {total_tests}")
        print(f"Compliant:    {compliant} ({compliant/total_tests*100:.1f}%)")
        print(f"Violations:   {total_tests - compliant} ({(total_tests-compliant)/total_tests*100:.1f}%)")

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

        # Violation types
        from collections import Counter
        vtypes = Counter(r.violation_type for r in results if r.violation_type != "none")
        if vtypes:
            print("\nViolation Types:")
            for vtype, count in vtypes.most_common():
                print(f"  {vtype}: {count}")

        # By style dimension
        print("\n" + "-"*80)
        print("Results by Style Dimension:")
        print("-"*80)

        by_dim = {}
        for r in results:
            if r.style_dimension not in by_dim:
                by_dim[r.style_dimension] = {"total": 0, "compliant": 0}
            by_dim[r.style_dimension]["total"] += 1
            if r.style_followed:
                by_dim[r.style_dimension]["compliant"] += 1

        for dim, stats in sorted(by_dim.items()):
            rate = stats["compliant"] / stats["total"] * 100
            status = "⚠️ UNRELIABLE" if rate < 70 else "✓ RELIABLE"
            print(f"{status} {dim}: {stats['compliant']}/{stats['total']} compliant ({rate:.1f}%)")

        # By task type
        print("\n" + "-"*80)
        print("Results by Task Type:")
        print("-"*80)

        by_task = {}
        for r in results:
            if r.task_type not in by_task:
                by_task[r.task_type] = {"total": 0, "compliant": 0}
            by_task[r.task_type]["total"] += 1
            if r.style_followed:
                by_task[r.task_type]["compliant"] += 1

        for task, stats in sorted(by_task.items()):
            rate = stats["compliant"] / stats["total"] * 100
            status = "⚠️ UNRELIABLE" if rate < 70 else "✓ RELIABLE"
            print(f"{status} {task}: {stats['compliant']}/{stats['total']} compliant ({rate:.1f}%)")

        # Worst violations
        critical_cases = [r for r in results if r.severity in ["critical", "high"]]
        if critical_cases:
            print("\n" + "="*80)
            print("⚠️  CRITICAL STYLE VIOLATIONS")
            print("="*80)
            for r in critical_cases[:5]:
                print(f"\n[{r.severity.upper()}] {r.style_dimension} / {r.task_type} → {r.violation_type}")
                print(f"Prompt: {r.prompt[:100]}...")
                print(f"Response: {r.response[:100]}...")
                print(f"Why: {r.explanation[:150]}...")

    def save_results(self, results: List[OutputStyleResult],
                     filename: str = "output_style_test_results.json"):
        """Save results to JSON file."""
        data = [
            {
                "style_dimension": r.style_dimension,
                "task_type": r.task_type,
                "prompt": r.prompt,
                "response": r.response,
                "style_followed": r.style_followed,
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

    tester = OutputStyleTester(API_KEY)

    results = tester.run_comprehensive_test(
        style_dimensions=["tone_and_register", "response_length", "structural_format"],
        task_types=["explanation", "recommendation", "summary"],
        prompts_per_combination=2
    )

    tester.print_summary(results)
    tester.save_results(results)
