import anthropic
import json
from typing import List, Dict
import time
import os
import re
from dataclasses import dataclass
from datetime import datetime

@dataclass
class AnalogicalReasoningResult:
    """Result of an analogical reasoning test."""
    analogy_type: str
    source_domain: str
    prompt: str
    response: str
    correct: bool
    error_type: str  # "surface_mapping", "false_analogy", "incomplete_transfer", "missed_disanalogy", "none"
    quality_score: int  # 0-100
    explanation: str
    timestamp: str

class AnalogicalReasoningTester:
    """
    Tests LLM capability at analogical reasoning: completing analogies,
    transferring solutions across domains, identifying structural similarities,
    spotting false analogies used in arguments, and generating novel analogies
    to explain complex concepts. Analogical reasoning underpins explanation,
    problem transfer, and creative thinking.
    """

    ANALOGY_TYPES = {
        "completion":           "A:B :: C:? — complete the analogy with the best-fitting answer",
        "solution_transfer":    "Apply a solution strategy from one domain to an analogous problem in another",
        "structural_mapping":   "Identify which element in domain B corresponds to a given element in domain A",
        "false_analogy_detect": "Identify why a persuasive analogy breaks down or is misleading",
        "explanatory_analogy":  "Generate an analogy to explain a complex concept to a non-expert",
        "cross_domain_insight": "Use an analogy to generate a novel hypothesis or approach",
        "disanalogy_spotting":  "Given two things presented as analogous, identify the key differences",
    }

    SOURCE_DOMAINS = [
        "biology_and_nature",
        "engineering_and_machines",
        "social_systems",
        "mathematics_and_logic",
        "everyday_objects",
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

    def generate_test_prompts(self, analogy_type: str,
                              source_domain: str,
                              num_prompts: int = 3) -> List[Dict]:
        """Generate analogical reasoning problems with expected answers."""

        type_desc = self.ANALOGY_TYPES.get(analogy_type, "General analogy")

        prompt = f"""Generate {num_prompts} analogical reasoning problems sourced from the "{source_domain}" domain testing "{analogy_type}".

Analogy type: {type_desc}
Source domain: {source_domain}

Each problem should:
1. Draw primarily from the "{source_domain}" domain as the source of the analogy
2. Test the specific analogy capability:
   - "completion": classic A:B::C:? format, with a defensible best answer
   - "solution_transfer": present a solved problem in {source_domain}, then a structurally identical unsolved problem in a different domain
   - "structural_mapping": describe two systems and ask which element of one maps to a given element of the other
   - "false_analogy_detect": present an argument that uses a flawed or misleading analogy, ask to explain why it fails
   - "explanatory_analogy": ask the model to explain a specific complex concept using an analogy from {source_domain}
   - "cross_domain_insight": describe a principle from {source_domain} and ask what novel insight it might yield in another field
   - "disanalogy_spotting": present two things as analogous and ask where the analogy breaks down
3. Have a clear, defensible correct answer or set of key points

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
            return [{"prompt": f"Test {analogy_type} from {source_domain}", "correct_answer": "unknown"}]

    def evaluate_response(self, prompt: str, response: str,
                          correct_answer: str, analogy_type: str) -> Dict:
        """Evaluate the quality of the model's analogical reasoning."""

        eval_prompt = f"""Evaluate this LLM's analogical reasoning response.

ANALOGY TYPE: {analogy_type} ({self.ANALOGY_TYPES.get(analogy_type, '')})

PROBLEM:
{prompt}

EXPECTED ANSWER / KEY POINTS:
{correct_answer}

LLM RESPONSE:
{response}

Assess:
1. Did the model map the correct structural relationships (not just surface features)?
2. Did it transfer the solution strategy appropriately without distorting it?
3. For "false_analogy_detect" / "disanalogy_spotting": did it identify the right breakdown points?
4. For "explanatory_analogy": is the analogy accurate and genuinely illuminating?
5. Assign a quality score 0-100 based on depth, accuracy, and insight of the analogy work.

Error types:
- "surface_mapping": mapped superficial features instead of structural relationships
- "false_analogy": generated or accepted a fundamentally misleading analogy
- "incomplete_transfer": partially transferred a solution but missed key structural elements
- "missed_disanalogy": failed to identify where a comparison breaks down
- "none": analogical reasoning is accurate and insightful

Return a JSON object with:
- "correct": true if the response captures the key correct answer points (score >= 70)
- "error_type": "surface_mapping"/"false_analogy"/"incomplete_transfer"/"missed_disanalogy"/"none"
- "quality_score": integer 0-100
- "explanation": assessment of the analogical reasoning quality

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
                "quality_score": 0,
                "explanation": f"Evaluation error: {e}"
            }

    def test_prompt(self, test_prompt: str, correct_answer: str, analogy_type: str) -> Dict:
        """Send an analogical reasoning problem and evaluate the response."""

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
        evaluation = self.evaluate_response(test_prompt, llm_response, correct_answer, analogy_type)

        return {
            "response": llm_response,
            "correct": evaluation.get("correct", False),
            "error_type": evaluation.get("error_type", "none"),
            "quality_score": evaluation.get("quality_score", 0),
            "explanation": evaluation.get("explanation", "")
        }

    def run_comprehensive_test(self,
                               analogy_types: List[str] = None,
                               source_domains: List[str] = None,
                               prompts_per_combination: int = 2) -> List[AnalogicalReasoningResult]:
        """Run comprehensive analogical reasoning tests."""

        if analogy_types is None:
            analogy_types = ["completion", "solution_transfer", "false_analogy_detect"]
        if source_domains is None:
            source_domains = ["biology_and_nature", "engineering_and_machines", "social_systems"]

        results = []
        total_tests = len(analogy_types) * len(source_domains) * prompts_per_combination

        print("="*80)
        print("ANALOGICAL REASONING CAPABILITY TEST")
        print("="*80)
        print(f"\nTesting {len(analogy_types)} analogy types × {len(source_domains)} source domains")
        print(f"Total tests: {total_tests}\n")

        test_num = 0
        for analogy_type in analogy_types:
            for source_domain in source_domains:
                print(f"\n{'='*80}")
                print(f"ANALOGY TYPE: {analogy_type} | SOURCE DOMAIN: {source_domain}")
                print(f"{'='*80}\n")

                print(f"Generating {prompts_per_combination} problems...")
                problems = self.generate_test_prompts(analogy_type, source_domain, prompts_per_combination)

                for problem in problems:
                    test_num += 1
                    test_prompt = problem.get("prompt", "")
                    correct_answer = problem.get("correct_answer", "")

                    print(f"\nTest {test_num}/{total_tests}")
                    print(f"Problem: {test_prompt[:100]}...")
                    print(f"Expected: {correct_answer[:60]}")

                    result = self.test_prompt(test_prompt, correct_answer, analogy_type)

                    test_result = AnalogicalReasoningResult(
                        analogy_type=analogy_type,
                        source_domain=source_domain,
                        prompt=test_prompt,
                        response=result["response"],
                        correct=result["correct"],
                        error_type=result["error_type"],
                        quality_score=result["quality_score"],
                        explanation=result["explanation"],
                        timestamp=datetime.now().isoformat()
                    )
                    results.append(test_result)

                    score = result["quality_score"]
                    status = f"✓ CORRECT (score: {score})" if result["correct"] else f"❌ WRONG (score: {score}, error: {result['error_type']})"
                    print(f"Result: {status}")
                    print(f"Explanation: {result['explanation'][:100]}...")

                    time.sleep(1)

        return results

    def print_summary(self, results: List[AnalogicalReasoningResult]):
        """Print summary of analogical reasoning test results."""

        print("\n" + "="*80)
        print("ANALOGICAL REASONING TEST SUMMARY")
        print("="*80)

        total_tests = len(results)
        correct = sum(1 for r in results if r.correct)
        avg_score = sum(r.quality_score for r in results) / total_tests if total_tests else 0

        print(f"\nTotal Tests:    {total_tests}")
        print(f"Correct:        {correct} ({correct/total_tests*100:.1f}%)")
        print(f"Incorrect:      {total_tests - correct} ({(total_tests-correct)/total_tests*100:.1f}%)")
        print(f"Average Quality Score: {avg_score:.1f}/100")

        # Error type breakdown
        from collections import Counter
        errors = Counter(r.error_type for r in results if r.error_type != "none")
        if errors:
            print("\nError Types:")
            for error, count in errors.most_common():
                print(f"  {error}: {count}")

        # By analogy type
        print("\n" + "-"*80)
        print("Results by Analogy Type:")
        print("-"*80)

        by_type = {}
        for r in results:
            if r.analogy_type not in by_type:
                by_type[r.analogy_type] = {"total": 0, "correct": 0, "scores": []}
            by_type[r.analogy_type]["total"] += 1
            by_type[r.analogy_type]["scores"].append(r.quality_score)
            if r.correct:
                by_type[r.analogy_type]["correct"] += 1

        for atype, stats in sorted(by_type.items()):
            acc = stats["correct"] / stats["total"] * 100
            avg = sum(stats["scores"]) / len(stats["scores"])
            status = "⚠️ WEAK" if acc < 60 else "✓ CAPABLE"
            print(f"{status} {atype}: {stats['correct']}/{stats['total']} correct, avg score {avg:.0f}")

        # By source domain
        print("\n" + "-"*80)
        print("Results by Source Domain:")
        print("-"*80)

        by_domain = {}
        for r in results:
            if r.source_domain not in by_domain:
                by_domain[r.source_domain] = {"total": 0, "correct": 0}
            by_domain[r.source_domain]["total"] += 1
            if r.correct:
                by_domain[r.source_domain]["correct"] += 1

        for domain, stats in sorted(by_domain.items()):
            acc = stats["correct"] / stats["total"] * 100
            status = "⚠️ WEAK" if acc < 60 else "✓ CAPABLE"
            print(f"{status} {domain}: {stats['correct']}/{stats['total']} correct ({acc:.1f}%)")

        # Worst performers
        worst = sorted(results, key=lambda r: r.quality_score)[:5]
        if worst and worst[0].quality_score < 50:
            print("\n" + "="*80)
            print("⚠️  LOWEST QUALITY ANALOGIES")
            print("="*80)
            for r in worst:
                if r.quality_score < 50:
                    print(f"\n[Score: {r.quality_score}] {r.analogy_type} / {r.source_domain} → {r.error_type}")
                    print(f"Problem: {r.prompt[:100]}...")
                    print(f"Why: {r.explanation[:150]}...")

    def save_results(self, results: List[AnalogicalReasoningResult],
                     filename: str = "analogical_reasoning_test_results.json"):
        """Save results to JSON file."""
        data = [
            {
                "analogy_type": r.analogy_type,
                "source_domain": r.source_domain,
                "prompt": r.prompt,
                "response": r.response,
                "correct": r.correct,
                "error_type": r.error_type,
                "quality_score": r.quality_score,
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

    tester = AnalogicalReasoningTester(API_KEY)

    results = tester.run_comprehensive_test(
        analogy_types=["completion", "solution_transfer", "false_analogy_detect"],
        source_domains=["biology_and_nature", "engineering_and_machines", "social_systems"],
        prompts_per_combination=2
    )

    tester.print_summary(results)
    tester.save_results(results)
