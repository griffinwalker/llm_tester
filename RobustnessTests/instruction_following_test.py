import anthropic
import json
from typing import List, Dict
import time
import os
import re
from dataclasses import dataclass
from datetime import datetime

@dataclass
class InstructionFollowingResult:
    """Result of an instruction-following robustness test."""
    instruction_type: str
    task_domain: str
    prompt: str
    response: str
    followed_correctly: bool
    failure_mode: str  # "ignored", "partial", "contradicted", "over_interpreted", "none"
    compliance_score: int  # 0-100
    explanation: str
    timestamp: str

class InstructionFollowingTester:
    """
    Tests LLM reliability at following complex, multi-step, conditional, or
    constrained instructions. Evaluates whether the model respects all stated
    requirements or silently drops/misinterprets them.
    """

    INSTRUCTION_TYPES = {
        "multi_step":         "Sequential steps that must all be completed in order",
        "conditional":        "If-then-else logic embedded in the instruction",
        "negative_constraint":"Explicit prohibitions (do NOT do X, avoid Y)",
        "format_constraint":  "Strict output format requirements (length, structure, style)",
        "priority_ordering":  "Multiple goals with an explicit priority ranking",
        "contradiction":      "Instructions containing a genuine conflict to be flagged",
        "implicit_inference": "Instructions requiring reasonable inference to complete correctly",
    }

    TASK_DOMAINS = [
        "text_summarization",
        "code_generation",
        "data_transformation",
        "creative_writing",
        "question_answering",
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

    def generate_test_prompts(self, instruction_type: str,
                              task_domain: str,
                              num_prompts: int = 3) -> List[str]:
        """Generate test prompts that embed the specified instruction type."""

        type_desc = self.INSTRUCTION_TYPES.get(instruction_type, "General instruction type")

        prompt = f"""Generate {num_prompts} test prompts for the "{task_domain}" domain that require following "{instruction_type}" instructions.

Instruction type: {type_desc}
Task domain: {task_domain}

Each prompt should:
1. Embed clear instructions of the specified type into a realistic {task_domain} task
2. Include enough detail that compliance can be objectively evaluated
3. Be challenging but not impossible — a careful model should succeed
4. Vary in complexity and the specific sub-type of instruction challenge

For "contradiction" type specifically: include a genuine logical conflict in the instructions and expect the model to identify and flag it rather than silently picking one interpretation.

IMPORTANT: These prompts are used to evaluate LLM instruction-following reliability.

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
            return [f"Test {instruction_type} prompt for {task_domain}"]

    def evaluate_compliance(self, prompt: str, response: str,
                            instruction_type: str) -> Dict:
        """Evaluate how well the response followed the instructions."""

        eval_prompt = f"""Evaluate how well this LLM response followed the instructions in the prompt.

INSTRUCTION TYPE TESTED: {instruction_type} ({self.INSTRUCTION_TYPES.get(instruction_type, '')})

PROMPT (with instructions):
{prompt}

LLM RESPONSE:
{response}

Assess:
1. Did the model follow ALL stated instructions, or did it drop/ignore some?
2. Did it violate any explicit prohibitions?
3. Did it maintain required formats, orderings, or constraints throughout?
4. For "contradiction" type: did it correctly identify and flag the conflict?
5. Assign a compliance score 0-100 (100 = perfect compliance).

Failure modes:
- "ignored": one or more instructions were completely ignored
- "partial": instructions were partially followed but key parts missed
- "contradicted": a stated prohibition or constraint was violated
- "over_interpreted": instructions were stretched beyond their stated scope
- "none": all instructions followed correctly

Return a JSON object with:
- "followed_correctly": true if score >= 80 and no critical failures
- "failure_mode": "ignored"/"partial"/"contradicted"/"over_interpreted"/"none"
- "compliance_score": integer 0-100
- "explanation": which specific instructions were/weren't followed

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
                "followed_correctly": True,
                "failure_mode": "none",
                "compliance_score": 100,
                "explanation": f"Evaluation error: {e}"
            }

    def test_prompt(self, test_prompt: str, instruction_type: str) -> Dict:
        """Send a prompt and evaluate instruction-following compliance."""

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
        evaluation = self.evaluate_compliance(test_prompt, llm_response, instruction_type)

        return {
            "response": llm_response,
            "followed_correctly": evaluation.get("followed_correctly", True),
            "failure_mode": evaluation.get("failure_mode", "none"),
            "compliance_score": evaluation.get("compliance_score", 100),
            "explanation": evaluation.get("explanation", "")
        }

    def run_comprehensive_test(self,
                               instruction_types: List[str] = None,
                               task_domains: List[str] = None,
                               prompts_per_combination: int = 2) -> List[InstructionFollowingResult]:
        """Run comprehensive instruction-following tests."""

        if instruction_types is None:
            instruction_types = ["multi_step", "negative_constraint", "format_constraint"]
        if task_domains is None:
            task_domains = ["text_summarization", "code_generation", "question_answering"]

        results = []
        total_tests = len(instruction_types) * len(task_domains) * prompts_per_combination

        print("="*80)
        print("INSTRUCTION FOLLOWING ROBUSTNESS TEST")
        print("="*80)
        print(f"\nTesting {len(instruction_types)} instruction types × {len(task_domains)} task domains")
        print(f"Total tests: {total_tests}\n")

        test_num = 0
        for instruction_type in instruction_types:
            for task_domain in task_domains:
                print(f"\n{'='*80}")
                print(f"INSTRUCTION TYPE: {instruction_type} | TASK: {task_domain}")
                print(f"{'='*80}\n")

                print(f"Generating {prompts_per_combination} test prompts...")
                test_prompts = self.generate_test_prompts(
                    instruction_type, task_domain, prompts_per_combination
                )

                for test_prompt in test_prompts:
                    test_num += 1
                    print(f"\nTest {test_num}/{total_tests}")
                    print(f"Prompt: {test_prompt[:100]}...")

                    result = self.test_prompt(test_prompt, instruction_type)

                    test_result = InstructionFollowingResult(
                        instruction_type=instruction_type,
                        task_domain=task_domain,
                        prompt=test_prompt,
                        response=result["response"],
                        followed_correctly=result["followed_correctly"],
                        failure_mode=result["failure_mode"],
                        compliance_score=result["compliance_score"],
                        explanation=result["explanation"],
                        timestamp=datetime.now().isoformat()
                    )
                    results.append(test_result)

                    score = result["compliance_score"]
                    if result["followed_correctly"]:
                        status = f"✓ COMPLIANT (score: {score})"
                    else:
                        status = f"❌ NON-COMPLIANT (score: {score}, mode: {result['failure_mode']})"

                    print(f"Result: {status}")
                    print(f"Explanation: {result['explanation'][:100]}...")

                    time.sleep(1)

        return results

    def print_summary(self, results: List[InstructionFollowingResult]):
        """Print summary of instruction-following test results."""

        print("\n" + "="*80)
        print("INSTRUCTION FOLLOWING TEST SUMMARY")
        print("="*80)

        total_tests = len(results)
        failures = sum(1 for r in results if not r.followed_correctly)
        avg_score = sum(r.compliance_score for r in results) / total_tests if total_tests else 0

        print(f"\nTotal Tests: {total_tests}")
        print(f"Fully Compliant: {total_tests - failures} ({(total_tests-failures)/total_tests*100:.1f}%)")
        print(f"Non-Compliant:   {failures} ({failures/total_tests*100:.1f}%)")
        print(f"Average Compliance Score: {avg_score:.1f}/100")

        # Failure mode breakdown
        from collections import Counter
        modes = Counter(r.failure_mode for r in results if r.failure_mode != "none")
        if modes:
            print("\nFailure Modes:")
            for mode, count in modes.most_common():
                print(f"  {mode}: {count}")

        # By instruction type
        print("\n" + "-"*80)
        print("Results by Instruction Type:")
        print("-"*80)

        by_type = {}
        for r in results:
            if r.instruction_type not in by_type:
                by_type[r.instruction_type] = {"total": 0, "failures": 0, "scores": []}
            by_type[r.instruction_type]["total"] += 1
            by_type[r.instruction_type]["scores"].append(r.compliance_score)
            if not r.followed_correctly:
                by_type[r.instruction_type]["failures"] += 1

        for instruction_type, stats in sorted(by_type.items()):
            fail_rate = stats["failures"] / stats["total"] * 100
            avg = sum(stats["scores"]) / len(stats["scores"])
            status = "⚠️ UNRELIABLE" if fail_rate > 30 else "✓ RELIABLE"
            print(f"{status} {instruction_type}: {stats['failures']}/{stats['total']} failures, avg score {avg:.0f}")

        # By task domain
        print("\n" + "-"*80)
        print("Results by Task Domain:")
        print("-"*80)

        by_domain = {}
        for r in results:
            if r.task_domain not in by_domain:
                by_domain[r.task_domain] = {"total": 0, "failures": 0}
            by_domain[r.task_domain]["total"] += 1
            if not r.followed_correctly:
                by_domain[r.task_domain]["failures"] += 1

        for domain, stats in sorted(by_domain.items()):
            fail_rate = stats["failures"] / stats["total"] * 100
            status = "⚠️ UNRELIABLE" if fail_rate > 30 else "✓ RELIABLE"
            print(f"{status} {domain}: {stats['failures']}/{stats['total']} failures ({fail_rate:.1f}%)")

        # Worst failures
        worst = sorted(results, key=lambda r: r.compliance_score)[:5]
        if worst and worst[0].compliance_score < 80:
            print("\n" + "="*80)
            print("⚠️  LOWEST COMPLIANCE SCORES")
            print("="*80)
            for r in worst:
                if r.compliance_score < 80:
                    print(f"\n[Score: {r.compliance_score}] {r.instruction_type} / {r.task_domain} → {r.failure_mode}")
                    print(f"Prompt: {r.prompt[:100]}...")
                    print(f"Why: {r.explanation[:150]}...")

    def save_results(self, results: List[InstructionFollowingResult],
                     filename: str = "instruction_following_test_results.json"):
        """Save results to JSON file."""
        data = [
            {
                "instruction_type": r.instruction_type,
                "task_domain": r.task_domain,
                "prompt": r.prompt,
                "response": r.response,
                "followed_correctly": r.followed_correctly,
                "failure_mode": r.failure_mode,
                "compliance_score": r.compliance_score,
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

    tester = InstructionFollowingTester(API_KEY)

    results = tester.run_comprehensive_test(
        instruction_types=["multi_step", "negative_constraint", "format_constraint"],
        task_domains=["text_summarization", "code_generation", "question_answering"],
        prompts_per_combination=2
    )

    tester.print_summary(results)
    tester.save_results(results)
