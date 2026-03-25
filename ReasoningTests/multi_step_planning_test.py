import anthropic
import json
from typing import List, Dict
import time
import os
import re
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PlanningResult:
    """Result of a multi-step planning test."""
    planning_type: str
    domain: str
    prompt: str
    response: str
    plan_valid: bool
    failure_mode: str  # "missing_step", "wrong_order", "infeasible_step", "goal_not_met", "none"
    completeness_score: int  # 0-100
    explanation: str
    timestamp: str

class MultiStepPlanningTester:
    """
    Tests LLM capability at multi-step planning: breaking goals into ordered
    steps, respecting dependencies and constraints, allocating resources, and
    verifying that plans actually achieve the stated goal. Tests both forward
    planning (given goal, produce steps) and plan evaluation (given a plan,
    find flaws).
    """

    PLANNING_TYPES = {
        "forward_planning":     "Given a goal and starting state, generate a valid step-by-step plan",
        "plan_critique":        "Identify flaws, missing steps, or ordering errors in a given plan",
        "constraint_planning":  "Plan under explicit resource, time, or ordering constraints",
        "contingency_planning": "Plan that includes branches for likely failure cases",
        "replanning":           "Adapt an existing plan after an unexpected mid-execution change",
        "dependency_ordering":  "Order tasks correctly given a set of dependency relationships",
        "resource_allocation":  "Assign limited resources to tasks to best achieve the goal",
    }

    DOMAINS = [
        "software_development",
        "project_management",
        "everyday_logistics",
        "scientific_experiment",
        "emergency_response",
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

    def generate_test_prompts(self, planning_type: str,
                              domain: str,
                              num_prompts: int = 3) -> List[Dict]:
        """Generate planning problems with evaluation criteria."""

        type_desc = self.PLANNING_TYPES.get(planning_type, "General planning")

        prompt = f"""Generate {num_prompts} planning problems in the "{domain}" domain testing "{planning_type}".

Planning type: {type_desc}
Domain: {domain}

Each problem should:
1. Present a clear goal or scenario in the "{domain}" domain
2. Include enough context for the model to generate or evaluate a plan
3. Be designed to test the specific planning capability:
   - "forward_planning": state a goal and any constraints, ask for a plan
   - "plan_critique": provide a plan with 1-2 subtle flaws, ask the model to find them
   - "constraint_planning": include real constraints (budget, time, ordering) that must be respected
   - "contingency_planning": describe a goal where at least one step has a realistic failure mode
   - "replanning": provide a plan in-progress, then describe an unexpected event requiring adaptation
   - "dependency_ordering": give a list of tasks with dependency rules, ask for a valid execution order
   - "resource_allocation": describe tasks and limited resources, ask for an optimal allocation
4. Have a clear evaluation criterion (what makes a plan valid/complete)

For "plan_critique" specifically: include the flawed plan in the prompt and describe what the correct plan should look like in the correct_answer.

Return ONLY a JSON array of objects with "prompt" and "correct_answer" keys.
"correct_answer" should describe what a good response includes (key steps, identified flaws, etc.).

Return format: [{{"prompt": "...", "correct_answer": "..."}}]"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2500,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text
            cleaned = self._extract_json(content)
            problems = json.loads(cleaned)
            return problems
        except Exception as e:
            print(f"Error generating prompts: {e}")
            return [{"prompt": f"Test {planning_type} in {domain}", "correct_answer": "unknown"}]

    def evaluate_plan(self, prompt: str, response: str,
                      correct_answer: str, planning_type: str) -> Dict:
        """Evaluate whether the model produced or assessed a valid plan."""

        eval_prompt = f"""Evaluate this LLM's multi-step planning response.

PLANNING TYPE: {planning_type} ({self.PLANNING_TYPES.get(planning_type, '')})

PROBLEM:
{prompt}

WHAT A GOOD RESPONSE SHOULD INCLUDE:
{correct_answer}

LLM RESPONSE:
{response}

Assess:
1. Does the plan cover all necessary steps to achieve the stated goal?
2. Are the steps in a valid, dependency-respecting order?
3. Are all steps actually feasible given the stated constraints?
4. For "plan_critique": did the model correctly identify the flaws in the plan?
5. For "contingency_planning": did the model include fallback paths?
6. Assign a completeness score 0-100 (100 = fully addresses all criteria).

Failure modes:
- "missing_step": one or more critical steps are absent from the plan
- "wrong_order": steps are present but in an invalid dependency order
- "infeasible_step": a step cannot be executed given the stated constraints
- "goal_not_met": all steps are plausible but they don't actually achieve the goal
- "none": plan is valid and complete

Return a JSON object with:
- "plan_valid": true if the plan is complete and achieves the goal (score >= 75)
- "failure_mode": "missing_step"/"wrong_order"/"infeasible_step"/"goal_not_met"/"none"
- "completeness_score": integer 0-100
- "explanation": specific assessment of plan quality

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
                "plan_valid": False,
                "failure_mode": "none",
                "completeness_score": 0,
                "explanation": f"Evaluation error: {e}"
            }

    def test_prompt(self, test_prompt: str, correct_answer: str, planning_type: str) -> Dict:
        """Send a planning problem and evaluate the response."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": test_prompt}]
            )
            llm_response = response.content[0].text
        except Exception as e:
            llm_response = f"Error: {e}"

        time.sleep(1)
        evaluation = self.evaluate_plan(test_prompt, llm_response, correct_answer, planning_type)

        return {
            "response": llm_response,
            "plan_valid": evaluation.get("plan_valid", False),
            "failure_mode": evaluation.get("failure_mode", "none"),
            "completeness_score": evaluation.get("completeness_score", 0),
            "explanation": evaluation.get("explanation", "")
        }

    def run_comprehensive_test(self,
                               planning_types: List[str] = None,
                               domains: List[str] = None,
                               prompts_per_combination: int = 2) -> List[PlanningResult]:
        """Run comprehensive multi-step planning tests."""

        if planning_types is None:
            planning_types = ["forward_planning", "plan_critique", "constraint_planning"]
        if domains is None:
            domains = ["software_development", "project_management", "everyday_logistics"]

        results = []
        total_tests = len(planning_types) * len(domains) * prompts_per_combination

        print("="*80)
        print("MULTI-STEP PLANNING CAPABILITY TEST")
        print("="*80)
        print(f"\nTesting {len(planning_types)} planning types × {len(domains)} domains")
        print(f"Total tests: {total_tests}\n")

        test_num = 0
        for planning_type in planning_types:
            for domain in domains:
                print(f"\n{'='*80}")
                print(f"PLANNING TYPE: {planning_type} | DOMAIN: {domain}")
                print(f"{'='*80}\n")

                print(f"Generating {prompts_per_combination} problems...")
                problems = self.generate_test_prompts(planning_type, domain, prompts_per_combination)

                for problem in problems:
                    test_num += 1
                    test_prompt = problem.get("prompt", "")
                    correct_answer = problem.get("correct_answer", "")

                    print(f"\nTest {test_num}/{total_tests}")
                    print(f"Problem: {test_prompt[:100]}...")

                    result = self.test_prompt(test_prompt, correct_answer, planning_type)

                    test_result = PlanningResult(
                        planning_type=planning_type,
                        domain=domain,
                        prompt=test_prompt,
                        response=result["response"],
                        plan_valid=result["plan_valid"],
                        failure_mode=result["failure_mode"],
                        completeness_score=result["completeness_score"],
                        explanation=result["explanation"],
                        timestamp=datetime.now().isoformat()
                    )
                    results.append(test_result)

                    score = result["completeness_score"]
                    if result["plan_valid"]:
                        status = f"✓ VALID PLAN (score: {score})"
                    else:
                        status = f"❌ FLAWED PLAN (score: {score}, mode: {result['failure_mode']})"

                    print(f"Result: {status}")
                    print(f"Explanation: {result['explanation'][:100]}...")

                    time.sleep(1)

        return results

    def print_summary(self, results: List[PlanningResult]):
        """Print summary of multi-step planning test results."""

        print("\n" + "="*80)
        print("MULTI-STEP PLANNING TEST SUMMARY")
        print("="*80)

        total_tests = len(results)
        valid_plans = sum(1 for r in results if r.plan_valid)
        avg_score = sum(r.completeness_score for r in results) / total_tests if total_tests else 0

        print(f"\nTotal Tests:           {total_tests}")
        print(f"Valid Plans:           {valid_plans} ({valid_plans/total_tests*100:.1f}%)")
        print(f"Flawed Plans:          {total_tests - valid_plans} ({(total_tests-valid_plans)/total_tests*100:.1f}%)")
        print(f"Average Completeness:  {avg_score:.1f}/100")

        # Failure modes
        from collections import Counter
        modes = Counter(r.failure_mode for r in results if r.failure_mode != "none")
        if modes:
            print("\nFailure Modes:")
            for mode, count in modes.most_common():
                print(f"  {mode}: {count}")

        # By planning type
        print("\n" + "-"*80)
        print("Results by Planning Type:")
        print("-"*80)

        by_type = {}
        for r in results:
            if r.planning_type not in by_type:
                by_type[r.planning_type] = {"total": 0, "valid": 0, "scores": []}
            by_type[r.planning_type]["total"] += 1
            by_type[r.planning_type]["scores"].append(r.completeness_score)
            if r.plan_valid:
                by_type[r.planning_type]["valid"] += 1

        for ptype, stats in sorted(by_type.items()):
            valid_rate = stats["valid"] / stats["total"] * 100
            avg = sum(stats["scores"]) / len(stats["scores"])
            status = "⚠️ WEAK" if valid_rate < 60 else "✓ CAPABLE"
            print(f"{status} {ptype}: {stats['valid']}/{stats['total']} valid, avg score {avg:.0f}")

        # By domain
        print("\n" + "-"*80)
        print("Results by Domain:")
        print("-"*80)

        by_domain = {}
        for r in results:
            if r.domain not in by_domain:
                by_domain[r.domain] = {"total": 0, "valid": 0}
            by_domain[r.domain]["total"] += 1
            if r.plan_valid:
                by_domain[r.domain]["valid"] += 1

        for domain, stats in sorted(by_domain.items()):
            valid_rate = stats["valid"] / stats["total"] * 100
            status = "⚠️ WEAK" if valid_rate < 60 else "✓ CAPABLE"
            print(f"{status} {domain}: {stats['valid']}/{stats['total']} valid ({valid_rate:.1f}%)")

        # Worst plans
        worst = sorted(results, key=lambda r: r.completeness_score)[:5]
        if worst and worst[0].completeness_score < 60:
            print("\n" + "="*80)
            print("⚠️  LOWEST QUALITY PLANS")
            print("="*80)
            for r in worst:
                if r.completeness_score < 60:
                    print(f"\n[Score: {r.completeness_score}] {r.planning_type} / {r.domain} → {r.failure_mode}")
                    print(f"Problem: {r.prompt[:100]}...")
                    print(f"Why: {r.explanation[:150]}...")

    def save_results(self, results: List[PlanningResult],
                     filename: str = "multi_step_planning_test_results.json"):
        """Save results to JSON file."""
        data = [
            {
                "planning_type": r.planning_type,
                "domain": r.domain,
                "prompt": r.prompt,
                "response": r.response,
                "plan_valid": r.plan_valid,
                "failure_mode": r.failure_mode,
                "completeness_score": r.completeness_score,
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

    tester = MultiStepPlanningTester(API_KEY)

    results = tester.run_comprehensive_test(
        planning_types=["forward_planning", "plan_critique", "constraint_planning"],
        domains=["software_development", "project_management", "everyday_logistics"],
        prompts_per_combination=2
    )

    tester.print_summary(results)
    tester.save_results(results)
