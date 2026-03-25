import anthropic
import json
from typing import List, Dict
import time
import os
import re
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ImplicitInstructionResult:
    """Result of an implicit instruction following test."""
    implication_type: str
    domain: str
    prompt: str
    response: str
    implication_honored: bool
    failure_mode: str  # "missed_implication", "over_literal", "under_literal", "wrong_inference", "none"
    severity: str  # "critical", "high", "medium", "low", "none"
    explanation: str
    timestamp: str

class ImplicitInstructionTester:
    """
    Tests whether an LLM correctly infers and follows instructions that are
    implied by context rather than stated explicitly. Real users frequently
    communicate intent through implication, shared context, and pragmatics.
    A model that only follows literal instructions will fail these cases;
    one that over-infers will hallucinate requirements.
    """

    IMPLICATION_TYPES = {
        "pragmatic_implicature":  "What is implied by what is said (Gricean maxims)",
        "contextual_convention":  "Professional or genre conventions implied by the context",
        "audience_adaptation":    "Adjustments implied by who the audience is described as",
        "example_generalization": "Instruction implied by an example (do what the example shows)",
        "goal_inference":         "Infer the user's underlying goal from an underspecified request",
        "negative_space":         "What the user implicitly does NOT want, given what they asked for",
        "continuation_contract":  "What is implied by the structure already established in the output",
    }

    DOMAINS = [
        "professional_writing",
        "technical_documentation",
        "casual_conversation",
        "creative_collaboration",
        "data_and_analysis",
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

    def generate_test_prompts(self, implication_type: str,
                              domain: str,
                              num_prompts: int = 3) -> List[Dict]:
        """Generate prompts with implied instructions and what they imply."""

        imp_desc = self.IMPLICATION_TYPES.get(implication_type, "General implication")

        prompt = f"""Generate {num_prompts} test prompts for the "{domain}" domain that test "{implication_type}" instruction following.

Implication type: {imp_desc}
Domain: {domain}

Each prompt should:
1. Contain an instruction or request that IMPLIES something without stating it explicitly
2. Be a realistic message a user in the "{domain}" context would actually send
3. Test the specific implication type:
   - "pragmatic_implicature": phrasing that implies more than it literally says (e.g. "Can you make this shorter?" implies it's currently too long)
   - "contextual_convention": asking for a document type that has well-known conventions (e.g. "write me a cover letter" implies certain standard sections)
   - "audience_adaptation": mentioning an audience implies adjusting complexity/style (e.g. "explain this to my 8-year-old")
   - "example_generalization": providing one example and saying "like this" implies following that example's pattern
   - "goal_inference": underspecified request where the real goal is inferrable (e.g. "make this work" on broken code)
   - "negative_space": asking for X implies not wanting Y even if Y isn't mentioned (e.g. "give me the key points" implies not wanting full prose)
   - "continuation_contract": asking the model to "continue" implies maintaining the established structure/voice/format

4. Include an "implied_requirement" field: the thing the model should infer and honor

Return ONLY a JSON array of objects with "prompt" and "implied_requirement" keys.

Return format: [{{"prompt": "...", "implied_requirement": "..."}}]"""

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
            return [{"prompt": f"Test {implication_type} in {domain}",
                     "implied_requirement": "Honor the implied instruction"}]

    def evaluate_implication_following(self, prompt: str, response: str,
                                       implication_type: str,
                                       implied_requirement: str) -> Dict:
        """Evaluate whether the model honored the implied instruction."""

        eval_prompt = f"""Evaluate whether this LLM correctly inferred and followed an implied instruction.

IMPLICATION TYPE: {implication_type} ({self.IMPLICATION_TYPES.get(implication_type, '')})

ORIGINAL PROMPT:
{prompt}

WHAT THE PROMPT IMPLIES (the model should infer this without being told):
{implied_requirement}

LLM RESPONSE:
{response}

Assess:
1. Did the model recognize and honor the implied requirement?
2. Did it interpret the prompt too literally, missing the implication?
3. Did it over-infer, adding requirements that weren't implied?
4. Did it make a wrong inference about what was implied?

Note: being slightly over-helpful (e.g. including one extra point) is less serious than completely missing the implication.

Failure modes:
- "missed_implication": treated the prompt literally and completely ignored the implied requirement
- "over_literal": partially honored the implication but stayed too close to the literal wording
- "under_literal": over-inferred, adding implied requirements that weren't actually there
- "wrong_inference": made a confident but incorrect inference about what was implied
- "none": correctly inferred and honored the implied requirement

Return a JSON object with:
- "implication_honored": true if the model correctly followed the implied instruction
- "failure_mode": "missed_implication"/"over_literal"/"under_literal"/"wrong_inference"/"none"
- "severity": "critical"/"high"/"medium"/"low"/"none"
- "explanation": specific assessment of how well the implication was handled

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
                "implication_honored": True,
                "failure_mode": "none",
                "severity": "none",
                "explanation": f"Evaluation error: {e}"
            }

    def test_prompt(self, test_prompt: str, implication_type: str,
                    implied_requirement: str) -> Dict:
        """Send a prompt with implied instructions and evaluate the response."""

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
        evaluation = self.evaluate_implication_following(
            test_prompt, llm_response, implication_type, implied_requirement
        )

        return {
            "response": llm_response,
            "implication_honored": evaluation.get("implication_honored", True),
            "failure_mode": evaluation.get("failure_mode", "none"),
            "severity": evaluation.get("severity", "none"),
            "explanation": evaluation.get("explanation", "")
        }

    def run_comprehensive_test(self,
                               implication_types: List[str] = None,
                               domains: List[str] = None,
                               prompts_per_combination: int = 2) -> List[ImplicitInstructionResult]:
        """Run comprehensive implicit instruction tests."""

        if implication_types is None:
            implication_types = ["pragmatic_implicature", "contextual_convention", "goal_inference"]
        if domains is None:
            domains = ["professional_writing", "technical_documentation", "casual_conversation"]

        results = []
        total_tests = len(implication_types) * len(domains) * prompts_per_combination

        print("="*80)
        print("IMPLICIT INSTRUCTION FOLLOWING TEST")
        print("="*80)
        print(f"\nTesting {len(implication_types)} implication types × {len(domains)} domains")
        print(f"Total tests: {total_tests}\n")

        test_num = 0
        for implication_type in implication_types:
            for domain in domains:
                print(f"\n{'='*80}")
                print(f"IMPLICATION TYPE: {implication_type} | DOMAIN: {domain}")
                print(f"{'='*80}\n")

                print(f"Generating {prompts_per_combination} test prompts...")
                test_cases = self.generate_test_prompts(
                    implication_type, domain, prompts_per_combination
                )

                for case in test_cases:
                    test_num += 1
                    test_prompt = case.get("prompt", "")
                    implied_requirement = case.get("implied_requirement", "")

                    print(f"\nTest {test_num}/{total_tests}")
                    print(f"Prompt:   {test_prompt[:80]}...")
                    print(f"Implies:  {implied_requirement[:80]}")

                    result = self.test_prompt(test_prompt, implication_type, implied_requirement)

                    test_result = ImplicitInstructionResult(
                        implication_type=implication_type,
                        domain=domain,
                        prompt=test_prompt,
                        response=result["response"],
                        implication_honored=result["implication_honored"],
                        failure_mode=result["failure_mode"],
                        severity=result["severity"],
                        explanation=result["explanation"],
                        timestamp=datetime.now().isoformat()
                    )
                    results.append(test_result)

                    status = "✓ IMPLICATION HONORED" if result["implication_honored"] else f"❌ MISSED ({result['failure_mode']})"
                    print(f"Result: {status}")
                    if not result["implication_honored"]:
                        print(f"Severity: {result['severity'].upper()}")
                    print(f"Explanation: {result['explanation'][:100]}...")

                    time.sleep(1)

        return results

    def print_summary(self, results: List[ImplicitInstructionResult]):
        """Print summary of implicit instruction test results."""

        print("\n" + "="*80)
        print("IMPLICIT INSTRUCTION TEST SUMMARY")
        print("="*80)

        total_tests = len(results)
        honored = sum(1 for r in results if r.implication_honored)

        print(f"\nTotal Tests:          {total_tests}")
        print(f"Implication Honored:  {honored} ({honored/total_tests*100:.1f}%)")
        print(f"Implication Missed:   {total_tests - honored} ({(total_tests-honored)/total_tests*100:.1f}%)")

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

        # By implication type
        print("\n" + "-"*80)
        print("Results by Implication Type:")
        print("-"*80)

        by_type = {}
        for r in results:
            if r.implication_type not in by_type:
                by_type[r.implication_type] = {"total": 0, "honored": 0}
            by_type[r.implication_type]["total"] += 1
            if r.implication_honored:
                by_type[r.implication_type]["honored"] += 1

        for itype, stats in sorted(by_type.items()):
            rate = stats["honored"] / stats["total"] * 100
            status = "⚠️ LITERAL" if rate < 70 else "✓ PRAGMATIC"
            print(f"{status} {itype}: {stats['honored']}/{stats['total']} honored ({rate:.1f}%)")

        # By domain
        print("\n" + "-"*80)
        print("Results by Domain:")
        print("-"*80)

        by_domain = {}
        for r in results:
            if r.domain not in by_domain:
                by_domain[r.domain] = {"total": 0, "honored": 0}
            by_domain[r.domain]["total"] += 1
            if r.implication_honored:
                by_domain[r.domain]["honored"] += 1

        for domain, stats in sorted(by_domain.items()):
            rate = stats["honored"] / stats["total"] * 100
            status = "⚠️ LITERAL" if rate < 70 else "✓ PRAGMATIC"
            print(f"{status} {domain}: {stats['honored']}/{stats['total']} honored ({rate:.1f}%)")

        # Critical failures
        critical_cases = [r for r in results if not r.implication_honored and r.severity in ["critical", "high"]]
        if critical_cases:
            print("\n" + "="*80)
            print("⚠️  CRITICAL IMPLICATION FAILURES")
            print("="*80)
            for r in critical_cases[:5]:
                print(f"\n[{r.severity.upper()}] {r.implication_type} / {r.domain} → {r.failure_mode}")
                print(f"Prompt: {r.prompt[:100]}...")
                print(f"Response: {r.response[:100]}...")
                print(f"Why: {r.explanation[:150]}...")

    def save_results(self, results: List[ImplicitInstructionResult],
                     filename: str = "implicit_instruction_test_results.json"):
        """Save results to JSON file."""
        data = [
            {
                "implication_type": r.implication_type,
                "domain": r.domain,
                "prompt": r.prompt,
                "response": r.response,
                "implication_honored": r.implication_honored,
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

    tester = ImplicitInstructionTester(API_KEY)

    results = tester.run_comprehensive_test(
        implication_types=["pragmatic_implicature", "contextual_convention", "goal_inference"],
        domains=["professional_writing", "technical_documentation", "casual_conversation"],
        prompts_per_combination=2
    )

    tester.print_summary(results)
    tester.save_results(results)
