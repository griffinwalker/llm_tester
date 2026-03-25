# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This project is a Python-based framework for testing LLM safety, alignment, and quality. It uses one LLM (typically Claude) to generate test cases and evaluate another LLM's responses across multiple safety and performance dimensions.

## Running Scripts

There is no build system or test runner. Scripts are run directly:

```bash
# Ensure API key is set
export ANTHROPIC_API_KEY='your-api-key'

# Main frameworks
python LLMTestSafety.py
python LLMEvaluation.py

# Specialized safety/alignment tests
python SafetyAlignmentTests/jailbreak_test.py
python SafetyAlignmentTests/toxicity_test.py
python SafetyAlignmentTests/prompt_injection_test.py
python SafetyAlignmentTests/bias_detection_test.py
python SafetyAlignmentTests/misinformation_test.py
python SafetyAlignmentTests/privacy_violation_test.py
```

Results are saved as JSON files (e.g., `safety_test_results.json`, `evaluation_results.json`) in the project root.

```bash
# Robustness & Reliability tests
python RobustnessTests/adversarial_input_test.py
python RobustnessTests/consistency_test.py
python RobustnessTests/instruction_following_test.py
python RobustnessTests/context_stress_test.py
python RobustnessTests/graceful_degradation_test.py
python RobustnessTests/knowledge_boundary_test.py
```

## Architecture

**Two-LLM pattern**: a Generator LLM creates diverse test variations, then an Evaluator LLM scores the target LLM's responses against criteria.

**Root-level frameworks:**
- `LLMTestSafety.py` — `LLMTester` class with predefined safety criteria sets (general, privacy_and_security, harmful_content, misinformation, bias_and_fairness). Generates test variations, queries target, evaluates responses.
- `LLMEvaluation.py` — `LLMEvaluator` class with AI-generated criteria; supports batch evaluation and multi-model comparison. Uses `EvaluationCriteria` and `EvaluationResult` dataclasses.

**`SafetyAlignmentTests/` specialized testers** (all share a pattern: define attack categories/techniques → generate prompts → evaluate severity):
- `jailbreak_test.py` — 8 jailbreak techniques × 6 harm categories
- `toxicity_test.py` — 7 toxicity types × 5 prompt categories
- `prompt_injection_test.py` — 6 injection techniques, evaluates severity (high/medium/low/none)
- `bias_detection_test.py` — 7 bias categories using paired scenarios (identical except for protected attribute) to detect differential treatment
- `misinformation_test.py` — 7 misinformation categories × 5 prompt strategies
- `privacy_violation_test.py` — 6 attack types testing for leakage of 10 sensitive data types
- `llm_test_framework.py` — shared base utilities (JSON extraction, API interaction)

**`RobustnessTests/` reliability testers** (same pattern as SafetyAlignmentTests):
- `adversarial_input_test.py` — 7 input types (unicode abuse, special chars, extreme length, etc.) × 5 topic categories; evaluates graceful handling vs crash/garbled/hallucination
- `consistency_test.py` — 7 variation types (rephrasing, negation, leading phrasing, etc.) × 5 topics; sends prompt pairs and evaluates substantive consistency
- `instruction_following_test.py` — 7 instruction types (multi-step, negative constraints, contradictions, etc.) × 5 domains; scores compliance 0-100
- `context_stress_test.py` — 7 stress types (needle-in-haystack, long doc retrieval, contradictory context, etc.) × 5 domains; evaluates correct context use
- `graceful_degradation_test.py` — 7 scenario types (impossible task, missing info, ambiguous request, etc.) × 5 domains; checks for false confidence vs graceful acknowledgment
- `knowledge_boundary_test.py` — 7 boundary types (post-cutoff events, obscure entities, false premises, etc.) × 5 domains; evaluates epistemic calibration and hallucination

**Key implementation details:**
- Hard-coded model: `claude-sonnet-4-20250514` across all modules
- 1-second delay between API calls for rate limiting
- JSON extraction utilities parse structured responses from the evaluator LLM
- Each specialized tester has an `if __name__ == "__main__"` block with runnable examples
