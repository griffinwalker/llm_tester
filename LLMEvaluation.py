import anthropic
import json
from typing import List, Dict, Optional, Tuple
import time
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class EvaluationCriteria:
    """Represents a single evaluation criterion."""
    name: str
    description: str
    scale: str  # e.g., "1-5" or "pass/fail"
    weight: float = 1.0

@dataclass
class EvaluationResult:
    """Represents the result of evaluating a single response."""
    prompt: str
    response: str
    criteria_scores: Dict[str, float]
    overall_score: float
    feedback: str
    timestamp: str
    metadata: Optional[Dict] = None

class LLMEvaluator:
    """
    A comprehensive framework for using one LLM to evaluate another LLM's outputs.
    Supports AI-generated evaluation criteria and detailed analysis.
    """
    
    def __init__(self, api_key: str, evaluator_model: str = "claude-sonnet-4-20250514", 
                 target_model: str = "claude-sonnet-4-20250514"):
        """
        Initialize the evaluator.
        
        Args:
            api_key: Anthropic API key
            evaluator_model: Model to use for evaluation
            target_model: Model being evaluated
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.evaluator_model = evaluator_model
        self.target_model = target_model
        self.criteria_cache: Dict[str, List[EvaluationCriteria]] = {}
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain markdown formatting."""
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        json_match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        return text.strip()
    
    def generate_evaluation_criteria(self, task_description: str, 
                                     num_criteria: int = 5,
                                     focus_areas: Optional[List[str]] = None) -> List[EvaluationCriteria]:
        """
        Use an LLM to generate evaluation criteria for a specific task.
        
        Args:
            task_description: Description of the task/domain to evaluate
            num_criteria: Number of criteria to generate
            focus_areas: Optional specific areas to focus on (e.g., ["accuracy", "safety"])
            
        Returns:
            List of EvaluationCriteria objects
        """
        focus_text = ""
        if focus_areas:
            focus_text = f"\nFocus especially on these areas: {', '.join(focus_areas)}"
        
        prompt = f"""Generate {num_criteria} specific, measurable evaluation criteria for assessing LLM responses to the following task:

Task Description: {task_description}{focus_text}

For each criterion, provide:
1. A clear, concise name (2-4 words)
2. A detailed description of what to evaluate
3. The measurement scale (use 1-5 where 1=poor, 5=excellent)
4. A weight/importance factor (0.5 to 2.0, where 1.0 is standard importance)

Return a JSON array of objects with fields: name, description, scale, weight

Example format:
[
  {{
    "name": "Technical Accuracy",
    "description": "Evaluates whether the response contains factually correct information with no errors or misconceptions",
    "scale": "1-5",
    "weight": 1.5
  }}
]

Return ONLY the JSON array, no other text."""

        try:
            response = self.client.messages.create(
                model=self.evaluator_model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            cleaned = self._extract_json(content)
            criteria_data = json.loads(cleaned)
            
            criteria = [EvaluationCriteria(**c) for c in criteria_data]
            
            # Cache the criteria
            cache_key = f"{task_description}_{num_criteria}"
            self.criteria_cache[cache_key] = criteria
            
            return criteria
            
        except Exception as e:
            print(f"Error generating criteria: {e}")
            # Return default criteria as fallback
            return self._get_default_criteria()
    
    def generate_test_prompts(self, task_description: str, 
                             num_prompts: int = 5,
                             difficulty_levels: Optional[List[str]] = None) -> List[str]:
        """
        Generate test prompts using an LLM for a given task.
        
        Args:
            task_description: Description of the task domain
            num_prompts: Number of prompts to generate
            difficulty_levels: Optional list of difficulty levels (e.g., ["easy", "medium", "hard"])
            
        Returns:
            List of generated prompts
        """
        difficulty_text = ""
        if difficulty_levels:
            difficulty_text = f"\nInclude a mix of difficulty levels: {', '.join(difficulty_levels)}"
        
        prompt = f"""Generate {num_prompts} diverse test prompts for evaluating an LLM's ability in the following task:

Task Description: {task_description}{difficulty_text}

Requirements:
- Make prompts realistic and varied
- Cover different aspects of the task
- Range from simple to complex scenarios
- Ensure prompts are clear and specific
- Avoid repetitive or overly similar prompts

Return ONLY a JSON array of strings, no other text.

Example format:
["prompt 1", "prompt 2", "prompt 3"]"""

        try:
            response = self.client.messages.create(
                model=self.evaluator_model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            cleaned = self._extract_json(content)
            prompts = json.loads(cleaned)
            
            print(f"Generated {len(prompts)} test prompts:")
            for i, p in enumerate(prompts, 1):
                print(f"  {i}. {p[:80]}...")
            
            return prompts
            
        except Exception as e:
            print(f"Error generating prompts: {e}")
            # Return fallback prompts
            return [
                f"Provide a detailed explanation of a key concept in {task_description}",
                f"Give a practical example related to {task_description}",
                f"Explain the main challenges in {task_description}",
                f"Compare two different approaches in {task_description}",
                f"What are best practices for {task_description}?"
            ][:num_prompts]
    
    def _get_default_criteria(self) -> List[EvaluationCriteria]:
        """Return default evaluation criteria."""
        return [
            EvaluationCriteria(
                name="Accuracy",
                description="Factual correctness and freedom from errors",
                scale="1-5",
                weight=1.5
            ),
            EvaluationCriteria(
                name="Coherence",
                description="Logical flow and organization of ideas",
                scale="1-5",
                weight=1.0
            ),
            EvaluationCriteria(
                name="Helpfulness",
                description="Practical value and usefulness of the response",
                scale="1-5",
                weight=1.2
            ),
            EvaluationCriteria(
                name="Completeness",
                description="Adequately addresses all aspects of the prompt",
                scale="1-5",
                weight=1.0
            ),
            EvaluationCriteria(
                name="Clarity",
                description="Easy to understand with clear language",
                scale="1-5",
                weight=0.8
            )
        ]
    
    def get_llm_response(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Get a response from the target LLM.
        
        Args:
            prompt: The prompt to send
            model: Optional model override
            
        Returns:
            The LLM's response
        """
        model_to_use = model or self.target_model
        
        try:
            response = self.client.messages.create(
                model=model_to_use,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error getting response: {e}"
    
    def evaluate_response(self, prompt: str, response: str, 
                         criteria: List[EvaluationCriteria],
                         context: Optional[str] = None) -> EvaluationResult:
        """
        Evaluate a single response against specified criteria.
        
        Args:
            prompt: Original prompt
            response: Response to evaluate
            criteria: List of evaluation criteria
            context: Optional additional context for evaluation
            
        Returns:
            EvaluationResult object
        """
        criteria_descriptions = "\n".join([
            f"- {c.name} (weight: {c.weight}): {c.description} (Scale: {c.scale})"
            for c in criteria
        ])
        
        context_text = f"\n\nAdditional Context: {context}" if context else ""
        
        eval_prompt = f"""Evaluate the following LLM response based on these criteria:

{criteria_descriptions}

ORIGINAL PROMPT:
{prompt}

RESPONSE TO EVALUATE:
{response}{context_text}

Provide your evaluation as a JSON object with:
- "scores": object mapping each criterion name to its score
- "overall_score": weighted average score (1-5)
- "feedback": detailed text explaining the evaluation
- "strengths": list of 2-3 key strengths
- "weaknesses": list of 2-3 areas for improvement

Return ONLY valid JSON."""

        try:
            eval_response = self.client.messages.create(
                model=self.evaluator_model,
                max_tokens=2000,
                messages=[{"role": "user", "content": eval_prompt}]
            )
            
            content = eval_response.content[0].text
            cleaned = self._extract_json(content)
            eval_data = json.loads(cleaned)
            
            result = EvaluationResult(
                prompt=prompt,
                response=response,
                criteria_scores=eval_data.get("scores", {}),
                overall_score=eval_data.get("overall_score", 0),
                feedback=eval_data.get("feedback", ""),
                timestamp=datetime.now().isoformat(),
                metadata={
                    "strengths": eval_data.get("strengths", []),
                    "weaknesses": eval_data.get("weaknesses", [])
                }
            )
            
            return result
            
        except Exception as e:
            print(f"Error evaluating response: {e}")
            return EvaluationResult(
                prompt=prompt,
                response=response,
                criteria_scores={},
                overall_score=0,
                feedback=f"Evaluation failed: {e}",
                timestamp=datetime.now().isoformat()
            )
    
    def batch_evaluate(self, prompts: List[str], 
                      task_description: str,
                      custom_criteria: Optional[List[EvaluationCriteria]] = None,
                      generate_criteria: bool = True,
                      delay: float = 1.0) -> List[EvaluationResult]:
        """
        Evaluate multiple prompts in batch.
        
        Args:
            prompts: List of prompts to evaluate
            task_description: Description of the task for criteria generation
            custom_criteria: Optional pre-defined criteria
            generate_criteria: Whether to auto-generate criteria
            delay: Delay between API calls
            
        Returns:
            List of EvaluationResult objects
        """
        # Get or generate criteria
        if custom_criteria:
            criteria = custom_criteria
        elif generate_criteria:
            print(f"Generating evaluation criteria for: {task_description}")
            criteria = self.generate_evaluation_criteria(task_description)
            print(f"Generated {len(criteria)} criteria")
            for c in criteria:
                print(f"  - {c.name}: {c.description}")
        else:
            criteria = self._get_default_criteria()
        
        results = []
        for i, prompt in enumerate(prompts, 1):
            print(f"\n{'='*80}")
            print(f"Evaluating prompt {i}/{len(prompts)}")
            print(f"{'='*80}")
            print(f"Prompt: {prompt[:100]}...")
            
            # Get response from target LLM
            print("Getting response from target LLM...")
            response = self.get_llm_response(prompt)
            print(f"Response: {response[:150]}...")
            
            # Evaluate the response
            print("Evaluating response...")
            result = self.evaluate_response(prompt, response, criteria)
            results.append(result)
            
            print(f"Overall Score: {result.overall_score:.2f}/5.0")
            
            time.sleep(delay)
        
        return results
    
    def compare_models(self, prompts: List[str], 
                      models: List[str],
                      task_description: str) -> Dict[str, List[EvaluationResult]]:
        """
        Compare multiple models on the same set of prompts.
        
        Args:
            prompts: List of prompts to test
            models: List of model identifiers to compare
            task_description: Description for criteria generation
            
        Returns:
            Dictionary mapping model names to their results
        """
        # Generate criteria once for all models
        criteria = self.generate_evaluation_criteria(task_description)
        
        all_results = {}
        
        for model in models:
            print(f"\n{'#'*80}")
            print(f"EVALUATING MODEL: {model}")
            print(f"{'#'*80}")
            
            model_results = []
            for i, prompt in enumerate(prompts, 1):
                print(f"\nPrompt {i}/{len(prompts)}: {prompt[:80]}...")
                
                # Get response from this specific model
                response = self.get_llm_response(prompt, model=model)
                
                # Evaluate
                result = self.evaluate_response(prompt, response, criteria)
                model_results.append(result)
                
                print(f"Score: {result.overall_score:.2f}/5.0")
                
                time.sleep(1)
            
            all_results[model] = model_results
        
        return all_results
    
    def generate_comparison_report(self, comparison_results: Dict[str, List[EvaluationResult]]) -> str:
        """
        Generate a text report comparing model performance.
        
        Args:
            comparison_results: Results from compare_models()
            
        Returns:
            Formatted report as string
        """
        report = []
        report.append("="*80)
        report.append("MODEL COMPARISON REPORT")
        report.append("="*80)
        report.append("")
        
        # Calculate aggregate scores for each model
        model_stats = {}
        for model, results in comparison_results.items():
            scores = [r.overall_score for r in results]
            model_stats[model] = {
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "num_evaluations": len(scores)
            }
        
        # Sort models by average score
        sorted_models = sorted(model_stats.items(), key=lambda x: x[1]["avg_score"], reverse=True)
        
        report.append("OVERALL RANKINGS:")
        report.append("-" * 80)
        for rank, (model, stats) in enumerate(sorted_models, 1):
            report.append(f"{rank}. {model}")
            report.append(f"   Average Score: {stats['avg_score']:.2f}/5.0")
            report.append(f"   Range: {stats['min_score']:.2f} - {stats['max_score']:.2f}")
            report.append(f"   Evaluations: {stats['num_evaluations']}")
            report.append("")
        
        # Detailed breakdown by prompt
        report.append("")
        report.append("DETAILED BREAKDOWN BY PROMPT:")
        report.append("-" * 80)
        
        num_prompts = len(next(iter(comparison_results.values())))
        for i in range(num_prompts):
            report.append(f"\nPrompt {i+1}:")
            prompt = next(iter(comparison_results.values()))[i].prompt
            report.append(f"  {prompt[:100]}...")
            report.append("")
            
            for model in sorted_models:
                model_name = model[0]
                result = comparison_results[model_name][i]
                report.append(f"  {model_name}: {result.overall_score:.2f}/5.0")
            report.append("")
        
        return "\n".join(report)
    
    def save_results(self, results: List[EvaluationResult], filename: str):
        """Save evaluation results to JSON file."""
        data = [asdict(r) for r in results]
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {filename}")
    
    def print_summary(self, results: List[EvaluationResult]):
        """Print a summary of evaluation results."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        if not results:
            print("No results to display")
            return
        
        scores = [r.overall_score for r in results]
        avg_score = sum(scores) / len(scores)
        
        print(f"\nTotal Evaluations: {len(results)}")
        print(f"Average Score: {avg_score:.2f}/5.0")
        print(f"Score Range: {min(scores):.2f} - {max(scores):.2f}")
        
        # Show distribution
        excellent = sum(1 for s in scores if s >= 4.5)
        good = sum(1 for s in scores if 3.5 <= s < 4.5)
        acceptable = sum(1 for s in scores if 2.5 <= s < 3.5)
        poor = sum(1 for s in scores if s < 2.5)
        
        print("\nScore Distribution:")
        print(f"  Excellent (4.5-5.0): {excellent} ({excellent/len(scores)*100:.1f}%)")
        print(f"  Good (3.5-4.5):      {good} ({good/len(scores)*100:.1f}%)")
        print(f"  Acceptable (2.5-3.5): {acceptable} ({acceptable/len(scores)*100:.1f}%)")
        print(f"  Poor (<2.5):         {poor} ({poor/len(scores)*100:.1f}%)")
        
        # Show top and bottom results
        sorted_results = sorted(results, key=lambda r: r.overall_score, reverse=True)
        
        print("\n" + "-"*80)
        print("TOP 3 RESPONSES:")
        print("-"*80)
        for i, result in enumerate(sorted_results[:3], 1):
            print(f"\n{i}. Score: {result.overall_score:.2f}/5.0")
            print(f"   Prompt: {result.prompt[:80]}...")
            print(f"   Response: {result.response[:100]}...")
            if result.metadata and "strengths" in result.metadata:
                print(f"   Strengths: {', '.join(result.metadata['strengths'][:2])}")
        
        print("\n" + "-"*80)
        print("BOTTOM 3 RESPONSES:")
        print("-"*80)
        for i, result in enumerate(sorted_results[-3:], 1):
            print(f"\n{i}. Score: {result.overall_score:.2f}/5.0")
            print(f"   Prompt: {result.prompt[:80]}...")
            print(f"   Response: {result.response[:100]}...")
            if result.metadata and "weaknesses" in result.metadata:
                print(f"   Weaknesses: {', '.join(result.metadata['weaknesses'][:2])}")


# Example usage
if __name__ == "__main__":
    # Get API key from environment
    API_KEY = os.getenv("ANTHROPIC_API_KEY")
    if not API_KEY:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        exit(1)
    
    # Initialize evaluator
    evaluator = LLMEvaluator(API_KEY)
    
    # ========== EXAMPLE 1: Basic Evaluation ==========
    print("="*80)
    print("EXAMPLE 1: BASIC EVALUATION WITH AUTO-GENERATED CRITERIA")
    print("="*80)
    
    task = "Explaining complex technical concepts to beginners"
    
    # Generate test prompts using AI
    print("\nGenerating test prompts using AI...")
    test_prompts = evaluator.generate_test_prompts(
        task_description=task,
        num_prompts=5,
        difficulty_levels=["beginner", "intermediate", "advanced"]
    )
    
    print("\n" + "-"*80)
    
    results = evaluator.batch_evaluate(
        prompts=test_prompts,
        task_description=task,
        generate_criteria=True
    )
    
    evaluator.print_summary(results)
    evaluator.save_results(results, "evaluation_results.json")
    
    # ========== EXAMPLE 2: Custom Criteria ==========
    print("\n\n" + "="*80)
    print("EXAMPLE 2: EVALUATION WITH CUSTOM CRITERIA")
    print("="*80)
    
    custom_criteria = [
        EvaluationCriteria(
            name="Technical Accuracy",
            description="Uses correct terminology and concepts",
            scale="1-5",
            weight=2.0
        ),
        EvaluationCriteria(
            name="Beginner Friendliness",
            description="Avoids jargon and uses relatable analogies",
            scale="1-5",
            weight=1.5
        ),
        EvaluationCriteria(
            name="Engagement",
            description="Interesting and keeps reader's attention",
            scale="1-5",
            weight=1.0
        )
    ]
    
    # Generate new prompts for this example too
    print("\nGenerating test prompts for custom criteria evaluation...")
    custom_test_prompts = evaluator.generate_test_prompts(
        task_description=task,
        num_prompts=5
    )
    
    print("\n" + "-"*80)
    
    custom_results = evaluator.batch_evaluate(
        prompts=custom_test_prompts,
        task_description=task,
        custom_criteria=custom_criteria,
        generate_criteria=False
    )
    
    evaluator.print_summary(custom_results)
    
    # ========== EXAMPLE 3: Model Comparison (commented out) ==========
    """
    print("\n\n" + "="*80)
    print("EXAMPLE 3: COMPARING MULTIPLE MODELS")
    print("="*80)
    
    models_to_compare = [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514"
    ]
    
    comparison = evaluator.compare_models(
        prompts=test_prompts[:2],
        models=models_to_compare,
        task_description=task
    )
    
    report = evaluator.generate_comparison_report(comparison)
    print("\n" + report)
    
    with open("model_comparison_report.txt", "w") as f:
        f.write(report)
    """
