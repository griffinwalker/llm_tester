import anthropic
import json
from typing import List, Dict
import time

class LLMTester:
    """
    A framework for using one LLM to generate test cases and evaluate another LLM's responses.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the tester with an Anthropic API key.
        
        Args:
            api_key: Your Anthropic API key
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"
    
    def generate_test_variations(self, base_topic: str, num_variations: int = 10) -> List[str]:
        """
        Generate multiple variations of questions about a topic.
        
        Args:
            base_topic: The topic or question to generate variations for
            num_variations: Number of variations to generate
            
        Returns:
            List of question variations
        """
        prompt = f"""Generate {num_variations} different ways to ask about or discuss the following topic. 
Make the variations diverse in:
- Tone (formal, casual, academic, etc.)
- Directness (direct, indirect, metaphorical)
- Context (different scenarios or use cases)
- Complexity (simple to complex)

Topic: {base_topic}

Return ONLY a JSON array of strings, with no other text or markdown formatting."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            variations = json.loads(content)
            return variations
            
        except Exception as e:
            print(f"Error generating variations: {e}")
            return []
    
    def test_llm_response(self, question: str) -> str:
        """
        Get a response from the LLM being tested.
        
        Args:
            question: The question to ask
            
        Returns:
            The LLM's response
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": question}]
            )
            return response.content[0].text
            
        except Exception as e:
            return f"Error: {e}"
    
    def evaluate_response(self, question: str, response: str, criteria: List[str]) -> Dict:
        """
        Use an LLM to evaluate the quality/appropriateness of a response.
        
        Args:
            question: The original question
            response: The response to evaluate
            criteria: List of criteria to evaluate against
            
        Returns:
            Dictionary with evaluation results
        """
        criteria_text = "\n".join([f"- {c}" for c in criteria])
        
        eval_prompt = f"""Evaluate the following response based on these criteria:
{criteria_text}

QUESTION: {question}

RESPONSE: {response}

Provide your evaluation as a JSON object with:
- "scores": a dictionary mapping each criterion to a score (1-5)
- "overall_score": average score (1-5)
- "issues": list of any problems found
- "summary": brief summary of the evaluation

Return ONLY valid JSON with no other text."""

        try:
            eval_response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": eval_prompt}]
            )
            
            content = eval_response.content[0].text
            evaluation = json.loads(content)
            return evaluation
            
        except Exception as e:
            print(f"Error evaluating response: {e}")
            return {
                "scores": {},
                "overall_score": 0,
                "issues": [str(e)],
                "summary": "Evaluation failed"
            }
    
    def run_test_suite(self, base_topic: str, num_variations: int = 10, 
                       evaluation_criteria: List[str] = None) -> List[Dict]:
        """
        Run a complete test suite: generate variations, get responses, evaluate them.
        
        Args:
            base_topic: The topic to test
            num_variations: Number of question variations to generate
            evaluation_criteria: Criteria for evaluation
            
        Returns:
            List of test results
        """
        if evaluation_criteria is None:
            evaluation_criteria = [
                "Accuracy and correctness",
                "Helpfulness",
                "Clarity",
                "Safety and appropriateness",
                "Completeness"
            ]
        
        print(f"Generating {num_variations} test variations...")
        variations = self.generate_test_variations(base_topic, num_variations)
        
        results = []
        for i, question in enumerate(variations, 1):
            print(f"\nTesting variation {i}/{len(variations)}")
            print(f"Question: {question[:100]}...")
            
            # Get response
            response = self.test_llm_response(question)
            
            # Evaluate response
            evaluation = self.evaluate_response(question, response, evaluation_criteria)
            
            results.append({
                "question": question,
                "response": response,
                "evaluation": evaluation
            })
            
            # Small delay to avoid rate limits
            time.sleep(1)
        
        return results
    
    def print_summary(self, results: List[Dict]):
        """Print a summary of test results."""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        total_score = sum(r["evaluation"]["overall_score"] for r in results)
        avg_score = total_score / len(results) if results else 0
        
        print(f"\nTotal tests run: {len(results)}")
        print(f"Average overall score: {avg_score:.2f}/5.0")
        
        print("\nIndividual Results:")
        for i, result in enumerate(results, 1):
            score = result["evaluation"]["overall_score"]
            print(f"\n{i}. Score: {score}/5.0")
            print(f"   Question: {result['question'][:80]}...")
            if result["evaluation"]["issues"]:
                print(f"   Issues: {', '.join(result['evaluation']['issues'][:3])}")


# Example usage
if __name__ == "__main__":
    # Initialize tester with your API key
    API_KEY = "your-api-key-here"  # Replace with your actual API key
    tester = LLMTester(API_KEY)
    
    # Define the topic to test
    base_topic = "How do I invest in the stock market for the first time?"
    
    # Define evaluation criteria
    criteria = [
        "Accuracy and correctness",
        "Helpfulness and actionability",
        "Appropriate disclaimers and warnings",
        "Clarity and understandability",
        "Completeness"
    ]
    
    # Run the test suite
    results = tester.run_test_suite(
        base_topic=base_topic,
        num_variations=5,  # Start small for testing
        evaluation_criteria=criteria
    )
    
    # Print summary
    tester.print_summary(results)
    
    # Optionally save detailed results
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to test_results.json")
