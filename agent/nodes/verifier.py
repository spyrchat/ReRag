"""
Verifier node for hallucination detection in RAG responses.
Checks if generated answers are faithful to the retrieved context.
"""

from typing import Dict, Any, List
from langchain_core.prompts import PromptTemplate
from logs.utils.logger import get_logger

logger = get_logger("verifier")


class AnswerVerifier:
    """
    Verifies if generated answers are grounded in the retrieved context.
    Detects hallucinations, unsupported claims, and fabricated information.
    """
    
    def __init__(self, llm):
        self.llm = llm
        
        self.verification_prompt = PromptTemplate.from_template(
            """You are verifying if an answer is faithful to its source material.

Question: {question}

Source context: {context}

Answer to verify: {answer}

Check for:
1. **Major hallucinations**: Specific technical details, code examples, or facts NOT in the source
2. **Critical fabrications**: Incorrect information that could mislead the user
3. **Minor paraphrasing is OK**: Natural reformulation of ideas from the source is acceptable

IMPORTANT: 
- Natural, conversational language is GOOD (e.g., "You can...", "A common approach...")
- Paraphrasing and synthesis of source information is ACCEPTABLE
- Only flag MAJOR issues: specific facts, code, or technical details not in source
- Don't flag the absence of phrases like "according to" or "based on" - those are fine to omit

Return JSON:
{{
    "is_faithful": true/false,
    "hallucination_detected": true/false,
    "confidence": 0.0-1.0,
    "severity": "none/minor/moderate/severe",
    "issues": ["list ONLY major fabrications"],
    "recommendation": "accept/revise/reject",
    "reasoning": "Brief explanation"
}}

Severity guide:
- none: Fully grounded, natural paraphrasing is fine
- minor: Small unnecessary additions, but core answer is correct
- moderate: Some specific claims not in source
- severe: Mostly fabricated information

JSON:"""
        )
    
    def verify(
        self, 
        question: str, 
        context: str, 
        answer: str
    ) -> Dict[str, Any]:
        """
        Verifies if answer is faithful to context.
        
        Args:
            question: Original user question
            context: Retrieved context (source of truth)
            answer: Generated answer to verify
            
        Returns:
            Dict with verification results
        """
        try:
            prompt = self.verification_prompt.format(
                question=question,
                context=context if context else "No context available",
                answer=answer
            )
            
            response = self.llm.invoke(prompt)
            result = self._parse_verification_response(response.content)
            
            # Only log if issues detected
            if result.get('hallucination_detected') or not result.get('is_faithful'):
                logger.warning(
                    f"[Verifier] Issues detected: "
                    f"faithful={result['is_faithful']}, "
                    f"severity={result.get('severity', 'unknown')}"
                )
            else:
                logger.debug(
                    f"[Verifier] ✓ Verification passed"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"[Verifier] Verification failed: {str(e)}")
            # Default to conservative verification on error
            return {
                'is_faithful': False,
                'hallucination_detected': True,
                'confidence': 0.0,
                'severity': 'unknown',
                'issues': [f"Verification error: {str(e)}"],
                'recommendation': 'revise',
                'reasoning': 'Verification process failed',
                'error': str(e)
            }
    
    def _parse_verification_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM verification response into structured format."""
        import json
        import re
        
        # Try to extract JSON from response
        try:
            # First try direct JSON parse
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON in code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Try to find raw JSON object
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
            
            # Fallback: conservative default
            logger.warning("[Verifier] Could not parse verification response, using conservative defaults")
            return {
                'is_faithful': False,
                'hallucination_detected': True,
                'confidence': 0.5,
                'severity': 'moderate',
                'issues': ['Could not parse verification response'],
                'recommendation': 'revise',
                'reasoning': 'Verification parsing failed'
            }


def make_verifier(llm):
    """
    Factory function to create a verifier node.
    
    Args:
        llm: Language model instance
        
    Returns:
        Verifier node function for the agent graph
    """
    verifier = AnswerVerifier(llm)
    
    def verifier_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verifier node that checks answer faithfulness.
        """
        question = state.get("question", "")
        context = state.get("context", "")
        answer = state.get("answer", "")
        
        if not answer:
            logger.warning("[Verifier] No answer to verify, skipping")
            return {
                **state,
                "verification": {
                    'is_faithful': True,
                    'hallucination_detected': False,
                    'confidence': 1.0,
                    'severity': 'none',
                    'issues': [],
                    'recommendation': 'accept',
                    'reasoning': 'No answer to verify'
                }
            }
        
        verification_result = verifier.verify(question, context, answer)
        
        return {
            **state,
            "verification": verification_result
        }
    
    return verifier_node
