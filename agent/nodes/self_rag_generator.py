"""
Self-RAG Generator with iterative refinement.
Generates answers and refines them based on verification feedback.
"""

from typing import Dict, Any, List
from langchain_core.prompts import PromptTemplate
from logs.utils.logger import get_logger
from agent.nodes.verifier import AnswerVerifier

logger = get_logger("self_rag_generator")


class SelfCorrectingGenerator:
    """
    Generator that can self-correct when hallucinations are detected.
    Implements iterative refinement with verification feedback loop.
    """

    def __init__(self, llm, verifier: AnswerVerifier, max_iterations: int = 3):
        self.llm = llm
        self.verifier = verifier
        self.max_iterations = max_iterations

        # Initial generation prompt
        self.generation_prompt = PromptTemplate.from_template(
            """You are a helpful programming expert answering a developer's question.

Question: {question}

Query Analysis:
{query_analysis}

Context (from Stack Overflow):
{context}

Instructions:
- Use the query analysis to understand what the user needs and the key concepts involved
- Follow the reasoning steps identified in the analysis
- Answer naturally and conversationally, as if you're helping a colleague
- Use the provided context as your primary source of information
- Include relevant code examples when helpful
- Address the specific information needs mentioned in the analysis
- If the context doesn't fully cover the question, acknowledge what's missing naturally (e.g., "One common approach is..." rather than "The context doesn't mention...")
- Do NOT use phrases like "based on the context" or "according to the sources" - just answer directly
- Focus on being helpful and practical

Your answer:"""
        )

        # Revision prompt with explicit feedback
        self.revision_prompt = PromptTemplate.from_template(
            """You're revising your answer because it contained some inaccurate information.

Original question: {question}

Query Analysis (what the user needs):
{query_analysis}

Available information (Stack Overflow):
{context}

Your previous answer:
{previous_answer}

Issues detected:
{feedback}

Please revise your answer to:
1. Remove any information not supported by the available context above
2. Keep the helpful and accurate parts that address the key concepts in the query analysis
3. Maintain a natural, conversational tone (no phrases like "based on the context")
4. If the context doesn't fully address something, briefly acknowledge it naturally (e.g., "For this specific case..." or "One approach that works well...")
5. Focus on being clear and practical

Revised answer:"""
        )

    def generate_with_verification(
        self,
        question: str,
        context: str,
        query_analysis: str = "No analysis available"
    ) -> Dict[str, Any]:
        """
        Generates answer with iterative verification and refinement.

        Args:
            question: User's question
            context: Retrieved context (source of truth)
            query_analysis: Analysis of the query (reasoning steps, key concepts)

        Returns:
            Dict with final answer, verification status, and iteration history
        """
        iteration_history = []
        current_answer = None

        for iteration in range(self.max_iterations):
            logger.debug(
                f"[SelfRAG] Iteration {iteration + 1}/{self.max_iterations}")

            if iteration == 0:
                # Initial generation
                prompt = self.generation_prompt.format(
                    context=context if context else "No context available",
                    question=question,
                    query_analysis=query_analysis
                )
                response = self.llm.invoke(prompt)
                current_answer = response.content.strip()
                logger.debug(
                    f"[SelfRAG] Generated initial answer ({len(current_answer)} chars)")
            else:
                # Revision with feedback from verification
                previous_verification = iteration_history[-1]['verification']
                feedback = self._format_feedback(previous_verification)

                prompt = self.revision_prompt.format(
                    question=question,
                    context=context if context else "No context available",
                    query_analysis=query_analysis,
                    previous_answer=current_answer,
                    feedback=feedback
                )
                response = self.llm.invoke(prompt)
                current_answer = response.content.strip()
                logger.debug(
                    f"[SelfRAG] Revised answer (iteration {iteration + 1})")

            # Verify the current answer
            verification = self.verifier.verify(
                question=question,
                context=context,
                answer=current_answer
            )

            # Store iteration metadata
            iteration_history.append({
                'iteration': iteration + 1,
                'answer': current_answer,
                'verification': verification,
                'action': 'initial' if iteration == 0 else 'revision'
            })

            # Check if we should stop
            if self._should_stop(verification, iteration):
                logger.debug(
                    f"[SelfRAG] Stopping: "
                    f"is_faithful={verification['is_faithful']}, "
                    f"iteration={iteration + 1}"
                )
                break

        # Prepare final result
        final_verification = iteration_history[-1]['verification']

        return {
            'final_answer': current_answer,
            'verification': final_verification,
            'iterations': len(iteration_history),
            'iteration_history': iteration_history,
            'converged': final_verification['is_faithful'],
            'generation_metadata': {
                'max_iterations_reached': len(iteration_history) == self.max_iterations,
                'hallucinations_corrected': any(
                    h['verification'].get('hallucination_detected', False)
                    for h in iteration_history[:-1]
                )
            }
        }

    def _format_feedback(self, verification: Dict[str, Any]) -> str:
        """Converts verification output to human-readable feedback."""
        if not verification.get('hallucination_detected'):
            return "No major issues found."

        feedback_parts = []

        if verification.get('issues'):
            feedback_parts.append("**Specific Issues Found:**")
            for issue in verification['issues']:
                feedback_parts.append(f"- {issue}")

        feedback_parts.append(
            f"\n**Severity:** {verification.get('severity', 'unknown')}"
        )
        feedback_parts.append(
            f"**Recommendation:** {verification.get('recommendation', 'revise')}"
        )

        if verification.get('reasoning'):
            feedback_parts.append(
                f"**Reasoning:** {verification['reasoning']}")

        return "\n".join(feedback_parts)

    def _should_stop(self, verification: Dict[str, Any], iteration: int) -> bool:
        """
        Determines if iterative refinement should stop.

        Stopping criteria:
        1. Verification passed (is_faithful = True)
        2. Reached max iterations
        3. High confidence with minor issues only
        """
        # Criterion 1: Passed verification
        if verification.get('is_faithful', False):
            return True

        # Criterion 2: Max iterations reached
        if iteration >= self.max_iterations - 1:
            logger.warning(
                f"[SelfRAG] Max iterations reached without convergence"
            )
            return True

        # Criterion 3: High confidence with minor issues only
        if (verification.get('confidence', 0) > 0.8 and
                verification.get('severity') == 'minor'):
            logger.info("[SelfRAG] Stopping with minor issues only")
            return True

        return False


def make_self_rag_generator(llm, max_iterations: int = 3):
    """
    Factory function to create a self-correcting generator node.

    Args:
        llm: Language model instance
        max_iterations: Maximum refinement iterations (default: 3)

    Returns:
        Self-RAG generator node function for the agent graph
    """
    # Create verifier
    verifier = AnswerVerifier(llm)

    # Create self-correcting generator
    self_rag_gen = SelfCorrectingGenerator(llm, verifier, max_iterations)

    def self_rag_generator_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Self-RAG generator node with verification loop.
        """
        question = state.get("question", "")
        context = state.get("context", "")
        query_analysis = state.get("query_analysis", "No analysis available")

        # Log query analysis for debugging
        logger.info(f"[SelfRAG] Query analysis received: {query_analysis[:200]}..." if len(query_analysis) > 200 else f"[SelfRAG] Query analysis: {query_analysis}")

        if not question:
            logger.error("[SelfRAG] No question provided")
            return {
                **state,
                "answer": "Error: No question provided",
                "self_rag_metadata": {
                    'iterations': 0,
                    'converged': False,
                    'error': 'No question provided'
                }
            }

        try:
            # Generate with verification loop
            result = self_rag_gen.generate_with_verification(
                question, context, query_analysis)

            # Log summary only if refinement happened
            if result['iterations'] > 1:
                logger.info(
                    f"[SelfRAG] Refined answer: "
                    f"{result['iterations']} iterations, "
                    f"converged={result['converged']}"
                )
            else:
                logger.debug(f"[SelfRAG] Answer accepted on first attempt")

            return {
                **state,
                "answer": result['final_answer'],
                "verification": result['verification'],
                "self_rag_metadata": {
                    'iterations': result['iterations'],
                    'converged': result['converged'],
                    'iteration_history': result['iteration_history'],
                    'hallucinations_corrected': result['generation_metadata']['hallucinations_corrected'],
                    'max_iterations_reached': result['generation_metadata']['max_iterations_reached']
                }
            }

        except Exception as e:
            logger.error(f"[SelfRAG] Generation failed: {str(e)}")
            return {
                **state,
                "answer": "I'm sorry, I couldn't generate an answer due to an internal error.",
                "error": str(e),
                "self_rag_metadata": {
                    'iterations': 0,
                    'converged': False,
                    'error': str(e)
                }
            }

    return self_rag_generator_node
