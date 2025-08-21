"""
Document validation and cleaning pipeline.
Implements "fail fast" validation with comprehensive checks.
"""
import re
import unicodedata
from typing import List, Set, Dict, Any
from collections import Counter

from langchain_core.documents import Document
from pipelines.contracts import ValidationResult, normalize_text, compute_content_hash


class DocumentValidator:
    """Validates and cleans documents before ingestion."""
    
    def __init__(self, config: Dict[str, Any]):
        self.min_char_length = config.get("min_char_length", 50)
        self.max_char_length = config.get("max_char_length", 1_000_000)
        self.min_token_estimate = config.get("min_token_estimate", 10)
        self.max_token_estimate = config.get("max_token_estimate", 100_000)
        self.allowed_languages = set(config.get("allowed_languages", ["en"]))
        self.remove_duplicates = config.get("remove_duplicates", True)
        self.normalize_unicode = config.get("normalize_unicode", True)
        self.clean_html = config.get("clean_html", True)
        self.preserve_code_blocks = config.get("preserve_code_blocks", True)
        
        # Content patterns to check
        self.suspicious_patterns = [
            r"(?i)lorem ipsum",  # Placeholder text
            r"^.{0,10}$",       # Too short content
            r"^(.)\1{50,}",     # Repeated characters
        ]
        
        # Character sets - be more permissive for HTML/code content
        self.allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        self.allowed_chars.update(" .,!?;:()[]{}\"'-\n\t")
        self.allowed_chars.update("<>/&=#@%*+_|\\`~$^")  # Add common HTML/code characters
        
        # Seen document hashes for deduplication
        self.seen_hashes: Set[str] = set()
    
    def validate_batch(self, documents: List[Document]) -> List[ValidationResult]:
        """Validate a batch of documents."""
        results = []
        
        for doc in documents:
            result = self.validate_document(doc)
            results.append(result)
        
        return results
    
    def validate_document(self, doc: Document) -> ValidationResult:
        """Validate a single document."""
        doc_id = doc.metadata.get("external_id", "unknown")
        errors = []
        warnings = []
        
        # Basic content checks
        content = doc.page_content or ""
        if not content.strip():
            errors.append("Empty content")
        
        # Length checks
        char_count = len(content)
        if char_count < self.min_char_length:
            errors.append(f"Content too short: {char_count} < {self.min_char_length} chars")
        
        if char_count > self.max_char_length:
            errors.append(f"Content too long: {char_count} > {self.max_char_length} chars")
        
        # Token estimate
        token_estimate = self._estimate_tokens(content)
        if token_estimate < self.min_token_estimate:
            warnings.append(f"Low token count estimate: {token_estimate}")
        
        if token_estimate > self.max_token_estimate:
            warnings.append(f"High token count estimate: {token_estimate}")
        
        # Content quality checks
        errors.extend(self._check_content_quality(content))
        warnings.extend(self._check_content_warnings(content))
        
        # Duplication check
        if self.remove_duplicates:
            content_hash = compute_content_hash(content)
            if content_hash in self.seen_hashes:
                errors.append(f"Duplicate content (hash: {content_hash[:12]})")
            else:
                self.seen_hashes.add(content_hash)
        
        # Metadata validation
        errors.extend(self._validate_metadata(doc.metadata))
        
        return ValidationResult(
            valid=len(errors) == 0,
            doc_id=doc_id,
            errors=errors,
            warnings=warnings
        )
    
    def clean_document(self, doc: Document) -> Document:
        """Clean and normalize document content."""
        content = doc.page_content or ""
        
        # Unicode normalization
        if self.normalize_unicode:
            content = unicodedata.normalize('NFKC', content)
        
        # HTML cleaning
        if self.clean_html:
            content = self._clean_html(content)
        
        # Text normalization
        content = self._normalize_text(content)
        
        # Update document
        cleaned_doc = Document(
            page_content=content,
            metadata=doc.metadata.copy()
        )
        
        # Update character count in metadata
        cleaned_doc.metadata["char_count"] = len(content)
        cleaned_doc.metadata["token_estimate"] = self._estimate_tokens(content)
        
        return cleaned_doc
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters for English)."""
        return len(text) // 4
    
    def _check_content_quality(self, content: str) -> List[str]:
        """Check for content quality issues."""
        errors = []
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, content):
                errors.append(f"Suspicious pattern detected: {pattern}")
        
        # Check character distribution
        char_counts = Counter(content)
        total_chars = len(content)
        
        # Check for excessive repetition
        if total_chars > 0:
            max_char_ratio = max(count / total_chars for count in char_counts.values())
            if max_char_ratio > 0.5:  # More than 50% of content is single character
                errors.append(f"Excessive character repetition: {max_char_ratio:.2%}")
        
        # Check for valid character set
        invalid_chars = set(content) - self.allowed_chars
        if invalid_chars:
            warnings_chars = list(invalid_chars)[:5]  # Show first 5
            errors.append(f"Invalid characters found: {warnings_chars}")
        
        return errors
    
    def _check_content_warnings(self, content: str) -> List[str]:
        """Check for content quality warnings."""
        warnings = []
        
        # Check for very long lines (might be malformed)
        lines = content.split('\n')
        long_lines = [i for i, line in enumerate(lines) if len(line) > 1000]
        if long_lines:
            warnings.append(f"Very long lines found at positions: {long_lines[:5]}")
        
        # Check for unusual whitespace patterns
        if re.search(r'\s{10,}', content):
            warnings.append("Excessive whitespace found")
        
        # Check language (basic heuristic)
        if self.allowed_languages and "en" in self.allowed_languages:
            english_ratio = self._estimate_english_ratio(content)
            if english_ratio < 0.7:
                warnings.append(f"Low English content ratio: {english_ratio:.2%}")
        
        return warnings
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """Validate required metadata fields."""
        errors = []
        
        required_fields = ["external_id", "source"]
        for field in required_fields:
            if field not in metadata or not metadata[field]:
                errors.append(f"Missing required metadata field: {field}")
        
        return errors
    
    def _clean_html(self, content: str) -> str:
        """Basic HTML cleaning."""
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', ' ', content)
        
        # Decode common HTML entities
        html_entities = {
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
            '&nbsp;': ' '
        }
        
        for entity, replacement in html_entities.items():
            content = content.replace(entity, replacement)
        
        return content
    
    def _normalize_text(self, content: str) -> str:
        """Normalize text content."""
        # Fix multiple whitespaces
        content = re.sub(r'\s+', ' ', content)
        
        # Fix multiple newlines (preserve paragraph structure)
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # Strip leading/trailing whitespace
        content = content.strip()
        
        return content
    
    def _estimate_english_ratio(self, content: str) -> float:
        """Rough estimation of English content ratio."""
        # Simple heuristic: check for common English words
        english_words = {
            'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that',
            'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'i'
        }
        
        words = re.findall(r'\b\w+\b', content.lower())
        if not words:
            return 0.0
        
        english_count = sum(1 for word in words if word in english_words)
        return english_count / len(words)
