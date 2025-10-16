"""
Advanced chunking strategies for different content types.
Matches embedder receptive field and preserves semantic coherence.
"""
import re
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from embedding.recursive_splitter import RecursiveSplitter


class ChunkingStrategy(ABC):
    """Abstract base for chunking strategies."""

    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        """Chunk documents into smaller pieces."""
        pass

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return strategy name for metadata."""
        pass


class RecursiveChunkingStrategy(ChunkingStrategy):
    """Recursive character-based chunking for general text."""

    def __init__(self, config: Dict[str, Any]):
        self.chunk_size = config.get("chunk_size", 500)
        self.chunk_overlap = config.get("chunk_overlap", 50)
        self.separators = config.get("separators", None)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        """Split documents using recursive character splitting."""
        chunks = self.splitter.split_documents(documents)

        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunking_strategy"] = self.strategy_name
            chunk.metadata["chunk_size"] = self.chunk_size
            chunk.metadata["chunk_overlap"] = self.chunk_overlap

        return chunks

    @property
    def strategy_name(self) -> str:
        return "recursive_character"


class SemanticChunkingStrategy(ChunkingStrategy):
    """Semantic chunking that preserves sentence boundaries."""

    def __init__(self, config: Dict[str, Any]):
        self.target_chunk_size = config.get("chunk_size", 500)
        self.max_chunk_size = config.get("max_chunk_size", 800)
        self.sentence_overlap = config.get("sentence_overlap", 1)

    def chunk(self, documents: List[Document]) -> List[Document]:
        """Split documents at sentence boundaries."""
        chunks = []

        for doc in documents:
            doc_chunks = self._chunk_document(doc)
            chunks.extend(doc_chunks)

        return chunks

    def _chunk_document(self, doc: Document) -> List[Document]:
        """Chunk a single document at sentence boundaries."""
        text = doc.page_content
        sentences = self._split_into_sentences(text)

        if not sentences:
            return [doc]

        chunks = []
        current_chunk = ""
        chunk_sentences = []

        for i, sentence in enumerate(sentences):
            # Check if adding this sentence would exceed target size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(potential_chunk) <= self.target_chunk_size or not current_chunk:
                current_chunk = potential_chunk
                chunk_sentences.append(sentence)
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(self._create_chunk(
                        doc, current_chunk, len(chunks)))

                # Start new chunk (with overlap)
                overlap_start = max(
                    0, len(chunk_sentences) - self.sentence_overlap)
                overlap_sentences = chunk_sentences[overlap_start:]

                current_chunk = " ".join(overlap_sentences + [sentence])
                chunk_sentences = overlap_sentences + [sentence]

        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk(doc, current_chunk, len(chunks)))

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Simple sentence splitting - can be enhanced with NLTK/spaCy
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _create_chunk(self, original_doc: Document, chunk_text: str, chunk_index: int) -> Document:
        """Create a chunk document with metadata."""
        metadata = original_doc.metadata.copy()
        metadata.update({
            "chunk_index": chunk_index,
            "chunking_strategy": self.strategy_name,
            "char_count": len(chunk_text)
        })

        return Document(page_content=chunk_text, metadata=metadata)

    @property
    def strategy_name(self) -> str:
        return "semantic_sentence"


class CodeAwareChunkingStrategy(ChunkingStrategy):
    """Chunking strategy that preserves code blocks and functions - never creates oversized chunks."""

    def __init__(self, config: Dict[str, Any]):
        self.chunk_size = config.get("chunk_size", 800)
        self.preserve_functions = config.get("preserve_functions", True)
        self.preserve_code_blocks = config.get("preserve_code_blocks", True)

        # Fallback to recursive splitting for non-code content
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=50
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        """Split documents preserving code structure - no oversized chunks."""
        chunks = []

        for doc in documents:
            if self._has_code_content(doc.page_content):
                doc_chunks = self._chunk_code_document(doc)
            else:
                doc_chunks = self.fallback_splitter.split_documents([doc])

            # Add metadata
            for i, chunk in enumerate(doc_chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunking_strategy"] = self.strategy_name

            chunks.extend(doc_chunks)

        return chunks

    def _chunk_code_document(self, doc: Document) -> List[Document]:
        """Chunk document with code-aware splitting - break large functions intelligently."""
        text = doc.page_content
        chunks = []

        # Find code blocks first
        code_blocks = list(re.finditer(
            r'```[\w]*\n(.*?)\n```', text, re.DOTALL))

        if code_blocks:
            return self._handle_explicit_code_blocks(doc, text, code_blocks)
        else:
            return self._split_by_functions(doc)

    def _handle_explicit_code_blocks(self, doc: Document, text: str, code_blocks) -> List[Document]:
        """Handle documents with explicit ```code``` blocks."""
        chunks = []
        last_end = 0

        for code_match in code_blocks:
            code_start = code_match.start()
            code_end = code_match.end()
            code_content = code_match.group(0)

            # Text before code block
            before_text = text[last_end:code_start].strip()
            if before_text:
                chunks.extend(self._split_text_safely(
                    doc, before_text, len(chunks)))

            # Handle code block - NEVER create oversized chunks
            if len(code_content) <= self.chunk_size:
                # Code fits - create single chunk
                chunks.append(self._create_chunk(
                    doc, code_content, len(chunks)))
            else:
                # Code too large - split intelligently by functions within the code block
                # Extract just the code without ```
                code_text = code_match.group(1)
                code_chunks = self._split_large_code_block(code_text)

                # Wrap each chunk back in code block format
                for code_chunk_text in code_chunks:
                    # Detect language from original block
                    original_block = code_match.group(0)
                    lang_match = re.match(r'```(\w*)', original_block)
                    lang = lang_match.group(
                        1) if lang_match and lang_match.group(1) else ''

                    wrapped_code = f"```{lang}\n{code_chunk_text}\n```"
                    chunks.append(self._create_chunk(
                        doc, wrapped_code, len(chunks)))

            last_end = code_end

        # Handle remaining text after last code block
        remaining_text = text[last_end:].strip()
        if remaining_text:
            chunks.extend(self._split_text_safely(
                doc, remaining_text, len(chunks)))

        return chunks

    def _split_large_code_block(self, code_text: str) -> List[str]:
        """Split large code block at intelligent boundaries - never break functions."""
        lines = code_text.split('\n')

        # Find function/class boundaries
        function_boundaries = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r'^(def |class |async def |function |public |private |protected )', stripped):
                function_boundaries.append(i)

        if not function_boundaries:
            # No functions found - split at logical boundaries (imports, comments, empty lines)
            return self._split_at_logical_boundaries(code_text)

        chunks = []
        current_chunk_lines = []

        for i, line in enumerate(lines):
            current_chunk_lines.append(line)

            # Check if we're at a function boundary and chunk is getting large
            if i in function_boundaries and len(current_chunk_lines) > 1:
                # Check current chunk size
                # Exclude current function start
                current_text = '\n'.join(current_chunk_lines[:-1])

                if len(current_text) >= self.chunk_size * 0.8:  # 80% threshold
                    # Save current chunk (without the new function)
                    if current_text.strip():
                        chunks.append(current_text)

                    # Start new chunk with the function
                    current_chunk_lines = [line]

        # Handle remaining lines
        if current_chunk_lines:
            remaining_text = '\n'.join(current_chunk_lines)
            if remaining_text.strip():
                # If this remaining chunk is still too large, split it at other boundaries
                if len(remaining_text) > self.chunk_size:
                    chunks.extend(
                        self._split_at_logical_boundaries(remaining_text))
                else:
                    chunks.append(remaining_text)

        return chunks if chunks else [code_text]

    def _split_at_logical_boundaries(self, code_text: str) -> List[str]:
        """Split code at logical boundaries (imports, comments, empty lines)."""
        lines = code_text.split('\n')
        chunks = []
        current_chunk_lines = []

        for i, line in enumerate(lines):
            current_chunk_lines.append(line)

            # Look for logical split points
            stripped = line.strip()
            is_logical_boundary = (
                # After import blocks
                (stripped.startswith(('import ', 'from ')) and
                 i + 1 < len(lines) and
                 not lines[i + 1].strip().startswith(('import ', 'from '))) or

                # After comment blocks
                (stripped.startswith('#') and
                 i + 1 < len(lines) and
                 not lines[i + 1].strip().startswith('#')) or

                # At significant empty lines (2+ consecutive)
                (not stripped and
                 i + 1 < len(lines) and
                 lines[i + 1].strip() and
                 len('\n'.join(current_chunk_lines)) >= self.chunk_size * 0.6)
            )

            if is_logical_boundary and len('\n'.join(current_chunk_lines)) >= self.chunk_size * 0.8:
                # Save current chunk
                chunk_text = '\n'.join(current_chunk_lines)
                if chunk_text.strip():
                    chunks.append(chunk_text)
                current_chunk_lines = []

        # Add final chunk
        if current_chunk_lines:
            chunk_text = '\n'.join(current_chunk_lines)
            if chunk_text.strip():
                chunks.append(chunk_text)

        return chunks if chunks else [code_text]

    def _split_by_functions(self, doc: Document) -> List[Document]:
        """Split by function/class boundaries - break large functions intelligently."""
        text = doc.page_content
        lines = text.split('\n')

        # Find function/class definitions
        function_starts = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r'^(def |class |function |async def |public |private |protected )', stripped):
                function_starts.append(i)

        if not function_starts:
            # No functions - use fallback
            return self.fallback_splitter.split_documents([doc])

        chunks = []

        for i, func_start in enumerate(function_starts):
            # Find function end
            func_end = function_starts[i + 1] if i + \
                1 < len(function_starts) else len(lines)

            # Extract complete function
            function_lines = lines[func_start:func_end]
            function_text = '\n'.join(function_lines)

            if len(function_text) <= self.chunk_size:
                # Function fits - create chunk
                chunks.append(self._create_chunk(
                    doc, function_text, len(chunks)))
            else:
                # Large function - split intelligently within the function
                sub_chunks = self._split_large_function(function_lines)
                for sub_chunk in sub_chunks:
                    chunks.append(self._create_chunk(
                        doc, sub_chunk, len(chunks)))

        return chunks

    def _split_large_function(self, function_lines: List[str]) -> List[str]:
        """Split large function at logical boundaries within the function."""
        if not function_lines:
            return []

        # Always keep function signature in first chunk
        func_signature = function_lines[0]
        remaining_lines = function_lines[1:]

        if not remaining_lines:
            return [func_signature]

        chunks = []
        current_chunk_lines = [func_signature]

        # Look for logical split points within function body
        for i, line in enumerate(remaining_lines):
            current_chunk_lines.append(line)

            stripped = line.strip()
            current_text = '\n'.join(current_chunk_lines)

            # Split at logical boundaries if chunk is getting large
            if len(current_text) >= self.chunk_size * 0.8:
                is_good_split_point = (
                    # After major control structures
                    stripped.startswith(('if ', 'for ', 'while ', 'try:', 'with ', 'elif ', 'else:')) or

                    # After return statements
                    stripped.startswith('return') or

                    # After significant comments
                    (stripped.startswith('#') and len(stripped) > 10) or

                    # At empty lines within reasonable size
                    (not stripped and
                     i + 1 < len(remaining_lines) and
                     remaining_lines[i + 1].strip())
                )

                if is_good_split_point:
                    # Save current chunk
                    chunks.append(current_text)

                    # Start new chunk (keep some context - last few lines)
                    context_lines = min(2, len(current_chunk_lines) - 1)
                    current_chunk_lines = current_chunk_lines[-context_lines:] if context_lines > 0 else [
                    ]

        # Add final chunk
        if current_chunk_lines:
            final_text = '\n'.join(current_chunk_lines)
            if final_text.strip() and final_text not in chunks:
                chunks.append(final_text)

        return chunks if chunks else ['\n'.join(function_lines)]

    def _split_text_safely(self, doc: Document, text: str, start_index: int) -> List[Document]:
        """Split text content using fallback splitter."""
        temp_doc = Document(page_content=text, metadata=doc.metadata.copy())
        text_chunks = self.fallback_splitter.split_documents([temp_doc])

        for i, chunk in enumerate(text_chunks):
            chunk.metadata["chunk_index"] = start_index + i
            chunk.metadata["chunking_strategy"] = self.strategy_name
            chunk.metadata["content_type"] = "text_only"

        return text_chunks

    def _has_code_content(self, text: str) -> bool:
        """Detect if text contains code with improved precision."""

        # Quick check for obvious code indicators
        obvious_code_patterns = [
            r'```[\w]*\n',              # Markdown code blocks
            r'def \w+\s*\(',           # Python functions
            r'function \w+\s*\(',      # JavaScript functions
            r'class \w+\s*[\(\{:]',    # Class with code syntax
            r'(import|from)\s+\w+.*\n\s*(import|from)',  # Multiple imports
            r'\{\s*\n.*\n\s*\}',       # Brace blocks
        ]

        for pattern in obvious_code_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True

        # Context-aware class detection
        if re.search(r'\bclass\s+\w+', text, re.IGNORECASE):
            # Check if "class" appears in code context
            code_context_indicators = [
                r'class \w+\s*:',          # Python class definition
                r'class \w+\s*\(',         # Python class with inheritance
                r'class \w+.*extends',     # Java/JS inheritance
                r'class \w+.*implements',  # Java interface
                r'(public|private)\s+class',  # Access modifiers
            ]

            for pattern in code_context_indicators:
                if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                    return True

            # If "class" found but no code context, check line density
            lines = text.split('\n')
            class_lines = [line for line in lines if re.search(
                r'\bclass\s+\w+', line, re.IGNORECASE)]

            # If multiple class definitions, likely code
            if len(class_lines) > 1:
                return True

            # Check surrounding context of class mentions
            for line in class_lines:
                # If class is part of code-like syntax
                if re.search(r'[{}();]|def |import |function ', line):
                    return True

        return False

    def _create_chunk(self, original_doc: Document, chunk_text: str, chunk_index: int) -> Document:
        """Create chunk with metadata."""
        metadata = original_doc.metadata.copy()
        metadata.update({
            "chunk_index": chunk_index,
            "chunking_strategy": self.strategy_name,
            "char_count": len(chunk_text),
            "has_code": self._has_code_content(chunk_text)
        })

        return Document(page_content=chunk_text, metadata=metadata)

    @property
    def strategy_name(self) -> str:
        return "code_aware"


class TableAwareChunkingStrategy(ChunkingStrategy):
    """Chunking strategy that preserves table structure."""

    def __init__(self, config: Dict[str, Any]):
        self.chunk_size = config.get("chunk_size", 1000)
        self.preserve_headers = config.get("preserve_headers", True)
        self.max_table_size = config.get("max_table_size", 2000)

        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=50
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        """Split documents preserving table structure."""
        chunks = []

        for doc in documents:
            if self._has_tables(doc.page_content):
                doc_chunks = self._chunk_table_document(doc)
            else:
                doc_chunks = self.fallback_splitter.split_documents([doc])

            # Add metadata
            for i, chunk in enumerate(doc_chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunking_strategy"] = self.strategy_name

            chunks.extend(doc_chunks)

        return chunks

    def _has_tables(self, text: str) -> bool:
        """Detect if text contains tables."""
        table_indicators = [
            r'\|.*\|.*\|',  # Markdown tables
            r'\t.*\t.*\t',  # Tab-separated
            r'┌─+┬─+┐',     # ASCII tables
            r'<table>',     # HTML tables
        ]

        for pattern in table_indicators:
            if re.search(pattern, text, re.MULTILINE):
                return True

        return False

    def _chunk_table_document(self, doc: Document) -> List[Document]:
        """Chunk document with table awareness."""
        text = doc.page_content

        # Find markdown tables
        table_pattern = r'(\|.*\|.*\n)+(\|[-:\s]+\|.*\n)?(\|.*\|.*\n)+'
        tables = list(re.finditer(table_pattern, text, re.MULTILINE))

        if not tables:
            return self.fallback_splitter.split_documents([doc])

        chunks = []
        last_end = 0

        for table_match in tables:
            # Add text before table
            before_table = text[last_end:table_match.start()].strip()
            if before_table:
                chunks.extend(self._split_text_chunk(
                    doc, before_table, len(chunks)))

            # Process table
            table_content = table_match.group(0)
            if len(table_content) <= self.max_table_size:
                # Keep table intact
                chunks.append(self._create_chunk(
                    doc, table_content, len(chunks)))
            else:
                # Split large table by rows
                chunks.extend(self._split_large_table(
                    doc, table_content, len(chunks)))

            last_end = table_match.end()

        # Add remaining text
        remaining_text = text[last_end:].strip()
        if remaining_text:
            chunks.extend(self._split_text_chunk(
                doc, remaining_text, len(chunks)))

        return chunks

    def _split_text_chunk(self, doc: Document, text: str, start_index: int) -> List[Document]:
        """Split non-table text using fallback splitter."""
        temp_doc = Document(page_content=text, metadata=doc.metadata.copy())
        text_chunks = self.fallback_splitter.split_documents([temp_doc])

        for i, chunk in enumerate(text_chunks):
            chunk.metadata["chunk_index"] = start_index + i
            chunk.metadata["chunking_strategy"] = self.strategy_name

        return text_chunks

    def _split_large_table(self, doc: Document, table_content: str, start_index: int) -> List[Document]:
        """Split large table by rows while preserving header."""
        lines = table_content.split('\n')
        header_lines = []
        data_lines = []

        # Identify header (first line + separator if exists)
        if lines:
            header_lines.append(lines[0])
            if len(lines) > 1 and re.match(r'\|[-:\s]+\|', lines[1]):
                header_lines.append(lines[1])
                data_lines = lines[2:]
            else:
                data_lines = lines[1:]

        chunks = []
        current_rows = header_lines.copy() if self.preserve_headers else []

        for line in data_lines:
            current_rows.append(line)
            chunk_text = '\n'.join(current_rows)

            if len(chunk_text) > self.chunk_size:
                # Save current chunk
                if len(current_rows) > len(header_lines):
                    chunks.append(self._create_chunk(
                        doc, chunk_text, start_index + len(chunks)))

                # Start new chunk with headers
                current_rows = header_lines.copy(
                ) + [line] if self.preserve_headers else [line]

        # Add final chunk
        if len(current_rows) > len(header_lines):
            chunk_text = '\n'.join(current_rows)
            chunks.append(self._create_chunk(
                doc, chunk_text, start_index + len(chunks)))

        return chunks

    def _create_chunk(self, original_doc: Document, chunk_text: str, chunk_index: int) -> Document:
        """Create chunk with metadata."""
        metadata = original_doc.metadata.copy()
        metadata.update({
            "chunk_index": chunk_index,
            "chunking_strategy": self.strategy_name,
            "char_count": len(chunk_text),
            "has_table": self._has_tables(chunk_text)
        })

        return Document(page_content=chunk_text, metadata=metadata)

    @property
    def strategy_name(self) -> str:
        return "table_aware"


class ChunkingStrategyFactory:
    """Factory for creating chunking strategies."""

    STRATEGIES = {
        "recursive": RecursiveChunkingStrategy,
        "semantic": SemanticChunkingStrategy,
        "code_aware": CodeAwareChunkingStrategy,
        "table_aware": TableAwareChunkingStrategy,
    }

    @classmethod
    def create_strategy(cls, strategy_name: str, config: Dict[str, Any]) -> ChunkingStrategy:
        """Create chunking strategy by name."""
        if strategy_name not in cls.STRATEGIES:
            available = ", ".join(cls.STRATEGIES.keys())
            raise ValueError(
                f"Unknown chunking strategy '{strategy_name}'. Available: {available}")

        strategy_class = cls.STRATEGIES[strategy_name]
        return strategy_class(config)

    @classmethod
    def get_strategy_for_content(cls, content: str, config: Dict[str, Any]) -> ChunkingStrategy:
        """Auto-select chunking strategy based on content analysis."""
        # Simple heuristics for auto-selection
        if cls._has_code_content(content):
            return cls.create_strategy("code_aware", config)
        elif cls._has_tables(content):
            return cls.create_strategy("table_aware", config)
        elif config.get("use_semantic", False):
            return cls.create_strategy("semantic", config)
        else:
            return cls.create_strategy("recursive", config)

    @staticmethod
    def _has_code_content(text: str) -> bool:
        """Detect code content."""
        code_patterns = [r'```', r'def \w+\(', r'function \w+\(', r'class \w+']
        return any(re.search(pattern, text) for pattern in code_patterns)

    @staticmethod
    def _has_tables(text: str) -> bool:
        """Detect table content."""
        table_patterns = [r'\|.*\|.*\|', r'\t.*\t.*\t']
        return any(re.search(pattern, text, re.MULTILINE) for pattern in table_patterns)
