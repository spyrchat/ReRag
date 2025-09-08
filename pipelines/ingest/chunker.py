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
                    chunks.append(self._create_chunk(doc, current_chunk, len(chunks)))
                
                # Start new chunk (with overlap)
                overlap_start = max(0, len(chunk_sentences) - self.sentence_overlap)
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
    """Chunking strategy that preserves code blocks and functions."""
    
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
        """Split documents preserving code structure."""
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
    
    def _has_code_content(self, text: str) -> bool:
        """Detect if text contains code."""
        code_indicators = [
            r'```',  # Markdown code blocks
            r'def \w+\(',  # Python functions
            r'function \w+\(',  # JavaScript functions
            r'class \w+',  # Class definitions
            r'import \w+',  # Import statements
            r'\{\s*\n.*\n\s*\}',  # Brace blocks
        ]
        
        for pattern in code_indicators:
            if re.search(pattern, text, re.MULTILINE):
                return True
        
        return False
    
    def _chunk_code_document(self, doc: Document) -> List[Document]:
        """Chunk document with code-aware splitting."""
        text = doc.page_content
        chunks = []
        
        # Find code blocks
        code_blocks = list(re.finditer(r'```[\w]*\n(.*?)\n```', text, re.DOTALL))
        
        if not code_blocks:
            # No explicit code blocks, use function-based splitting
            return self._split_by_functions(doc)
        
        last_end = 0
        for i, match in enumerate(code_blocks):
            # Add text before code block
            before_code = text[last_end:match.start()].strip()
            if before_code:
                chunks.append(self._create_chunk(doc, before_code, len(chunks)))
            
            # Add code block (keep intact if not too large)
            code_content = match.group(0)
            if len(code_content) <= self.chunk_size:
                chunks.append(self._create_chunk(doc, code_content, len(chunks)))
            else:
                # Split large code blocks
                code_chunks = self.fallback_splitter.create_documents([code_content])
                for code_chunk in code_chunks:
                    code_chunk.metadata = doc.metadata.copy()
                    chunks.append(code_chunk)
            
            last_end = match.end()
        
        # Add remaining text
        remaining_text = text[last_end:].strip()
        if remaining_text:
            chunks.append(self._create_chunk(doc, remaining_text, len(chunks)))
        
        return chunks
    
    def _split_by_functions(self, doc: Document) -> List[Document]:
        """Split by function/class boundaries."""
        text = doc.page_content
        
        # Find function/class definitions
        function_pattern = r'^(def |class |function |async def )'
        lines = text.split('\n')
        
        chunks = []
        current_chunk_lines = []
        
        for line in lines:
            if re.match(function_pattern, line.strip()) and current_chunk_lines:
                # Start new chunk at function boundary
                chunk_text = '\n'.join(current_chunk_lines)
                if chunk_text.strip():
                    chunks.append(self._create_chunk(doc, chunk_text, len(chunks)))
                current_chunk_lines = [line]
            else:
                current_chunk_lines.append(line)
                
                # Check size limit
                chunk_text = '\n'.join(current_chunk_lines)
                if len(chunk_text) > self.chunk_size:
                    chunks.append(self._create_chunk(doc, chunk_text, len(chunks)))
                    current_chunk_lines = []
        
        # Add final chunk
        if current_chunk_lines:
            chunk_text = '\n'.join(current_chunk_lines)
            if chunk_text.strip():
                chunks.append(self._create_chunk(doc, chunk_text, len(chunks)))
        
        return chunks if chunks else [doc]
    
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
                chunks.extend(self._split_text_chunk(doc, before_table, len(chunks)))
            
            # Process table
            table_content = table_match.group(0)
            if len(table_content) <= self.max_table_size:
                # Keep table intact
                chunks.append(self._create_chunk(doc, table_content, len(chunks)))
            else:
                # Split large table by rows
                chunks.extend(self._split_large_table(doc, table_content, len(chunks)))
            
            last_end = table_match.end()
        
        # Add remaining text
        remaining_text = text[last_end:].strip()
        if remaining_text:
            chunks.extend(self._split_text_chunk(doc, remaining_text, len(chunks)))
        
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
                    chunks.append(self._create_chunk(doc, chunk_text, start_index + len(chunks)))
                
                # Start new chunk with headers
                current_rows = header_lines.copy() + [line] if self.preserve_headers else [line]
        
        # Add final chunk
        if len(current_rows) > len(header_lines):
            chunk_text = '\n'.join(current_rows)
            chunks.append(self._create_chunk(doc, chunk_text, start_index + len(chunks)))
        
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
            raise ValueError(f"Unknown chunking strategy '{strategy_name}'. Available: {available}")
        
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
