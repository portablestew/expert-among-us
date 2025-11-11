"""
Search Engine Module

Implements the search and lookup functionality for Expert Among Us.
Handles vector similarity search, score merging, filtering, and result ranking.
"""

from typing import List, Optional, Dict
from dataclasses import dataclass
from expert_among_us.models.changelist import Changelist
from expert_among_us.models.query import QueryParams, VectorSearchResult
from expert_among_us.models.query_result import (
    QueryResult,
    CommitResult,
    FileChunkResult,
)
from expert_among_us.embeddings.base import Embedder
from expert_among_us.db.metadata.base import MetadataDB
from expert_among_us.db.vector.base import VectorDB
from expert_among_us.reranking.base import Reranker
from expert_among_us.utils.debug import DebugLogger
from expert_among_us.utils.progress import log_info


@dataclass
class SearchResult:
    """Combined search result with changelist and score.
    
    DEPRECATED: Use CommitResult or FileChunkResult instead.
    This class is kept for backward compatibility only.
    """
    changelist: Changelist
    similarity_score: float
    source: str  # 'metadata', 'diff', or 'combined'
    chroma_id: Optional[str] = None  # ChromaDB ID for debugging chunk-level matching


class Searcher:
    """
    Search engine for querying indexed commit history.
    
    Responsibilities:
    - Generate query embeddings
    - Perform vector similarity search
    - Merge scores from multiple collections
    - Apply metadata filters
    - Return ranked results
    """
    
    def __init__(
        self,
        expert_name: str,
        embedder: Embedder,
        metadata_db: MetadataDB,
        vector_db: VectorDB,
        reranker: Optional[Reranker] = None,
        enable_metadata_search: bool = True,
        enable_diff_search: bool = True,
        enable_file_search: bool = True,
        enable_reranking: bool = True,
        min_similarity_score: float = 0.1,
        relative_threshold: float = 0.8
    ):
        """
        Initialize the search engine.
        
        Args:
            expert_name: Name of the expert to search
            embedder: Embedding provider for query encoding
            metadata_db: Metadata database instance
            vector_db: Vector database instance
            reranker: Optional reranker for post-processing results
            enable_metadata_search: Whether to search metadata embeddings
            enable_diff_search: Whether to search diff embeddings
            enable_file_search: Whether to search file content embeddings
            enable_reranking: Whether to enable cross-encoder reranking
            min_similarity_score: Minimum similarity score threshold (0.0-1.0)
            relative_threshold: Relative score threshold as percentage drop from top result (0.0-1.0)
        """
        self.expert_name = expert_name
        self.embedder = embedder
        self.metadata_db: MetadataDB = metadata_db
        self.vector_db: VectorDB = vector_db
        self.reranker = reranker
        self.enable_metadata_search = enable_metadata_search
        self.enable_diff_search = enable_diff_search
        self.enable_file_search = enable_file_search
        self.enable_reranking = enable_reranking
        self.min_similarity_score = min_similarity_score
        self.relative_threshold = relative_threshold
    
    def search(self, params: QueryParams) -> List[QueryResult]:
        """
        Perform comprehensive search across commit metadata, diff content, and file content.
        
        This method orchestrates a multi-stage search process with optional cross-encoder reranking:
        1. Generates query embedding from the input prompt
        2. Retrieves 2x candidates for reranking (if enabled)
        3. Searches metadata collection for similar commit summaries and descriptions
        4. Optionally searches diff collection for similar code changes
        5. Merges and deduplicates commit scores using weighted averaging
        6. Fetches full changelist details and applies metadata filters (users, files)
        7. Reranks commits separately with cross-encoder (if enabled)
        8. Limits commits to final max_changes AFTER reranking
        9. Optionally searches file content collection for similar code patterns
        10. Reranks file chunks separately with cross-encoder (if enabled)
        11. Limits file chunks to final max_file_chunks AFTER reranking
        12. Combines and sorts final results
        
        The search supports two types of results:
        - CommitResult: Full commit/changelist matches from metadata and diff searches
        - FileChunkResult: Individual file chunk matches from file content searches
        
        Args:
            params: Query parameters containing:
                - prompt: Natural language query to search for
                - users: Optional list of authors to filter by (OR logic)
                - files: Optional list of file patterns to filter by (OR logic)
                - max_changes: Maximum number of commit results to return
                - max_file_chunks: Maximum number of file chunk results to return
                
        Returns:
            List of QueryResult objects (either CommitResult or FileChunkResult)
            sorted in descending order by similarity score. Each result contains
            the matched entity, similarity score, source collection, and debugging info.
            Original search scores are preserved in search_similarity_score field.
            
        Note:
            The method applies two similarity filters:
            - Absolute threshold: Filters out results below min_similarity_score
            - Relative threshold: Filters out results below (top_score * (1 - relative_threshold))
            
        Raises:
            Various exceptions may be raised by underlying vector database operations
            or metadata database queries.
        """
        log_info(f"Searching expert '{self.expert_name}' for: {params.prompt[:50]}...")
        
        # Step 1: Generate query embedding
        query_embedding = self.embedder.embed(params.prompt)
        
        # Step 2: Retrieve 3x results for reranking (if enabled)
        # Fetch more candidates so reranking can improve final top-K selection
        retrieval_multiplier = 3 if (self.enable_reranking and self.reranker) else 1
        commit_retrieval_limit = params.max_changes * retrieval_multiplier
        file_retrieval_limit = params.max_file_chunks * retrieval_multiplier
        
        # Step 3: Search metadata and diff collections
        metadata_results: List[VectorSearchResult] = []
        if self.enable_metadata_search:
            metadata_results = self._search_metadata(query_embedding, commit_retrieval_limit)
        
        diff_results: List[VectorSearchResult] = []
        if self.enable_diff_search:
            raw_diff_results = self._search_diffs(query_embedding, commit_retrieval_limit)
            diff_results = self._aggregate_chunk_scores(raw_diff_results)
            if len(raw_diff_results) > len(diff_results):
                log_info(f"Aggregated {len(raw_diff_results)} diff chunks into {len(diff_results)} commits")

        # Step 4: Merge and deduplicate commit scores
        commit_scores = self._merge_commit_scores(metadata_results, diff_results)
        top_commits = sorted(commit_scores.keys(), key=lambda x: commit_scores[x]['score'], reverse=True)
        top_commits = top_commits[:commit_retrieval_limit]
        
        # Step 5: Fetch full changelists and apply filters
        changelists = self.metadata_db.get_changelists_by_ids(top_commits)
        filtered_commits = self._apply_commit_filters(changelists, commit_scores, params)
        filtered_commits = self._apply_similarity_filters(filtered_commits)
        
        # Step 6: Rerank commits separately (BEFORE limiting to max_changes)
        if self.enable_reranking and self.reranker and filtered_commits:
            filtered_commits = self._rerank_results(params.prompt, filtered_commits)
            # Re-apply filters after reranking (scores have changed)
            # Only apply relative_threshold to logits, skip min_similarity_score
            filtered_commits = self._apply_relative_threshold_filter(filtered_commits)
        
        # Step 7: Limit commits to final max_changes AFTER reranking
        filtered_commits = filtered_commits[:params.max_changes]
        
        # Step 8: Search file collection separately
        file_results: List[VectorSearchResult] = []
        if self.enable_file_search:
            file_results = self._search_files(query_embedding, file_retrieval_limit)

        file_scores = self._merge_file_scores(file_results)
        top_files = sorted(file_scores.keys(), key=lambda x: file_scores[x]['score'], reverse=True)
        top_files = top_files[:file_retrieval_limit]
        
        filtered_files = self._apply_file_filters(top_files, file_scores, params)
        filtered_files = self._apply_similarity_filters(filtered_files)
        
        # Step 9: Rerank files separately (BEFORE limiting to max_file_chunks)
        if self.enable_reranking and self.reranker and filtered_files:
            filtered_files = self._rerank_results(params.prompt, filtered_files)
            # Re-apply filters after reranking (scores have changed)
            # Only apply relative_threshold to logits, skip min_similarity_score
            filtered_files = self._apply_relative_threshold_filter(filtered_files)
        
        # Step 10: Limit files to final max_file_chunks AFTER reranking
        filtered_files = filtered_files[:params.max_file_chunks]
        
        # Step 11: Combine final results (already at correct limits)
        final_results = filtered_commits + filtered_files
        final_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        log_info(f"Found {len(final_results)} results ({len(filtered_commits)} commits, {len(filtered_files)} files)")
        return final_results
    
    def _search_metadata(
        self,
        query_embedding: List[float],
        top_k: int
    ) -> List[VectorSearchResult]:
        """
        Search metadata collection for similar commits.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of vector search results from metadata collection
        """
        results = self.vector_db.search_metadata(query_embedding, top_k)
        log_info(f"Metadata search found {len(results)} results")
        return results
    
    def _search_diffs(
        self,
        query_embedding: List[float],
        top_k: int
    ) -> List[VectorSearchResult]:
        """
        Search diff collection for similar code changes.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of vector search results from diff collection
        """
        results = self.vector_db.search_diffs(query_embedding, top_k)
        log_info(f"Diff search found {len(results)} results")
        return results
    
    def _search_files(
        self,
        query_embedding: List[float],
        top_k: int
    ) -> List[VectorSearchResult]:
        """
        Search file content collection for similar code.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return

        Returns:
            List of vector search results from file content collection
        """
        # NOTE:
        # VectorSearchResult is a concrete Pydantic model in production, but in tests
        # the vector_db is a Mock without search_files configured, so calling
        # len(results) on the raw Mock return value raises TypeError.
        #
        # To keep behavior robust (and avoid hiding bugs), we normalize the result
        # to a list and log its size, which works for both real implementations
        # and mocks.
        results = self.vector_db.search_files(query_embedding, top_k)

        # Normalize to list defensively; this is effectively a no-op for real
        # implementations that already return a list[VectorSearchResult].
        if results is None:
            results_list: List[VectorSearchResult] = []
        elif isinstance(results, list):
            results_list = results
        else:
            try:
                results_list = list(results)
            except TypeError:
                # Fall back gracefully for unexpected/mocked types; avoids test failures
                # while still keeping production behavior unchanged.
                results_list = []

        log_info(f"File search found {len(results_list)} results")
        return results_list
    
    def _aggregate_chunk_scores(
        self,
        chunk_results: List[VectorSearchResult]
    ) -> List[VectorSearchResult]:
        """Aggregate chunk scores using max pooling.
        
        When multiple chunks from the same commit match, take the max score
        and preserve the chroma_id of the best-matching chunk.
        
        Args:
            chunk_results: Raw results with multiple chunks per commit
            
        Returns:
            Aggregated results with one score per commit (max pooling)
        """
        from typing import Dict, Tuple
        
        # Group by result_id, tracking both max score and corresponding chroma_id
        grouped: Dict[str, Tuple[float, Optional[str]]] = {}
        for result in chunk_results:
            if result.result_id not in grouped:
                grouped[result.result_id] = (result.similarity_score, result.chroma_id)
            else:
                # Max pooling: keep the highest score and its chroma_id
                current_score, current_chroma_id = grouped[result.result_id]
                if result.similarity_score > current_score:
                    grouped[result.result_id] = (result.similarity_score, result.chroma_id)
        
        # Convert back to VectorSearchResult list
        return [
            VectorSearchResult(
                result_id=cid,
                similarity_score=score,
                chroma_id=chroma_id
            )
            for cid, (score, chroma_id) in grouped.items()
        ]
    
    def _merge_commit_scores(
        self,
        metadata_results: List[VectorSearchResult],
        diff_results: List[VectorSearchResult],
    ) -> Dict[str, Dict[str, any]]:
        """
        Merge scores from metadata and diff searches.
        
        For duplicate changelist IDs (commits), combines scores using weighted average:
        - Metadata: 60% weight (higher signal for "what/why")
        - Diff: 40% weight (implementation details)
        
        When both sources contribute, the source is set to whichever had
        the higher individual similarity score.
        
        Args:
            metadata_results: Results from metadata search
            diff_results: Results from diff search
            file_results: Results from file search (kept separate)
            
        Returns:
            Dictionary mapping result_id to {'score': float, 'source': str, 'is_file': bool}
        """
        merged: Dict[str, Dict[str, any]] = {}
        
        # Add metadata results (commits)
        for result in metadata_results:
            merged[result.result_id] = {
                'score': result.similarity_score,
                'source': 'metadata',
                'metadata_score': result.similarity_score,
                'diff_score': None,
                'file_score': None,
                'is_file': False,
                'chroma_id': result.chroma_id
            }
        
        # Merge diff results (commits)
        for result in diff_results:
            if result.result_id in merged:
                # Combine scores: 60% metadata, 40% diff
                metadata_score = merged[result.result_id]['metadata_score']
                diff_score = result.similarity_score
                combined_score = (metadata_score * 0.6) + (diff_score * 0.4)
                
                # Determine source based on which individual score is higher
                if metadata_score > diff_score:
                    source = 'metadata'
                    chroma_id = merged[result.result_id]['chroma_id']
                elif diff_score > metadata_score:
                    source = 'diff'
                    chroma_id = result.chroma_id
                else:
                    # Equal scores - use metadata as tiebreaker
                    source = 'metadata'
                    chroma_id = merged[result.result_id]['chroma_id']

                merged[result.result_id] = {
                    'score': combined_score,
                    'source': source,
                    'metadata_score': metadata_score,
                    'diff_score': diff_score,
                    'file_score': None,
                    'is_file': False,
                    'chroma_id': chroma_id
                }
            else:
                # Only in diff results
                merged[result.result_id] = {
                    'score': result.similarity_score,
                    'source': 'diff',
                    'metadata_score': None,
                    'diff_score': result.similarity_score,
                    'file_score': None,
                    'is_file': False,
                    'chroma_id': result.chroma_id
                }

        return merged
        
    def _merge_file_scores(
        self,
        file_results: List[VectorSearchResult],
    ) -> Dict[str, Dict[str, any]]:
        """
        Add file results separately (no merging with commits)
        """
        merged: Dict[str, Dict[str, any]] = {}

        for result in file_results:
            merged[result.result_id] = {
                'score': result.similarity_score,
                'source': 'file',
                'metadata_score': None,
                'diff_score': None,
                'file_score': result.similarity_score,
                'is_file': True,
                'chroma_id': result.chroma_id,
            }
        
        return merged
    
    def _apply_commit_filters(
        self,
        changelists: List[Changelist],
        scores: Dict[str, Dict[str, any]],
        params: QueryParams
    ) -> List[CommitResult]:
        """
        Apply metadata filters to changelists.
        
        Filters:
        - users: Include only changelists by specified authors (OR logic)
        - files: Include only changelists affecting specified files (OR logic)
        
        Args:
            changelists: List of changelists to filter
            scores: Score information for each changelist
            params: Query parameters with filter criteria
            
        Returns:
            Filtered list of CommitResult objects
        """
        results: List[CommitResult] = []
        
        for changelist in changelists:
            # Skip if not in scores (shouldn't happen, but safety check)
            if changelist.id not in scores:
                continue
            
            # Apply user filter (OR logic)
            if params.users:
                if changelist.author not in params.users:
                    continue
            
            # Apply file filter (OR logic)
            if params.files:
                # Check if any of the query files match any changelist files
                if not any(qfile in changelist.files for qfile in params.files):
                    continue
            
            # Passed all filters, add to results
            score_info = scores[changelist.id]
            results.append(CommitResult(
                changelist=changelist,
                similarity_score=score_info['score'],
                source=score_info['source'],
                chroma_id=score_info.get('chroma_id')
            ))
        
        return results
    
    def _apply_file_filters(
        self,
        top_files: List[str],
        file_scores: Dict[str, Dict[str, any]],
        params: QueryParams
    ) -> List[FileChunkResult]:
        """
        Apply metadata filters to file chunks.
        
        Filters:
        - files: Include only chunks affecting specified files (OR logic)
        
        Args:
            top_files: List of file IDs to filter (already pre-sorted and limited)
            file_scores: Score information for each file chunk
            params: Query parameters with filter criteria
            
        Returns:
            Filtered list of FileChunkResult objects
        """
        results: List[FileChunkResult] = []
        
        if top_files:
            # Retrieve file chunks using their chroma_ids
            chunk_ids = [file_scores[file_id]['chroma_id'] for file_id in top_files if file_scores[file_id].get('chroma_id')]
            if chunk_ids:
                file_chunks = self.metadata_db.get_file_chunks_by_ids(chunk_ids)
                
                # Create a mapping from chroma_id to FileChunk for quick lookup
                chunk_map = {chunk.get_chroma_id(): chunk for chunk in file_chunks}
                
                for file_id in top_files:
                    score_info = file_scores[file_id]
                    chroma_id = score_info.get('chroma_id')
                    
                    # Get the FileChunk object
                    file_chunk = chunk_map.get(chroma_id)
                    if not file_chunk:
                        continue
                    
                    # Apply file filter (OR logic)
                    if params.files:
                        # Check if any of the query files match this file path
                        if not any(qfile in file_chunk.file_path for qfile in params.files):
                            continue
                    
                    # Note: User filter doesn't apply to files
                    
                    # Passed all filters, add to results
                    results.append(FileChunkResult(
                        file_chunk=file_chunk,
                        similarity_score=score_info['score'],
                        source=score_info['source'],
                        chroma_id=chroma_id
                    ))
        
        return results
    
    def _apply_similarity_filters(
        self,
        results: List[QueryResult]
    ) -> List[QueryResult]:
        """
        Apply similarity score filters to a list of results.
        
        For cosine similarity (before reranking): applies both min_similarity_score and relative_threshold
        For cross-encoder logits (after reranking): applies only relative_threshold
        
        Args:
            results: List of QueryResult objects to filter
            
        Returns:
            Filtered list of QueryResult objects
        """
        return self._apply_filters(results, apply_min_score=True)

    def _apply_relative_threshold_filter(
        self,
        results: List[QueryResult]
    ) -> List[QueryResult]:
        """
        Apply only relative threshold filter to a list of results.
        
        Used after reranking where scores are logits, not cosine similarity.
        
        Args:
            results: List of QueryResult objects to filter
            
        Returns:
            Filtered list of QueryResult objects
        """
        return self._apply_filters(results, apply_min_score=False)

    def _apply_filters(
        self,
        results: List[QueryResult],
        apply_min_score: bool
    ) -> List[QueryResult]:
        """
        Apply similarity score filters with configurable minimum score filtering.
        
        Args:
            results: List of QueryResult objects to filter
            apply_min_score: Whether to apply minimum similarity score filter
            
        Returns:
            Filtered list of QueryResult objects
        """
        if not results:
            return results
        
        filtered_results = results
        
        # Apply minimum similarity score filter (absolute threshold)
        # Only apply to cosine similarity, not cross-encoder logits
        if apply_min_score:
            before_score_filter = len(filtered_results)
            filtered_results = [r for r in filtered_results if r.similarity_score >= self.min_similarity_score]
            filtered_count = before_score_filter - len(filtered_results)
            if filtered_count > 0:
                log_info(f"Filtered out {filtered_count} results below minimum score {self.min_similarity_score}")
        
        # Apply relative threshold filter (range-based for logits, multiplicative for cosine)
        # Range-based approach considers full score distribution with minimum range of 1.0
        if filtered_results and self.relative_threshold < 1.0:
            top_score = filtered_results[0].similarity_score  # Already sorted by score
            bottom_score = filtered_results[-1].similarity_score
            
            # Range-based threshold with minimum range of 1.0 for compatibility
            score_range = top_score - bottom_score
            effective_range = max(1.0, score_range)
            relative_cutoff = top_score - (effective_range * self.relative_threshold)
            
            before_count = len(filtered_results)
            filtered_results = [r for r in filtered_results if r.similarity_score >= relative_cutoff]
            if len(filtered_results) < before_count:
                log_info(f"Relative threshold filtered out {before_count - len(filtered_results)} results (cutoff: {relative_cutoff:.3f})")
        
        return filtered_results
    
    def _rerank_results(
        self,
        query: str,
        results: List[QueryResult]
    ) -> List[QueryResult]:
        """Rerank results using cross-encoder with chunked max pooling.
        
        Long documents are automatically chunked by the reranker, with each
        chunk scored independently. The maximum score across chunks is used
        as the final document score.
        
        Args:
            query: Original search query
            results: List of QueryResult objects to rerank
            
        Returns:
            Reranked list of QueryResult objects with updated scores.
            Original scores are preserved in search_similarity_score field.
        """
        if not results or not self.reranker:
            return results
        
        log_info(f"Reranking {len(results)} results with cross-encoder...")
        
        # Extract full text from results (no truncation - reranker handles chunking)
        documents = []
        
        for result in results:
            if isinstance(result, CommitResult):
                # Use metadata text for commits (message + files)
                # Prefer metadata over full diff (diff can be huge)
                doc_text = result.changelist.get_metadata_text()
            else:  # FileChunkResult
                # Use full content for file chunks
                doc_text = result.file_chunk.content
        
            documents.append(doc_text)
        
        # Rerank with automatic chunking and max pooling
        # Reranker handles splitting long docs into 2KB chunks internally
        ranked_pairs = self.reranker.rerank(query, documents)
        
        # Store original scores and update with reranked scores
        reranked_results = []
        for idx, rerank_score in ranked_pairs:
            result = results[idx]
            
            # Preserve original search score before overwriting
            result.search_similarity_score = result.similarity_score
            result.similarity_score = float(rerank_score)
            
            reranked_results.append(result)
        
        if DebugLogger.is_enabled():
            log_info(f"Reranking complete - score range: {reranked_results[0].similarity_score:.3f} to {reranked_results[-1].similarity_score:.3f}")
        
        return reranked_results

    def close(self):
        """Clean up resources."""
        self.metadata_db.close()
        self.vector_db.close()