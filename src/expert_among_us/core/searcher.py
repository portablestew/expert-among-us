"""
Search Engine Module

Implements the search and lookup functionality for Expert Among Us.
Handles vector similarity search, score merging, filtering, and result ranking.
"""

from typing import List, Optional, Dict, Set
from dataclasses import dataclass
from expert_among_us.models.changelist import Changelist
from expert_among_us.models.query import QueryParams, VectorSearchResult
from expert_among_us.embeddings.base import Embedder
from expert_among_us.db.metadata.sqlite import SQLiteMetadataDB
from expert_among_us.db.vector.chroma import ChromaVectorDB
from expert_among_us.utils.progress import log_info, log_warning


@dataclass
class SearchResult:
    """Combined search result with changelist and score."""
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
        metadata_db: SQLiteMetadataDB,
        vector_db: ChromaVectorDB,
        enable_metadata_search: bool = True,
        enable_diff_search: bool = True,
        min_similarity_score: float = 0.3,
        relative_threshold: float = 0.3
    ):
        """
        Initialize the search engine.
        
        Args:
            expert_name: Name of the expert to search
            embedder: Embedding provider for query encoding
            metadata_db: Metadata database instance
            vector_db: Vector database instance
            enable_metadata_search: Whether to search metadata embeddings
            enable_diff_search: Whether to search diff embeddings
            min_similarity_score: Minimum similarity score threshold (0.0-1.0)
            relative_threshold: Relative score threshold as percentage drop from top result (0.0-1.0)
        """
        self.expert_name = expert_name
        self.embedder = embedder
        self.metadata_db = metadata_db
        self.vector_db = vector_db
        self.enable_metadata_search = enable_metadata_search
        self.enable_diff_search = enable_diff_search
        self.min_similarity_score = min_similarity_score
        self.relative_threshold = relative_threshold
    
    def search(self, params: QueryParams) -> List[SearchResult]:
        """
        Perform search with the given parameters.
        
        Process:
        1. Generate embedding for query
        2. Search metadata collection
        3. Optionally search diff collection
        4. Merge and deduplicate scores
        5. Fetch full changelists
        6. Apply filters
        7. Return ranked results
        
        Args:
            params: Query parameters including prompt and filters
            
        Returns:
            List of SearchResult objects ordered by similarity score
        """
        log_info(f"Searching expert '{self.expert_name}' for: {params.prompt[:50]}...")
        
        # Step 1: Generate query embedding
        query_embedding = self.embedder.embed(params.prompt)
        
        # Step 2: Search metadata collection (primary)
        # Get 2*max_changes to allow for filtering
        search_limit = params.max_changes * 2
        if self.enable_metadata_search:
            metadata_results = self._search_metadata(query_embedding, search_limit)
        else:
            metadata_results = []
        
        # Step 3: Optionally search diff collection (secondary)
        diff_results: List[VectorSearchResult] = []
        if self.enable_diff_search:
            log_info("Searching diff embeddings...")
            raw_diff_results = self._search_diffs(query_embedding, search_limit)
            
            # Aggregate chunk scores using max pooling
            diff_results = self._aggregate_chunk_scores(raw_diff_results)
            if len(raw_diff_results) > len(diff_results):
                log_info(f"Aggregated {len(raw_diff_results)} chunks into {len(diff_results)} commits")
        
        # Step 4: Merge and deduplicate scores
        merged_scores = self._merge_scores(metadata_results, diff_results)
        
        # Step 5: Get top K changelist IDs (K > N to allow for filtering)
        top_ids = sorted(merged_scores.keys(), key=lambda x: merged_scores[x]['score'], reverse=True)
        top_ids = top_ids[:search_limit]
        
        # Step 6: Fetch full changelists from metadata DB
        changelists = self.metadata_db.get_changelists_by_ids(top_ids)
        
        # Step 7: Apply metadata filters
        filtered_results = self._apply_filters(changelists, merged_scores, params)
        
        # Step 8: Apply minimum similarity score filter (absolute threshold)
        before_score_filter = len(filtered_results)
        filtered_results = [r for r in filtered_results if r.similarity_score >= self.min_similarity_score]
        filtered_count = before_score_filter - len(filtered_results)
        if filtered_count > 0:
            log_info(f"Filtered out {filtered_count} results below minimum score {self.min_similarity_score}")
        
        # Step 9: Apply relative threshold filter (percentage of top score)
        if filtered_results and self.relative_threshold < 1.0:
            top_score = filtered_results[0].similarity_score  # Already sorted by score
            relative_cutoff = top_score * (1 - self.relative_threshold)
            before_count = len(filtered_results)
            filtered_results = [r for r in filtered_results if r.similarity_score >= relative_cutoff]
            if len(filtered_results) < before_count:
                log_info(f"Relative threshold filtered out {before_count - len(filtered_results)} results (cutoff: {relative_cutoff:.3f})")
        
        # Step 10: Sort by final score and limit to max_changes
        filtered_results.sort(key=lambda x: x.similarity_score, reverse=True)
        final_results = filtered_results[:params.max_changes]
        
        log_info(f"Found {len(final_results)} results after filtering")
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
        
        # Group by changelist_id, tracking both max score and corresponding chroma_id
        grouped: Dict[str, Tuple[float, Optional[str]]] = {}
        for result in chunk_results:
            if result.changelist_id not in grouped:
                grouped[result.changelist_id] = (result.similarity_score, result.chroma_id)
            else:
                # Max pooling: keep the highest score and its chroma_id
                current_score, current_chroma_id = grouped[result.changelist_id]
                if result.similarity_score > current_score:
                    grouped[result.changelist_id] = (result.similarity_score, result.chroma_id)
        
        # Convert back to VectorSearchResult list
        return [
            VectorSearchResult(
                changelist_id=cid,
                similarity_score=score,
                chroma_id=chroma_id
            )
            for cid, (score, chroma_id) in grouped.items()
        ]
    
    def _merge_scores(
        self,
        metadata_results: List[VectorSearchResult],
        diff_results: List[VectorSearchResult]
    ) -> Dict[str, Dict[str, any]]:
        """
        Merge scores from metadata and diff searches.
        
        For duplicate changelist IDs, combines scores using weighted average:
        - Metadata: 60% weight (higher signal for "what/why")
        - Diff: 40% weight (implementation details)
        
        When both sources contribute, the source is set to whichever had
        the higher individual similarity score.
        
        Args:
            metadata_results: Results from metadata search
            diff_results: Results from diff search
            
        Returns:
            Dictionary mapping changelist_id to {'score': float, 'source': str}
        """
        merged: Dict[str, Dict[str, any]] = {}
        
        # Add metadata results
        for result in metadata_results:
            merged[result.changelist_id] = {
                'score': result.similarity_score,
                'source': 'metadata',
                'metadata_score': result.similarity_score,
                'diff_score': None,
                'chroma_id': result.chroma_id
            }
        
        # Merge diff results
        for result in diff_results:
            if result.changelist_id in merged:
                # Combine scores: 60% metadata, 40% diff
                metadata_score = merged[result.changelist_id]['metadata_score']
                diff_score = result.similarity_score
                combined_score = (metadata_score * 0.6) + (diff_score * 0.4)
                
                # Determine source based on which individual score is higher
                if metadata_score > diff_score:
                    source = 'metadata'
                    chroma_id = merged[result.changelist_id]['chroma_id']
                elif diff_score > metadata_score:
                    source = 'diff'
                    chroma_id = result.chroma_id
                else:
                    # Equal scores - use metadata as tiebreaker
                    source = 'metadata'
                    chroma_id = merged[result.changelist_id]['chroma_id']
                
                merged[result.changelist_id] = {
                    'score': combined_score,
                    'source': source,
                    'metadata_score': metadata_score,
                    'diff_score': diff_score,
                    'chroma_id': chroma_id
                }
            else:
                # Only in diff results
                merged[result.changelist_id] = {
                    'score': result.similarity_score,
                    'source': 'diff',
                    'metadata_score': None,
                    'diff_score': result.similarity_score,
                    'chroma_id': result.chroma_id
                }
        
        return merged
    
    def _apply_filters(
        self,
        changelists: List[Changelist],
        scores: Dict[str, Dict[str, any]],
        params: QueryParams
    ) -> List[SearchResult]:
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
            Filtered list of SearchResult objects
        """
        results: List[SearchResult] = []
        
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
            results.append(SearchResult(
                changelist=changelist,
                similarity_score=score_info['score'],
                source=score_info['source'],
                chroma_id=score_info.get('chroma_id')
            ))
        
        return results
    
    def close(self):
        """Clean up resources."""
        self.metadata_db.close()
        self.vector_db.close()