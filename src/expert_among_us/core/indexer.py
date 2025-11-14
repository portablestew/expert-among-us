from typing import List, Optional
from expert_among_us.models.changelist import Changelist
from expert_among_us.models.file_chunk import FileChunk
from expert_among_us.embeddings.base import Embedder
from expert_among_us.vcs.base import VCSProvider
from expert_among_us.db.metadata.base import MetadataDB
from expert_among_us.db.vector.base import VectorDB
from expert_among_us.utils.chunking import chunk_text_with_lines
from expert_among_us.utils.truncate import is_binary_file
from expert_among_us.utils.sanitization import TextSanitizer
from expert_among_us.utils.progress import console, log_info
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn, TaskID, TimeElapsedColumn, TimeRemainingColumn


class Indexer:
    """Unified indexer that works against an abstract VCSProvider.

    All VCS access must go through the injected VCSProvider instance to keep
    the indexing pipeline pluggable and independent of concrete VCS types.
    """

    def __init__(
        self,
        expert_config: dict,
        vcs: VCSProvider,
        metadata_db: MetadataDB,
        vector_db: VectorDB,
        embedder: Embedder,
        settings,
    ):
        """Create a new Indexer.

        Args:
            expert_config: Expert configuration dictionary.
            vcs: Concrete VCS provider implementing VCSProvider.
            metadata_db: Metadata database instance.
            vector_db: Vector database instance.
            embedder: Embedding provider instance.
        """
        self.expert_config = expert_config
        self.vcs: VCSProvider = vcs
        self.metadata_db: MetadataDB = metadata_db
        self.vector_db: VectorDB = vector_db
        self.embedder = embedder
        self.settings = settings
        
        # Initialize text sanitizer for removing high-entropy patterns
        # Sanitization happens before embedding but after SQLite storage,
        # so search results show original content while embeddings are clean
        self.sanitizer = TextSanitizer()

        # Static description constants for progress tasks
        self._TASK_DESC_FILE_READ = "[cyan]  ├─ Reading files from VCS"
        self._TASK_DESC_FILE_EMBED = "[cyan]  ├─ Embedding file chunks"
        self._TASK_DESC_FILE_STORE = "[cyan]  ├─ Storing file data"
        self._TASK_DESC_COMMIT_META = "[cyan]  ├─ Embedding commit metadata"
        self._TASK_DESC_COMMIT_DIFF = "[cyan]  ├─ Embedding diff chunks"
        self._TASK_DESC_COMMIT_STORE = "[cyan]  └─ Storing commit data"

        # Progress task IDs for persistent multi-level display
        self._overall_task: Optional[TaskID] = None
        # File operation tasks
        self._file_read_task: Optional[TaskID] = None
        self._file_embed_task: Optional[TaskID] = None
        self._file_store_task: Optional[TaskID] = None
        # Commit operation tasks
        self._commit_meta_task: Optional[TaskID] = None
        self._commit_diff_task: Optional[TaskID] = None
        self._commit_store_task: Optional[TaskID] = None

        # Rich Progress is opt-in and used ONLY for:
        # - Processing files into databases
        # - Processing commits into databases
        # Guard all uses with `if self.progress` so it can be disabled easily.
        self.progress: Progress | None = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

    def _add_arrow(self, desc: str) -> str:
        """Add ➤ arrow to indicate active task"""
        return desc.replace('  ├─', '➤ ├─').replace('  └─', '➤ └─')

    def index_unified(self, batch_size: int = 100, start_after: Optional[str] = None):
        """Index both files and commits in a single pass.

        Respects the max_commits limit from expert_config by tracking the total
        number of processed commits (including any previously indexed ones) and
        stopping once the configured cap is reached.

        Args:
            batch_size: Maximum commits per batch
            start_after: Optional commit hash to start indexing from (for testing specific commits)
        """
        workspace_path=self.expert_config["workspace_path"]
        
        # Get starting point
        # Override with start_after if provided (for testing specific commits)
        if start_after is not None:
            last_processed_id = start_after
        else:
            last_processed_id = self.metadata_db.get_last_processed_commit_hash(
                self.expert_config['name']
            )

        # Respect global max_commits across runs: treat already indexed commits as part of the total.
        max_commits = int(self.expert_config.get("max_commits", 10000))
        already_indexed = self.metadata_db.get_commit_count(self.expert_config["name"])

        # Total available commits according to VCS; clamp max_commits to this so we don't
        # overrun or show misleading progress when there are fewer commits than the cap.
        total_available = self.vcs.get_total_commit_count(
            workspace_path=workspace_path,
            subdirs=self.expert_config.get("subdirs"),
        )

        if isinstance(total_available, int) and total_available > 0:
            max_commits = min(max_commits, total_available)
        
        if batch_size > max_commits:
            batch_size = max_commits

        total_commits = already_indexed

        # If we've already reached or exceeded the cap, do nothing.
        if total_commits >= max_commits:
            console.print(f"[green]Already indexed {total_commits}/{max_commits} commits")
            return

        # Intro line before any processing: show constraints and starting point.
        console.print(
            f"[green]Indexing {self.expert_config['name']}: {already_indexed}/{max_commits} commits = {last_processed_id or 'OLDEST'}, batch_size={batch_size}\n"
        )

        # Single progress context for entire indexing operation
        with self.progress:
            # Create persistent progress tasks
            self._overall_task = self.progress.add_task(
                f"[green]Indexing commits: {workspace_path}",
                total=max_commits,
                completed=already_indexed
            )
            # File operation tasks (total=1 prevents scrolling, start=True+stop prevents pulse and stops timer)
            self._file_read_task = self.progress.add_task(
                self._TASK_DESC_FILE_READ,
                total=1,
                visible=True,
                start=True
            )
            self.progress.stop_task(self._file_read_task)
            
            self._file_embed_task = self.progress.add_task(
                self._TASK_DESC_FILE_EMBED,
                total=1,
                visible=True,
                start=True
            )
            self.progress.stop_task(self._file_embed_task)
            
            self._file_store_task = self.progress.add_task(
                self._TASK_DESC_FILE_STORE,
                total=1,
                visible=True,
                start=True
            )
            self.progress.stop_task(self._file_store_task)
            
            # Commit operation tasks (total=1 prevents scrolling, start=True+stop prevents pulse and stops timer)
            self._commit_meta_task = self.progress.add_task(
                self._TASK_DESC_COMMIT_META,
                total=1,
                visible=True,
                start=True
            )
            self.progress.stop_task(self._commit_meta_task)
            
            self._commit_diff_task = self.progress.add_task(
                self._TASK_DESC_COMMIT_DIFF,
                total=1,
                visible=True,
                start=True
            )
            self.progress.stop_task(self._commit_diff_task)
            
            self._commit_store_task = self.progress.add_task(
                self._TASK_DESC_COMMIT_STORE,
                total=1,
                visible=True,
                start=True
            )
            self.progress.stop_task(self._commit_store_task)
            
            batch_num = 0
            
            while total_commits < max_commits:
                # Fetch next batch of commits
                batch = self.vcs.get_commits_after(
                    workspace_path=workspace_path,
                    after_hash=last_processed_id,
                    batch_size=batch_size,
                    subdirs=self.expert_config.get('subdirs')
                )
                
                if not batch:
                    console.print(f"[green]All commits processed: {total_commits}/{max_commits}")
                    break
                
                # Print starting point on first batch only
                if batch_num == 0:
                    start_commit = batch[0]  # Oldest commit in first batch
                    console.print(
                        f"[green]🚀 Start: commit {start_commit.id}, "
                        f"{start_commit.timestamp.strftime('%Y-%m-%d %H:%M:%S%z')}"
                    )
                
                batch_num += 1
                
                # Reset all subtasks to inactive state for the new batch
                # total=1 prevents scrolling, start=True+stop prevents pulse and stops timer
                self.progress.update(self._file_read_task, description=self._TASK_DESC_FILE_READ, completed=0, total=1, start=True)
                self.progress.stop_task(self._file_read_task)
                
                self.progress.update(self._file_embed_task, description=self._TASK_DESC_FILE_EMBED, completed=0, total=1, start=True)
                self.progress.stop_task(self._file_embed_task)
                
                self.progress.update(self._file_store_task, description=self._TASK_DESC_FILE_STORE, completed=0, total=1, start=True)
                self.progress.stop_task(self._file_store_task)
                
                self.progress.update(self._commit_meta_task, description=self._TASK_DESC_COMMIT_META, completed=0, total=1, start=True)
                self.progress.stop_task(self._commit_meta_task)
                
                self.progress.update(self._commit_diff_task, description=self._TASK_DESC_COMMIT_DIFF, completed=0, total=1, start=True)
                self.progress.stop_task(self._commit_diff_task)
                
                self.progress.update(self._commit_store_task, description=self._TASK_DESC_COMMIT_STORE, completed=0, total=1, start=True)
                self.progress.stop_task(self._commit_store_task)

                # Enforce max_commits across batches by trimming this batch if needed.
                remaining_allowed = max_commits - total_commits
                if remaining_allowed <= 0:
                    break
                if len(batch) > remaining_allowed:
                    batch = batch[:remaining_allowed]
                
                # Collect unique file paths from batch
                file_paths = set()
                for commit in batch:
                    file_paths.update(commit.files)
                
                # Index files with progress
                if file_paths:
                    existing, missing = self._check_file_existence(file_paths, batch[-1].id)

                    if existing:
                        self._index_files_at_commit(existing, batch[-1].id)

                    if missing:
                        for file_path in missing:
                            self._delete_file_chunks(file_path)

                # Index commits with batched embeddings + progress
                self._index_commit_batch(batch)

                # Update tracking
                final_commit = batch[-1]
                last_processed_id = final_commit.id
                total_commits += len(batch)
                self.metadata_db.update_last_processed_commit(
                    self.expert_config['name'],
                    last_processed_id,
                )

                # Batch summary with final commit hash (continuation point) and its timestamp.
                console.print(
                    f"[green]✅ Batch {batch_num}: "
                    f"commit {final_commit.id}, "
                    f"{final_commit.timestamp.strftime('%Y-%m-%d %H:%M:%S%z')}"
                )
                
                # Update overall progress
                self.progress.update(
                    self._overall_task,
                    completed=total_commits
                )

        # If loop exited because total_commits hit max_commits, print a clear message.
        if total_commits >= max_commits:
            console.print(
                f"[green]Reached max_commits={max_commits} "
                f"({total_commits}/{max_commits} commits indexed)"
            )
    
    def _check_file_existence(self, file_paths: set[str], revision_id: str) -> tuple[set[str], set[str]]:
        """Check which files exist at the specified revision."""
        
        tracked_files = set(self.vcs.get_tracked_files_at_commit(
            self.expert_config['workspace_path'],
            revision_id,
            subdirs=self.expert_config.get('subdirs')
        ))
        
        existing = file_paths & tracked_files
        missing = file_paths - tracked_files
        
        return existing, missing
    
    def _delete_file_chunks(self, file_path: str):
        """Remove file chunks from both databases.
        
        Uses sqlite as source of truth: fetches chunk IDs from sqlite,
        deletes them from Chroma, then removes from sqlite. This ensures
        both databases stay in sync even when Chroma lacks metadata.
        """
        # Get chunk IDs from sqlite (source of truth)
        chunk_ids = self.metadata_db.get_file_chunk_ids(file_path)
        
        # Delete from Chroma using exact IDs
        if chunk_ids:
            self.vector_db.delete_file_chunks(chunk_ids)
        
        # Delete from sqlite
        self.metadata_db.delete_file_chunks_by_path(file_path)
     
    def _index_files_at_commit(self, file_paths: set[str], revision_id: str, min_file_size=128) -> None:
        """Index files at a specific commit using batched VCS reads.

        File reading semantics:
        - Fetch all file contents for the given revision_id in a single batched call
          via VCSProvider.get_files_content_at_commit().
        - Binary files and missing files are skipped (treated as None).
        - Text content is chunked and stored as before.
        """
        if not file_paths:
            return

        # Task 1: File reading with progress
        self.progress.start_task(self._file_read_task)
        self.progress.update(
            self._file_read_task,
            description=self._add_arrow(self._TASK_DESC_FILE_READ),
            total=len(file_paths),
            completed=0
        )

        # Define progress callback for VCS provider
        def update_file_progress(current: int, total: int) -> None:
            """Called by VCS provider after each batch of files is read."""
            self.progress.update(self._file_read_task, completed=current)

        # Batched read from VCS with progress tracking
        contents_by_path = self.vcs.get_files_content_at_commit(
            workspace_path=self.expert_config["workspace_path"],
            file_paths=list(file_paths),
            commit_hash=revision_id,
            progress_callback=update_file_progress,
        )

        # Remove arrow on completion
        self.progress.update(self._file_read_task, description=self._TASK_DESC_FILE_READ)
        self.progress.stop_task(self._file_read_task)

        file_chunks_map: dict[str, list[tuple[str, int, int]]] = {}
        total_chunks = 0

        # Normalize and filter results:
        # - Ensure an entry for every requested file (defensive if provider misbehaves).
        # - Skip None/binary files, or files that are too small.
        for file_path in file_paths:
            content = contents_by_path.get(file_path)
            if not content or len(content) < min_file_size:
                continue

            # Defensive binary check in case provider didn't filter.
            try:
                if is_binary_file(content.encode("utf-8", errors="ignore")):
                    continue
            except Exception:
                # If binary detection fails, skip conservatively.
                continue

            chunks = chunk_text_with_lines(content, chunk_size=8192)
            if chunks:
                file_chunks_map[file_path] = chunks
                total_chunks += len(chunks)

        if total_chunks == 0:
            return

        # Build FileChunk objects (CPU-only, relatively fast)
        all_chunks: list[FileChunk] = []
        for file_path, chunks in file_chunks_map.items():
            for idx, (text, line_start, line_end) in enumerate(chunks):
                all_chunks.append(
                    FileChunk(
                        file_path=file_path,
                        chunk_index=idx,
                        content=text,
                        line_start=line_start,
                        line_end=line_end,
                        revision_id=revision_id,
                    )
                )

        # Task 2: Embedding with progress
        self.progress.start_task(self._file_embed_task)
        self.progress.update(
            self._file_embed_task,
            description=self._add_arrow(self._TASK_DESC_FILE_EMBED),
            total=len(all_chunks),
            completed=0
        )
        
        def update_embed_progress(current: int, total: int) -> None:
            self.progress.update(self._file_embed_task, completed=current)
        
        # Sanitize content before embedding to reduce high-entropy noise (if enabled)
        if self.settings.enable_sanitization:
            sanitized_contents = [self.sanitizer.sanitize(c.content) for c in all_chunks]
        else:
            sanitized_contents = [c.content for c in all_chunks]
        
        embeddings = self.embedder.embed_batch(
            sanitized_contents,
            progress_callback=update_embed_progress
        )
        
        # Remove arrow on completion
        self.progress.update(self._file_embed_task, description=self._TASK_DESC_FILE_EMBED)
        self.progress.stop_task(self._file_embed_task)

        # Task 3: Storage with progress
        self.progress.start_task(self._file_store_task)
        self.progress.update(
            self._file_store_task,
            description=self._add_arrow(self._TASK_DESC_FILE_STORE),
            total=len(file_chunks_map),
            completed=0
        )
        
        self._store_file_chunks(all_chunks, embeddings, self._file_store_task)
        
        # Remove arrow on completion
        self.progress.update(self._file_store_task, description=self._TASK_DESC_FILE_STORE)
        self.progress.stop_task(self._file_store_task)
    
    def _store_file_chunks(self, chunks: List[FileChunk], embeddings: List[List[float]], task_id: TaskID):
        """Store file chunks in both databases.

        Args:
            chunks: List of file chunks to store
            embeddings: Corresponding embeddings for each chunk
            task_id: Progress task ID for tracking storage progress
        """
        from collections import defaultdict

        if not chunks:
            return

        # Group chunks by file for metadata DB
        chunks_by_file: dict[str, list[FileChunk]] = defaultdict(list)
        for chunk in chunks:
            chunks_by_file[chunk.file_path].append(chunk)

        # Map embeddings back to chunks by chroma_id
        vector_by_id: dict[str, list[float]] = {}
        for chunk, embedding in zip(chunks, embeddings):
            vector_by_id[chunk.get_chroma_id()] = embedding

        # Insert per-file metadata, chunks, and vectors
        for file_path, file_chunks in chunks_by_file.items():
            first_chunk = file_chunks[0]
            file_size = len(first_chunk.content.encode("utf-8"))

            self.metadata_db.insert_file_content(
                file_path=file_path,
                expert_name=self.expert_config["name"],
                revision_id=first_chunk.revision_id,
                file_size=file_size,
                chunk_count=len(file_chunks),
            )

            self.metadata_db.insert_file_chunks(file_chunks)

            vectors_for_file: list[tuple[str, list[float]]] = []
            for chunk in file_chunks:
                vec = vector_by_id.get(chunk.get_chroma_id())
                if vec is not None:
                    vectors_for_file.append((chunk.get_chroma_id(), vec))

            if vectors_for_file:
                self.vector_db.insert_files(vectors_for_file)

            self.progress.update(task_id, advance=1)
    
    def _index_commit_batch(self, batch: list[Changelist]) -> None:
        """Process commits with batched embeddings.

        Uses progress tasks for each phase: metadata embedding, diff embedding, and storage.
        """
        if not batch:
            return

        # Prepare texts for batched embedding
        metadata_texts: list[str] = []
        diff_chunk_texts: list[str] = []
        diff_chunk_keys: list[tuple[str, int]] = []  # (commit_id, chunk_index)

        for commit in batch:
            # Metadata text (single entry per commit)
            metadata_texts.append(commit.get_metadata_text())

            # Diff chunks (0..N per commit)
            diff_chunks = chunk_text_with_lines(commit.diff, chunk_size=8192)
            for idx, (chunk_text, _, _) in enumerate(diff_chunks):
                diff_chunk_texts.append(chunk_text)
                diff_chunk_keys.append((commit.id, idx))

        # Task 1: Metadata embeddings
        self.progress.start_task(self._commit_meta_task)
        self.progress.update(
            self._commit_meta_task,
            description=self._add_arrow(self._TASK_DESC_COMMIT_META),
            total=len(metadata_texts),
            completed=0
        )
        
        def update_meta_progress(current: int, total: int) -> None:
            self.progress.update(self._commit_meta_task, completed=current)
        
        # Sanitize metadata before embedding to reduce high-entropy noise (if enabled)
        if self.settings.enable_sanitization:
            sanitized_metadata = [self.sanitizer.sanitize(text) for text in metadata_texts]
        else:
            sanitized_metadata = metadata_texts
        
        metadata_embeddings = self.embedder.embed_batch(
            sanitized_metadata,
            progress_callback=update_meta_progress
        )
        
        # Remove arrow on completion
        self.progress.update(self._commit_meta_task, description=self._TASK_DESC_COMMIT_META)
        self.progress.stop_task(self._commit_meta_task)

        # Task 2: Diff embeddings (conditional)
        diff_embeddings: list[list[float]] = []
        if diff_chunk_texts:
            self.progress.start_task(self._commit_diff_task)
            self.progress.update(
                self._commit_diff_task,
                description=self._add_arrow(self._TASK_DESC_COMMIT_DIFF),
                total=len(diff_chunk_texts),
                completed=0
            )
            
            def update_diff_progress(current: int, total: int) -> None:
                self.progress.update(self._commit_diff_task, completed=current)
            
            # Sanitize diff chunks before embedding to reduce high-entropy noise (if enabled)
            if self.settings.enable_sanitization:
                sanitized_diffs = [self.sanitizer.sanitize(text) for text in diff_chunk_texts]
            else:
                sanitized_diffs = diff_chunk_texts
            
            diff_embeddings = self.embedder.embed_batch(
                sanitized_diffs,
                progress_callback=update_diff_progress
            )
            
            # Remove arrow on completion
            self.progress.update(self._commit_diff_task, description=self._TASK_DESC_COMMIT_DIFF)
            self.progress.stop_task(self._commit_diff_task)

        # Build commit_id -> diff vectors mapping
        from collections import defaultdict
        commit_diff_vectors: dict[str, list[tuple[str, list[float]]]] = defaultdict(list)
        for (commit_id, chunk_idx), emb in zip(diff_chunk_keys, diff_embeddings):
            vector_id = f"{commit_id}_chunk_{chunk_idx}"
            commit_diff_vectors[commit_id].append((vector_id, emb))

        # Task 3: Storage
        self.progress.start_task(self._commit_store_task)
        self.progress.update(
            self._commit_store_task,
            description=self._add_arrow(self._TASK_DESC_COMMIT_STORE),
            total=len(batch),
            completed=0
        )

        # Store all metadata + diff vectors, updating progress per commit
        for idx, commit in enumerate(batch):
            metadata_emb = metadata_embeddings[idx]

            # Store metadata
            self.metadata_db.insert_changelists([commit])
            self.vector_db.insert_metadata([(commit.id, metadata_emb)])

            # Store diffs for this commit, if any
            diff_vectors = commit_diff_vectors.get(commit.id)
            if diff_vectors:
                self.vector_db.insert_diffs(diff_vectors)

            self.progress.update(self._commit_store_task, advance=1)

        # Remove arrow on completion
        self.progress.update(self._commit_store_task, description=self._TASK_DESC_COMMIT_STORE)
        self.progress.stop_task(self._commit_store_task)