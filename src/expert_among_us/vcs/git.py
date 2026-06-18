import subprocess
from pathlib import Path
from typing import Callable, Optional
from datetime import datetime, timezone, timedelta

from expert_among_us.vcs.base import VCSProvider
from expert_among_us.models.changelist import Changelist
from expert_among_us.utils.truncate import filter_binary_from_diff, is_binary_file, should_index_file, filter_diff_by_extensions, compact_diff, truncate_to_bytes
from expert_among_us.utils.debug import DebugLogger

# Maximum commits to fetch details for in a single git operation
MAX_COMMITS_PER_BATCH = 50
 
class Git(VCSProvider):
    """Git VCS provider implementation."""
 
    def __init__(self, settings):
        """Initialize Git provider.
        
        Args:
            settings: Settings instance with indexing configuration (required)
        """
        self._settings = settings

        # In-memory commit hash cache for efficient chronological pagination.
        # These are implementation details of the Git provider and are not part
        # of the public VCSProvider interface.
        #
        # Design:
        # - _hash_cache: ordered list of commit hashes (oldest → newest)
        # - _hash_index: mapping commit_hash → index in _hash_cache
        # - _hash_cache_key: project_root
        #
        # get_commits_after() becomes stateless w.r.t. position:
        # - Indexer (or caller) provides after_hash as the cursor
        # - We look up after_hash in _hash_index to find the next slice
        self._hash_cache: list[str] | None = None
        self._hash_index: dict[str, int] | None = None
        self._hash_cache_key: str | None = None  # project_root
    
    @staticmethod
    def detect(project_root: str) -> bool:
        """Detect if this project uses Git."""
        git_dir = Path(project_root) / ".git"
        return git_dir.exists() and git_dir.is_dir()
    
    def get_commits_after(
        self,
        project_root: str,
        after_hash: str | None,
        batch_size: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[Changelist]:
        """Get commits after a specific hash in chronological order (oldest → newest).

        This is the primary commit traversal method used by the unified indexer.

        The implementation uses a two-phase strategy for efficiency and correctness:
        - Phase 1 (once per project_root): fetch all matching commit hashes in
          chronological order and cache them.
        - Phase 2 (per call): slice the next `batch_size` hashes from the cache and
          fetch full commit details only for those hashes.

        This avoids flip-flopping caused by combining --reverse with -n over ranges,
        and ensures stable, contiguous pagination from oldest to newest.
        """
        if batch_size == 0:
            return []

        # Ensure the ordered hash cache for this project_root is built. This is
        # shared with get_total_commit_count() so the full history is walked only
        # once per project_root.
        self._ensure_hash_cache(project_root)

        if DebugLogger.is_enabled():
            from expert_among_us.utils.progress import console as progress_console
            progress_console.print(
                f"[dim]Git.get_commits_after: {len(self._hash_cache or [])} cached commits "
                f"(after={after_hash or 'START'})[/dim]"
            )

        # Determine starting index based on after_hash cursor
        if after_hash:
            # If after_hash is unknown for this project_root view,
            # validate whether it's a real commit or truly invalid.
            if not self._hash_index:
                return []
            start_idx = self._hash_index.get(after_hash)
            if start_idx is None:
                # Hash not in our cache - validate it's a real commit to catch user errors
                validate_cmd = ["git", "-C", project_root, "cat-file", "-e", after_hash]
                validate_result = subprocess.run(
                    validate_cmd,
                    capture_output=True,
                    text=True,
                )
                if validate_result.returncode != 0:
                    # Invalid commit hash - raise to match legacy behavior and catch errors
                    raise subprocess.CalledProcessError(
                        validate_result.returncode,
                        validate_cmd,
                        validate_result.stdout,
                        validate_result.stderr,
                    )
                # Valid commit but not in our filtered view - return empty
                return []
            start = start_idx + 1  # strictly after the cursor
        else:
            # No cursor: start from the beginning
            start = 0

        if not self._hash_cache:
            return []

        end = start + batch_size
        batch_hashes = self._hash_cache[start:end]

        if not batch_hashes:
            # No more commits after the given cursor.
            return []

        # Fetch full commit details for this batch only
        # Progress callback is handled inside _fetch_commits_by_hashes via sub-batching
        changelists = self._fetch_commits_by_hashes(
            project_root=project_root,
            hashes=batch_hashes,
            progress_callback=progress_callback,
        )

        return changelists
    
    def _invalidate_cache(self) -> None:
        """Clear commit-hash cache and index."""
        self._hash_cache = None
        self._hash_index = None
        self._hash_cache_key = None

    def _ensure_hash_cache(self, project_root: str) -> list[str]:
        """Return the cached ordered commit-hash list for ``project_root``.

        Builds the cache on first use (and whenever project_root changes), then
        reuses it. The same list backs both pagination (get_commits_after) and
        the total count (get_total_commit_count), so the full history is walked
        only once per project_root. Mirrors the Perforce provider's caching of
        ``_fetch_all_changelist_numbers``.

        Raises:
            subprocess.CalledProcessError: if the underlying ``git log`` fails
                (e.g. an empty repository with no commits).
        """
        if self._hash_cache is None or self._hash_cache_key != project_root:
            hashes = self._fetch_all_hashes(project_root=project_root)
            self._hash_cache = hashes
            # Build index for O(1) lookup of positions
            self._hash_index = {
                commit_hash: idx for idx, commit_hash in enumerate(hashes)
            }
            self._hash_cache_key = project_root

            if DebugLogger.is_enabled():
                from expert_among_us.utils.progress import console as progress_console
                progress_console.print(
                    f"[dim]Git._ensure_hash_cache: cached {len(hashes)} commits "
                    f"for {project_root}[/dim]"
                )
        return self._hash_cache

    def _fetch_all_hashes(
        self,
        project_root: str,
    ) -> list[str]:
        """Fetch all commit hashes in chronological order (oldest → newest) for the
        given project root.

        This is a lightweight operation (hashes only) and is used to build a stable,
        contiguous commit sequence for pagination.
        """
        cmd = [
            "git",
            "-C",
            project_root,
            "log",
            "--no-merges",
            "--reverse",
            "--format=%H",
        ]

        if DebugLogger.is_enabled():
            cmd_str = " ".join(str(part) for part in cmd)
            from expert_among_us.utils.progress import console as progress_console
            progress_console.print(f"[dim]Git._fetch_all_hashes: {cmd_str}[/dim]")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            # Preserve legacy semantics for callers/tests that expect a failure
            # (e.g. invalid ranges) to raise CalledProcessError.
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                result.stdout,
                result.stderr,
            )

        hashes: list[str] = [
            line.strip()
            for line in result.stdout.strip().split("\n")
            if line.strip()
        ]
        return hashes

    def _fetch_commits_by_hashes(
        self,
        project_root: str,
        hashes: list[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[Changelist]:
        """Fetch full commit details using sub-batching for stability.
        
        Processes commits in batches of MAX_COMMITS_PER_BATCH to prevent
        timeouts and memory issues with large commit sets.
        
        Args:
            project_root: Path to git repository
            hashes: List of commit hashes to fetch
            progress_callback: Optional callback(current, total) for progress updates
            
        Returns:
            List of Changelist objects for the requested commits
        """
        if not hashes:
            return []
        
        all_changelists: list[Changelist] = []
        total_hashes = len(hashes)
        
        # Process in sub-batches for stability and progress tracking
        for batch_start in range(0, len(hashes), MAX_COMMITS_PER_BATCH):
            batch_end = min(batch_start + MAX_COMMITS_PER_BATCH, len(hashes))
            batch_hashes = hashes[batch_start:batch_end]
            
            try:
                batch_changelists = self._fetch_single_commit_batch(
                    project_root=project_root,
                    hashes=batch_hashes,
                    embed_diffs=self._settings.embed_diffs,
                )
                all_changelists.extend(batch_changelists)
            except Exception as e:
                if DebugLogger.is_enabled():
                    from expert_among_us.utils.progress import console as progress_console
                    progress_console.print(
                        f"[dim red]Git._fetch_commits_by_hashes: "
                        f"error in batch {batch_start}-{batch_end}: {e}[/dim red]"
                    )
                # Continue with next batch despite error
            
            # Report progress after each sub-batch
            if progress_callback:
                try:
                    progress_callback(batch_end, total_hashes)
                except Exception:
                    pass  # Don't let callback errors disrupt processing
        
        return all_changelists
    
    def _fetch_single_commit_batch(
        self,
        project_root: str,
        hashes: list[str],
        embed_diffs: bool = True,
    ) -> list[Changelist]:
        """Fetch full commit details for a single batch of commits.

        Strategy:
        - One `git log` to get metadata lines for the requested hashes.
        - One `git show` (or equivalent) to batch diffs for all requested commits (if embed_diffs is True).
        - One `git show` to batch name-status for all requested commits.
        - Then assemble Changelist objects from these pre-fetched maps.

        This avoids issuing per-commit git-show processes while preserving
        existing semantics, including binary filtering and skipping empty diffs.
        """
        if not hashes:
            return []

        # 1) Fetch metadata for the specific hashes (one line per commit).
        meta_cmd = [
            "git",
            "-C",
            project_root,
            "log",
            "--no-merges",
            "--pretty=format:%H|%an|%ae|%at|%s",
            "--reverse",
            "--no-walk",
        ]
        # Restrict to the provided commits only, preserving the order enforced by --reverse.
        meta_cmd.extend(hashes)

        if DebugLogger.is_enabled():
            from expert_among_us.utils.progress import console as progress_console
            meta_cmd_str = " ".join(str(part) for part in meta_cmd)
            progress_console.print(
                f"[dim]Git._fetch_commits_by_hashes: metadata for {len(hashes)} commits via {meta_cmd_str}[/dim]"
            )

        meta_result = subprocess.run(
            meta_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if meta_result.returncode != 0:
            raise subprocess.CalledProcessError(
                meta_result.returncode,
                meta_cmd,
                meta_result.stdout,
                meta_result.stderr,
            )

        # Parse metadata lines into a stable list
        meta_lines = [
            line.strip()
            for line in meta_result.stdout.strip().split("\n")
            if line.strip()
        ]

        # 2) Conditionally fetch diffs for all commits (only when embed_diffs is True)
        diff_by_commit: dict[str, str] = {}
        
        if embed_diffs:
            diff_cmd = [
                "git",
                "-C",
                project_root,
                "show",
                "--no-merges",
                "--format=commit %H",
                "--patch",
            ]
            diff_cmd.extend(hashes)

            if DebugLogger.is_enabled():
                from expert_among_us.utils.progress import console as progress_console
                diff_cmd_str = " ".join(str(part) for part in diff_cmd)
                progress_console.print(
                    f"[dim]Git._fetch_commits_by_hashes: diffs via {diff_cmd_str}[/dim]"
                )

            diff_result = subprocess.run(
                diff_cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if diff_result.returncode != 0:
                raise subprocess.CalledProcessError(
                    diff_result.returncode,
                    diff_cmd,
                    diff_result.stdout,
                    diff_result.stderr,
                )

            current_hash: str | None = None
            current_lines: list[str] = []

            for line in diff_result.stdout.splitlines():
                if line.startswith("commit "):
                    # Flush previous commit block
                    if current_hash is not None:
                        diff_by_commit[current_hash] = "\n".join(current_lines).lstrip("\n")
                    # Start new block
                    current_hash = line[len("commit ") :].strip()
                    current_lines = []
                else:
                    current_lines.append(line)
            # Flush final commit block, if any
            if current_hash is not None:
                diff_by_commit[current_hash] = "\n".join(current_lines).lstrip("\n")

        # 3) Batch fetch name-status (changed files) for all commits.
        files_cmd = [
            "git",
            "-C",
            project_root,
            "show",
            "--no-merges",
            "--name-status",
            "--format=commit %H",
        ]
        files_cmd.extend(hashes)

        if DebugLogger.is_enabled():
            from expert_among_us.utils.progress import console as progress_console
            files_cmd_str = " ".join(str(part) for part in files_cmd)
            progress_console.print(
                f"[dim]Git._fetch_commits_by_hashes: files via {files_cmd_str}[/dim]"
            )

        files_result = subprocess.run(
            files_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if files_result.returncode != 0:
            raise subprocess.CalledProcessError(
                files_result.returncode,
                files_cmd,
                files_result.stdout,
                files_result.stderr,
            )

        files_by_commit: dict[str, list[str]] = {}
        omitted_by_commit: dict[str, int] = {}
        current_hash = None
        current_files: list[str] = []
        current_omitted = 0

        for line in files_result.stdout.splitlines():
            if line.startswith("commit "):
                # Flush previous
                if current_hash is not None:
                    files_by_commit[current_hash] = current_files
                    omitted_by_commit[current_hash] = current_omitted
                current_hash = line[len("commit ") :].strip()
                current_files = []
                current_omitted = 0
            else:
                stripped = line.strip()
                if not stripped:
                    continue
                # NAME-STATUS line: "STATUS<TAB>path", or for renames/copies
                # "R100<TAB>old<TAB>new" / "C100<TAB>old<TAB>new".
                fields = stripped.split("\t")
                if len(fields) < 2:
                    continue
                status = fields[0]
                path = fields[-1]  # new path for renames/copies, else the path
                # Filter by extension if specified
                if not should_index_file(path, self._settings.allowed_file_extensions):
                    continue
                code = status[0] if status else ""
                if code in ("R", "C"):
                    # Pure rename/copy (similarity 100) has no content change; keep
                    # only renames/copies that also modified the file.
                    score_str = status[1:]
                    score = int(score_str) if score_str.isdigit() else 0
                    if score >= 100:
                        current_omitted += 1
                        continue
                # A (add), M (modify), T (typechange), D (delete), and modified
                # renames/copies are content-bearing; deletes are kept so
                # HEAD-deletion cleanup still fires.
                current_files.append(path)
        if current_hash is not None:
            files_by_commit[current_hash] = current_files
            omitted_by_commit[current_hash] = current_omitted

        # 4) Assemble Changelist objects using the metadata + batched diffs/files.
        changelists: list[Changelist] = []

        for meta_line in meta_lines:
            parts = meta_line.split("|", 4)
            if len(parts) < 5:
                continue

            commit_hash, author_name, author_email, timestamp_str, message = parts

            try:
                timestamp = datetime.fromtimestamp(int(timestamp_str), tz=timezone.utc)
            except (TypeError, ValueError):
                # Skip malformed lines defensively
                continue

            # Handle diff based on embed_diffs flag
            diff = ""
            if embed_diffs:
                raw_diff = diff_by_commit.get(commit_hash, "")
                
                # Step 1: Filter binary content
                diff, binary_files = filter_binary_from_diff(raw_diff)
                
                # Step 2: Filter by extensions (BEFORE compacting - needs diff headers)
                if self._settings.allowed_file_extensions:
                    diff = filter_diff_by_extensions(diff, self._settings.allowed_file_extensions)
                
                # Step 3: Apply compact transformation (after all filtering)
                if self._settings.compact_diffs:
                    diff = compact_diff(diff, max_line_bytes=self._settings.compact_diff_max_line_bytes)
                
                # Step 4: Truncate individual commit diffs to limit
                if diff:
                    diff, was_truncated = truncate_to_bytes(diff, self._settings.max_diff_bytes_per_commit)
                    if was_truncated:
                        diff += "\n\n[TRUNCATED - commit diff exceeded limit]"

            # NOTE: commits with an empty diff (empty commits, binary-only
            # changes, or files excluded by the extension filter) are
            # intentionally retained. The message + file list are still useful
            # for semantic search. Dropping them would also let a run of
            # >= batch_size empty-diff commits look like end-of-history
            # (get_commits_after returns an empty batch), prematurely halting
            # indexing before later indexable commits.

            # Lookup pre-fetched file list for this commit
            files = files_by_commit.get(commit_hash, [])

            changelist = Changelist(
                id=commit_hash,
                expert_name="",  # Will be set by the caller
                project_name="",  # Will be set by the caller
                timestamp=timestamp,
                author=f"{author_name} <{author_email}>",
                message=message,
                diff=diff,
                files=files,
                omitted_file_count=omitted_by_commit.get(commit_hash, 0),
            )
            changelists.append(changelist)

        return changelists
    
    def get_latest_commit_time(
        self,
        project_root: str,
    ) -> Optional[datetime]:
        """Get the timestamp of the most recent commit."""
        cmd = ["git", "-C", project_root, "log", "-1", "--format=%at"]
        
        # Use errors='replace' to handle non-UTF-8 characters gracefully
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if result.returncode != 0 or not result.stdout.strip():
            return None
        
        timestamp_str = result.stdout.strip()
        # Create timezone-aware datetime in UTC
        return datetime.fromtimestamp(int(timestamp_str), tz=timezone.utc)

    def get_total_commit_count(
        self,
        project_root: str,
    ) -> int:
        """Return the total number of commits considered for indexing.

        Reuses the cached, ordered hash list — the same set traversed by
        get_commits_after() via ``git log --no-merges`` — so the count is
        consistent with what actually gets indexed and the full history is
        walked only once per project_root (the count is then a zero-cost
        ``len()`` on the in-memory list).

        Semantics:
        - Matches get_commits_after(): excludes merge commits; HEAD history.
        - Returns 0 for an empty repository (no commits).

        NOTE: This counts HEAD history, not ``--all`` refs. That is intentional:
        it must agree with the commits the indexer can actually traverse, so the
        derived "skipped" stat does not report phantom commits living only on
        unmerged branches.
        """
        try:
            return len(self._ensure_hash_cache(project_root))
        except subprocess.CalledProcessError:
            # Empty repository: `git log` exits non-zero when there are no commits.
            return 0
    
    def get_file_content_at_commit(
        self,
        project_root: str,
        file_path: str,
        commit_hash: str,
    ) -> Optional[str]:
        """Get file content at a specific commit hash.

        This is a thin wrapper around the batched get_files_content_at_commit()
        for backward compatibility and to preserve existing semantics.

        Args:
            project_root: Path to git repository
            file_path: Relative path to file
            commit_hash: Git commit hash

        Returns:
            File content as string, or None if file doesn't exist or is binary
        """
        results = self.get_files_content_at_commit(
            project_root=project_root,
            file_paths=[file_path],
            commit_hash=commit_hash,
        )
        return results.get(file_path)

    def get_files_content_at_commit(
        self,
        project_root: str,
        file_paths: list[str],
        commit_hash: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict[str, Optional[str]]:
        """Get content for multiple files at a specific commit using a single git process.

        Implementation details:
        - Uses `git cat-file --batch` to read all requested objects via one subprocess.
        - Sends `{commit_hash}:{path}` object refs via stdin (in chunks of 100 for memory safety).
        - Parses responses of form:
            `<sha> <type> <size>`
            `<content bytes...>`
          Or:
            `<ref> missing`
        - Treats:
            - Missing objects as None
            - Non-blobs as None
            - Binary blobs (per is_binary_file) as None
        - Always returns a dict entry for each requested path.
        - On any parsing/IO error for a given entry, falls back to None for that file.
        - Calls progress_callback(current, total) after each batch if provided.
        """
        # Normalize and handle trivial cases up-front.
        if not file_paths:
            return {}

        # Ensure deterministic mapping and avoid duplicate work.
        # Maintain original order for consistent debug logging, but only query unique paths.
        unique_paths: list[str] = []
        seen: set[str] = set()
        for p in file_paths:
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)

        # Prepare result with default None for all requested paths.
        results: dict[str, Optional[str]] = {p: None for p in unique_paths}

        # Build all refs once; we will stream them in batches to a single git cat-file process.
        refs: list[str] = [f"{commit_hash}:{p}" for p in unique_paths]

        # Start git cat-file --batch once; use binary mode to faithfully handle all content.
        cmd = ["git", "-C", project_root, "cat-file", "--batch"]

        if DebugLogger.is_enabled():
            from expert_among_us.utils.progress import console as progress_console
            cmd_str = " ".join(str(part) for part in cmd)
            progress_console.print(
                f"[dim]Git.get_files_content_at_commit: {len(unique_paths)} files via {cmd_str}[/dim]"
            )

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError:
            # If git cannot be executed at all, we already initialized results with None.
            return results

        assert proc.stdin is not None
        assert proc.stdout is not None

        # Helper to safely close stdin when done writing.
        def _close_stdin_safely() -> None:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass

        # Process refs in batches to avoid buffer deadlocks.
        # For each batch: write refs to stdin, read responses from stdout.
        # This interleaved approach prevents the git cat-file process from blocking
        # on a full stdout buffer while we're still writing to stdin.
        batch_size = 50
        try:
            for batch_start in range(0, len(refs), batch_size):
                batch_end = min(batch_start + batch_size, len(refs))
                batch_refs = refs[batch_start:batch_end]
                batch_paths = unique_paths[batch_start:batch_end]
                
                # Write this batch of refs to stdin
                data = ("\n".join(batch_refs) + "\n").encode("utf-8", errors="replace")
                proc.stdin.write(data)
                proc.stdin.flush()
                
                # Immediately read responses for this batch to avoid buffer deadlock
                for path, ref in zip(batch_paths, batch_refs):
                    header_bytes = proc.stdout.readline()
                    if not header_bytes:
                        # Unexpected EOF; remaining entries stay as None.
                        break

                    header = header_bytes.decode("utf-8", errors="replace").rstrip("\n")

                    # Missing entry: "<ref> missing"
                    if header.endswith(" missing"):
                        # Already defaulted to None.
                        continue

                    parts = header.split()
                    if len(parts) != 3:
                        # Malformed header; cannot trust following bytes for this entry.
                        continue

                    _sha, obj_type, size_str = parts
                    try:
                        size = int(size_str)
                    except ValueError:
                        # Invalid size; skip this entry.
                        continue

                    # Read the object body and trailing newline as raw bytes.
                    body = proc.stdout.read(size)
                    # After the body, git writes a single newline separator.
                    _ = proc.stdout.read(1)

                    if obj_type != "blob":
                        # Non-file objects are not considered file contents.
                        continue

                    if not body:
                        # Empty or missing content; leave as None.
                        continue

                    # Filter binary content BEFORE decoding.
                    try:
                        if is_binary_file(body):
                            # Binary blobs map to None.
                            continue
                    except Exception:
                        # If binary detection fails, be safe and treat as None.
                        continue

                    try:
                        text = body.decode("utf-8", errors="replace")
                    except Exception:
                        # Decoding failure; treat as None.
                        continue

                    results[path] = text
                
                # Report progress after processing each batch
                if progress_callback:
                    try:
                        progress_callback(batch_end, len(refs))
                    except Exception:
                        # Ignore callback errors to prevent disrupting file reading
                        pass
        except Exception:
            # On any unexpected error, results accumulated so far remain;
            # the rest stay as None.
            if DebugLogger.is_enabled():
                from expert_among_us.utils.progress import console as progress_console
                progress_console.print(
                    "[dim red]Git.get_files_content_at_commit: error while processing batch; remaining files left as None[/dim red]"
                )
        finally:
            # Close stdin to signal completion and ensure the process is reaped.
            _close_stdin_safely()
            try:
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        return results
    
    def get_tracked_files_at_commit(
        self,
        project_root: str,
        commit_hash: str,
    ) -> list[str]:
        """Get list of tracked files at a specific commit.

        Args:
            project_root: Path to git repository.
            commit_hash: Git commit hash.

        Returns:
            List of file paths (relative to the project root) tracked at the commit.
        """
        cmd = ["git", "-C", project_root, "ls-tree", "-r", "--name-only", commit_hash]

        if DebugLogger.is_enabled():
            from expert_among_us.utils.progress import console as progress_console
            cmd_str = " ".join(str(part) for part in cmd)
            progress_console.print(f"[dim]Git.get_tracked_files_at_commit: {cmd_str}[/dim]")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            return []

        files = [
            line.strip()
            for line in result.stdout.strip().split("\n")
            if line.strip()
        ]
        return files
    
    def get_commit_position(self, commit_id: Optional[str]) -> tuple[int, int]:
        """Get position of commit in ordered sequence for progress tracking.
        
        Args:
            commit_id: Commit hash, or None for start position
            
        Returns:
            Tuple of (commits_considered, total_commits)
        """
        # If no cache, return (0, 0)
        if not self._hash_cache:
            return (0, 0)
        
        # If commit_id is None, starting from beginning
        if not commit_id:
            return (0, len(self._hash_cache))
        
        # If no index built yet, return (0, total)
        if not self._hash_index:
            return (0, len(self._hash_cache))
        
        # Look up position and add +1 to convert from 0-based index to 1-based count
        position = self._hash_index.get(commit_id, -1)
        if position == -1:
            # Commit not in cache, return 0 (will be validated elsewhere)
            return (0, len(self._hash_cache))
        
        return (position + 1, len(self._hash_cache))
    