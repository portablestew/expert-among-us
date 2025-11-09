import subprocess
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone, timedelta

from expert_among_us.vcs.base import VCSProvider
from expert_among_us.models.changelist import Changelist
from expert_among_us.utils.truncate import filter_binary_from_diff, is_binary_file
from expert_among_us.utils.debug import DebugLogger
 
class Git(VCSProvider):
    """Git VCS provider implementation."""
 
    def __init__(self, debug_logger: Optional[callable] = None):
        # debug_logger is deprecated; we keep it only for backward compatibility.
        # Newer code paths rely on DebugLogger.is_enabled() directly.
        self._debug_logger = debug_logger

        # In-memory commit hash cache for efficient chronological pagination.
        # These are implementation details of the Git provider and are not part
        # of the public VCSProvider interface.
        #
        # Design:
        # - _hash_cache: ordered list of commit hashes (oldest → newest)
        # - _hash_index: mapping commit_hash → index in _hash_cache
        # - _hash_cache_key: (workspace_path, subdirs_tuple)
        #
        # get_commits_after() becomes stateless w.r.t. position:
        # - Indexer (or caller) provides after_hash as the cursor
        # - We look up after_hash in _hash_index to find the next slice
        self._hash_cache: list[str] | None = None
        self._hash_index: dict[str, int] | None = None
        self._hash_cache_key: tuple | None = None  # (workspace_path, subdirs_tuple)
    
    @staticmethod
    def detect(workspace_path: str) -> bool:
        """Detect if this workspace uses Git."""
        git_dir = Path(workspace_path) / ".git"
        return git_dir.exists() and git_dir.is_dir()
    
    def get_commits_after(
        self,
        workspace_path: str,
        after_hash: str | None,
        batch_size: int,
        subdirs: Optional[list[str]] = None,
    ) -> list[Changelist]:
        """Get commits after a specific hash in chronological order (oldest → newest).

        This is the primary commit traversal method used by the unified indexer.

        The implementation uses a two-phase strategy for efficiency and correctness:
        - Phase 1 (once per (workspace_path, after_hash, subdirs) tuple): fetch all
          matching commit hashes in chronological order and cache them.
        - Phase 2 (per call): slice the next `batch_size` hashes from the cache and
          fetch full commit details only for those hashes.

        This avoids flip-flopping caused by combining --reverse with -n over ranges,
        and ensures stable, contiguous pagination from oldest to newest.
        """
        if batch_size == 0:
            return []

        # Normalize subdirs for cache key (after_hash is NOT part of the key;
        # it is treated as a cursor into the global ordered sequence).
        subdirs_tuple = tuple(sorted(subdirs)) if subdirs else None
        cache_key = (workspace_path, subdirs_tuple)

        # (Re)build cache if it does not exist or the parameters changed
        if self._hash_cache is None or self._hash_cache_key != cache_key:
            self._hash_cache = self._fetch_all_hashes(
                workspace_path=workspace_path,
                subdirs=subdirs,
            )
            # Build index for O(1) lookup of positions
            self._hash_index = {
                commit_hash: idx for idx, commit_hash in enumerate(self._hash_cache)
            }
            self._hash_cache_key = cache_key

            if DebugLogger.is_enabled():
                from expert_among_us.utils.progress import console as progress_console
                progress_console.print(
                    f"[dim]Git.get_commits_after: cached {len(self._hash_cache)} commits "
                    f"(after={after_hash or 'START'}, subdirs={subdirs_tuple})[/dim]"
                )

        # Determine starting index based on after_hash cursor
        if after_hash:
            # If after_hash is unknown for this (workspace_path, subdirs) view,
            # validate whether it's a real commit or truly invalid.
            if not self._hash_index:
                return []
            start_idx = self._hash_index.get(after_hash)
            if start_idx is None:
                # Hash not in our cache - validate it's a real commit to catch user errors
                validate_cmd = ["git", "-C", workspace_path, "cat-file", "-e", after_hash]
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
        changelists = self._fetch_commits_by_hashes(
            workspace_path=workspace_path,
            hashes=batch_hashes,
            subdirs=subdirs,
        )

        return changelists
    
    def _invalidate_cache(self) -> None:
        """Clear commit-hash cache and index."""
        self._hash_cache = None
        self._hash_index = None
        self._hash_cache_key = None

    def _fetch_all_hashes(
        self,
        workspace_path: str,
        subdirs: Optional[list[str]] = None,
    ) -> list[str]:
        """Fetch all commit hashes in chronological order (oldest → newest) for the
        given workspace/subdir filter.

        This is a lightweight operation (hashes only) and is used to build a stable,
        contiguous commit sequence for pagination.
        """
        cmd = [
            "git",
            "-C",
            workspace_path,
            "log",
            "--no-merges",
            "--reverse",
            "--format=%H",
        ]

        if subdirs:
            cmd.append("--")
            cmd.extend(subdirs)

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
        workspace_path: str,
        hashes: list[str],
        subdirs: Optional[list[str]] = None,
    ) -> list[Changelist]:
        """Fetch full commit details for the given hashes using batched git calls.

        Strategy:
        - One `git log` to get metadata lines for the requested hashes.
        - One `git show` (or equivalent) to batch diffs for all requested commits.
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
            workspace_path,
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

        # 2) Batch fetch diffs for all commits.
        # We use a custom header "commit <hash>" to delimit sections.
        diff_cmd = [
            "git",
            "-C",
            workspace_path,
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

        diff_by_commit: dict[str, str] = {}
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
            workspace_path,
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
        current_hash = None
        current_files: list[str] = []

        for line in files_result.stdout.splitlines():
            if line.startswith("commit "):
                # Flush previous
                if current_hash is not None:
                    files_by_commit[current_hash] = current_files
                current_hash = line[len("commit ") :].strip()
                current_files = []
            else:
                stripped = line.strip()
                if not stripped:
                    continue
                # Expect NAME-STATUS line: "STATUS<TAB>path"
                parts = stripped.split("\t", 1)
                if len(parts) == 2:
                    _status, path = parts
                    current_files.append(path)
        if current_hash is not None:
            files_by_commit[current_hash] = current_files

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

            # Lookup pre-fetched diff; default to empty string if missing
            raw_diff = diff_by_commit.get(commit_hash, "")
            # Filter binary content from diff
            diff, binary_files = filter_binary_from_diff(raw_diff)

            # Skip commits with empty diffs (consistent with previous behavior)
            if not diff or not diff.strip():
                continue

            # Lookup pre-fetched file list for this commit
            files = files_by_commit.get(commit_hash, [])

            changelist = Changelist(
                id=commit_hash,
                expert_name="",  # Will be set by the caller
                timestamp=timestamp,
                author=f"{author_name} <{author_email}>",
                message=message,
                diff=diff,
                files=files,
            )
            changelists.append(changelist)

        return changelists

    # Legacy pagination API preserved for backward compatibility with existing tests.
    # New code paths should use get_commits_after() instead.
    def get_commits_page(
        self,
        workspace_path: str,
        since_hash: Optional[str],
        page: int,
        page_size: int,
        subdirs: Optional[list[str]] = None,
    ) -> list[Changelist]:
        """Legacy paginated commit retrieval used by older tests.

        Semantics preserved:
        - Uses `git log` with optional `since_hash` to define the range.
        - Raises subprocess.CalledProcessError when git reports an invalid range
          (e.g. nonexistent since_hash), as expected by tests.
        """
        if page_size <= 0:
            return []

        # Base git log command; keep behavior close to original implementation.
        cmd = [
            "git",
            "-C",
            workspace_path,
            "log",
            "--no-merges",
            "--pretty=format:%H|%an|%ae|%at|%s",
        ]

        if since_hash:
            # For the legacy API, use since_hash as a lower bound; invalid hashes
            # will cause git log to fail, which we propagate to match tests.
            cmd.append(f"{since_hash}..HEAD")

        if subdirs:
            cmd.append("--")
            cmd.extend(subdirs)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            # Preserve legacy behavior: surface invalid range as CalledProcessError
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                result.stdout,
                result.stderr,
            )

        # Paginate lines manually
        lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
        start = page * page_size
        end = start + page_size
        page_lines = lines[start:end]

        return self._parse_commits(workspace_path, "\n".join(page_lines))

    def _parse_commits(self, workspace_path: str, stdout: str) -> list[Changelist]:
        """Parse commit output and return list of Changelist objects.

        Deprecated compatibility helper for older tests that exercise the
        legacy pagination APIs. Newer code paths should rely on the batched
        _fetch_commits_by_hashes() implementation instead.
        """
        changelists: list[Changelist] = []
        for line in stdout.strip().split("\n"):
            if not line:
                continue

            parts = line.split("|", 4)
            if len(parts) < 5:
                continue

            commit_hash, author_name, author_email, timestamp_str, message = parts
            try:
                # Create timezone-aware datetime in UTC
                timestamp = datetime.fromtimestamp(int(timestamp_str), tz=timezone.utc)
            except (TypeError, ValueError):
                continue

            # Get diff and files for this commit via git show, matching previous behavior.
            diff_cmd = ["git", "-C", workspace_path, "show", "--format=", commit_hash]
            diff_result = subprocess.run(
                diff_cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            diff = diff_result.stdout if diff_result.returncode == 0 else ""

            # Filter binary content from diff
            diff, _binary_files = filter_binary_from_diff(diff)

            # Skip commits with empty diffs (merge commits are already excluded by --no-merges)
            if not diff or not diff.strip():
                continue

            files_cmd = [
                "git",
                "-C",
                workspace_path,
                "show",
                "--name-status",
                "--format=",
                commit_hash,
            ]
            files_result = subprocess.run(
                files_cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            files: list[str] = []
            if files_result.returncode == 0:
                for file_line in files_result.stdout.strip().split("\n"):
                    if not file_line:
                        continue
                    parts = file_line.split("\t", 1)
                    if len(parts) == 2:
                        _status, filepath = parts
                        files.append(filepath)

            changelist = Changelist(
                id=commit_hash,
                expert_name="",  # Will be set by the caller
                timestamp=timestamp,
                author=f"{author_name} <{author_email}>",
                message=message,
                diff=diff,
                files=files,
            )
            changelists.append(changelist)

        return changelists
    
    def get_latest_commit_time(
        self,
        workspace_path: str,
        subdirs: Optional[list[str]] = None,
    ) -> Optional[datetime]:
        """Get the timestamp of the most recent commit."""
        cmd = ["git", "-C", workspace_path, "log", "-1", "--format=%at"]
        
        if subdirs:
            cmd.append("--")
            cmd.extend(subdirs)
        
        # Use errors='replace' to handle non-UTF-8 characters gracefully
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if result.returncode != 0 or not result.stdout.strip():
            return None
        
        timestamp_str = result.stdout.strip()
        # Create timezone-aware datetime in UTC
        return datetime.fromtimestamp(int(timestamp_str), tz=timezone.utc)

    def get_total_commit_count(
        self,
        workspace_path: str,
        subdirs: Optional[list[str]] = None,
    ) -> int:
        """Return total number of non-merge commits considered by this provider.

        Semantics:
        - Matches get_commits_after(): excludes merge commits via --no-merges.
        - When subdirs is provided, restricts to commits that touch those paths.
        - Returns 0 on any git error.
        """
        cmd = [
            "git",
            "-C",
            workspace_path,
            "rev-list",
            "--all",
            "--no-merges",
            "--count",
        ]

        if subdirs:
            cmd.append("--")
            cmd.extend(subdirs)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            return 0

        output = result.stdout.strip()
        try:
            return int(output) if output else 0
        except ValueError:
            return 0
    
    def get_file_content_at_commit(
        self,
        workspace_path: str,
        file_path: str,
        commit_hash: str,
    ) -> Optional[str]:
        """Get file content at a specific commit hash.

        This is a thin wrapper around the batched get_files_content_at_commit()
        for backward compatibility and to preserve existing semantics.

        Args:
            workspace_path: Path to git repository
            file_path: Relative path to file
            commit_hash: Git commit hash

        Returns:
            File content as string, or None if file doesn't exist or is binary
        """
        results = self.get_files_content_at_commit(
            workspace_path=workspace_path,
            file_paths=[file_path],
            commit_hash=commit_hash,
        )
        return results.get(file_path)

    def get_files_content_at_commit(
        self,
        workspace_path: str,
        file_paths: list[str],
        commit_hash: str,
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
        cmd = ["git", "-C", workspace_path, "cat-file", "--batch"]

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
        workspace_path: str,
        commit_hash: str,
        subdirs: Optional[list[str]] = None,
    ) -> list[str]:
        """Get list of tracked files at a specific commit.

        Args:
            workspace_path: Path to git repository.
            commit_hash: Git commit hash.
            subdirs: Optional subdirectories to filter by.

        Returns:
            List of file paths (relative to workspace root) tracked at the commit.
        """
        cmd = ["git", "-C", workspace_path, "ls-tree", "-r", "--name-only", commit_hash]

        if subdirs:
            cmd.append("--")
            cmd.extend(subdirs)

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
    