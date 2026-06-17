
"""Perforce VCS provider implementation.

This module provides a Perforce implementation of the VCSProvider interface,
enabling Expert Among Us to work with Perforce repositories. It uses the p4
CLI to interact with Perforce servers.
"""

import gc
import shutil
import socket
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Callable, Optional
from datetime import datetime, timezone

from expert_among_us.vcs.base import VCSProvider
from expert_among_us.models.changelist import Changelist
from expert_among_us.utils.truncate import filter_binary_from_diff, is_binary_file, should_index_file, compact_diff, truncate_to_bytes
from expert_among_us.utils.debug import DebugLogger


class DescribeResult(Enum):
    """Result status from p4 describe operation."""
    SUCCESS = "success"
    SIZE_LIMIT = "size_limit"
    CORRUPTION = "corruption"

# Automated users to exclude from changelist queries
# These patterns use Perforce wildcard syntax (* for multiple chars)
EXCLUDED_AUTOMATED_USERS = (
    # !!! Removed -- sometimes build jobs deliver useful commits, e.g. "preflight and commit" jobs
    #"*jenkins*",
    #"*builder*",
    #"lumbery@*",
)

# Maximum number of files to fetch diffs for per changelist
# Prevents timeouts from huge merges/refactors/codegen commits
# CLs with more files will have diffs for only the first MAX_FILES_PER_CL files (alphabetically)
MAX_FILES_PER_CL = 200

# Maximum output size from p4 describe command per batch
# Prevents timeouts and memory issues from huge changelists (e.g., 2.8 GB commits)
# Batches exceeding this limit will be split via binary search
MAX_DESCRIBE_OUTPUT_BYTES = 15 * 1024 * 1024  # 15 MB


class Perforce(VCSProvider):
    """Perforce VCS provider implementation.
    
    Implements the VCSProvider interface for Perforce repositories.
    Uses p4 CLI commands to interact with Perforce server.
    
    Architecture:
    - Changelist cache (_cl_cache): ordered list of CL numbers (oldest → newest)
    - Index (_cl_index): mapping CL number → position in cache
    - Cache key (_cl_cache_key): project_root for invalidation
    
    This mirrors the Git provider's caching strategy for efficient pagination.
    """
    
    def __init__(self, settings):
        """Initialize Perforce provider.
        
        Args:
            settings: Settings instance with indexing configuration (required)
        """
        self._settings = settings
        self._debug_logger = None  # Deprecated, kept for backward compatibility
        
        # Changelist number cache for efficient chronological pagination
        # Design matches Git._hash_cache pattern:
        # - _cl_cache: ordered list of CL numbers (oldest → newest)
        # - _cl_index: mapping cl_number → index in _cl_cache
        # - _cl_cache_key: project_root
        self._cl_cache: list[str] | None = None
        self._cl_index: dict[str, int] | None = None
        self._cl_cache_key: str | None = None
        
        # Cache for all user workspaces (shared between detect() and _get_workspace_mapping())
        # List of dicts with keys: host, root, client, depot_root
        self._user_workspaces: list[dict] | None = None
        
        # Cache for depot root → local root mapping (project_root → (depot_root, local_root))
        # This avoids calling p4 where for every file conversion
        self._workspace_mapping_cache: dict[str, tuple[str, str]] = {}

        # Cache for the depot prefix corresponding to a project_root
        # (project_root → "//depot/.../<project-subpath>"). Used to convert
        # depot paths into project-root-relative paths without recomputing per file.
        self._project_depot_prefix_cache: dict[str, Optional[str]] = {}
        
        # Circuit breaker for consecutive corrupt changelists
        self._consecutive_corrupt_cls = 0
        self._max_consecutive_corrupt_cls = 10
    
    def _get_all_user_workspaces(self) -> list[dict]:
        """Fetch all user workspaces with depot mappings (cached).
        
        Uses `p4 clients --me` to get basic workspace info, then `p4 client -o`
        to get depot root from each client's View field. Results are cached for
        performance since this is shared between detect() and _get_workspace_mapping().
        
        Returns:
            List of dicts with keys: host, root, client, depot_root
            Example: [
                {
                    "host": "my-machine",
                    "root": "C:\\work\\depot\\main",
                    "client": "user_main",
                    "depot_root": "//depot/main"
                }
            ]
        """
        if self._user_workspaces is not None:
            return self._user_workspaces
        
        try:
            # Step 1: Get basic workspace info (host, root, client name)
            result = subprocess.run(
                ["p4", "-z", "tag", "-F", "%Host% %Root% %client%", "clients", "--me"],
                capture_output=True,
                timeout=15,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            
            if result.returncode != 0:
                self._user_workspaces = []
                return []
            
            workspaces = []
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(maxsplit=2)
                if len(parts) == 3:
                    host, root, client = parts
                    
                    # Step 2: Get depot root from client spec
                    depot_root = self._get_client_depot_root(client)
                    
                    workspaces.append({
                        "host": host,
                        "root": root,
                        "client": client,
                        "depot_root": depot_root
                    })
            
            self._user_workspaces = workspaces
            return workspaces
            
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            self._user_workspaces = []
            return []
    
    def _get_client_depot_root(self, client_name: str) -> str:
        """Get depot root from client spec View field.
        
        Parses the View section of `p4 client -o` output and returns the shortest
        positive (non-exclusion) depot mapping, which represents the workspace's
        main depot root.
        
        Example View:
            View:
                //depot/main/... //client/...
                -//depot/main/dev/Assets/... //client/dev/Assets/...
        
        Returns: "//depot/main" (shortest positive mapping, /... stripped)
        
        Args:
            client_name: Name of the Perforce client
            
        Returns:
            Depot root path (e.g., "//depot/main"), or empty string if not found
        """
        try:
            result = subprocess.run(
                ["p4", "client", "-o", client_name],
                capture_output=True,
                text=True,
                timeout=10,
                encoding="utf-8",
                errors="replace"
            )
            
            if result.returncode != 0:
                return ""
            
            # Collect all positive (non-exclusion) depot paths from View
            depot_paths = []
            in_view = False
            
            for line in result.stdout.splitlines():
                if line.startswith("View:"):
                    in_view = True
                    continue
                
                if in_view:
                    stripped = line.strip()
                    
                    # Check if still in View section (indented lines)
                    if line.startswith(("\t", " ")) and stripped:
                        # Skip exclusion rules (start with -)
                        if stripped.startswith("-"):
                            continue
                        
                        # Parse mapping: "//depot/path/... //client/path/..."
                        parts = stripped.split()
                        if len(parts) >= 2:
                            depot_path = parts[0].rstrip("/...")
                            depot_paths.append(depot_path)
                    elif stripped and not line.startswith(("\t", " ")):
                        # End of View section (reached non-indented line)
                        break
            
            if not depot_paths:
                return ""
            
            # Return shortest path (closest to root, most general mapping)
            return min(depot_paths, key=len)
            
        except (subprocess.TimeoutExpired, OSError):
            return ""
    
    def _run_subprocess_with_retry(
        self,
        cmd: list[str],
        project_root: str,
        max_retries: int = 5,
        operation_name: str = "subprocess",
        timeout: Optional[int] = None,
        **subprocess_kwargs
    ) -> subprocess.CompletedProcess:
        """Run subprocess with exponential backoff retry and cleanup.
        
        Designed for memory-intensive operations (large file reads, diffs) that may
        fail due to resource exhaustion or transient errors. Includes explicit
        cleanup to prevent memory accumulation.
        
        Args:
            cmd: Command and arguments to execute
            project_root: Working directory for the command
            max_retries: Maximum number of retry attempts (default: 5)
            operation_name: Description of operation for error messages
            timeout: Optional timeout in seconds
            **subprocess_kwargs: Additional arguments passed to subprocess.run
            
        Returns:
            subprocess.CompletedProcess on success
            
        Raises:
            RuntimeError: After all retries exhausted with details of last error
        """
        last_error = None
        result = None
        
        for attempt in range(max_retries):
            try:
                result = subprocess.run(
                    cmd,
                    cwd=project_root,
                    capture_output=True,
                    timeout=timeout,
                    **subprocess_kwargs
                )
                
                if result.returncode != 0:
                    # Non-zero exit code - retry
                    last_error = f"Command returned exit code {result.returncode}"
                    if attempt < max_retries - 1:
                        if DebugLogger.is_enabled():
                            from expert_among_us.utils.progress import console as progress_console
                            progress_console.print(
                                f"[yellow]Retry {attempt + 1}/{max_retries} for {operation_name}: {last_error}[/yellow]"
                            )
                        time.sleep(2 ** attempt)  # 1s, 2s, 4s, 8s, 16s backoff
                        continue
                else:
                    # Success - return result
                    return result
                    
            except (OSError, subprocess.TimeoutExpired) as e:
                last_error = f"{type(e).__name__}: {str(e)}"
                if attempt < max_retries - 1:
                    if DebugLogger.is_enabled():
                        from expert_among_us.utils.progress import console as progress_console
                        progress_console.print(
                            f"[yellow]Retry {attempt + 1}/{max_retries} for {operation_name}: {last_error}[/yellow]"
                        )
                    time.sleep(2 ** attempt)  # 1s, 2s, 4s, 8s, 16s backoff
                    continue
                # Final attempt failed - will raise below
            finally:
                # Explicit cleanup after each attempt
                if result is not None:
                    del result
                    result = None
                gc.collect()
            
            # If we reach here on the final attempt, raise error
            if attempt == max_retries - 1:
                cmd_str = " ".join(str(part) for part in cmd[:5])  # Show first 5 args
                if len(cmd) > 5:
                    cmd_str += f" ... ({len(cmd)} total args)"
                raise RuntimeError(
                    f"Failed to execute {operation_name} after {max_retries} attempts. "
                    f"Command: {cmd_str}. "
                    f"Last error: {last_error}"
                )
        
        # Should never reach here, but satisfy type checker
        raise RuntimeError(f"Unexpected state in _run_subprocess_with_retry for {operation_name}")
    
    @staticmethod
    def detect(project_root: str) -> bool:
        """Detect if workspace is a Perforce client workspace.
        
        Uses `p4 clients --me` to find all user workspaces and matches by hostname
        and root path. This approach works regardless of whether P4CLIENT is set,
        unlike `p4 info` which uses the default client when P4CLIENT is not set.
        
        Args:
            project_root: Path to the project root directory to check
            
        Returns:
            True if Perforce is detected and workspace is valid, False otherwise
        """
        # Check if p4 command is available
        if not shutil.which("p4"):
            return False
        
        try:
            # Create temporary provider instance to use cached workspace discovery
            from expert_among_us.config.settings import Settings
            provider = Perforce(Settings())
            workspaces = provider._get_all_user_workspaces()
            
            if not workspaces:
                return False
            
            # Get current hostname for matching
            current_host = socket.gethostname().lower()
            
            # Normalize project_root for comparison
            workspace_normalized = str(Path(project_root).resolve())
            
            # Check if project_root matches any user workspace
            for ws in workspaces:
                # Match by hostname (case-insensitive)
                if ws["host"].lower() == current_host:
                    # Normalize root path
                    try:
                        root_normalized = str(Path(ws["root"]).resolve())
                        
                        # Check if project_root is within or equal to root
                        # This allows subdirectories of a client workspace to be detected
                        Path(workspace_normalized).relative_to(root_normalized)
                        return True
                    except (ValueError, OSError):
                        # project_root is not under root, or invalid path
                        continue
            
            return False
            
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False
    
    def get_commits_after(
        self,
        project_root: str,
        after_hash: str | None,
        batch_size: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[Changelist]:
        """Get changelists after a specific CL number in chronological order (oldest → newest).
        
        This is the primary changelist traversal method used by the unified indexer.
        
        The implementation uses a two-phase strategy matching Git:
        - Phase 1 (once per project_root): fetch all matching CL numbers in
          chronological order and cache them.
        - Phase 2 (per call): slice the next `batch_size` CLs from the cache and
          fetch full changelist details only for those CLs.
        
        Args:
            project_root: Path to the project root (the indexed directory)
            after_hash: Get changelists after this CL number (None = from beginning)
            batch_size: Maximum number of changelists to return
            progress_callback: Optional callback(current, total) called after each sub-batch
            
        Returns:
            List of Changelist objects in chronological order (oldest → newest)
        """
        if batch_size == 0:
            return []
        
        # Fetch CL numbers (will use cache if available)
        cl_numbers = self._fetch_all_changelist_numbers(
            project_root=project_root,
        )
        
        if DebugLogger.is_enabled():
            from expert_among_us.utils.progress import console as progress_console
            progress_console.print(
                f"[dim]Perforce.get_commits_after: using {len(cl_numbers)} cached changelists "
                f"(after={after_hash or 'START'})[/dim]"
            )
        
        # Determine starting index based on after_hash cursor
        if after_hash:
            # If after_hash is unknown for this project_root view,
            # validate whether it's a real CL or truly invalid.
            if not self._cl_index:
                return []
            start_idx = self._cl_index.get(after_hash)
            if start_idx is None:
                # CL not in our cache - validate it's a real CL to catch user errors
                validate_cmd = ["p4", "changes", "-m", "1", f"@={after_hash}"]
                validate_result = subprocess.run(
                    validate_cmd,
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                if validate_result.returncode != 0 or not validate_result.stdout.strip():
                    # Invalid CL number - raise to match Git behavior and catch errors
                    raise subprocess.CalledProcessError(
                        validate_result.returncode or 1,
                        validate_cmd,
                        validate_result.stdout,
                        validate_result.stderr,
                    )
                # Valid CL but not in our filtered view - return empty
                return []
            start = start_idx + 1  # strictly after the cursor
        else:
            # No cursor: start from the beginning
            start = 0
        
        if not cl_numbers:
            return []
        
        end = start + batch_size
        batch_cl_numbers = cl_numbers[start:end]
        
        if not batch_cl_numbers:
            # No more changelists after the given cursor
            return []
        
        # Fetch full changelist details for this batch only
        changelists = self._fetch_changelists_by_numbers(
            project_root=project_root,
            cl_numbers=batch_cl_numbers,
            progress_callback=progress_callback,
        )
        
        return changelists
    
    def _invalidate_cache(self) -> None:
        """Clear changelist cache and index."""
        self._cl_cache = None
        self._cl_index = None
        self._cl_cache_key = None
    
    def _fetch_all_changelist_numbers(
        self,
        project_root: str,
    ) -> list[str]:
        """Fetch all CL numbers under the project root (with caching).
        
        Caches results keyed by project_root to avoid redundant p4 calls.
        Returns cached results when available.
        
        Uses `p4 changes -s submitted [path...]` to get all submitted changelists.
        Perforce returns newest first, so we reverse to get oldest → newest order.
        
        Args:
            project_root: Path to the project root (the indexed directory)
            
        Returns:
            List of CL numbers as strings, ordered oldest → newest
        """
        # Check cache first
        cache_key = project_root
        
        if self._cl_cache is not None and self._cl_cache_key == cache_key:
            return self._cl_cache
        
        # Cache miss - fetch from Perforce
        cmd = ["p4", "changes", "-s", "submitted"]
        
        # Add automated user exclusions
        if EXCLUDED_AUTOMATED_USERS:
            cmd.append("-E")
            for user_pattern in EXCLUDED_AUTOMATED_USERS:
                cmd.append(f"-u-{user_pattern}")
        
        # Query all submitted changelists under the project root
        depot_path = self._local_to_depot_path(project_root, None)
        cmd.append(depot_path)
        
        if DebugLogger.is_enabled():
            cmd_str = " ".join(str(part) for part in cmd)
            from expert_among_us.utils.progress import console as progress_console
            progress_console.print(f"[dim]Perforce._fetch_all_changelist_numbers: {cmd_str}[/dim]")
        
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,  # Longer timeout than detect() since this can legitimately take time
        )
        
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                result.stdout,
                result.stderr,
            )
        
        # Parse: "Change 12345 on 2024/01/15 14:30:00 by user@client ..."
        # Collect changelist numbers for sorting
        cl_entries = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("Change "):
                parts = line.split()
                if len(parts) >= 2:
                    cl_number = parts[1]
                    cl_entries.append(cl_number)
        
        # Sort by changelist number (chronological order)
        cl_entries.sort(key=lambda x: int(x))
        
        # Deduplicate adjacent entries (duplicates are now adjacent after sort)
        cl_numbers = []
        prev_cl_number = None
        for cl_number in cl_entries:
            if cl_number != prev_cl_number:
                cl_numbers.append(cl_number)
                prev_cl_number = cl_number
        
        # Update cache before returning
        self._cl_cache = cl_numbers
        self._cl_index = {cl_num: idx for idx, cl_num in enumerate(cl_numbers)}
        self._cl_cache_key = cache_key
        
        return cl_numbers
    
    def _fetch_changelists_by_numbers(
        self,
        project_root: str,
        cl_numbers: list[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[Changelist]:
        """Fetch full details for specific CLs (batched with sub-batching).
        
        Uses `p4 describe -du` to get metadata + diffs for changelists.
        Processes in sub-batches to avoid timeouts with large changelist sets.
        
        Args:
            project_root: Path to the project root (the indexed directory)
            cl_numbers: List of CL numbers to fetch
            progress_callback: Optional callback(current, total) called after each sub-batch
            
        Returns:
            List of Changelist objects with full details
        """
        if not cl_numbers:
            return []
        
        # Restrict indexed files to those under the project root. The depot
        # prefix for the project root is the single filtering boundary.
        depot_path = self._local_to_depot_path(project_root, None)
        depot_prefixes = [depot_path.rstrip("/...")]
        
        # Process in sub-batches to avoid timeouts
        SUB_BATCH_SIZE = 50
        SUB_BATCH_TIMEOUT = 60
        all_changelists = []
        
        for batch_start in range(0, len(cl_numbers), SUB_BATCH_SIZE):
            batch_end = min(batch_start + SUB_BATCH_SIZE, len(cl_numbers))
            sub_batch = cl_numbers[batch_start:batch_end]
            
            # Fetch this sub-batch
            sub_changelists = self._fetch_single_describe_batch(
                project_root=project_root,
                cl_numbers=sub_batch,
                depot_prefixes=depot_prefixes,
                timeout=SUB_BATCH_TIMEOUT,
                max_output_bytes=MAX_DESCRIBE_OUTPUT_BYTES,
                embed_diffs=self._settings.embed_diffs,
            )
            
            all_changelists.extend(sub_changelists)
            
            # Report progress after processing each sub-batch
            if progress_callback:
                try:
                    progress_callback(batch_end, len(cl_numbers))
                except Exception:
                    # Ignore callback errors to prevent disrupting processing
                    pass
        
        return all_changelists
    
    def _fetch_single_describe_batch(
        self,
        project_root: str,
        cl_numbers: list[str],
        depot_prefixes: Optional[list[str]],
        timeout: int,
        max_output_bytes: int = MAX_DESCRIBE_OUTPUT_BYTES,
        embed_diffs: bool = True,
    ) -> list[Changelist]:
        """Fetch full details for a single batch of CLs with size limit fallback.
        
        Uses streaming size limits to prevent timeouts from huge changelists.
        If a batch exceeds the size limit, either accepts truncation (single CL)
        or uses binary search to split the batch (multiple CLs).
        
        Args:
            project_root: Path to the project root (the indexed directory)
            cl_numbers: List of CL numbers to fetch (sub-batch)
            depot_prefixes: Optional depot path prefixes for filtering
            timeout: Timeout in seconds for the p4 describe command
            max_output_bytes: Maximum output size in bytes (default 10 MB)
            embed_diffs: Whether to fetch diffs (True) or only metadata (False)
            
        Returns:
            List of Changelist objects for this batch
        """
        if DebugLogger.is_enabled():
            from expert_among_us.utils.progress import console as progress_console
            progress_console.print(
                f"[dim]Perforce._fetch_single_describe_batch: "
                f"{cl_numbers[0]}..{cl_numbers[-1]} "
                f"({len(cl_numbers)} CLs, timeout={timeout}s)[/dim]"
            )
        
        # Try batch with size limit
        output, result = self._run_describe_with_size_limit(
            project_root, cl_numbers, timeout, max_output_bytes, embed_diffs
        )
        
        if result == DescribeResult.SUCCESS:
            # Success - parse normally
            parsed = self._parse_describe_output(
                output, project_root, depot_prefixes, embed_diffs
            )
            # Explicit cleanup after large describe operation
            del output
            gc.collect()
            self._consecutive_corrupt_cls = 0  # Reset circuit breaker
            return parsed
        
        # Not success - SIZE_LIMIT or CORRUPTION
        if DebugLogger.is_enabled():
            from expert_among_us.utils.progress import console as progress_console
            progress_console.print(
                f"[yellow]Batch {cl_numbers[0]}..{cl_numbers[-1]} {result.value}, "
                f"max {max_output_bytes / (1024*1024):.1f} MB[/yellow]"
            )
        
        if len(cl_numbers) == 1:
            # Single CL handling
            if result == DescribeResult.CORRUPTION:
                # CORRUPTION: Skip entirely (output is invalid/empty)
                self._consecutive_corrupt_cls += 1
                if self._consecutive_corrupt_cls >= self._max_consecutive_corrupt_cls:
                    raise RuntimeError(
                        f"Too many consecutive corrupt CLs ({self._consecutive_corrupt_cls}). "
                        "Depot needs 'p4 verify' or 'p4d -xU'."
                    )
                from expert_among_us.utils.progress import console as progress_console
                progress_console.print(f"[yellow]Skipping corrupt CL {cl_numbers[0]}[/yellow]")
                del output
                gc.collect()
                return []
            else:
                # SIZE_LIMIT: Use truncated output (acceptable per existing behavior)
                parsed = self._parse_describe_output(
                    output, project_root, depot_prefixes, embed_diffs
                )
                del output
                gc.collect()
                return parsed
        
        # Multiple CLs - binary search (same for both SIZE_LIMIT and CORRUPTION)
        del output
        gc.collect()
        return self._fetch_with_binary_search(
            project_root, cl_numbers, depot_prefixes,
            timeout, max_output_bytes, embed_diffs
        )
    
    def _run_describe_with_size_limit(
        self,
        project_root: str,
        cl_numbers: list[str],
        timeout: int,
        max_bytes: int = MAX_DESCRIBE_OUTPUT_BYTES,
        embed_diffs: bool = True,
    ) -> tuple[str, DescribeResult]:
        """Run p4 describe with streaming output size limit.
        
        Streams subprocess output and enforces a hard byte limit to prevent
        memory issues and timeouts from huge changelists. Adds truncation marker
        when limit is exceeded.
        
        Args:
            project_root: Path to the project root (the indexed directory)
            cl_numbers: List of CL numbers to describe
            timeout: Command timeout in seconds
            max_bytes: Maximum output size in bytes (default 10 MB)
            embed_diffs: Whether to fetch diffs (True) or only metadata (False)
            
        Returns:
            Tuple of (output, result_status)
            - SUCCESS: Full output retrieved
            - SIZE_LIMIT: Output truncated, needs splitting
            - CORRUPTION: Depot corruption, needs isolation
            
        Raises:
            subprocess.CalledProcessError: On p4 command failure
            subprocess.TimeoutExpired: On timeout (always fatal)
        """
        # Conditionally include diffs based on embed_diffs flag
        if embed_diffs:
            cmd = ["p4", "describe", "-du", "-m", str(MAX_FILES_PER_CL)]
        else:
            cmd = ["p4", "describe", "-s"]
        cmd.extend(cl_numbers)
        
        process = subprocess.Popen(
            cmd,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        
        output_parts = []
        total_bytes = 0
        truncated = False
        
        try:
            # Stream output with size checking
            while True:
                chunk = process.stdout.read(8192)  # 8 KB chunks
                if not chunk:
                    break
                
                chunk_bytes = len(chunk.encode('utf-8'))
                
                if total_bytes + chunk_bytes > max_bytes:
                    # Will exceed limit - truncate and stop reading
                    remaining = max_bytes - total_bytes
                    if remaining > 0:
                        # Partial chunk fits
                        encoded = chunk.encode('utf-8')[:remaining]
                        output_parts.append(encoded.decode('utf-8', errors='ignore'))
                    truncated = True
                    
                    # Kill process immediately to prevent pipe deadlock
                    # (process would block when stdout buffer fills)
                    process.kill()
                    break
                
                output_parts.append(chunk)
                total_bytes += chunk_bytes
            
            # Wait for process to complete (returns immediately if killed)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise  # Always fatal - let caller handle
            
            # Add truncation marker if output was cut off
            if truncated:
                output_parts.append("\n\n[TRUNCATED - exceeded size limit]")
                return ''.join(output_parts), DescribeResult.SIZE_LIMIT
            elif process.returncode != 0:
                # Only raise on real errors, not our intentional kill
                stderr_output = process.stderr.read()
                
                # Check for corruption patterns
                corruption_patterns = [
                    "Revision table out of sync with index!",
                ]
                
                if any(pattern in stderr_output for pattern in corruption_patterns):
                    if DebugLogger.is_enabled():
                        from expert_among_us.utils.progress import console as progress_console
                        progress_console.print(
                            f"[yellow]Corruption detected in CLs {cl_numbers}: {stderr_output.strip()}[/yellow]"
                        )
                    return "", DescribeResult.CORRUPTION
                
                # Non-corruption error
                raise subprocess.CalledProcessError(
                    process.returncode, cmd,
                    ''.join(output_parts),
                    stderr_output
                )
            
            return ''.join(output_parts), DescribeResult.SUCCESS
            
        finally:
            # Ensure process cleanup
            if process.poll() is None:
                process.kill()
                process.wait()
    
    def _fetch_with_binary_search(
        self,
        project_root: str,
        cl_numbers: list[str],
        depot_prefixes: Optional[list[str]],
        timeout: int,
        max_output_bytes: int,
        embed_diffs: bool = True,
    ) -> list[Changelist]:
        """Split batch using binary search to isolate problematic CLs.
        
        Recursively processes CLs in smaller batches until all succeed.
        Base case (single CL) is handled by caller which accepts truncation.
        
        Args:
            project_root: Path to the project root (the indexed directory)
            cl_numbers: List of CL numbers (must be > 1)
            depot_prefixes: Optional depot path prefixes for filtering
            timeout: Command timeout in seconds
            max_output_bytes: Maximum output size per batch
            embed_diffs: Whether to fetch diffs (True) or only metadata (False)
            
        Returns:
            List of Changelist objects from all sub-batches
        """
        if len(cl_numbers) <= 1:
            # Base case - caller handles single CL truncation
            return self._fetch_single_describe_batch(
                project_root, cl_numbers, depot_prefixes,
                timeout, max_output_bytes, embed_diffs
            )
        
        # Split in half
        mid = len(cl_numbers) // 2
        first_half = cl_numbers[:mid]
        second_half = cl_numbers[mid:]
        
        # Process each half recursively
        results = []
        results.extend(
            self._fetch_single_describe_batch(
                project_root, first_half, depot_prefixes,
                timeout, max_output_bytes, embed_diffs
            )
        )
        results.extend(
            self._fetch_single_describe_batch(
                project_root, second_half, depot_prefixes,
                timeout, max_output_bytes, embed_diffs
            )
        )
        
        return results
    
    def _parse_describe_output(
        self,
        output: str,
        project_root: str,
        depot_prefixes: Optional[list[str]] = None,
        embed_diffs: bool = True,
    ) -> list[Changelist]:
        """Parse output from `p4 describe` command (with or without diffs).
        
        Filters files and diffs to only include those matching depot_prefixes.
        
        Output format:
        ```
        Change 12345 by user@client on 2024/01/15 14:30:00
        
            Commit message here
        
        Affected files ...
        
        ... //depot/src/file.cpp#42 edit
        
        Differences ...
        
        ==== //depot/src/file.cpp#42 (text) ====
        
        ... diff content ...
        
        Change 12346 by user@client on 2024/01/15 15:00:00
        ...
        ```
        
        Args:
            output: Raw output from p4 describe command
            project_root: Path to the project root (for context)
            depot_prefixes: Optional list of depot path prefixes to filter files
            
        Returns:
            List of parsed Changelist objects
        """
        changelists = []
        lines = output.splitlines()
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Look for changelist header: "Change 12345 by user@client on 2024/01/15 14:30:00"
            if line.startswith("Change "):
                parts = line.split()
                if len(parts) < 7:
                    i += 1
                    continue
                
                # Parse header: Change <num> by <user>@<client> on <date> <time>
                cl_number = parts[1]
                author = parts[3]  # user@client
                date_str = parts[5]  # 2024/01/15
                time_str = parts[6]  # 14:30:00
                
                # Parse timestamp
                try:
                    timestamp = datetime.strptime(
                        f"{date_str} {time_str}",
                        "%Y/%m/%d %H:%M:%S"
                    ).replace(tzinfo=timezone.utc)
                except (ValueError, IndexError):
                    i += 1
                    continue
                
                i += 1
                
                # Skip empty line after header
                if i < len(lines) and not lines[i].strip():
                    i += 1
                
                # Collect commit message (indented lines)
                message_lines = []
                while i < len(lines):
                    if lines[i].startswith("\t") or (lines[i].startswith(" ") and lines[i].strip()):
                        message_lines.append(lines[i].strip())
                        i += 1
                    else:
                        break
                
                message = "\n".join(message_lines) if message_lines else ""
                
                # Skip to "Affected files ..." section
                while i < len(lines) and not lines[i].startswith("Affected files"):
                    i += 1
                
                if i < len(lines):
                    i += 1  # Skip "Affected files ..." line
                
                # Skip empty line
                if i < len(lines) and not lines[i].strip():
                    i += 1
                
                # Collect affected files
                files = []
                while i < len(lines):
                    line = lines[i]
                    # File lines start with "... //depot/path#rev action"
                    if line.startswith("... //"):
                        parts = line.split()
                        if len(parts) >= 2:
                            # Extract depot path
                            depot_path = parts[1]
                            # Remove revision number (#42)
                            if "#" in depot_path:
                                depot_path = depot_path.split("#")[0]
                            
                            # Filter by depot prefixes if provided
                            if depot_prefixes:
                                matches = any(
                                    depot_path.startswith(prefix)
                                    for prefix in depot_prefixes
                                )
                                if not matches:
                                    i += 1
                                    continue  # Skip this file
                            
                            # Filter by file extension
                            if not should_index_file(depot_path, self._settings.allowed_file_extensions):
                                i += 1
                                continue
                            
                            # Convert to project-root-relative path
                            relative_path = self._depot_to_relative_path(project_root, depot_path)
                            if relative_path:
                                files.append(relative_path)
                        i += 1
                    else:
                        break
                
                # Skip to "Differences ..." section (or next changelist if no diffs)
                while i < len(lines) and not lines[i].startswith("Differences") and not lines[i].startswith("Change "):
                    i += 1
                
                if i < len(lines) and lines[i].startswith("Differences"):
                    i += 1  # Skip "Differences ..." line
                    
                    # Skip empty line after "Differences"
                    if i < len(lines) and not lines[i].strip():
                        i += 1
                
                # Collect diff until next changelist or EOF
                # If filtering by depot_prefixes, only include diff sections for matching files
                diff_lines = []
                current_file_matches = True  # Track if current diff section matches filter
                
                while i < len(lines):
                    line = lines[i]
                    
                    if line.startswith("Change "):
                        # Start of next changelist
                        break
                    
                    # Check for diff file headers: ==== //depot/path/file.cpp#42 (text) ====
                    if line.startswith("==== //"):
                        if depot_prefixes:
                            # Extract depot path from header
                            parts = line.split()
                            if len(parts) >= 2:
                                depot_spec = parts[1]  # //depot/path/file.cpp#42
                                depot_path = depot_spec.split("#")[0] if "#" in depot_spec else depot_spec
                                
                                # Check if this file matches our filter
                                current_file_matches = any(
                                    depot_path.startswith(prefix)
                                    for prefix in depot_prefixes
                                )
                            else:
                                current_file_matches = True
                        else:
                            current_file_matches = True
                        
                        # Also filter by extension
                        if current_file_matches and self._settings.allowed_file_extensions:
                            current_file_matches = should_index_file(
                                depot_path, self._settings.allowed_file_extensions
                            )
                    
                    # Only include lines if current file matches filter
                    if current_file_matches:
                        diff_lines.append(line)
                    
                    i += 1
                
                diff = "\n".join(diff_lines)
                
                # Step 1: Filter binary content from diff
                diff, _binary_files = filter_binary_from_diff(diff)
                
                # Step 2: Apply compact transformation (after all filtering)
                # Note: Extension filtering for Perforce already happens earlier via should_index_file()
                # on individual files (lines 1025-1027, 1080-1083), so no additional filtering needed here
                if self._settings.compact_diffs:
                    diff = compact_diff(diff, max_line_bytes=self._settings.compact_diff_max_line_bytes)
                
                # Step 3: Truncate individual commit diffs to limit
                if diff:
                    diff, was_truncated = truncate_to_bytes(diff, self._settings.max_diff_bytes_per_commit)
                    if was_truncated:
                        diff += "\n\n[TRUNCATED - commit diff exceeded limit]"
                
                # Only skip empty diffs when we expected diffs
                # (if embed_diffs was False, diff will be empty but that's intentional)
                if embed_diffs and (not diff or not diff.strip()):
                    continue
                
                changelist = Changelist(
                    id=cl_number,
                    expert_name="",  # Will be set by caller
                    project_name="",  # Will be set by caller
                    timestamp=timestamp,
                    author=author,
                    message=message if message else f"Changelist {cl_number}",
                    diff=diff,
                    files=files if files else [],
                )
                changelists.append(changelist)
            else:
                i += 1
        
        return changelists
    
    def _local_to_depot_path(self, project_root: str, local_subdir: str) -> str:
        """Convert a local path to depot path syntax using the cached client mapping.
        
        Fast string substitution approach - only calls p4 once per project root.
        Uses the cached Perforce client workspace mapping in reverse (local → depot).
        
        Args:
            project_root: Path to the project root (the indexed directory)
            local_subdir: Subdirectory relative to project_root (None for the root itself)
            
        Returns:
            Depot path with recursive wildcard (e.g., "//depot/src/engine/...")
        """
        depot_root, local_root = self._get_workspace_mapping(project_root)
        
        if depot_root and local_root:
            # Build full local path
            local_path = Path(project_root) / local_subdir if local_subdir else Path(project_root)
            local_path_normalized = str(local_path.resolve())
            
            # Try to get relative path from local_root
            try:
                # Normalize local_root for comparison
                local_root_normalized = str(Path(local_root).resolve())
                relative_path = str(Path(local_path_normalized).relative_to(local_root_normalized))
                
                # String substitution: replace local root with depot root
                # Use forward slashes for depot paths
                relative_path = relative_path.replace("\\", "/")
                depot_path = f"{depot_root}/{relative_path}/..."
                return depot_path
            except ValueError:
                # local_path is not under local_root, fall through to fallback
                pass
        
        # Fallback: assume standard depot mapping
        return f"//depot/{local_subdir}/..."
    
    def _get_workspace_mapping(self, project_root: str) -> tuple[str, str]:
        """Get depot root and local root mapping for workspace (cached).
        
        Uses cached workspace data from `_get_all_user_workspaces()` to find the
        matching workspace by path, providing depot root and local root for path
        conversion. This avoids calling `p4 where` which fails on workspace roots.
        
        Args:
            project_root: Path to the project root (the indexed directory)
            
        Returns:
            Tuple of (depot_root, local_root) for string substitution
        """
        if project_root in self._workspace_mapping_cache:
            return self._workspace_mapping_cache[project_root]
        
        try:
            # Get all user workspaces with depot mappings
            workspaces = self._get_all_user_workspaces()
            
            if not workspaces:
                # Fallback: no workspaces found
                self._workspace_mapping_cache[project_root] = ("", project_root)
                return ("", project_root)
            
            # Get current hostname for matching
            current_host = socket.gethostname().lower()
            
            # Normalize project_root for comparison
            workspace_normalized = str(Path(project_root).resolve())
            
            # Find matching workspace
            for ws in workspaces:
                # Match by hostname (case-insensitive)
                if ws["host"].lower() == current_host:
                    try:
                        # Normalize root path
                        root_normalized = str(Path(ws["root"]).resolve())
                        
                        # Check if project_root is within or equal to this workspace root
                        Path(workspace_normalized).relative_to(root_normalized)
                        
                        # Found matching workspace - cache and return
                        mapping = (ws["depot_root"], ws["root"])
                        self._workspace_mapping_cache[project_root] = mapping
                        
                        if DebugLogger.is_enabled():
                            from expert_among_us.utils.progress import console as progress_console
                            progress_console.print(
                                f"[dim]Perforce._get_workspace_mapping: cached mapping "
                                f"depot='{ws['depot_root']}' → local='{ws['root']}'[/dim]"
                            )
                        
                        return mapping
                    except (ValueError, OSError):
                        # project_root is not under this root
                        continue
            
            # No matching workspace found - fallback
            self._workspace_mapping_cache[project_root] = ("", project_root)
            return ("", project_root)
            
        except Exception:
            # Fallback on any error
            self._workspace_mapping_cache[project_root] = ("", project_root)
            return ("", project_root)
    
    def _get_project_depot_prefix(self, project_root: str) -> Optional[str]:
        """Return the depot path prefix corresponding to ``project_root``.

        This is the depot path that maps to the indexed project root, e.g.
        ``//depot/main/my-project``. Depot paths are made relative to this
        prefix so that stored paths are relative to the project root, not to
        the broader Perforce client workspace.

        The result is memoized per project_root.

        Args:
            project_root: Path to the project root (the indexed directory)

        Returns:
            Depot prefix without a trailing wildcard, or None if it cannot be
            determined.
        """
        if project_root in self._project_depot_prefix_cache:
            return self._project_depot_prefix_cache[project_root]

        depot_path = self._local_to_depot_path(project_root, None)
        # _local_to_depot_path appends a recursive wildcard ("/...").
        prefix = depot_path.rstrip("/...") if depot_path else None
        self._project_depot_prefix_cache[project_root] = prefix
        return prefix

    def _depot_to_relative_path(self, project_root: str, depot_path: str) -> Optional[str]:
        r"""Convert a depot path to a path relative to ``project_root``.

        Honors the VCSProvider contract that file paths are relative to the
        project root (the indexed directory), not the broader Perforce client
        workspace. Uses the cached project→depot prefix and a fast string strip.

        Args:
            project_root: Path to the project root (the indexed directory)
            depot_path: Depot path (e.g., "//depot/main/my-project/src/module/File.cpp")

        Returns:
            Project-root-relative path with forward slashes
            (e.g., "src/module/File.cpp"), or None if the depot path is outside
            the project root.
        """
        project_depot_prefix = self._get_project_depot_prefix(project_root)

        if project_depot_prefix and depot_path.startswith(project_depot_prefix):
            relative_path = depot_path[len(project_depot_prefix):].lstrip("/")
            return relative_path.replace("\\", "/")

        return None
    
    def get_tracked_files_at_commit(
        self,
        project_root: str,
        commit_hash: str,
    ) -> list[str]:
        """Get list of tracked files at a specific changelist.
        
        Uses `p4 files [path...]@CL` to list all files as they existed at that changelist.
        Note: Uses @ (not @=) to get the state of files at that changelist, not just
        files modified in that specific changelist.
        
        Args:
            project_root: Path to the project root (the indexed directory)
            commit_hash: Changelist number
            
        Returns:
            List of file paths (relative to the project root) tracked at the changelist
        """
        cmd = ["p4", "files"]
        
        # Query all files under the project root
        depot_path = self._local_to_depot_path(project_root, None)
        cmd.append(depot_path)
        cmd.append(f"{depot_path}@{commit_hash}")
        
        if DebugLogger.is_enabled():
            from expert_among_us.utils.progress import console as progress_console
            cmd_str = " ".join(str(part) for part in cmd)
            progress_console.print(f"[dim]Perforce.get_tracked_files_at_commit: {cmd_str}[/dim]")
        
        # Use helper function for retry logic with cleanup
        # Re-raise on failure - returning empty list would cause indexer to delete all file chunks
        result = self._run_subprocess_with_retry(
            cmd=cmd,
            project_root=project_root,
            operation_name=f"p4 files at commit {commit_hash}",
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        
        # Parse output: "//depot/src/file.cpp#42 - edit change 12345 (text)"
        files = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("//"):
                parts = line.split("#", 1)
                if parts:
                    depot_path = parts[0]
                    relative_path = self._depot_to_relative_path(project_root, depot_path)
                    if relative_path:
                        files.append(relative_path)
        
        return files
    
    def get_file_content_at_commit(
        self,
        project_root: str,
        file_path: str,
        commit_hash: str,
    ) -> Optional[str]:
        """Get file content at a specific changelist.
        
        This is a thin wrapper around get_files_content_at_commit()
        for backward compatibility.
        
        Args:
            project_root: Path to the project root (the indexed directory)
            file_path: Relative path to file
            commit_hash: Changelist number
            
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
        """Get content for multiple files at a specific changelist (batched operation).
        
        Uses `p4 print -q` to fetch file contents in batches.
        
        Args:
            project_root: Path to the project root (the indexed directory)
            file_paths: List of relative file paths to fetch
            commit_hash: Changelist number to read from
            progress_callback: Optional callback(current, total) called after each batch.
                             Receives the number of files processed so far and total files.
            
        Returns:
            Dictionary mapping file_path -> content (or None if missing/binary)
        """
        if not file_paths:
            return {}
        
        # Normalize and deduplicate paths
        unique_paths: list[str] = []
        seen: set[str] = set()
        for p in file_paths:
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)
        
        # Prepare result with default None for all requested paths
        results: dict[str, Optional[str]] = {p: None for p in unique_paths}
        
        # Process in batches to avoid command line length limits
        batch_size = 50
        
        for batch_start in range(0, len(unique_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(unique_paths))
            batch_paths = unique_paths[batch_start:batch_end]
            
            # Build p4 print command with depot paths
            cmd = ["p4", "print"]
            depot_specs = []
            for relative_path in batch_paths:
                depot_path = self._local_to_depot_path(project_root, relative_path)
                # Remove /... suffix for specific file
                depot_path = depot_path.rstrip("/...")
                depot_specs.append(f"{depot_path}@{commit_hash}")
            
            cmd.extend(depot_specs)
            
            if DebugLogger.is_enabled():
                from expert_among_us.utils.progress import console as progress_console
                cmd_str = " ".join(str(part) for part in cmd)
                progress_console.print(
                    f"[dim]Perforce.get_files_content_at_commit: {len(batch_paths)} files via {cmd_str}[/dim]"
                )
            
            # Use helper function for retry logic with cleanup
            try:
                result = self._run_subprocess_with_retry(
                    cmd=cmd,
                    project_root=project_root,
                    operation_name=f"p4 print batch ({len(batch_paths)} files)",
                    encoding="utf-8",
                    errors="replace",
                )
                
                # Parse output
                self._parse_print_output(result.stdout, batch_paths, results, project_root)
                
            except RuntimeError as e:
                # Re-raise with more context about which files failed
                raise RuntimeError(
                    f"{str(e)} Batch files: {batch_paths[0]}...{batch_paths[-1]}"
                ) from e
            
            # Report progress after processing each batch
            if progress_callback:
                try:
                    progress_callback(batch_end, len(unique_paths))
                except Exception:
                    # Ignore callback errors to prevent disrupting file reading
                    pass
        
        return results
    
    def _parse_print_output(
        self,
        output: str,
        batch_paths: list[str],
        results: dict[str, Optional[str]],
        project_root: str,
    ) -> None:
        """Parse output from `p4 print -q` command.
        
        Updates results dict in-place with file contents.
        
        Args:
            output: Raw output from p4 print
            batch_paths: List of project-root-relative paths that were queried
            results: Results dictionary to update
            project_root: Project root for path conversion
        """
        lines = output.split("\n")
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Look for file header: "//depot/path/file.cpp#42 - edit change 12345 (text)"
            # It will match the cached depot root path
            relative_path = self._parse_relative_path_line(project_root, line)
            if relative_path:
                i += 1

                # Check for binary marker (case-insensitive to be safe)
                if "(binary)" in line.lower():
                    # Binary file, skip content and save bandwidth
                    if DebugLogger.is_enabled():
                        from expert_among_us.utils.progress import console as progress_console
                        progress_console.print(f"[dim]Perforce: Skipping binary file {relative_path}[/dim]")
                    i += 1
                    # Skip until next file header or EOF
                    while i < len(lines) and not self._parse_relative_path_line(project_root, lines[i]):
                        i += 1
                    continue
                
                # Skip empty line after header
                if i < len(lines) and not lines[i].strip():
                    i += 1
                
                # Collect file content until next file header or EOF
                content_lines = []
                while i < len(lines):
                    if self._parse_relative_path_line(project_root, lines[i]):
                        # Start of next file
                        break
                    content_lines.append(lines[i])
                    i += 1
                
                content = "\n".join(content_lines)
                
                # Check if content is binary
                try:
                    content_bytes = content.encode("utf-8", errors="replace")
                    if is_binary_file(content_bytes):
                        # Binary content, leave as None
                        continue
                except Exception:
                    # Error during binary check, leave as None
                    continue
                
                # Store content for matching relative path
                if relative_path in results:
                    results[relative_path] = content
            else:
                i += 1
    
    def _parse_relative_path_line(self, project_root: str, line: str) -> Optional[str]:
        """Return the project-root-relative path if the line contains a depot path, else None."""
        depot_path = line.split("#")[0] if "#" in line else None
        return self._depot_to_relative_path(project_root, depot_path) if depot_path else None
    
    def get_latest_commit_time(
        self,
        project_root: str,
    ) -> Optional[datetime]:
        """Get the timestamp of the most recent changelist.
        
        Uses `p4 changes -m 1 -s submitted [path...]` to get the latest CL.
        
        Args:
            project_root: Path to the project root (the indexed directory)
            
        Returns:
            Datetime of the most recent changelist, or None if no changelists found
        """
        cmd = ["p4", "changes", "-m", "1", "-s", "submitted"]
        
        depot_path = self._local_to_depot_path(project_root, None)
        cmd.append(depot_path)
        
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        
        if result.returncode != 0 or not result.stdout.strip():
            return None
        
        # Parse first line: "Change 12345 on 2024/01/15 14:30:00 by user@client ..."
        line = result.stdout.strip().split("\n")[0]
        parts = line.split()
        
        if len(parts) < 7:
            return None
        
        # Extract date and time: parts[5] = 2024/01/15, parts[6] = 14:30:00
        try:
            date_str = parts[5]
            time_str = parts[6]
            timestamp = datetime.strptime(
                f"{date_str} {time_str}",
                "%Y/%m/%d %H:%M:%S"
            ).replace(tzinfo=timezone.utc)
            return timestamp
        except (ValueError, IndexError):
            return None
    
    def get_total_commit_count(
        self,
        project_root: str,
    ) -> int:
        """Return the total number of changelists to consider for indexing.
        
        Uses _fetch_all_changelist_numbers() which caches results.
        
        Args:
            project_root: Path to the project root (the indexed directory).
            
        Returns:
            Integer count of changelists (deduplicated).
        """
        cl_numbers = self._fetch_all_changelist_numbers(
            project_root=project_root,
        )
        return len(cl_numbers)
    
    def get_commit_position(self, commit_id: Optional[str]) -> tuple[int, int]:
        """Get position of changelist in ordered sequence for progress tracking.
        
        Args:
            commit_id: Changelist number, or None for start position
            
        Returns:
            Tuple of (commits_considered, total_commits)
        """
        # If no cache, return (0, 0)
        if not self._cl_cache:
            return (0, 0)
        
        # If commit_id is None, starting from beginning
        if not commit_id:
            return (0, len(self._cl_cache))
        
        # If no index built yet, return (0, total)
        if not self._cl_index:
            return (0, len(self._cl_cache))
        
        # Look up position and add +1 to convert from 0-based index to 1-based count
        position = self._cl_index.get(commit_id, -1)
        if position == -1:
            # CL not in cache, return 0 (will be validated elsewhere)
            return (0, len(self._cl_cache))
        
        return (position + 1, len(self._cl_cache))