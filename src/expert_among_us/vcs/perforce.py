
"""Perforce VCS provider implementation.

This module provides a Perforce implementation of the VCSProvider interface,
enabling Expert Among Us to work with Perforce repositories. It uses the p4
CLI to interact with Perforce servers.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional
from datetime import datetime, timezone

from expert_among_us.vcs.base import VCSProvider
from expert_among_us.models.changelist import Changelist
from expert_among_us.utils.truncate import filter_binary_from_diff, is_binary_file
from expert_among_us.utils.debug import DebugLogger


class Perforce(VCSProvider):
    """Perforce VCS provider implementation.
    
    Implements the VCSProvider interface for Perforce repositories.
    Uses p4 CLI commands to interact with Perforce server.
    
    Architecture:
    - Changelist cache (_cl_cache): ordered list of CL numbers (oldest → newest)
    - Index (_cl_index): mapping CL number → position in cache
    - Cache key (_cl_cache_key): (workspace_path, subdirs_tuple) for invalidation
    
    This mirrors the Git provider's caching strategy for efficient pagination.
    """
    
    def __init__(self, debug_logger: Optional[callable] = None):
        """Initialize Perforce provider.
        
        Args:
            debug_logger: Deprecated, kept for backward compatibility.
                         Use DebugLogger.is_enabled() for debug logging.
        """
        self._debug_logger = debug_logger
        
        # Changelist number cache for efficient chronological pagination
        # Design matches Git._hash_cache pattern:
        # - _cl_cache: ordered list of CL numbers (oldest → newest)
        # - _cl_index: mapping cl_number → index in _cl_cache
        # - _cl_cache_key: (workspace_path, subdirs_tuple)
        self._cl_cache: list[str] | None = None
        self._cl_index: dict[str, int] | None = None
        self._cl_cache_key: tuple | None = None
        
        # Cache for depot root → local root mapping (workspace_path → (depot_root, local_root))
        # This avoids calling p4 where for every file conversion
        self._workspace_mapping_cache: dict[str, tuple[str, str]] = {}
    
    @staticmethod
    def detect(workspace_path: str) -> bool:
        """Detect if workspace is a Perforce client workspace.
        
        Uses `p4 info` to verify the workspace is configured for Perforce,
        and checks that the Client root matches the workspace path.
        
        Args:
            workspace_path: Path to the workspace directory to check
            
        Returns:
            True if Perforce is detected and workspace is valid, False otherwise
        """
        # Check if p4 command is available
        if not shutil.which("p4"):
            return False
        
        # Try p4 info to verify client workspace
        try:
            result = subprocess.run(
                ["p4", "info"],
                cwd=workspace_path,
                capture_output=True,
                timeout=5,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            
            if result.returncode != 0:
                return False
            
            # Parse output for "Client root:" field and verify it matches workspace_path
            for line in result.stdout.splitlines():
                if line.startswith("Client root:"):
                    # Extract the client root path from the line
                    # Format: "Client root: /path/to/workspace" or "Client root: C:\path\to\workspace"
                    client_root = line.split(":", 1)[1].strip()
                    
                    # Normalize paths for comparison (resolve symlinks, normalize separators)
                    workspace_normalized = str(Path(workspace_path).resolve())
                    client_root_normalized = str(Path(client_root).resolve())
                    
                    # Check if the workspace path is within or equal to the client root
                    # This allows subdirectories of a client workspace to be detected
                    try:
                        Path(workspace_normalized).relative_to(client_root_normalized)
                        return True
                    except ValueError:
                        # workspace_path is not under client_root
                        return False
            
            # No "Client root:" found in output
            return False
            
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False
    
    def get_commits_after(
        self,
        workspace_path: str,
        after_hash: str | None,
        batch_size: int,
        subdirs: Optional[list[str]] = None,
    ) -> list[Changelist]:
        """Get changelists after a specific CL number in chronological order (oldest → newest).
        
        This is the primary changelist traversal method used by the unified indexer.
        
        The implementation uses a two-phase strategy matching Git:
        - Phase 1 (once per (workspace_path, subdirs) tuple): fetch all
          matching CL numbers in chronological order and cache them.
        - Phase 2 (per call): slice the next `batch_size` CLs from the cache and
          fetch full changelist details only for those CLs.
        
        Args:
            workspace_path: Path to the workspace/repository
            after_hash: Get changelists after this CL number (None = from beginning)
            batch_size: Maximum number of changelists to return
            subdirs: Optional list of subdirectories to filter changelists by
            
        Returns:
            List of Changelist objects in chronological order (oldest → newest)
        """
        if batch_size == 0:
            return []
        
        # Normalize subdirs for cache key (after_hash is NOT part of the key;
        # it is treated as a cursor into the global ordered sequence).
        subdirs_tuple = tuple(sorted(subdirs)) if subdirs else None
        cache_key = (workspace_path, subdirs_tuple)
        
        # (Re)build cache if it does not exist or the parameters changed
        if self._cl_cache is None or self._cl_cache_key != cache_key:
            self._cl_cache = self._fetch_all_changelist_numbers(
                workspace_path=workspace_path,
                subdirs=subdirs,
            )
            # Build index for O(1) lookup of positions
            self._cl_index = {
                cl_num: idx for idx, cl_num in enumerate(self._cl_cache)
            }
            self._cl_cache_key = cache_key
            
            if DebugLogger.is_enabled():
                from expert_among_us.utils.progress import console as progress_console
                progress_console.print(
                    f"[dim]Perforce.get_commits_after: cached {len(self._cl_cache)} changelists "
                    f"(after={after_hash or 'START'}, subdirs={subdirs_tuple})[/dim]"
                )
        
        # Determine starting index based on after_hash cursor
        if after_hash:
            # If after_hash is unknown for this (workspace_path, subdirs) view,
            # validate whether it's a real CL or truly invalid.
            if not self._cl_index:
                return []
            start_idx = self._cl_index.get(after_hash)
            if start_idx is None:
                # CL not in our cache - validate it's a real CL to catch user errors
                validate_cmd = ["p4", "changes", "-m", "1", f"@={after_hash}"]
                validate_result = subprocess.run(
                    validate_cmd,
                    cwd=workspace_path,
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
        
        if not self._cl_cache:
            return []
        
        end = start + batch_size
        batch_cl_numbers = self._cl_cache[start:end]
        
        if not batch_cl_numbers:
            # No more changelists after the given cursor
            return []
        
        # Fetch full changelist details for this batch only
        changelists = self._fetch_changelists_by_numbers(
            workspace_path=workspace_path,
            cl_numbers=batch_cl_numbers,
            subdirs=subdirs,
        )
        
        return changelists
    
    def _invalidate_cache(self) -> None:
        """Clear changelist cache and index."""
        self._cl_cache = None
        self._cl_index = None
        self._cl_cache_key = None
    
    def _fetch_all_changelist_numbers(
        self,
        workspace_path: str,
        subdirs: Optional[list[str]] = None,
    ) -> list[str]:
        """Fetch all CL numbers matching filters (lightweight operation).
        
        Uses `p4 changes -s submitted [paths...]` to get all submitted changelists.
        Perforce returns newest first, so we reverse to get oldest → newest order.
        
        Args:
            workspace_path: Path to the workspace
            subdirs: Optional list of subdirectories to filter by
            
        Returns:
            List of CL numbers as strings, ordered oldest → newest
        """
        cmd = ["p4", "changes", "-s", "submitted"]
        
        if subdirs:
            # Convert local paths to depot paths
            for subdir in subdirs:
                depot_path = self._local_to_depot_path(workspace_path, subdir)
                cmd.append(depot_path)
        else:
            # Query all submitted changelists
            cmd.append("//...")
        
        if DebugLogger.is_enabled():
            cmd_str = " ".join(str(part) for part in cmd)
            from expert_among_us.utils.progress import console as progress_console
            progress_console.print(f"[dim]Perforce._fetch_all_changelist_numbers: {cmd_str}[/dim]")
        
        result = subprocess.run(
            cmd,
            cwd=workspace_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,  # Longer timeout than detect() since this can legitimately take time
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
        
        return cl_numbers
    
    def _fetch_changelists_by_numbers(
        self,
        workspace_path: str,
        cl_numbers: list[str],
        subdirs: Optional[list[str]] = None,
    ) -> list[Changelist]:
        """Fetch full details for specific CLs (batched).
        
        Uses `p4 describe -du <cl1> <cl2> ...` to get metadata + diffs for
        multiple changelists in a single command.
        
        Args:
            workspace_path: Path to the workspace
            cl_numbers: List of CL numbers to fetch
            subdirs: Optional subdirectories (for filtering files, if needed)
            
        Returns:
            List of Changelist objects with full details
        """
        if not cl_numbers:
            return []
        
        # Convert subdirs to depot path prefixes for filtering
        depot_prefixes = None
        if subdirs:
            depot_prefixes = []
            for subdir in subdirs:
                depot_path = self._local_to_depot_path(workspace_path, subdir)
                # Remove trailing /... to get prefix for matching
                prefix = depot_path.rstrip("/...")
                depot_prefixes.append(prefix)
        
        # Batch describe command for all CLs
        cmd = ["p4", "describe", "-du"]
        cmd.extend(cl_numbers)
        
        if DebugLogger.is_enabled():
            from expert_among_us.utils.progress import console as progress_console
            cmd_str = " ".join(str(part) for part in cmd)
            progress_console.print(
                f"[dim]Perforce._fetch_changelists_by_numbers: {len(cl_numbers)} CLs via {cmd_str}[/dim]"
            )
        
        result = subprocess.run(
            cmd,
            cwd=workspace_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                result.stdout,
                result.stderr,
            )
        
        # Parse the describe output into individual changelists
        changelists = self._parse_describe_output(
            result.stdout,
            workspace_path,
            depot_prefixes=depot_prefixes
        )
        
        return changelists
    
    def _parse_describe_output(
        self,
        output: str,
        workspace_path: str,
        depot_prefixes: Optional[list[str]] = None,
    ) -> list[Changelist]:
        """Parse output from `p4 describe -du` command.
        
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
            workspace_path: Path to workspace (for context)
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
                            
                            # Convert to local path
                            local_path = self._depot_to_local_path(workspace_path, depot_path)
                            if local_path:
                                files.append(local_path)
                        i += 1
                    else:
                        break
                
                # Skip to "Differences ..." section
                while i < len(lines) and not lines[i].startswith("Differences"):
                    i += 1
                
                if i < len(lines):
                    i += 1  # Skip "Differences ..." line
                
                # Skip empty line
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
                    
                    # Only include lines if current file matches filter
                    if current_file_matches:
                        diff_lines.append(line)
                    
                    i += 1
                
                diff = "\n".join(diff_lines)
                
                # Filter binary content from diff
                diff, _binary_files = filter_binary_from_diff(diff)
                
                # Skip changelists with empty diffs (consistent with Git)
                if not diff or not diff.strip():
                    continue
                
                changelist = Changelist(
                    id=cl_number,
                    expert_name="",  # Will be set by caller
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
    
    def _local_to_depot_path(self, workspace_path: str, local_subdir: str) -> str:
        """Convert local path to depot path syntax using cached workspace mapping.
        
        Fast string substitution approach - only calls p4 once per workspace.
        Uses the cached workspace mapping in reverse (local → depot).
        
        Args:
            workspace_path: Path to workspace
            local_subdir: Local subdirectory relative to workspace
            
        Returns:
            Depot path with recursive wildcard (e.g., "//depot/src/engine/...")
        """
        depot_root, local_root = self._get_workspace_mapping(workspace_path)
        
        if depot_root and local_root:
            # Build full local path
            local_path = Path(workspace_path) / local_subdir
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
    
    def _get_workspace_mapping(self, workspace_path: str) -> tuple[str, str]:
        """Get depot root and local root mapping for workspace (cached).
        
        Uses `p4 where` on the workspace root to determine the mapping between
        depot paths and local paths. This is called once per workspace and cached
        for performance.
        
        Args:
            workspace_path: Path to workspace
            
        Returns:
            Tuple of (depot_root, local_root) for string substitution
        """
        if workspace_path in self._workspace_mapping_cache:
            return self._workspace_mapping_cache[workspace_path]
        
        try:
            # Use p4 where on workspace root to get the mapping
            result = subprocess.run(
                ["p4", "where", workspace_path],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if lines:
                    parts = lines[0].split()
                    if len(parts) >= 3:
                        # Extract depot and local roots, removing /... suffix
                        depot_root = parts[0].rstrip("/...")
                        local_root = parts[2].rstrip("/...").rstrip("\\")
                        
                        self._workspace_mapping_cache[workspace_path] = (depot_root, local_root)
                        
                        if DebugLogger.is_enabled():
                            from expert_among_us.utils.progress import console as progress_console
                            progress_console.print(
                                f"[dim]Perforce._get_workspace_mapping: cached mapping "
                                f"depot='{depot_root}' → local='{local_root}'[/dim]"
                            )
                        
                        return (depot_root, local_root)
        except (subprocess.TimeoutExpired, OSError):
            pass
        
        # Fallback: no mapping available
        self._workspace_mapping_cache[workspace_path] = ("", workspace_path)
        return ("", workspace_path)
    
    def _depot_to_local_path(self, workspace_path: str, depot_path: str) -> str:
        r"""Convert depot path to local path using cached workspace mapping.
        
        Fast string substitution approach - only calls p4 once per workspace.
        
        Args:
            workspace_path: Path to workspace
            depot_path: Depot path (e.g., "//javelin/mainline/dev/GameCode/Game/VersionTrack.h")
            
        Returns:
            Full local path (e.g., "C:\Perforce\Javelin\mainline\dev\GameCode\Game\VersionTrack.h")
        """
        depot_root, local_root = self._get_workspace_mapping(workspace_path)
        
        if depot_root and depot_path.startswith(depot_root):
            # String substitution: replace depot root with local root
            relative_path = depot_path[len(depot_root):].lstrip("/")
            return str(Path(local_root) / relative_path)
        
        return None
    
    def get_tracked_files_at_commit(
        self,
        workspace_path: str,
        commit_hash: str,
        subdirs: Optional[list[str]] = None,
    ) -> list[str]:
        """Get list of tracked files at a specific changelist.
        
        Uses `p4 files [path...]@CL` to list all files as they existed at that changelist.
        Note: Uses @ (not @=) to get the state of files at that changelist, not just
        files modified in that specific changelist.
        
        Args:
            workspace_path: Path to the workspace/repository
            commit_hash: Changelist number
            subdirs: Optional list of subdirectories to filter by
            
        Returns:
            List of file paths (relative to workspace root) tracked at the changelist
        """
        cmd = ["p4", "files"]
        
        if subdirs:
            # Query specific subdirectories
            for subdir in subdirs:
                depot_path = self._local_to_depot_path(workspace_path, subdir)
                # Remove trailing /... for the @ syntax
                depot_path = depot_path.rstrip("/...")
                cmd.append(f"{depot_path}/...@{commit_hash}")
        else:
            # Query all files
            cmd.append(f"//...@{commit_hash}")
        
        if DebugLogger.is_enabled():
            from expert_among_us.utils.progress import console as progress_console
            cmd_str = " ".join(str(part) for part in cmd)
            progress_console.print(f"[dim]Perforce.get_tracked_files_at_commit: {cmd_str}[/dim]")
        
        result = subprocess.run(
            cmd,
            cwd=workspace_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        
        if result.returncode != 0:
            return []
        
        # Parse output: "//depot/src/file.cpp#42 - edit change 12345 (text)"
        files = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("//"):
                parts = line.split("#", 1)
                if parts:
                    depot_path = parts[0]
                    local_path = self._depot_to_local_path(workspace_path, depot_path)
                    if local_path:
                        files.append(local_path)
        
        return files
    
    def get_file_content_at_commit(
        self,
        workspace_path: str,
        file_path: str,
        commit_hash: str,
    ) -> Optional[str]:
        """Get file content at a specific changelist.
        
        This is a thin wrapper around get_files_content_at_commit()
        for backward compatibility.
        
        Args:
            workspace_path: Path to workspace
            file_path: Relative path to file
            commit_hash: Changelist number
            
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
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict[str, Optional[str]]:
        """Get content for multiple files at a specific changelist (batched operation).
        
        Uses `p4 print -q` to fetch file contents in batches.
        
        Args:
            workspace_path: Path to the workspace/repository
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
            for local_path in batch_paths:
                depot_path = self._local_to_depot_path(workspace_path, local_path)
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
            
            try:
                result = subprocess.run(
                    cmd,
                    cwd=workspace_path,
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                )
                
                if result.returncode != 0:
                    # Failed to fetch this batch, leave as None
                    continue
                
                # Parse output which has format:
                # //depot/path/file.cpp#42 - edit change 12345 (text)
                # 
                # <file content>
                # 
                # //depot/path/file2.cpp#15 - edit change 12345 (text)
                # 
                # <file content>
                
                self._parse_print_output(result.stdout, batch_paths, results, workspace_path)
                
                # Report progress after processing each batch
                if progress_callback:
                    try:
                        progress_callback(batch_end, len(unique_paths))
                    except Exception:
                        # Ignore callback errors to prevent disrupting file reading
                        pass
                
            except (OSError, subprocess.TimeoutExpired):
                # Skip this batch on error
                continue
        
        return results
    
    def _parse_print_output(
        self,
        output: str,
        batch_paths: list[str],
        results: dict[str, Optional[str]],
        workspace_path: str,
    ) -> None:
        """Parse output from `p4 print -q` command.
        
        Updates results dict in-place with file contents.
        
        Args:
            output: Raw output from p4 print
            batch_paths: List of local paths that were queried
            results: Results dictionary to update
            workspace_path: Workspace path for path conversion
        """
        lines = output.split("\n")
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Look for file header: "//depot/path/file.cpp#42 - edit change 12345 (text)"
            # It will match the cached depot root path
            local_path = self._parse_local_path_line(workspace_path, line)
            if local_path:
                i += 1

                # Check for binary marker (case-insensitive to be safe)
                if "(binary)" in line.lower():
                    # Binary file, skip content and save bandwidth
                    if DebugLogger.is_enabled():
                        from expert_among_us.utils.progress import console as progress_console
                        progress_console.print(f"[dim]Perforce: Skipping binary file {local_path}[/dim]")
                    i += 1
                    # Skip until next file header or EOF
                    while i < len(lines) and not self._parse_local_path_line(workspace_path, lines[i]):
                        i += 1
                    continue
                
                # Skip empty line after header
                if i < len(lines) and not lines[i].strip():
                    i += 1
                
                # Collect file content until next file header or EOF
                content_lines = []
                while i < len(lines):
                    if self._parse_local_path_line(workspace_path, lines[i]):
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
                
                # Store content for matching local path
                if local_path in results:
                    results[local_path] = content
            else:
                i += 1
    
    def _parse_local_path_line(self, workspace_path: str, line: str):
        """Returns local path if the line contains a depot path, else return None"""
        depot_path = line.split("#")[0] if "#" in line else None
        return self._depot_to_local_path(workspace_path, depot_path) if depot_path else None
    
    def get_latest_commit_time(
        self,
        workspace_path: str,
        subdirs: Optional[list[str]] = None,
    ) -> Optional[datetime]:
        """Get the timestamp of the most recent changelist.
        
        Uses `p4 changes -m 1 -s submitted [paths...]` to get the latest CL.
        
        Args:
            workspace_path: Path to the workspace/repository
            subdirs: Optional list of subdirectories to filter changelists by
            
        Returns:
            Datetime of the most recent changelist, or None if no changelists found
        """
        cmd = ["p4", "changes", "-m", "1", "-s", "submitted"]
        
        if subdirs:
            for subdir in subdirs:
                depot_path = self._local_to_depot_path(workspace_path, subdir)
                cmd.append(depot_path)
        else:
            cmd.append("//...")
        
        result = subprocess.run(
            cmd,
            cwd=workspace_path,
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
        workspace_path: str,
        subdirs: Optional[list[str]] = None,
    ) -> int:
        """Return the total number of changelists to consider for indexing.
        
        Reuses _fetch_all_changelist_numbers() to ensure consistent deduplication
        and leverage existing cache.
        
        Args:
            workspace_path: Path to the workspace / repository root.
            subdirs: Optional list of subdirectories to filter by.
            
        Returns:
            Integer count of changelists (deduplicated).
        """
        # Reuse cache if available
        subdirs_tuple = tuple(sorted(subdirs)) if subdirs else None
        cache_key = (workspace_path, subdirs_tuple)
        
        if self._cl_cache is not None and self._cl_cache_key == cache_key:
            return len(self._cl_cache)
        
        # Build cache and return count
        cl_numbers = self._fetch_all_changelist_numbers(
            workspace_path=workspace_path,
            subdirs=subdirs,
        )
        return len(cl_numbers)