"""
Property-based test for expert listing completeness.

Tests Property 16 from the design document using Hypothesis.

**Validates: Requirements 1.5, 8.1, 8.2, 8.3**
"""

import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest
from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st

from expert_among_us.db.metadata.sqlite import SQLiteMetadataDB
from expert_among_us.models.changelist import Changelist
from expert_among_us.api.operations import list_experts


# --- Helpers ---

@contextmanager
def _make_data_dir():
    """Context manager that creates a temporary data directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _create_expert_with_projects(data_dir: Path, expert_name: str, projects: list[dict]):
    """Create an expert with the given projects and changelists in a temp data dir.
    
    Args:
        data_dir: Root data directory (list_experts expects data_dir/data/<expert>/metadata.db)
        expert_name: Name of the expert to create
        projects: List of dicts with keys: name, commit_count, project_root, vcs_type
    """
    # Ensure the directory structure exists for SQLite to create the DB file
    db_dir = data_dir / "data" / expert_name
    db_dir.mkdir(parents=True, exist_ok=True)

    db = SQLiteMetadataDB(expert_name, data_dir=data_dir)
    db.initialize()
    db.create_expert(expert_name)

    for proj in projects:
        db.create_project(
            expert_name=expert_name,
            project_name=proj["name"],
            project_root=proj["project_root"],
            vcs_type=proj["vcs_type"],
        )

        # Insert the specified number of changelists for this project
        changelists = []
        for i in range(proj["commit_count"]):
            cl = Changelist(
                id=f"{expert_name}-{proj['name']}-{i}-{uuid.uuid4().hex[:8]}",
                expert_name=expert_name,
                project_name=proj["name"],
                timestamp=datetime.now(timezone.utc),
                author="test-author",
                message=f"Commit {i} for project {proj['name']}",
                diff=f"diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
                files=[f"{proj['name']}/file_{i}.py"],
            )
            changelists.append(cl)

        if changelists:
            db.insert_changelists(changelists)

    db.close()


# --- Strategies ---

# Strategy for valid project names
valid_name_start = st.sampled_from(
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
)
valid_name_rest = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-",
    min_size=0,
    max_size=15,
)


@st.composite
def valid_project_names(draw):
    """Generate valid project names matching [a-zA-Z0-9][a-zA-Z0-9_-]*."""
    start = draw(valid_name_start)
    rest = draw(valid_name_rest)
    return start + rest


# Strategy for commit counts per project (keep small for test speed)
commit_counts = st.integers(min_value=0, max_value=5)

# Strategy for VCS types
vcs_types = st.sampled_from(["git", "p4"])


@st.composite
def project_configs(draw):
    """Generate a project config dict with a unique name and commit count."""
    name = draw(valid_project_names())
    commit_count = draw(commit_counts)
    vcs_type = draw(vcs_types)
    return {
        "name": name,
        "commit_count": commit_count,
        "project_root": f"/repos/{name}",
        "vcs_type": vcs_type,
    }


@st.composite
def unique_project_lists(draw):
    """Generate a list of 1-5 projects with unique names."""
    n = draw(st.integers(min_value=1, max_value=5))
    projects = []
    seen_names = set()
    for _ in range(n):
        proj = draw(project_configs())
        # Ensure uniqueness by appending index if needed
        base_name = proj["name"]
        while proj["name"] in seen_names:
            proj["name"] = base_name + draw(
                st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=4)
            )
        seen_names.add(proj["name"])
        projects.append(proj)
    return projects


# --- Property 16: Expert Listing Completeness ---

class TestProperty16ExpertListingCompleteness:
    """
    Property 16: Expert Listing Completeness

    For any expert with N projects, listing that expert should return exactly N
    project entries, and the total commit count should equal the sum of individual
    project commit counts.

    **Validates: Requirements 1.5, 8.1, 8.2, 8.3**
    """

    @given(projects=unique_project_lists())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_listing_returns_correct_project_count_and_total_commits(self, projects):
        """Listing an expert returns exactly N projects with correct total_commit_count."""
        expert_name = "listing-test-expert"

        with _make_data_dir() as data_dir:
            _create_expert_with_projects(data_dir, expert_name, projects)

            # Call list_experts with the temp data directory
            experts = list_experts(data_dir=data_dir)

            # Should find exactly one expert
            assert len(experts) == 1
            expert_info = experts[0]

            # Verify project count
            assert len(expert_info.projects) == len(projects)

            # Verify total_commit_count equals sum of individual project commit counts
            expected_total = sum(p["commit_count"] for p in projects)
            assert expert_info.total_commit_count == expected_total

            # Verify total_commit_count equals sum of per-project counts from the listing
            assert expert_info.total_commit_count == sum(
                p.commit_count for p in expert_info.projects
            )

    @given(projects=unique_project_lists())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_individual_project_commit_counts_match(self, projects):
        """Each project in the listing has the correct individual commit count."""
        expert_name = "listing-counts-expert"

        with _make_data_dir() as data_dir:
            _create_expert_with_projects(data_dir, expert_name, projects)

            experts = list_experts(data_dir=data_dir)
            assert len(experts) == 1
            expert_info = experts[0]

            # Build a lookup of expected commit counts by project name
            expected_counts = {p["name"]: p["commit_count"] for p in projects}

            # Verify each project's commit count matches what was inserted
            for proj_info in expert_info.projects:
                assert proj_info.name in expected_counts, (
                    f"Unexpected project {proj_info.name} in listing"
                )
                assert proj_info.commit_count == expected_counts[proj_info.name], (
                    f"Project {proj_info.name}: expected {expected_counts[proj_info.name]} "
                    f"commits, got {proj_info.commit_count}"
                )
