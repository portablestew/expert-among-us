"""
Property-based tests for schema and CRUD operations.

Tests Properties 3, 14, and 15 from the design document using Hypothesis.

**Validates: Requirements 1.2, 1.4, 2.1, 2.2, 2.3, 7.2, 7.3**
"""

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st

from expert_among_us.models.expert import _validate_identifier_name, _NAME_PATTERN
from expert_among_us.db.metadata.sqlite import SQLiteMetadataDB


# --- Shared DB Fixture ---
# A single temp directory and DB is created once for the module and reused
# across all hypothesis examples. Tables are truncated between examples to
# provide isolation without the filesystem overhead of recreating DBs.

_shared_tmpdir = None
_shared_db = None


def _get_shared_db():
    """Get the shared DB instance, creating it on first call."""
    global _shared_tmpdir, _shared_db
    if _shared_db is None:
        _shared_tmpdir = tempfile.mkdtemp(prefix="prop_test_")
        expert_name = "prop_test_expert"
        db_path = Path(_shared_tmpdir) / "data" / expert_name / "metadata.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        _shared_db = SQLiteMetadataDB(expert_name)
        _shared_db.db_path = str(db_path)
        _shared_db.initialize()
        _shared_db.create_expert("prop_test_expert")
    return _shared_db


def _reset_db(db):
    """Truncate project-related tables to reset state between examples."""
    cursor = db.conn.cursor()
    cursor.execute("DELETE FROM changelist_files")
    cursor.execute("DELETE FROM file_chunks")
    cursor.execute("DELETE FROM file_contents")
    cursor.execute("DELETE FROM changelists")
    cursor.execute("DELETE FROM projects")
    db.conn.commit()


# --- Strategies ---

# Strategy for valid project names: alphanumeric, hyphens, underscores (can start with any)
valid_name_start = st.sampled_from(
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
)
valid_name_rest = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
    min_size=0,
    max_size=50,
)

@st.composite
def valid_project_names(draw):
    """Generate valid project names matching [a-zA-Z0-9_-][a-zA-Z0-9_-]*."""
    start = draw(valid_name_start)
    rest = draw(valid_name_rest)
    return start + rest


# Strategy for invalid project names
@st.composite
def invalid_project_names(draw):
    """Generate names that should fail validation."""
    category = draw(st.sampled_from([
        "empty",
        "contains_slash",
        "contains_backslash",
        "contains_space",
        "contains_special",
        "whitespace_only",
    ]))

    if category == "empty":
        return ""
    elif category == "contains_slash":
        prefix = draw(valid_project_names())
        suffix = draw(st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz",
            min_size=1,
            max_size=10,
        ))
        return prefix + "/" + suffix
    elif category == "contains_backslash":
        prefix = draw(valid_project_names())
        suffix = draw(st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz",
            min_size=1,
            max_size=10,
        ))
        return prefix + "\\" + suffix
    elif category == "contains_space":
        prefix = draw(valid_project_names())
        suffix = draw(st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz",
            min_size=1,
            max_size=10,
        ))
        return prefix + " " + suffix
    elif category == "contains_special":
        prefix = draw(valid_project_names())
        special = draw(st.sampled_from(list("!@#$%^&*()+=[]{}|;:',.<>?`~")))
        return prefix + special
    elif category == "whitespace_only":
        return draw(st.text(
            alphabet=" \t\n\r",
            min_size=1,
            max_size=10,
        ))


# Strategy for workspace paths (as strings for DB tests)
workspace_paths = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789/_-.",
    min_size=1,
    max_size=100,
).map(lambda s: "/" + s)

# Strategy for VCS types
vcs_types = st.sampled_from(["git", "p4"])

# Strategy for subdirectory lists
subdirs_strategy = st.lists(
    st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789/_-.",
        min_size=1,
        max_size=30,
    ),
    min_size=0,
    max_size=5,
)


# --- Property 3: Project Name Validation (simple unit tests) ---

class TestProperty3ProjectNameValidation:
    """
    Property 3: Project Name Validation

    Names must contain only alphanumeric characters, hyphens, and underscores.
    Path separators, empty strings, and special characters are rejected.

    **Validates: Requirements 2.1, 2.2, 2.3**
    """

    def test_valid_names_accepted(self):
        """Valid names are accepted."""
        for name in ["my-project", "_sharpmake_", "Repo123", "a", "-leading", "under_score"]:
            result = _validate_identifier_name(name, "Project name")
            assert result == name

    def test_invalid_names_rejected(self):
        """Invalid names are rejected."""
        for name in ["", " ", "has space", "has/slash", "has\\back", "name@here", "dot.name"]:
            with pytest.raises(ValueError):
                _validate_identifier_name(name, "Project name")


# --- Property 14: Composite PK Uniqueness ---

class TestProperty14CompositePKUniqueness:
    """
    Property 14: Composite PK Uniqueness

    For any expert, attempting to create two projects with the same name should
    not error (INSERT OR IGNORE), while creating projects with different names
    should succeed independently.

    **Validates: Requirements 1.2, 1.4**
    """

    @given(
        project_name=valid_project_names(),
        workspace1=workspace_paths,
        workspace2=workspace_paths,
        vcs1=vcs_types,
        vcs2=vcs_types,
    )
    @settings(max_examples=50)
    def test_duplicate_project_name_is_noop(self, project_name, workspace1, workspace2, vcs1, vcs2):
        """Creating the same project twice does not error (INSERT OR IGNORE semantics)."""
        db = _get_shared_db()
        _reset_db(db)

        # First creation
        db.create_project(
            expert_name="prop_test_expert",
            project_name=project_name,
            workspace_path=workspace1,
            subdirs=[],
            vcs_type=vcs1,
        )

        # Second creation with potentially different values — should not error
        db.create_project(
            expert_name="prop_test_expert",
            project_name=project_name,
            workspace_path=workspace2,
            subdirs=["src"],
            vcs_type=vcs2,
        )

        # Original values preserved (INSERT OR IGNORE keeps first)
        project = db.get_project("prop_test_expert", project_name)
        assert project is not None
        assert project["name"] == project_name
        assert project["workspace_path"] == workspace1
        assert project["vcs_type"] == vcs1

    @given(
        name1=valid_project_names(),
        name2=valid_project_names(),
        workspace1=workspace_paths,
        workspace2=workspace_paths,
        vcs1=vcs_types,
        vcs2=vcs_types,
    )
    @settings(max_examples=50)
    def test_different_names_coexist(self, name1, name2, workspace1, workspace2, vcs1, vcs2):
        """Projects with different names can both be created successfully."""
        assume(name1 != name2)

        db = _get_shared_db()
        _reset_db(db)

        db.create_project(
            expert_name="prop_test_expert",
            project_name=name1,
            workspace_path=workspace1,
            subdirs=[],
            vcs_type=vcs1,
        )
        db.create_project(
            expert_name="prop_test_expert",
            project_name=name2,
            workspace_path=workspace2,
            subdirs=[],
            vcs_type=vcs2,
        )

        proj1 = db.get_project("prop_test_expert", name1)
        proj2 = db.get_project("prop_test_expert", name2)

        assert proj1 is not None
        assert proj2 is not None
        assert proj1["name"] == name1
        assert proj2["name"] == name2


# --- Property 15: Project CRUD Round-Trip ---

class TestProperty15ProjectCRUDRoundTrip:
    """
    Property 15: Project CRUD Round-Trip

    For any valid project configuration (name, expert_name, workspace_path,
    subdirs, vcs_type), creating the project and then retrieving it should
    return all stored fields with their original values.

    **Validates: Requirements 1.3, 7.2, 7.3**
    """

    @given(
        project_name=valid_project_names(),
        workspace_path=workspace_paths,
        subdirs=subdirs_strategy,
        vcs_type=vcs_types,
    )
    @settings(max_examples=50)
    def test_create_then_retrieve_matches(self, project_name, workspace_path, subdirs, vcs_type):
        """Creating a project then retrieving it returns the same values."""
        # Filter out subdirs with commas since they're stored comma-separated
        # and would be split incorrectly on retrieval
        subdirs = [s for s in subdirs if "," not in s]

        db = _get_shared_db()
        _reset_db(db)

        db.create_project(
            expert_name="prop_test_expert",
            project_name=project_name,
            workspace_path=workspace_path,
            subdirs=subdirs,
            vcs_type=vcs_type,
        )

        retrieved = db.get_project("prop_test_expert", project_name)

        assert retrieved is not None
        assert retrieved["name"] == project_name
        assert retrieved["expert_name"] == "prop_test_expert"
        assert retrieved["workspace_path"] == workspace_path
        assert retrieved["vcs_type"] == vcs_type
        # subdirs round-trip: stored as comma-separated, retrieved as list
        # Empty strings in subdirs get filtered out on retrieval
        expected_subdirs = [s.strip() for s in subdirs if s.strip()]
        assert retrieved["subdirs"] == expected_subdirs
        # Default values
        assert retrieved["has_vector_metadata"] is True
        assert retrieved["last_processed_commit_hash"] is None
        assert retrieved["first_processed_commit_hash"] is None
