"""
Property-based tests for searcher filtering (Properties 5, 6, 7, 8, 9).

Tests the Searcher's where clause construction, startsWith file filtering,
project filter extraction, legacy expert safety, and expansion threading.

**Validates: Requirements 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3, 6.4, 10.1, 10.3, 11.1, 11.2, 11.3**
"""

from datetime import datetime
from typing import List, Optional
from unittest.mock import MagicMock, patch, call

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from expert_among_us.core.searcher import Searcher
from expert_among_us.models.changelist import Changelist
from expert_among_us.models.query import QueryParams


# --- Strategies ---

# Valid project names: starts with alphanumeric, then alphanumeric/hyphens/underscores
valid_name_start = st.sampled_from(
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
)
valid_name_rest = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
    min_size=0,
    max_size=20,
)


@st.composite
def valid_project_names(draw):
    """Generate valid project names matching [a-zA-Z0-9][a-zA-Z0-9_-]*."""
    start = draw(valid_name_start)
    rest = draw(valid_name_rest)
    return start + rest


# File path segments (no slashes)
file_path_segment = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-.",
    min_size=1,
    max_size=15,
)


@st.composite
def relative_file_paths(draw):
    """Generate relative file paths like 'src/main.py' or 'README.md'."""
    segments = draw(st.lists(file_path_segment, min_size=1, max_size=4))
    return "/".join(segments)


@st.composite
def prefixed_file_paths(draw, project_names):
    """Generate file paths prefixed with one of the given project names."""
    project = draw(st.sampled_from(project_names))
    rel_path = draw(relative_file_paths())
    return f"{project}/{rel_path}"


@st.composite
def changelists_with_files(draw, project_names):
    """Generate a Changelist with files prefixed by a known project name."""
    project = draw(st.sampled_from(project_names))
    num_files = draw(st.integers(min_value=1, max_value=5))
    files = []
    for _ in range(num_files):
        rel_path = draw(relative_file_paths())
        files.append(f"{project}/{rel_path}")

    cl_id = draw(st.text(
        alphabet="abcdef0123456789",
        min_size=8,
        max_size=12,
    ))
    author = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz",
        min_size=3,
        max_size=10,
    ))
    message = draw(st.text(min_size=5, max_size=50))

    return Changelist(
        id=cl_id,
        expert_name="test-expert",
        project_name=project,
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        author=author,
        message=message,
        diff="",
        files=files,
    )


def _make_mock_metadata_db(known_projects: List[str]):
    """Create a mock MetadataDB that returns given project names from list_projects."""
    mock_db = MagicMock()
    mock_db.list_projects.return_value = [{"name": p} for p in known_projects]
    return mock_db


def _make_mock_vector_db():
    """Create a mock VectorDB with all search methods returning empty lists."""
    mock_db = MagicMock()
    mock_db.search_metadata.return_value = []
    mock_db.search_diffs.return_value = []
    mock_db.search_files.return_value = []
    return mock_db


def _make_mock_embedder():
    """Create a mock Embedder."""
    mock = MagicMock()
    mock.embed.return_value = [0.1] * 1024
    return mock


def _make_searcher(known_projects: List[str], has_vector_metadata: bool = True):
    """Create a Searcher with mocked dependencies and known projects."""
    metadata_db = _make_mock_metadata_db(known_projects)
    vector_db = _make_mock_vector_db()
    embedder = _make_mock_embedder()

    searcher = Searcher(
        expert_name="test-expert",
        embedder=embedder,
        metadata_db=metadata_db,
        vector_db=vector_db,
        has_vector_metadata=has_vector_metadata,
        enable_query_expansion=False,
        enable_reranking=False,
    )
    return searcher


# --- Property 5: Query Scoping Soundness ---

class TestProperty5QueryScopingSoundness:
    """
    Property 5: Query Scoping Soundness

    For any query with a project filter specifying projects [X, Y, ...], all
    returned results should have project metadata matching one of those projects.
    For any query without a project filter, results from any project in the
    expert should be eligible.

    **Validates: Requirements 4.3, 4.4**
    """

    @given(
        known_projects=st.lists(valid_project_names(), min_size=2, max_size=5, unique=True),
    )
    @settings(max_examples=50)
    def test_filtered_query_builds_where_clause_with_only_specified_projects(self, known_projects):
        """When files have known project prefixes, where clause scopes to those projects."""
        searcher = _make_searcher(known_projects)

        # Pick a subset of projects to filter on
        filter_projects = known_projects[:2]
        files = [f"{p}/" for p in filter_projects]

        where_clause = searcher._build_where_clause(files)

        assert where_clause is not None
        assert "project" in where_clause
        assert "$in" in where_clause["project"]

        # All projects in the where clause must be from our filter set
        for proj in where_clause["project"]["$in"]:
            assert proj in filter_projects

    @given(
        known_projects=st.lists(valid_project_names(), min_size=1, max_size=5, unique=True),
    )
    @settings(max_examples=50)
    def test_unfiltered_query_returns_none_where_clause(self, known_projects):
        """When no files are provided, no where clause is built (full search)."""
        searcher = _make_searcher(known_projects)

        where_clause = searcher._build_where_clause(None)
        assert where_clause is None

    @given(
        known_projects=st.lists(valid_project_names(), min_size=1, max_size=5, unique=True),
    )
    @settings(max_examples=50)
    def test_empty_files_list_returns_none_where_clause(self, known_projects):
        """When files list is empty, no where clause is built."""
        searcher = _make_searcher(known_projects)

        where_clause = searcher._build_where_clause([])
        assert where_clause is None


# --- Property 6: StartsWith File Filter Correctness ---

class TestProperty6StartsWithFileFilterCorrectness:
    """
    Property 6: StartsWith File Filter Correctness

    For any list of changelists and any list of query file paths, the startsWith
    filter should include a changelist if and only if at least one of its file
    paths starts with at least one of the query file paths.

    **Validates: Requirements 5.1, 5.2, 5.3, 5.4**
    """

    @given(
        known_projects=st.lists(valid_project_names(), min_size=2, max_size=4, unique=True),
        data=st.data(),
    )
    @settings(max_examples=50)
    def test_startswith_includes_matching_changelists(self, known_projects, data):
        """A changelist with a file matching a query path prefix is included."""
        searcher = _make_searcher(known_projects)

        # Generate changelists
        changelists = data.draw(
            st.lists(changelists_with_files(known_projects), min_size=1, max_size=5)
        )

        # Pick a query file path that is a prefix of at least one changelist's file
        target_cl = data.draw(st.sampled_from(changelists))
        target_file = data.draw(st.sampled_from(target_cl.files))
        # Use a prefix of the target file (at least the project/ part)
        prefix_end = target_file.index("/") + 1  # include the slash
        query_prefix = target_file[:prefix_end]

        # Build scores dict for _apply_commit_filters
        scores = {
            cl.id: {"score": 0.8, "source": "metadata", "chroma_id": None, "embedding": None}
            for cl in changelists
        }

        params = QueryParams(
            prompt="test query",
            max_changes=20,
            max_file_chunks=10,
            files=[query_prefix],
        )

        results = searcher._apply_commit_filters(changelists, scores, params)

        # The target changelist must be included (its file starts with query_prefix)
        result_ids = {r.changelist.id for r in results}
        assert target_cl.id in result_ids

    @given(
        known_projects=st.lists(valid_project_names(), min_size=2, max_size=4, unique=True),
        data=st.data(),
    )
    @settings(max_examples=50)
    def test_startswith_excludes_non_matching_changelists(self, known_projects, data):
        """A changelist with no file matching any query path prefix is excluded."""
        assume(len(known_projects) >= 2)
        searcher = _make_searcher(known_projects)

        # Generate changelists all from project A
        project_a = known_projects[0]
        project_b = known_projects[1]

        # Ensure project names are different
        assume(project_a != project_b)

        # Build a changelist with only project_a files
        files_a = [f"{project_a}/src/file1.py", f"{project_a}/src/file2.py"]
        cl = Changelist(
            id="abc12345",
            expert_name="test-expert",
            project_name=project_a,
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            author="testuser",
            message="test commit message",
            diff="",
            files=files_a,
        )

        # Query prefix is for project_b only
        query_prefix = f"{project_b}/"

        # Ensure project_a files don't start with project_b prefix
        assume(not any(f.startswith(query_prefix) for f in files_a))

        scores = {
            cl.id: {"score": 0.8, "source": "metadata", "chroma_id": None, "embedding": None}
        }

        params = QueryParams(
            prompt="test query",
            max_changes=20,
            max_file_chunks=10,
            files=[query_prefix],
        )

        results = searcher._apply_commit_filters([cl], scores, params)

        # The changelist must NOT be included
        assert len(results) == 0

    @given(
        known_projects=st.lists(valid_project_names(), min_size=1, max_size=4, unique=True),
        data=st.data(),
    )
    @settings(max_examples=50)
    def test_no_files_filter_includes_all_changelists(self, known_projects, data):
        """When no files filter is provided, all changelists pass the filter."""
        searcher = _make_searcher(known_projects)

        changelists = data.draw(
            st.lists(changelists_with_files(known_projects), min_size=1, max_size=5)
        )

        scores = {
            cl.id: {"score": 0.8, "source": "metadata", "chroma_id": None, "embedding": None}
            for cl in changelists
        }

        params = QueryParams(
            prompt="test query",
            max_changes=20,
            max_file_chunks=10,
            files=None,
        )

        results = searcher._apply_commit_filters(changelists, scores, params)

        # All changelists should be included
        assert len(results) == len(changelists)

    @given(
        known_projects=st.lists(valid_project_names(), min_size=1, max_size=4, unique=True),
        data=st.data(),
    )
    @settings(max_examples=50)
    def test_startswith_biconditional(self, known_projects, data):
        """A changelist is included iff at least one file starts with a query path."""
        searcher = _make_searcher(known_projects)

        changelists = data.draw(
            st.lists(changelists_with_files(known_projects), min_size=1, max_size=5)
        )

        # Ensure unique IDs (duplicates would cause dict merging issues)
        seen_ids = set()
        unique_changelists = []
        for cl in changelists:
            if cl.id not in seen_ids:
                seen_ids.add(cl.id)
                unique_changelists.append(cl)
        changelists = unique_changelists
        assume(len(changelists) >= 1)

        # Generate query files from known projects
        query_files = data.draw(
            st.lists(
                st.sampled_from([f"{p}/" for p in known_projects]),
                min_size=1,
                max_size=3,
            )
        )

        scores = {
            cl.id: {"score": 0.8, "source": "metadata", "chroma_id": None, "embedding": None}
            for cl in changelists
        }

        params = QueryParams(
            prompt="test query",
            max_changes=20,
            max_file_chunks=10,
            files=query_files,
        )

        results = searcher._apply_commit_filters(changelists, scores, params)
        result_ids = {r.changelist.id for r in results}

        # Verify biconditional: included iff matches
        for cl in changelists:
            expected_match = any(
                any(cf.startswith(qf) for cf in cl.files)
                for qf in query_files
            )
            if expected_match:
                assert cl.id in result_ids, (
                    f"CL {cl.id} should be included (has matching file)"
                )
            else:
                assert cl.id not in result_ids, (
                    f"CL {cl.id} should be excluded (no matching file)"
                )


# --- Property 7: Project Filter Extraction Soundness ---

class TestProperty7ProjectFilterExtractionSoundness:
    """
    Property 7: Project Filter Extraction Soundness

    For any list of file paths and list of known projects, the extracted where
    clause should: (a) contain only project names that appear in the known
    projects list, (b) contain only project names that appear as a prefix in
    at least one file path, and (c) be None if no file prefixes match any
    known project.

    **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
    """

    @given(
        known_projects=st.lists(valid_project_names(), min_size=1, max_size=5, unique=True),
        data=st.data(),
    )
    @settings(max_examples=50)
    def test_where_clause_only_contains_known_projects(self, known_projects, data):
        """Where clause only includes project names from the known projects list."""
        searcher = _make_searcher(known_projects)

        # Generate file paths, some with known project prefixes, some without
        files = data.draw(st.lists(
            st.one_of(
                # Files with known project prefix
                st.builds(
                    lambda p, f: f"{p}/{f}",
                    st.sampled_from(known_projects),
                    relative_file_paths(),
                ),
                # Files without known project prefix
                relative_file_paths(),
            ),
            min_size=1,
            max_size=5,
        ))

        where_clause = searcher._build_where_clause(files)

        if where_clause is not None:
            projects_in_clause = where_clause["project"]["$in"]
            for proj in projects_in_clause:
                assert proj in known_projects, (
                    f"Project '{proj}' in where clause but not in known projects"
                )

    @given(
        known_projects=st.lists(valid_project_names(), min_size=1, max_size=5, unique=True),
        data=st.data(),
    )
    @settings(max_examples=50)
    def test_where_clause_projects_appear_as_file_prefixes(self, known_projects, data):
        """Every project in the where clause has at least one file path starting with it."""
        searcher = _make_searcher(known_projects)

        files = data.draw(st.lists(
            st.one_of(
                st.builds(
                    lambda p, f: f"{p}/{f}",
                    st.sampled_from(known_projects),
                    relative_file_paths(),
                ),
                relative_file_paths(),
            ),
            min_size=1,
            max_size=5,
        ))

        where_clause = searcher._build_where_clause(files)

        if where_clause is not None:
            projects_in_clause = where_clause["project"]["$in"]
            for proj in projects_in_clause:
                # At least one file must have this project as a prefix
                has_prefix = any(
                    f.rstrip("/").split("/")[0] == proj
                    for f in files
                )
                assert has_prefix, (
                    f"Project '{proj}' in where clause but no file has it as prefix"
                )

    @given(
        known_projects=st.lists(valid_project_names(), min_size=1, max_size=5, unique=True),
        data=st.data(),
    )
    @settings(max_examples=50)
    def test_none_when_no_prefixes_match(self, known_projects, data):
        """Where clause is None when no file prefix matches any known project."""
        searcher = _make_searcher(known_projects)

        # Generate file paths that definitely don't start with any known project
        # Use a prefix that's unlikely to match any generated project name
        non_matching_files = data.draw(st.lists(
            st.builds(
                lambda f: f"zzz-nonexistent-project/{f}",
                relative_file_paths(),
            ),
            min_size=1,
            max_size=5,
        ))

        # Make sure none of our non-matching prefixes happen to match
        assume(not any(
            f.rstrip("/").split("/")[0] in known_projects
            for f in non_matching_files
        ))

        where_clause = searcher._build_where_clause(non_matching_files)
        assert where_clause is None

    @given(
        known_projects=st.lists(valid_project_names(), min_size=1, max_size=5, unique=True),
    )
    @settings(max_examples=50)
    def test_none_when_no_files_provided(self, known_projects):
        """Where clause is None when files parameter is None or empty."""
        searcher = _make_searcher(known_projects)

        assert searcher._build_where_clause(None) is None
        assert searcher._build_where_clause([]) is None


# --- Property 8: Legacy Expert Safety ---

class TestProperty8LegacyExpertSafety:
    """
    Property 8: Legacy Expert Safety

    For any expert where has_vector_metadata is False, queries should never
    pass a where clause to ChromaDB, regardless of the files parameter provided.

    **Validates: Requirements 10.1, 10.3**
    """

    @given(
        known_projects=st.lists(valid_project_names(), min_size=1, max_size=5, unique=True),
        data=st.data(),
    )
    @settings(max_examples=50)
    def test_legacy_expert_never_produces_where_clause(self, known_projects, data):
        """has_vector_metadata=False always returns None from _build_where_clause."""
        searcher = _make_searcher(known_projects, has_vector_metadata=False)

        # Generate files that would normally produce a where clause
        files = data.draw(st.one_of(
            st.none(),
            st.just([]),
            st.lists(
                st.builds(
                    lambda p, f: f"{p}/{f}",
                    st.sampled_from(known_projects),
                    relative_file_paths(),
                ),
                min_size=1,
                max_size=5,
            ),
        ))

        where_clause = searcher._build_where_clause(files)
        assert where_clause is None, (
            f"Legacy expert (has_vector_metadata=False) should never produce a where clause, "
            f"but got: {where_clause}"
        )

    @given(
        known_projects=st.lists(valid_project_names(), min_size=1, max_size=3, unique=True),
    )
    @settings(max_examples=100)
    def test_legacy_expert_with_project_prefixed_files_still_returns_none(self, known_projects):
        """Even with perfect project-prefixed file paths, legacy returns None."""
        searcher = _make_searcher(known_projects, has_vector_metadata=False)

        # Explicitly provide files that match known projects
        files = [f"{p}/src/main.py" for p in known_projects]

        where_clause = searcher._build_where_clause(files)
        assert where_clause is None


# --- Property 9: Where Clause Expansion Threading ---

class TestProperty9WhereClauseExpansionThreading:
    """
    Property 9: Where Clause Expansion Threading

    For any query that produces a where clause, that same where clause should
    be passed to every iterative expansion search call. For any query with no
    where clause, no expansion call should introduce one.

    **Validates: Requirements 11.1, 11.2, 11.3**
    """

    @given(
        known_projects=st.lists(valid_project_names(), min_size=2, max_size=4, unique=True),
    )
    @settings(max_examples=100)
    def test_where_clause_passed_to_all_search_methods(self, known_projects):
        """The where clause built by search() is forwarded to all vector search calls."""
        metadata_db = _make_mock_metadata_db(known_projects)
        vector_db = _make_mock_vector_db()
        embedder = _make_mock_embedder()

        # Make metadata_db.get_changelists_by_ids return empty to short-circuit
        metadata_db.get_changelists_by_ids.return_value = []

        searcher = Searcher(
            expert_name="test-expert",
            embedder=embedder,
            metadata_db=metadata_db,
            vector_db=vector_db,
            has_vector_metadata=True,
            enable_query_expansion=False,
            enable_reranking=False,
            enable_metadata_search=True,
            enable_diff_search=True,
            enable_file_search=True,
        )

        # Query with project-scoped files
        files = [f"{known_projects[0]}/src/handler.py"]
        params = QueryParams(
            prompt="test query",
            max_changes=10,
            max_file_chunks=5,
            files=files,
        )

        expected_where = {"project": {"$in": [known_projects[0]]}}

        searcher.search(params)

        # Verify search_metadata was called with the where clause
        vector_db.search_metadata.assert_called()
        metadata_call_kwargs = vector_db.search_metadata.call_args
        assert metadata_call_kwargs.kwargs.get("where") == expected_where or \
            (len(metadata_call_kwargs.args) > 3 and metadata_call_kwargs.args[3] == expected_where), \
            f"search_metadata not called with expected where clause"

        # Verify search_diffs was called with the where clause
        vector_db.search_diffs.assert_called()
        diffs_call_kwargs = vector_db.search_diffs.call_args
        assert diffs_call_kwargs.kwargs.get("where") == expected_where or \
            (len(diffs_call_kwargs.args) > 3 and diffs_call_kwargs.args[3] == expected_where), \
            f"search_diffs not called with expected where clause"

        # Verify search_files was called with the where clause
        vector_db.search_files.assert_called()
        files_call_kwargs = vector_db.search_files.call_args
        assert files_call_kwargs.kwargs.get("where") == expected_where or \
            (len(files_call_kwargs.args) > 3 and files_call_kwargs.args[3] == expected_where), \
            f"search_files not called with expected where clause"

    @given(
        known_projects=st.lists(valid_project_names(), min_size=1, max_size=4, unique=True),
    )
    @settings(max_examples=100)
    def test_no_where_clause_when_no_files_filter(self, known_projects):
        """When no files filter is provided, no where clause is passed to searches."""
        metadata_db = _make_mock_metadata_db(known_projects)
        vector_db = _make_mock_vector_db()
        embedder = _make_mock_embedder()

        metadata_db.get_changelists_by_ids.return_value = []

        searcher = Searcher(
            expert_name="test-expert",
            embedder=embedder,
            metadata_db=metadata_db,
            vector_db=vector_db,
            has_vector_metadata=True,
            enable_query_expansion=False,
            enable_reranking=False,
            enable_metadata_search=True,
            enable_diff_search=True,
            enable_file_search=True,
        )

        params = QueryParams(
            prompt="test query",
            max_changes=10,
            max_file_chunks=5,
            files=None,
        )

        searcher.search(params)

        # Verify all search calls have where=None
        for call_obj in vector_db.search_metadata.call_args_list:
            assert call_obj.kwargs.get("where") is None, \
                "search_metadata should not have a where clause when files is None"

        for call_obj in vector_db.search_diffs.call_args_list:
            assert call_obj.kwargs.get("where") is None, \
                "search_diffs should not have a where clause when files is None"

        for call_obj in vector_db.search_files.call_args_list:
            assert call_obj.kwargs.get("where") is None, \
                "search_files should not have a where clause when files is None"

    @given(
        known_projects=st.lists(valid_project_names(), min_size=2, max_size=4, unique=True),
    )
    @settings(max_examples=100)
    def test_same_where_clause_in_all_expansion_calls(self, known_projects):
        """When expansion is enabled, the same where clause is threaded to expansion searches."""
        metadata_db = _make_mock_metadata_db(known_projects)
        vector_db = _make_mock_vector_db()
        embedder = _make_mock_embedder()

        metadata_db.get_changelists_by_ids.return_value = []

        searcher = Searcher(
            expert_name="test-expert",
            embedder=embedder,
            metadata_db=metadata_db,
            vector_db=vector_db,
            has_vector_metadata=True,
            enable_query_expansion=True,  # Enable expansion
            enable_reranking=False,
            enable_metadata_search=True,
            enable_diff_search=True,
            enable_file_search=True,
        )

        files = [f"{known_projects[0]}/src/handler.py"]
        params = QueryParams(
            prompt="test query",
            max_changes=10,
            max_file_chunks=5,
            files=files,
        )

        expected_where = {"project": {"$in": [known_projects[0]]}}

        searcher.search(params)

        # All calls to search methods should use the same where clause
        for call_obj in vector_db.search_metadata.call_args_list:
            where_val = call_obj.kwargs.get("where")
            assert where_val == expected_where, (
                f"Expected where={expected_where}, got where={where_val} in search_metadata call"
            )

        for call_obj in vector_db.search_diffs.call_args_list:
            where_val = call_obj.kwargs.get("where")
            assert where_val == expected_where, (
                f"Expected where={expected_where}, got where={where_val} in search_diffs call"
            )

        for call_obj in vector_db.search_files.call_args_list:
            where_val = call_obj.kwargs.get("where")
            assert where_val == expected_where, (
                f"Expected where={expected_where}, got where={where_val} in search_files call"
            )
