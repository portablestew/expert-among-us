"""
Property-based tests for path prefixing (Properties 1 and 2).

Tests the Indexer's static path prefixing utilities using Hypothesis.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from expert_among_us.core.indexer import Indexer


# --- Strategies ---

# Valid project names: starts with alphanumeric, then alphanumeric/hyphens/underscores
valid_name_start = st.sampled_from(
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
)
valid_name_rest = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
    min_size=0,
    max_size=30,
)


@st.composite
def valid_project_names(draw):
    """Generate valid project names matching [a-zA-Z0-9][a-zA-Z0-9_-]*."""
    start = draw(valid_name_start)
    rest = draw(valid_name_rest)
    return start + rest


# Relative file paths (no leading slash, forward-slash separated)
file_path_segment = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-.",
    min_size=1,
    max_size=20,
)

@st.composite
def relative_file_paths(draw):
    """Generate relative file paths like 'src/main.py' or 'README.md'."""
    segments = draw(st.lists(file_path_segment, min_size=1, max_size=5))
    return "/".join(segments)


# Lists of file paths
file_path_lists = st.lists(relative_file_paths(), min_size=1, max_size=10)


# Strategy for unified diff lines
@st.composite
def unified_diffs(draw):
    """Generate realistic unified diff strings with various line types."""
    lines = []
    num_hunks = draw(st.integers(min_value=1, max_value=3))

    for _ in range(num_hunks):
        file_path = draw(relative_file_paths())
        is_new_file = draw(st.booleans())
        is_deleted_file = draw(st.booleans())

        # Add the --- line
        if is_new_file:
            lines.append("--- /dev/null")
        else:
            lines.append(f"--- a/{file_path}")

        # Add the +++ line
        if is_deleted_file:
            lines.append("+++ /dev/null")
        else:
            lines.append(f"+++ b/{file_path}")

        # Add a hunk header
        lines.append("@@ -1,5 +1,5 @@")

        # Add some context/change lines
        num_lines = draw(st.integers(min_value=1, max_value=5))
        for _ in range(num_lines):
            line_type = draw(st.sampled_from([" ", "+", "-"]))
            content = draw(st.text(
                alphabet="abcdefghijklmnopqrstuvwxyz0123456789 =;(){}",
                min_size=0,
                max_size=40,
            ))
            lines.append(f"{line_type}{content}")

    return "\n".join(lines)


# --- Property 1: Path Prefix Round-Trip ---

class TestProperty1PathPrefixRoundTrip:
    """
    Property 1: Path Prefix Round-Trip

    For any valid file path and valid project name, prefixing the path with the
    project name and then stripping the prefix should yield the original path.
    Additionally, every prefixed path must start with `project_name + "/"`.

    **Validates: Requirements 3.1, 3.5**
    """

    @given(
        files=file_path_lists,
        project_name=valid_project_names(),
    )
    @settings(max_examples=50)
    def test_every_prefixed_path_starts_with_project_name(self, files, project_name):
        """Every prefixed path starts with 'project_name/'."""
        prefixed = Indexer.prefix_file_paths(files, project_name)

        for p in prefixed:
            assert p.startswith(f"{project_name}/"), (
                f"Prefixed path '{p}' does not start with '{project_name}/'"
            )

    @given(
        files=file_path_lists,
        project_name=valid_project_names(),
    )
    @settings(max_examples=50)
    def test_strip_prefix_returns_original(self, files, project_name):
        """Stripping the project prefix from each prefixed path yields the original."""
        prefixed = Indexer.prefix_file_paths(files, project_name)
        prefix = f"{project_name}/"

        stripped = [p[len(prefix):] for p in prefixed]
        assert stripped == files

    @given(
        files=file_path_lists,
        project_name=valid_project_names(),
    )
    @settings(max_examples=50)
    def test_prefix_preserves_list_length(self, files, project_name):
        """Prefixing preserves the number of files in the list."""
        prefixed = Indexer.prefix_file_paths(files, project_name)
        assert len(prefixed) == len(files)


# --- Property 2: Diff Rewrite Correctness ---

class TestProperty2DiffRewriteCorrectness:
    """
    Property 2: Diff Rewrite Correctness

    For any unified diff string and valid project name, rewriting the diff
    should: (a) produce the same number of lines as the input, (b) transform
    all `--- a/X` lines to `--- a/{project_name}/X`, (c) transform all
    `+++ b/X` lines to `+++ b/{project_name}/X`, and (d) leave `/dev/null`
    lines and non-diff-path lines unchanged.

    **Validates: Requirements 3.2, 3.3, 3.4**
    """

    @given(
        diff=unified_diffs(),
        project_name=valid_project_names(),
    )
    @settings(max_examples=50)
    def test_line_count_preserved(self, diff, project_name):
        """Rewriting a diff preserves the total number of lines."""
        result = Indexer.rewrite_diff_paths(diff, project_name)

        original_lines = diff.split("\n")
        result_lines = result.split("\n")
        assert len(result_lines) == len(original_lines)

    @given(
        diff=unified_diffs(),
        project_name=valid_project_names(),
    )
    @settings(max_examples=50)
    def test_minus_lines_rewritten(self, diff, project_name):
        """All '--- a/X' lines become '--- a/{project_name}/X'."""
        result = Indexer.rewrite_diff_paths(diff, project_name)

        original_lines = diff.split("\n")
        result_lines = result.split("\n")

        for orig, rewritten in zip(original_lines, result_lines):
            if orig.startswith("--- a/"):
                original_path = orig[6:]  # strip '--- a/'
                expected = f"--- a/{project_name}/{original_path}"
                assert rewritten == expected, (
                    f"Expected '{expected}', got '{rewritten}'"
                )

    @given(
        diff=unified_diffs(),
        project_name=valid_project_names(),
    )
    @settings(max_examples=50)
    def test_plus_lines_rewritten(self, diff, project_name):
        """All '+++ b/X' lines become '+++ b/{project_name}/X'."""
        result = Indexer.rewrite_diff_paths(diff, project_name)

        original_lines = diff.split("\n")
        result_lines = result.split("\n")

        for orig, rewritten in zip(original_lines, result_lines):
            if orig.startswith("+++ b/"):
                original_path = orig[6:]  # strip '+++ b/'
                expected = f"+++ b/{project_name}/{original_path}"
                assert rewritten == expected, (
                    f"Expected '{expected}', got '{rewritten}'"
                )

    @given(
        diff=unified_diffs(),
        project_name=valid_project_names(),
    )
    @settings(max_examples=50)
    def test_dev_null_lines_unchanged(self, diff, project_name):
        """'/dev/null' lines are left unchanged."""
        result = Indexer.rewrite_diff_paths(diff, project_name)

        original_lines = diff.split("\n")
        result_lines = result.split("\n")

        for orig, rewritten in zip(original_lines, result_lines):
            if orig.startswith("--- /dev/null") or orig.startswith("+++ /dev/null"):
                assert rewritten == orig, (
                    f"/dev/null line changed: '{orig}' → '{rewritten}'"
                )

    @given(
        diff=unified_diffs(),
        project_name=valid_project_names(),
    )
    @settings(max_examples=50)
    def test_other_lines_unchanged(self, diff, project_name):
        """Lines that are not diff path headers are left unchanged."""
        result = Indexer.rewrite_diff_paths(diff, project_name)

        original_lines = diff.split("\n")
        result_lines = result.split("\n")

        for orig, rewritten in zip(original_lines, result_lines):
            is_diff_path_line = (
                orig.startswith("--- a/")
                or orig.startswith("+++ b/")
                or orig.startswith("--- /dev/null")
                or orig.startswith("+++ /dev/null")
            )
            if not is_diff_path_line:
                assert rewritten == orig, (
                    f"Non-path line changed: '{orig}' → '{rewritten}'"
                )
