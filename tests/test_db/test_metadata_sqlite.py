"""
Comprehensive tests for SQLiteMetadataDB covering:
- Database initialization and schema creation
- Expert CRUD operations
- Changelist operations (insertion and retrieval)
- Query operations
- Prompt caching
- Connection management
- Edge cases
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import os

from expert_among_us.db.metadata.sqlite import SQLiteMetadataDB
from expert_among_us.models.changelist import Changelist


@pytest.fixture
def temp_db():
    """Fixture providing a temporary database with a unique expert name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a temporary home directory structure
        expert_name = "test_expert"
        db_path = Path(tmpdir) / "data" / expert_name / "metadata.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use context manager for proper resource cleanup
        with SQLiteMetadataDB(expert_name) as db:
            db.db_path = str(db_path)
            db.initialize()
            # Create a default expert and project for tests that insert changelists
            db.create_expert("test_expert")
            cursor = db.conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO projects (expert_name, name, workspace_path, subdirs, vcs_type)
                VALUES (?, ?, ?, ?, ?)
            """, ("test_expert", "test-project", "/path/to/repo", "", "git"))
            db.conn.commit()
            yield db


@pytest.fixture
def sample_changelist():
    """Fixture providing a sample Changelist object."""
    return Changelist(
        id="abc123def456",
        expert_name="test_expert",
        project_name="test-project",
        timestamp=datetime.now(),
        author="John Doe",
        message="Fixed bug in authentication module",
        diff="diff --git a/auth.py...",
        files=["src/auth.py", "tests/test_auth.py"]
    )


class TestDatabaseInitialization:
    """Tests for database initialization and schema creation."""

    def test_db_schema_created(self, temp_db):
        """Verify that database schema is properly created with all required tables."""
        cursor = temp_db.conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        )
        tables = cursor.fetchall()
        table_names = [t[0] for t in tables]
        
        # Verify required tables exist
        assert "experts" in table_names
        assert "projects" in table_names
        assert "changelists" in table_names
        assert "changelist_files" in table_names

    def test_db_reopens_existing_database(self):
        """Verify that opening an existing database doesn't reset it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            expert_name = "reopen_test"
            db_path = Path(tmpdir) / "data" / expert_name / "metadata.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create and add data
            db1 = SQLiteMetadataDB(expert_name)
            db1.db_path = str(db_path)
            db1.initialize()
            db1.create_expert("test_expert")
            db1.close()

            # Reopen and verify data persists
            db2 = SQLiteMetadataDB(expert_name)
            db2.db_path = str(db_path)
            db2.initialize()
            retrieved = db2.get_expert("test_expert")
            assert retrieved is not None
            assert retrieved["name"] == "test_expert"
            db2.close()


class TestExpertOperations:
    """Tests for Expert CRUD operations."""

    def test_create_expert(self, temp_db):
        """Verify that an expert can be created and stored."""
        temp_db.create_expert(
            name="new_expert",
            description="A test expert"
        )
        
        retrieved = temp_db.get_expert("new_expert")
        assert retrieved is not None
        assert retrieved["name"] == "new_expert"
        assert retrieved["description"] == "A test expert"

    def test_get_nonexistent_expert(self, temp_db):
        """Verify that getting a non-existent expert returns None."""
        result = temp_db.get_expert("nonexistent_expert")
        assert result is None

    def test_update_expert_index_time(self, temp_db):
        """Verify that expert index time can be updated."""
        new_time = datetime.now()
        temp_db.update_expert_index_time("test_expert", new_time)
        
        retrieved = temp_db.get_expert("test_expert")
        assert retrieved["last_indexed_at"] is not None

    def test_create_multiple_experts(self, temp_db):
        """Verify that multiple experts can be created and retrieved independently."""
        temp_db.create_expert("expert1", description="First expert")
        temp_db.create_expert("expert2", description="Second expert")
        
        retrieved1 = temp_db.get_expert("expert1")
        retrieved2 = temp_db.get_expert("expert2")
        
        assert retrieved1["name"] == "expert1"
        assert retrieved2["name"] == "expert2"
        assert retrieved1["description"] == "First expert"
        assert retrieved2["description"] == "Second expert"



class TestChangelistOperations:
    """Tests for Changelist insertion and retrieval operations."""

    def test_insert_single_changelist(self, temp_db, sample_changelist):
        """Verify that a single changelist can be inserted."""
        temp_db.insert_changelists([sample_changelist])
        
        retrieved = temp_db.get_changelist(sample_changelist.id)
        assert retrieved is not None
        assert retrieved.id == sample_changelist.id
        assert retrieved.author == sample_changelist.author

    def test_insert_batch_changelists(self, temp_db):
        """Verify that multiple changelists can be inserted in batch."""
        changelists = [
            Changelist(
                id=f"id_{i}",
                expert_name="test_expert",
                project_name="test-project",
                timestamp=datetime.now(),
                author=f"Author {i}",
                message=f"Change {i}",
                diff=f"diff {i}",
                files=[f"file_{i}.py"]
            )
            for i in range(5)
        ]
        
        temp_db.insert_changelists(changelists)
        
        for changelist in changelists:
            retrieved = temp_db.get_changelist(changelist.id)
            assert retrieved is not None
            assert retrieved.id == changelist.id

    def test_get_changelist_nonexistent(self, temp_db):
        """Verify that getting a non-existent changelist returns None."""
        result = temp_db.get_changelist("nonexistent_id")
        assert result is None

    def test_get_changelists_by_ids(self, temp_db):
        """Verify that multiple changelists can be retrieved by ID list."""
        changelists = [
            Changelist(
                id=f"id_{i}",
                expert_name="test_expert",
                project_name="test-project",
                timestamp=datetime.now(),
                author=f"Author {i}",
                message=f"Change {i}",
                diff=f"diff {i}",
                files=[f"file_{i}.py"]
            )
            for i in range(3)
        ]
        temp_db.insert_changelists(changelists)
        
        ids = [f"id_{i}" for i in range(3)]
        retrieved = temp_db.get_changelists_by_ids(ids)
        
        assert len(retrieved) == 3
        retrieved_ids = {cl.id for cl in retrieved}
        assert retrieved_ids == set(ids)

    def test_get_changelists_by_ids_partial_match(self, temp_db):
        """Verify that get_changelists_by_ids handles partial matches correctly."""
        changelists = [
            Changelist(
                id="exists_1",
                expert_name="test_expert",
                project_name="test-project",
                timestamp=datetime.now(),
                author="Author",
                message="Change",
                diff="diff",
                files=["file.py"]
            ),
            Changelist(
                id="exists_2",
                expert_name="test_expert",
                project_name="test-project",
                timestamp=datetime.now(),
                author="Author",
                message="Change",
                diff="diff",
                files=["file.py"]
            )
        ]
        temp_db.insert_changelists(changelists)
        
        ids = ["exists_1", "exists_2", "does_not_exist"]
        retrieved = temp_db.get_changelists_by_ids(ids)
        
        # Should only return existing ones
        assert len(retrieved) == 2



class TestQueryOperations:
    """Tests for query operations like filtering by author and files."""

    def test_query_changelists_by_author(self, temp_db):
        """Verify that changelists can be queried by author."""
        changelists = [
            Changelist(
                id="id_1",
                expert_name="test_expert",
                project_name="test-project",
                timestamp=datetime.now(),
                author="John Doe",
                message="Change 1",
                diff="diff 1",
                files=["file1.py"]
            ),
            Changelist(
                id="id_2",
                expert_name="test_expert",
                project_name="test-project",
                timestamp=datetime.now(),
                author="Jane Smith",
                message="Change 2",
                diff="diff 2",
                files=["file2.py"]
            ),
            Changelist(
                id="id_3",
                expert_name="test_expert",
                project_name="test-project",
                timestamp=datetime.now(),
                author="John Doe",
                message="Change 3",
                diff="diff 3",
                files=["file3.py"]
            )
        ]
        temp_db.insert_changelists(changelists)
        
        # Query for John Doe's changes
        result_ids = temp_db.query_changelists_by_author("John Doe")
        assert len(result_ids) == 2
        assert "id_1" in result_ids
        assert "id_3" in result_ids

    def test_query_changelists_by_files_single_file(self, temp_db):
        """Verify that changelists can be queried by files containing a specific file."""
        changelists = [
            Changelist(
                id="id_1",
                expert_name="test_expert",
                project_name="test-project",
                timestamp=datetime.now(),
                author="Author",
                message="Change 1",
                diff="diff 1",
                files=["src/main.py", "src/utils.py"]
            ),
            Changelist(
                id="id_2",
                expert_name="test_expert",
                project_name="test-project",
                timestamp=datetime.now(),
                author="Author",
                message="Change 2",
                diff="diff 2",
                files=["tests/test_main.py"]
            ),
            Changelist(
                id="id_3",
                expert_name="test_expert",
                project_name="test-project",
                timestamp=datetime.now(),
                author="Author",
                message="Change 3",
                diff="diff 3",
                files=["src/main.py", "docs/README.md"]
            )
        ]
        temp_db.insert_changelists(changelists)
        
        # Query for changes to src/main.py
        result_ids = temp_db.query_changelists_by_files(["src/main.py"])
        assert len(result_ids) == 2
        assert "id_1" in result_ids
        assert "id_3" in result_ids

    def test_query_changelists_by_files_multiple_files_or_logic(self, temp_db):
        """Verify that querying by multiple files uses OR logic (matches any file)."""
        changelists = [
            Changelist(
                id="id_1",
                expert_name="test_expert",
                project_name="test-project",
                timestamp=datetime.now(),
                author="Author",
                message="Change 1",
                diff="diff 1",
                files=["src/main.py"]
            ),
            Changelist(
                id="id_2",
                expert_name="test_expert",
                project_name="test-project",
                timestamp=datetime.now(),
                author="Author",
                message="Change 2",
                diff="diff 2",
                files=["src/utils.py"]
            ),
            Changelist(
                id="id_3",
                expert_name="test_expert",
                project_name="test-project",
                timestamp=datetime.now(),
                author="Author",
                message="Change 3",
                diff="diff 3",
                files=["tests/test.py"]
            )
        ]
        temp_db.insert_changelists(changelists)
        
        # Query for changes to src/main.py OR src/utils.py
        result_ids = temp_db.query_changelists_by_files(["src/main.py", "src/utils.py"])
        assert len(result_ids) == 2
        assert "id_1" in result_ids
        assert "id_2" in result_ids

    def test_query_changelists_by_files_no_matches(self, temp_db):
        """Verify that querying returns empty list when no files match."""
        changelist = Changelist(
            id="id_1",
            expert_name="test_expert",
            project_name="test-project",
            timestamp=datetime.now(),
            author="Author",
            message="Change 1",
            diff="diff 1",
            files=["src/main.py"]
        )
        temp_db.insert_changelists([changelist])
        
        result_ids = temp_db.query_changelists_by_files(["nonexistent.py"])
        assert result_ids == []

    def test_query_changelists_by_files_prefix_matching(self, temp_db):
        """Verify that querying uses startsWith/prefix matching (LIKE) semantics."""
        # Create additional projects needed for this test
        cursor = temp_db.conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO projects (expert_name, name, workspace_path, subdirs, vcs_type)
            VALUES (?, ?, ?, ?, ?)
        """, ("test_expert", "payment-service", "/path/to/payment", "", "git"))
        cursor.execute("""
            INSERT OR IGNORE INTO projects (expert_name, name, workspace_path, subdirs, vcs_type)
            VALUES (?, ?, ?, ?, ?)
        """, ("test_expert", "user-service", "/path/to/users", "", "git"))
        temp_db.conn.commit()

        changelists = [
            Changelist(
                id="id_1",
                expert_name="test_expert",
                project_name="payment-service",
                timestamp=datetime.now(),
                author="Author",
                message="Change 1",
                diff="diff 1",
                files=["payment-service/src/handler.py", "payment-service/src/utils.py"]
            ),
            Changelist(
                id="id_2",
                expert_name="test_expert",
                project_name="user-service",
                timestamp=datetime.now(),
                author="Author",
                message="Change 2",
                diff="diff 2",
                files=["user-service/src/auth.py"]
            ),
            Changelist(
                id="id_3",
                expert_name="test_expert",
                project_name="payment-service",
                timestamp=datetime.now(),
                author="Author",
                message="Change 3",
                diff="diff 3",
                files=["payment-service/tests/test_handler.py"]
            ),
        ]
        temp_db.insert_changelists(changelists)

        # Prefix query: match all files within payment-service project
        result_ids = temp_db.query_changelists_by_files(["payment-service/"])
        assert len(result_ids) == 2
        assert "id_1" in result_ids
        assert "id_3" in result_ids

        # Prefix query: match specific subdirectory
        result_ids = temp_db.query_changelists_by_files(["payment-service/src/"])
        assert len(result_ids) == 1
        assert "id_1" in result_ids

        # Exact-style match still works (path acts as prefix of itself)
        result_ids = temp_db.query_changelists_by_files(["user-service/src/auth.py"])
        assert len(result_ids) == 1
        assert "id_2" in result_ids

        # Multiple prefixes use OR logic
        result_ids = temp_db.query_changelists_by_files(["payment-service/tests/", "user-service/"])
        assert len(result_ids) == 2
        assert "id_2" in result_ids
        assert "id_3" in result_ids

    def test_query_changelists_by_author_no_results(self, temp_db):
        """Verify that querying by author returns empty list when no matches."""
        result_ids = temp_db.query_changelists_by_author("Nonexistent Author")
        assert result_ids == []



class TestPromptCaching:
    """Tests for prompt caching operations."""

    def test_cache_and_get_prompt(self, temp_db, sample_changelist):
        """Verify that a prompt can be cached and retrieved."""
        # First insert a changelist
        temp_db.insert_changelists([sample_changelist])
        
        prompt = "This is a generated prompt for the changelist"
        temp_db.cache_generated_prompt(sample_changelist.id, prompt)
        
        retrieved = temp_db.get_generated_prompt(sample_changelist.id)
        assert retrieved == prompt

    def test_get_nonexistent_cached_prompt(self, temp_db):
        """Verify that getting a non-cached prompt returns None."""
        result = temp_db.get_generated_prompt("nonexistent_id")
        assert result is None

    def test_cache_overwrites_existing(self, temp_db, sample_changelist):
        """Verify that caching with the same ID overwrites the previous prompt."""
        temp_db.insert_changelists([sample_changelist])
        
        prompt1 = "Original prompt"
        prompt2 = "Updated prompt"
        
        temp_db.cache_generated_prompt(sample_changelist.id, prompt1)
        temp_db.cache_generated_prompt(sample_changelist.id, prompt2)
        
        retrieved = temp_db.get_generated_prompt(sample_changelist.id)
        assert retrieved == prompt2


class TestConnectionManagement:
    """Tests for database connection management."""

    def test_close_connection(self):
        """Verify that database connection can be closed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            expert_name = "close_test"
            db_path = Path(tmpdir) / "data" / expert_name / "metadata.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            db = SQLiteMetadataDB(expert_name)
            db.db_path = str(db_path)
            db.initialize()
            db.close()
            
            # Attempting to use the connection should raise an error
            with pytest.raises(Exception):
                cursor = db.conn.cursor()
                cursor.execute("SELECT 1")


class TestProjectOperations:
    """Tests for Project CRUD operations."""

    def test_create_project(self, temp_db):
        """Verify that a project can be created and stored."""
        temp_db.create_project(
            expert_name="test_expert",
            project_name="my-project",
            workspace_path="/repos/my-project",
            subdirs=["src", "tests"],
            vcs_type="git"
        )
        
        project = temp_db.get_project("test_expert", "my-project")
        assert project is not None
        assert project["name"] == "my-project"
        assert project["expert_name"] == "test_expert"
        assert project["workspace_path"] == "/repos/my-project"
        assert project["subdirs"] == ["src", "tests"]
        assert project["vcs_type"] == "git"
        assert project["has_vector_metadata"] is True
        assert project["last_processed_commit_hash"] is None
        assert project["first_processed_commit_hash"] is None

    def test_create_project_empty_subdirs(self, temp_db):
        """Verify that a project with no subdirs is handled correctly."""
        temp_db.create_project(
            expert_name="test_expert",
            project_name="no-subdirs",
            workspace_path="/repos/no-subdirs",
            subdirs=[],
            vcs_type="p4"
        )
        
        project = temp_db.get_project("test_expert", "no-subdirs")
        assert project is not None
        assert project["subdirs"] == []
        assert project["vcs_type"] == "p4"

    def test_create_project_idempotent(self, temp_db):
        """Verify that creating the same project twice is a no-op (INSERT OR IGNORE)."""
        temp_db.create_project(
            expert_name="test_expert",
            project_name="idempotent-proj",
            workspace_path="/repos/first",
            subdirs=[],
            vcs_type="git"
        )
        # Second creation with different path should be ignored
        temp_db.create_project(
            expert_name="test_expert",
            project_name="idempotent-proj",
            workspace_path="/repos/second",
            subdirs=["new-dir"],
            vcs_type="p4"
        )
        
        project = temp_db.get_project("test_expert", "idempotent-proj")
        assert project is not None
        # Original values preserved due to INSERT OR IGNORE
        assert project["workspace_path"] == "/repos/first"
        assert project["vcs_type"] == "git"

    def test_get_project_nonexistent(self, temp_db):
        """Verify that getting a non-existent project returns None."""
        result = temp_db.get_project("test_expert", "nonexistent")
        assert result is None

    def test_get_project_wrong_expert(self, temp_db):
        """Verify that projects are scoped to their expert."""
        temp_db.create_project(
            expert_name="test_expert",
            project_name="scoped-proj",
            workspace_path="/repos/scoped",
            subdirs=[],
            vcs_type="git"
        )
        # Different expert should not find this project
        result = temp_db.get_project("other_expert", "scoped-proj")
        assert result is None

    def test_list_projects_empty(self, temp_db):
        """Verify that listing projects for an expert with no projects (after removing fixture project) works."""
        temp_db.create_expert("empty_expert")
        projects = temp_db.list_projects("empty_expert")
        assert projects == []

    def test_list_projects_multiple(self, temp_db):
        """Verify that list_projects returns all projects for an expert."""
        temp_db.create_project("test_expert", "alpha", "/repos/alpha", [], "git")
        temp_db.create_project("test_expert", "beta", "/repos/beta", ["src"], "p4")
        temp_db.create_project("test_expert", "gamma", "/repos/gamma", [], "git")
        
        projects = temp_db.list_projects("test_expert")
        # Should include the fixture project + 3 new ones
        project_names = [p["name"] for p in projects]
        assert "alpha" in project_names
        assert "beta" in project_names
        assert "gamma" in project_names

    def test_list_projects_ordered_by_name(self, temp_db):
        """Verify that list_projects returns projects in alphabetical order."""
        temp_db.create_project("test_expert", "zebra", "/repos/z", [], "git")
        temp_db.create_project("test_expert", "apple", "/repos/a", [], "git")
        
        projects = temp_db.list_projects("test_expert")
        names = [p["name"] for p in projects]
        assert names == sorted(names)

    def test_update_project_last_processed(self, temp_db):
        """Verify that update_project_last_processed updates commit hash and timestamps."""
        temp_db.create_project("test_expert", "indexed-proj", "/repos/indexed", [], "git")
        
        temp_db.update_project_last_processed("test_expert", "indexed-proj", "abc123")
        
        project = temp_db.get_project("test_expert", "indexed-proj")
        assert project["last_processed_commit_hash"] == "abc123"
        assert project["first_processed_commit_hash"] == "abc123"
        assert project["last_indexed_at"] is not None

    def test_update_project_last_processed_preserves_first_hash(self, temp_db):
        """Verify that first_processed_commit_hash is only set once."""
        temp_db.create_project("test_expert", "multi-index", "/repos/multi", [], "git")
        
        temp_db.update_project_last_processed("test_expert", "multi-index", "first_hash")
        temp_db.update_project_last_processed("test_expert", "multi-index", "second_hash")
        
        project = temp_db.get_project("test_expert", "multi-index")
        assert project["last_processed_commit_hash"] == "second_hash"
        assert project["first_processed_commit_hash"] == "first_hash"

    def test_get_project_commit_count_empty(self, temp_db):
        """Verify that commit count is 0 for a project with no changelists."""
        temp_db.create_project("test_expert", "empty-proj", "/repos/empty", [], "git")
        
        count = temp_db.get_project_commit_count("test_expert", "empty-proj")
        assert count == 0

    def test_get_project_commit_count_with_changelists(self, temp_db):
        """Verify that commit count returns correct number for a project."""
        # test-project already exists in fixture
        changelists = [
            Changelist(
                id=f"proj_count_{i}",
                expert_name="test_expert",
                project_name="test-project",
                timestamp=datetime.now(),
                author="Author",
                message=f"Change {i}",
                diff=f"diff {i}",
                files=[f"file_{i}.py"]
            )
            for i in range(3)
        ]
        temp_db.insert_changelists(changelists)
        
        count = temp_db.get_project_commit_count("test_expert", "test-project")
        assert count == 3

    def test_get_project_commit_count_scoped_to_project(self, temp_db):
        """Verify that commit count is scoped to the specific project."""
        temp_db.create_project("test_expert", "proj-a", "/repos/a", [], "git")
        temp_db.create_project("test_expert", "proj-b", "/repos/b", [], "git")
        
        changelists_a = [
            Changelist(
                id=f"a_{i}",
                expert_name="test_expert",
                project_name="proj-a",
                timestamp=datetime.now(),
                author="Author",
                message=f"A change {i}",
                diff=f"diff {i}",
                files=[f"file_{i}.py"]
            )
            for i in range(2)
        ]
        changelists_b = [
            Changelist(
                id=f"b_{i}",
                expert_name="test_expert",
                project_name="proj-b",
                timestamp=datetime.now(),
                author="Author",
                message=f"B change {i}",
                diff=f"diff {i}",
                files=[f"file_{i}.py"]
            )
            for i in range(5)
        ]
        temp_db.insert_changelists(changelists_a)
        temp_db.insert_changelists(changelists_b)
        
        assert temp_db.get_project_commit_count("test_expert", "proj-a") == 2
        assert temp_db.get_project_commit_count("test_expert", "proj-b") == 5

    def test_delete_project(self, temp_db):
        """Verify that deleting a project removes it from the database."""
        temp_db.create_project("test_expert", "to-delete", "/repos/del", [], "git")
        
        temp_db.delete_project("test_expert", "to-delete")
        
        project = temp_db.get_project("test_expert", "to-delete")
        assert project is None

    def test_delete_project_cascades_changelists(self, temp_db):
        """Verify that deleting a project removes its changelists."""
        temp_db.create_project("test_expert", "cascade-proj", "/repos/cascade", [], "git")
        
        changelists = [
            Changelist(
                id=f"cascade_{i}",
                expert_name="test_expert",
                project_name="cascade-proj",
                timestamp=datetime.now(),
                author="Author",
                message=f"Change {i}",
                diff=f"diff {i}",
                files=[f"file_{i}.py"]
            )
            for i in range(3)
        ]
        temp_db.insert_changelists(changelists)
        
        # Verify changelists exist
        assert temp_db.get_project_commit_count("test_expert", "cascade-proj") == 3
        
        temp_db.delete_project("test_expert", "cascade-proj")
        
        # Changelists should be gone
        assert temp_db.get_project_commit_count("test_expert", "cascade-proj") == 0
        for i in range(3):
            assert temp_db.get_changelist(f"cascade_{i}") is None

    def test_delete_project_isolation(self, temp_db):
        """Verify that deleting a project does not affect other projects."""
        temp_db.create_project("test_expert", "keep-proj", "/repos/keep", [], "git")
        temp_db.create_project("test_expert", "delete-proj", "/repos/del", [], "git")
        
        keep_cl = Changelist(
            id="keep_cl",
            expert_name="test_expert",
            project_name="keep-proj",
            timestamp=datetime.now(),
            author="Author",
            message="Keep this",
            diff="diff keep",
            files=["keep.py"]
        )
        delete_cl = Changelist(
            id="delete_cl",
            expert_name="test_expert",
            project_name="delete-proj",
            timestamp=datetime.now(),
            author="Author",
            message="Delete this",
            diff="diff delete",
            files=["delete.py"]
        )
        temp_db.insert_changelists([keep_cl, delete_cl])
        
        temp_db.delete_project("test_expert", "delete-proj")
        
        # keep-proj and its changelists should still exist
        assert temp_db.get_project("test_expert", "keep-proj") is not None
        assert temp_db.get_changelist("keep_cl") is not None
        assert temp_db.get_project_commit_count("test_expert", "keep-proj") == 1
        
        # delete-proj should be gone
        assert temp_db.get_project("test_expert", "delete-proj") is None
        assert temp_db.get_changelist("delete_cl") is None


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_changelist_with_empty_files_list_allowed(self, temp_db):
        """Empty files list is allowed for metadata-only or diff-only changelists."""
        changelist = Changelist(
            id="id_empty_files",
            expert_name="test_expert",
            project_name="test-project",
            timestamp=datetime.now(),
            author="Author",
            message="Change with no files",
            diff="diff",
            files=[]
        )
        temp_db.insert_changelists([changelist])
        retrieved = temp_db.get_changelist("id_empty_files")
        assert retrieved is not None
        assert retrieved.files == []

    def test_changelist_with_special_characters(self, temp_db):
        """Verify that changelists with special characters in text fields are handled."""
        changelist = Changelist(
            id="id_special",
            expert_name="test_expert",
            project_name="test-project",
            timestamp=datetime.now(),
            author="O'Brien & Co.",
            message='Fixed bug with "quotes" and \'apostrophes\'',
            diff="diff content",
            files=["file-with-dashes.py", "file_with_underscores.py"]
        )
        temp_db.insert_changelists([changelist])
        
        retrieved = temp_db.get_changelist("id_special")
        assert retrieved is not None
        assert retrieved.author == "O'Brien & Co."
        assert '"quotes"' in retrieved.message

    def test_expert_with_description(self, temp_db):
        """Verify that experts with descriptions are handled correctly."""
        temp_db.create_expert(
            "special_expert",
            description="A description with 'quotes' and \"double quotes\""
        )
        
        retrieved = temp_db.get_expert("special_expert")
        assert retrieved is not None
        assert retrieved["description"] == "A description with 'quotes' and \"double quotes\""

    def test_insert_duplicate_changelist_overwrites(self, temp_db):
        """Verify that inserting duplicate changelist ID overwrites."""
        changelist1 = Changelist(
            id="duplicate_id",
            expert_name="test_expert",
            project_name="test-project",
            timestamp=datetime.now(),
            author="Author 1",
            message="Message 1",
            diff="diff 1",
            files=["file1.py"]
        )
        changelist2 = Changelist(
            id="duplicate_id",
            expert_name="test_expert",
            project_name="test-project",
            timestamp=datetime.now(),
            author="Author 2",
            message="Message 2",
            diff="diff 2",
            files=["file2.py"]
        )
        
        temp_db.insert_changelists([changelist1])
        temp_db.insert_changelists([changelist2])
        
        retrieved = temp_db.get_changelist("duplicate_id")
        assert retrieved.author == "Author 2"
        assert retrieved.message == "Message 2"

    def test_query_operations_consistency(self, temp_db):
        """Verify that query operations maintain consistency."""
        changelist = Changelist(
            id="id_1",
            expert_name="test_expert",
            project_name="test-project",
            timestamp=datetime.now(),
            author="Test Author",
            message="Test change",
            diff="diff",
            files=["test.py"]
        )
        temp_db.insert_changelists([changelist])
        
        # Query by author and by files should return same changelist
        author_ids = temp_db.query_changelists_by_author("Test Author")
        file_ids = temp_db.query_changelists_by_files(["test.py"])
        
        assert "id_1" in author_ids
        assert "id_1" in file_ids