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
            yield db


@pytest.fixture
def sample_changelist():
    """Fixture providing a sample Changelist object."""
    return Changelist(
        id="abc123def456",
        expert_name="test_expert",
        timestamp=datetime.now(),
        author="John Doe",
        message="Fixed bug in authentication module",
        diff="diff --git a/auth.py...",
        files=["src/auth.py", "tests/test_auth.py"]
    )


class TestDatabaseInitialization:
    """Tests for database initialization and schema creation."""

    def test_db_initialization_creates_file(self):
        """Verify that database initialization creates a database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            expert_name = "init_test"
            db_path = Path(tmpdir) / "data" / expert_name / "metadata.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            db = SQLiteMetadataDB(expert_name)
            db.db_path = str(db_path)
            
            assert not db_path.exists()
            db.initialize()
            assert db_path.exists()
            db.close()

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
            db1.create_expert("test_expert", "/path/to/repo", [], "git")
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
            workspace_path="/home/user/repo",
            subdirs=["src", "tests"],
            vcs_type="git"
        )
        
        retrieved = temp_db.get_expert("new_expert")
        assert retrieved is not None
        assert retrieved["name"] == "new_expert"
        assert retrieved["workspace_path"] == "/home/user/repo"
        assert retrieved["subdirs"] == ["src", "tests"]
        assert retrieved["vcs_type"] == "git"

    def test_get_nonexistent_expert(self, temp_db):
        """Verify that getting a non-existent expert returns None."""
        result = temp_db.get_expert("nonexistent_expert")
        assert result is None

    def test_update_expert_index_time(self, temp_db):
        """Verify that expert index time can be updated."""
        temp_db.create_expert("test_expert", "/path", [], "git")
        
        new_time = datetime.now()
        temp_db.update_expert_index_time("test_expert", new_time)
        
        retrieved = temp_db.get_expert("test_expert")
        assert retrieved["last_indexed_at"] is not None

    def test_create_multiple_experts(self, temp_db):
        """Verify that multiple experts can be created and retrieved independently."""
        temp_db.create_expert("expert1", "/path1", [], "git")
        temp_db.create_expert("expert2", "/path2", ["src"], "git")
        
        retrieved1 = temp_db.get_expert("expert1")
        retrieved2 = temp_db.get_expert("expert2")
        
        assert retrieved1["name"] == "expert1"
        assert retrieved2["name"] == "expert2"
        assert retrieved2["subdirs"] == ["src"]

    def test_create_expert_with_empty_subdirs(self, temp_db):
        """Verify that experts with empty subdirs list are handled."""
        temp_db.create_expert("expert_no_subdirs", "/path", [], "git")
        
        retrieved = temp_db.get_expert("expert_no_subdirs")
        assert retrieved["subdirs"] == []


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
                timestamp=datetime.now(),
                author="Author",
                message="Change",
                diff="diff",
                files=["file.py"]
            ),
            Changelist(
                id="exists_2",
                expert_name="test_expert",
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

    def test_get_changelists_by_ids_empty_list(self, temp_db):
        """Verify that get_changelists_by_ids handles empty ID list."""
        retrieved = temp_db.get_changelists_by_ids([])
        assert retrieved == []


class TestQueryOperations:
    """Tests for query operations like filtering by author and files."""

    def test_query_changelists_by_author(self, temp_db):
        """Verify that changelists can be queried by author."""
        changelists = [
            Changelist(
                id="id_1",
                expert_name="test_expert",
                timestamp=datetime.now(),
                author="John Doe",
                message="Change 1",
                diff="diff 1",
                files=["file1.py"]
            ),
            Changelist(
                id="id_2",
                expert_name="test_expert",
                timestamp=datetime.now(),
                author="Jane Smith",
                message="Change 2",
                diff="diff 2",
                files=["file2.py"]
            ),
            Changelist(
                id="id_3",
                expert_name="test_expert",
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
                timestamp=datetime.now(),
                author="Author",
                message="Change 1",
                diff="diff 1",
                files=["src/main.py", "src/utils.py"]
            ),
            Changelist(
                id="id_2",
                expert_name="test_expert",
                timestamp=datetime.now(),
                author="Author",
                message="Change 2",
                diff="diff 2",
                files=["tests/test_main.py"]
            ),
            Changelist(
                id="id_3",
                expert_name="test_expert",
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
                timestamp=datetime.now(),
                author="Author",
                message="Change 1",
                diff="diff 1",
                files=["src/main.py"]
            ),
            Changelist(
                id="id_2",
                expert_name="test_expert",
                timestamp=datetime.now(),
                author="Author",
                message="Change 2",
                diff="diff 2",
                files=["src/utils.py"]
            ),
            Changelist(
                id="id_3",
                expert_name="test_expert",
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
            timestamp=datetime.now(),
            author="Author",
            message="Change 1",
            diff="diff 1",
            files=["src/main.py"]
        )
        temp_db.insert_changelists([changelist])
        
        result_ids = temp_db.query_changelists_by_files(["nonexistent.py"])
        assert result_ids == []

    def test_query_changelists_by_author_no_results(self, temp_db):
        """Verify that querying by author returns empty list when no matches."""
        result_ids = temp_db.query_changelists_by_author("Nonexistent Author")
        assert result_ids == []

    def test_query_changelists_by_files_empty_list(self, temp_db):
        """Verify that querying by empty files list returns empty result."""
        result_ids = temp_db.query_changelists_by_files([])
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


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_changelist_with_empty_files_list(self, temp_db):
        """Verify that inserting a changelist requires at least one file."""
        # The Changelist model validates that files cannot be empty
        with pytest.raises(Exception):
            changelist = Changelist(
                id="id_empty_files",
                expert_name="test_expert",
                timestamp=datetime.now(),
                author="Author",
                message="Change with no files",
                diff="diff",
                files=[]
            )

    def test_changelist_with_special_characters(self, temp_db):
        """Verify that changelists with special characters in text fields are handled."""
        changelist = Changelist(
            id="id_special",
            expert_name="test_expert",
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

    def test_expert_with_special_characters_in_path(self, temp_db):
        """Verify that experts with special characters in paths are handled."""
        temp_db.create_expert(
            "special_expert",
            "/path/with spaces/and-dashes",
            ["sub dir/with/slashes"],
            "git"
        )
        
        retrieved = temp_db.get_expert("special_expert")
        assert retrieved is not None
        assert retrieved["workspace_path"] == "/path/with spaces/and-dashes"

    def test_insert_duplicate_changelist_overwrites(self, temp_db):
        """Verify that inserting duplicate changelist ID overwrites."""
        changelist1 = Changelist(
            id="duplicate_id",
            expert_name="test_expert",
            timestamp=datetime.now(),
            author="Author 1",
            message="Message 1",
            diff="diff 1",
            files=["file1.py"]
        )
        changelist2 = Changelist(
            id="duplicate_id",
            expert_name="test_expert",
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