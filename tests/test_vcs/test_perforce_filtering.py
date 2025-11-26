"""Tests for Perforce workspace-based file filtering.

These tests verify that file filtering works correctly when subdirs is None,
ensuring only files within the workspace_path are included in diffs.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from expert_among_us.vcs.perforce import Perforce
from expert_among_us.config.settings import Settings


@pytest.fixture
def settings():
    """Fixture providing a Settings instance for tests."""
    return Settings()


@pytest.fixture
def perforce_provider(settings):
    """Fixture providing a Perforce provider instance."""
    return Perforce(settings)


def create_mock_popen_process(output: str, returncode: int = 0):
    """Helper to create a mock Popen process with streaming output."""
    mock_process = Mock()
    
    # Simulate streaming read: return chunks, then empty string for EOF
    chunks = [output[i:i+8192] for i in range(0, len(output), 8192)]
    mock_process.stdout.read = Mock(side_effect=chunks + [""])
    
    mock_process.returncode = returncode
    mock_process.poll = Mock(return_value=returncode)
    mock_process.wait = Mock(return_value=None)
    mock_process.kill = Mock()
    mock_process.stderr = Mock()
    mock_process.stderr.read = Mock(return_value="")
    
    return mock_process


class TestWorkspaceFiltering:
    """Tests for automatic file filtering based on workspace_path."""
    
    def test_depot_prefixes_created_when_subdirs_none(self, perforce_provider, tmp_path):
        """Verify depot_prefixes is created from workspace_path when subdirs is None."""
        with patch('socket.gethostname', return_value='test-host'):
            with patch('subprocess.run') as mock_run, patch('subprocess.Popen') as mock_popen:
                # Mock workspace discovery
                mock_run.side_effect = [
                    Mock(returncode=0, stdout=f"test-host {tmp_path} test-client"),
                    Mock(returncode=0, stdout="View:\n\t//depot/localization/... //test-client/..."),
                ]
                
                # Mock p4 describe output with files both inside and outside localization dir
                describe_output = """Change 12345 by user@client on 2024/01/15 14:00:00

\tMixed commit

Affected files ...

... //depot/localization/en-US/strings.xml#1 edit
... //depot/ui/styles.css#1 edit

Differences ...

==== //depot/localization/en-US/strings.xml#1 (text) ====

+<string>Hello</string>

==== //depot/ui/styles.css#1 (text) ====

+.button { color: blue; }"""
                
                mock_popen.return_value = create_mock_popen_process(describe_output)
                
                # Call with subdirs=None
                changelists = perforce_provider._fetch_changelists_by_numbers(
                    workspace_path=str(tmp_path),
                    cl_numbers=["12345"],
                    subdirs=None  # Key: subdirs is None
                )
                
                # Verify changelist was created
                assert len(changelists) == 1
                cl = changelists[0]
                
                # Key assertion: Only localization file should be in the changelist
                # The UI file should have been filtered out by depot_prefixes
                assert len(cl.files) == 1
                assert "strings.xml" in cl.files[0]
                assert "styles.css" not in str(cl.files)
                
                # Verify diff only contains localization file changes
                assert "Hello" in cl.diff
                assert "button" not in cl.diff
    
    def test_depot_prefixes_respects_explicit_subdirs(self, perforce_provider, tmp_path):
        """Verify depot_prefixes uses explicit subdirs when provided."""
        with patch('socket.gethostname', return_value='test-host'):
            with patch('subprocess.run') as mock_run, patch('subprocess.Popen') as mock_popen:
                # Mock workspace discovery
                mock_run.side_effect = [
                    Mock(returncode=0, stdout=f"test-host {tmp_path} test-client"),
                    Mock(returncode=0, stdout="View:\n\t//depot/... //test-client/..."),
                ]
                
                # Mock p4 describe with files in different subdirs
                describe_output = """Change 12345 by user@client on 2024/01/15 14:00:00

\tMulti-dir commit

Affected files ...

... //depot/game/player.cpp#1 edit
... //depot/localization/strings.xml#1 edit
... //depot/tools/build.sh#1 edit

Differences ...

==== //depot/game/player.cpp#1 (text) ====

+void jump() {}

==== //depot/localization/strings.xml#1 (text) ====

+<string>Jump</string>

==== //depot/tools/build.sh#1 (text) ====

+make all"""
                
                mock_popen.return_value = create_mock_popen_process(describe_output)
                
                # Call with explicit subdirs
                changelists = perforce_provider._fetch_changelists_by_numbers(
                    workspace_path=str(tmp_path),
                    cl_numbers=["12345"],
                    subdirs=["game", "localization"]  # Only these subdirs
                )
                
                # Verify both game and localization files are included, but not tools
                assert len(changelists) == 1
                cl = changelists[0]
                assert len(cl.files) == 2
                assert any("player.cpp" in f for f in cl.files)
                assert any("strings.xml" in f for f in cl.files)
                assert not any("build.sh" in f for f in cl.files)
                
                # Verify diff includes game and localization but not tools
                assert "jump()" in cl.diff
                assert "Jump" in cl.diff
                assert "make all" not in cl.diff
    
    def test_filtering_prevents_cross_directory_pollution(self, perforce_provider, tmp_path):
        """Verify the fix prevents cross-directory file pollution in diffs."""
        with patch('socket.gethostname', return_value='test-host'):
            with patch('subprocess.run') as mock_run, patch('subprocess.Popen') as mock_popen:
                # Mock workspace at localization directory
                loc_path = tmp_path / "Assets" / "Localization"
                loc_path.mkdir(parents=True)
                
                mock_run.side_effect = [
                    Mock(returncode=0, stdout=f"test-host {loc_path} test-client"),
                    Mock(returncode=0, stdout="View:\n\t//depot/Assets/Localization/... //test-client/..."),
                ]
                
                # Mock commit that touches many files (UI, scripts, images, localization)
                describe_output = """Change 1076268 by user@client on 2024/01/15 14:00:00

\tRaid screen skinning

Affected files ...

... //depot/Assets/Localization/en-US/GroupsText.loc.xml#97 edit
... //depot/Assets/LyShineUI/_Common/UIStyle.lua#42 edit
... //depot/Assets/LyShineUI/Images/Icon_GroupLeader.png#1 add
... //depot/Assets/LyShineUI/Images/Icons/Raid/icon_army.png#1 add

Differences ...

==== //depot/Assets/Localization/en-US/GroupsText.loc.xml#97 (text) ====

@@ -97,3 +97,4 @@
-<string key="ui_raid_war_header">War Raid Groups</string>
+<string key="ui_raid_war_header">Army for war.</string>

==== //depot/Assets/LyShineUI/_Common/UIStyle.lua#42 (text) ====

@@ -10,3 +10,3 @@
-local theme = "dark"
+local theme = "light"

==== //depot/Assets/LyShineUI/Images/Icon_GroupLeader.png#1 (binary) ====

[Binary content]"""
                
                mock_popen.return_value = create_mock_popen_process(describe_output)
                
                # Index with workspace_path = localization dir, subdirs = None
                changelists = perforce_provider._fetch_changelists_by_numbers(
                    workspace_path=str(loc_path),
                    cl_numbers=["1076268"],
                    subdirs=None  # Should auto-filter by workspace_path
                )
                
                assert len(changelists) == 1
                cl = changelists[0]
                
                # CRITICAL: Only localization file should be included
                assert len(cl.files) == 1
                assert "GroupsText.loc.xml" in cl.files[0]
                assert not any("UIStyle.lua" in f for f in cl.files)
                assert not any("Icon_GroupLeader.png" in f for f in cl.files)
                
                # CRITICAL: Only localization diff should be present
                assert "Army for war" in cl.diff
                assert "theme" not in cl.diff  # UI file diff excluded
                assert "Binary" not in cl.diff  # Image file excluded