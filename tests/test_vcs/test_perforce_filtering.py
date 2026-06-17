"""Tests for Perforce project-root-based file filtering.

These tests verify that file filtering works correctly using the project root
as the boundary, ensuring only files within the project root are included in diffs.
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
    """Tests for automatic file filtering based on project_root."""
    
    def test_depot_prefixes_created_from_project_root(self, perforce_provider, tmp_path):
        """Verify depot_prefixes is derived from project_root as the filter boundary."""
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
                
                changelists = perforce_provider._fetch_changelists_by_numbers(
                    project_root=str(tmp_path),
                    cl_numbers=["12345"],
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
                
                # Index with project_root = localization dir
                changelists = perforce_provider._fetch_changelists_by_numbers(
                    project_root=str(loc_path),
                    cl_numbers=["1076268"],
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