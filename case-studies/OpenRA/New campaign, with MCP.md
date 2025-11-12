Task
What is the process to add a new Red Alert campaign?

## Complete conversation

41.7k / 200.0k

API Request...
$0.0000

Roo said
I'll help you understand the process of adding a new Red Alert campaign to OpenRA. Let me start by consulting the expert to learn about how campaigns are structured and implemented in this codebase.



API Request
$0.0000

Roo said
I'll consult the OpenRA expert to learn about the campaign implementation process.


Roo wants to use a tool on the expert-among-us MCP server
expert-among-us

prompt
[Tool string omitted to avoid biasing the summary analysis]
{
  "expert_name": "OpenRA",
  "prompt": "How are Red Alert campaigns structured and implemented? What is the process for adding a new campaign including mission definitions, briefings, and campaign progression?"
}
[See embedded MCP response below]


API Request
$0.0000

Roo said
I'll examine some actual campaign files to provide you with concrete examples and verify the structure.


Roo wants to read multiple files

API Request
$0.0000

Roo said
I'll examine the missions.yaml file and a Lua script example to provide you with the complete picture.


Roo wants to read this file

API Request
$0.0000

Roo said
I'll look at a Lua script example to complete the picture.


Roo wants to read this file

API Request
$0.0000

Task Completed
Process to Add a New Red Alert Campaign
Based on the OpenRA codebase structure, here's the complete process for adding a new Red Alert campaign:

1. Create Mission Directory Structure
For each mission, create a directory under mods/ra/maps/:

mods/ra/maps/{campaign-name}-{number}/
├── map.yaml          # Map configuration and actor placement
├── map.bin           # Binary map data
├── map.png           # Preview image
├── rules.yaml        # Mission-specific rule overrides
├── weapons.yaml      # (Optional) Custom weapons
├── {mission}.lua     # Main mission logic
└── {mission}-AI.lua  # (Optional) AI behavior

2. Configure Map Players
In map.yaml, define players with campaign-specific settings:

Players:
  PlayerReference@Greece:
    Name: Greece
    Playable: True
    Required: True
    LockFaction: True
    Faction: allies
  PlayerReference@USSR:
    Name: USSR
    Bot: campaign  # Critical: enables campaign AI
    Faction: soviet
    Enemies: Greece

Important: Always set Bot: campaign for AI players (see campaign-rules.yaml).

3. Create Mission Rules
In rules.yaml, include campaign rules and define mission logic:

Rules: ra|rules/campaign-rules.yaml, ra|rules/campaign-tooltips.yaml, ra|rules/campaign-palettes.yaml, rules.yaml

World:
  LuaScript:
    Scripts: campaign.lua, utils.lua, {mission}.lua
  MissionData:
    Briefing: "Mission objectives text..."
    BackgroundVideo: prolog.vqa
    BriefingVideo: ally1.vqa
    StartVideo: landing.vqa
    WinVideo: snowbomb.vqa
    LossVideo: bmap.vqa

The campaign-rules.yaml provides:

Objective-based victory conditions (replaces conquest)
Fog of war enforcement
Campaign AI bot configuration
Disabled multiplayer options
4. Implement Mission Logic
Create {mission}.lua with this standard structure:

-- Define mission constants
InsertionHelicopterType = "tran.insertion"
EnemyUnits = { Unit1, Unit2, Unit3 }

-- Initialization function
WorldLoaded = function()
  player = Player.GetPlayer("Greece")
  enemy = Player.GetPlayer("USSR")
  
  InitObjectives(player)
  CreateObjectives()
  SetupTriggers()
end

-- Define objectives
CreateObjectives = function()
  PrimaryObj = AddPrimaryObjective(player, "destroy-enemy-base")
  SecondaryObj = AddSecondaryObjective(player, "rescue-civilians")
end

-- Setup triggers for mission events
SetupTriggers = function()
  Trigger.OnKilled(ImportantActor, MissionFailed)
  Trigger.OnAllKilled(EnemyUnits, MissionSuccess)
end

-- Per-frame logic (optional)
Tick = function()
  -- Resource drain, condition checks, etc.
end

Key patterns from allies01.lua:

Use InitObjectives() to initialize the objectives system
Add objectives with AddPrimaryObjective() and AddSecondaryObjective()
Use Trigger.OnKilled(), Trigger.OnAllKilled(), etc. for events
Mark objectives complete with player.MarkCompletedObjective()
5. Register Campaign in missions.yaml
Add your mission to mods/ra/missions.yaml:

Allied Campaign:
  allies-01
  allies-02
  your-new-mission  # Add here

Soviet Campaign:
  soviet-01
  soviet-02
  your-new-mission  # Or here

Mission IDs must match the map directory name.

6. Add Translation Strings
Add objective and dialogue text to mods/ra/fluent/lua.ftl or mods/ra/fluent/campaign.ftl:

## your-new-mission
destroy-enemy-base = Destroy the Soviet base.
rescue-civilians = Rescue the civilians before they are executed.
tanya-rules-of-engagement = Use caution near civilians and allied structures.

Reference these in Lua with:

UserInterface.GetFluentMessage("destroy-enemy-base")

Key Configuration Requirements
Mission-Specific Actor Variants
Create custom actors in rules.yaml for campaign-specific behavior:

TRAN.Extraction:
  Inherits: TRAN
  RevealsShroud:
    Range: 0c0
  Cargo:
    Types: Einstein
    MaxWeight: 1

Disable Units/Buildings
Control tech progression in rules.yaml:

TSLA:
  Buildable:
    Prerequisites: ~disabled

Campaign Rule Inheritance
The campaign-rules.yaml automatically provides:

Fog of war locked on
Starting cash locked at 0
Build radius enforcement
Campaign AI bot type
Objective-based victory
Testing Checklist
Verify mission appears in Mission Browser
Test all difficulty levels (if implemented)
Confirm all objective completion/failure paths work
Check that AI behavior functions correctly
Verify reinforcements and scripted events trigger properly
Test edge cases (units destroyed, buildings sold)
Ensure translations display correctly
Common Pitfalls
Forgetting Bot: campaign: AI won't function without this in map.yaml
Rule inclusion order: Always include campaign rules before mission rules in rules.yaml
Mission not appearing: Ensure directory name matches entry in missions.yaml
Victory conditions: Campaign uses objectives, not conquest - remove -ConquestVictoryConditions and add MissionObjectives via campaign-rules.yaml


## Embedded MCP response

Red Alert Campaign Structure & Implementation
Based on the historical commit patterns in this codebase, here's how Red Alert campaigns are structured:

Core Components
1. Mission Map Files (per mission)
map.yaml - Map definition with actors, players, triggers
map.bin - Binary map data
map.png - Preview image
rules.yaml - Mission-specific rule overrides
weapons.yaml - (optional) Custom weapons
notifications.yaml - (optional) Custom notifications
Lua scripts:
{mission}.lua - Main mission logic
{mission}-AI.lua - AI behavior (if needed)
{mission}-reinforcements.lua - (optional) Reinforcement logic
2. Campaign-Wide Files
Located in mods/ra/:

rules/campaign-rules.yaml - Shared campaign rule modifications
rules/campaign-tooltips.yaml - Tooltip customizations for campaign
rules/campaign-palettes.yaml - Campaign-specific palettes
languages/campaign/en.ftl - Campaign text/translations
languages/lua/en.ftl - Lua script translations
3. Campaign Progression
Defined in mods/ra/mod.yaml:

Missions:
  OpenRA.Mods.Common.LoadScreens.BlankLoadScreen:
    Allies:
      allies-01: Allies Mission 1
      allies-02: Allies Mission 2
      # ...
    Soviet:
      soviet-01: Soviet Mission 1
      soviet-02: Soviet Mission 2
      # ...

Key Patterns from Historical Commits
Mission Rules Structure
From the code examples, missions typically:

Disable unused units/buildings:
TSLA:
  Buildable:
    Prerequisites: ~disabled

Add campaign-specific behaviors:
World:
  LuaScript:
    Scripts: campaign.lua, utils.lua, {mission}.lua
  MissionData:
    Briefing: "Mission text..."
    BriefingVideo: video.vqa
    WinVideo: video.vqa
    LossVideo: video.vqa

Create difficulty dropdowns:
ScriptLobbyDropdown@difficulty:
  ID: difficulty
  Label: dropdown-difficulty.label
  Values:
    easy: options-difficulty.easy
    normal: options-difficulty.normal
    hard: options-difficulty.hard
  Default: normal

Campaign-Specific Actors
From the "capturable husks" commit, campaigns often need:

Custom actor variants (e.g., E7.noautotarget, TRAN.IN)
Mission-specific buildings (e.g., MISS repurposed as Prison)
Scripted units with special behaviors
AI Implementation
The "campaign bot" commits show:

All AI players should have Bot: campaign in map.yaml
Separate AI logic in {mission}-AI.lua files
AI uses shared campaign utilities
Process for Adding a New Campaign
Step 1: Create Mission Directory
mods/ra/maps/{campaign}-{number}/
├── map.yaml
├── map.bin
├── map.png
├── rules.yaml
├── {mission}.lua
└── {mission}-AI.lua (if needed)

Step 2: Configure Map Players
In map.yaml:

Players:
  PlayerReference@Greece:
    Name: Greece
    Playable: True
    Required: True
    Faction: allies
  PlayerReference@USSR:
    Name: USSR
    Bot: campaign  # Important!
    Faction: soviet
    Enemies: Greece

Step 3: Create Mission Rules
In rules.yaml:

Rules: ra|rules/campaign-rules.yaml, ra|rules/campaign-tooltips.yaml, ra|rules/campaign-palettes.yaml, rules.yaml

World:
  LuaScript:
    Scripts: campaign.lua, utils.lua, {mission}.lua
  MissionData:
    Briefing: "..."
    BriefingVideo: video.vqa

Step 4: Implement Mission Logic
In {mission}.lua:

-- Standard initialization
WorldLoaded = function()
  player = Player.GetPlayer("Greece")
  enemy = Player.GetPlayer("USSR")
  
  InitObjectives(player)
  CreateObjectives()
  SetupTriggers()
end

CreateObjectives = function()
  PrimaryObj = AddPrimaryObjective(player, "objective-text")
  SecondaryObj = AddSecondaryObjective(player, "optional-objective")
end

Tick = function()
  -- Per-frame logic
  if CheckVictory() then
    player.MarkCompletedObjective(PrimaryObj)
  end
end

Step 5: Add to Campaign Progression
In mods/ra/mod.yaml:

Missions:
  OpenRA.Mods.Common.LoadScreens.BlankLoadScreen:
    Allies:
      allies-{number}: Allies Mission {number}

Step 6: Add Translations
In mods/ra/languages/lua/en.ftl:

## allies-{number}
objective-text = Destroy the enemy base.
optional-objective = Find the secret documents.

Important Regression Traps
1. Civilian Building Balance
From the "Revert balance changes" commit, civilian buildings have campaign-specific properties:

FCOM:
  Capturable:
    Types: ~disabled
  -GivesBuildableArea:
  -BaseProvider:

Trap: Don't let campaign changes leak into multiplayer balance.

2. Bot Configuration
From "Enable campaign bot" commits, ALL scripted AI players need:

Bot: campaign

Trap: Missing this causes AI to not function properly in campaigns.

3. Rule File Inclusion Order
Always include in this order:

Rules: ra|rules/campaign-rules.yaml, ra|rules/campaign-tooltips.yaml, ra|rules/campaign-palettes.yaml, rules.yaml

Trap: Wrong order can cause rule conflicts or missing features.

4. Objective Linking
From the Negotiations mission, co-op missions need careful objective coordination:

-- Link victory between cooperative players
Trigger.OnPlayerWon(England, function()
  -- Trigger Greece's victory notification
end)

Trap: Not properly linking objectives can cause premature victory/defeat.

Relevant Existing Files
mods/ra/rules/campaign-rules.yaml - Base campaign rules
mods/ra/maps/allies-05a/rules.yaml - Good example of complex mission setup
mods/ra/maps/intervention/rules.yaml - Shows prerequisite manipulation
mods/ra/maps/negotiations/negotiations.lua - Complex mission scripting example
mods/ra/maps/negotiations/negotiations-ai.lua - Advanced AI implementation
mods/cnc/maps/gdi01/rules.yaml - Simpler campaign structure (C&C reference)
mods/ra/languages/lua/en.ftl - Translation patterns
Testing Strategy
Based on commit patterns:

Test all difficulty levels separately
Verify AI behavior doesn't break with rule changes
Check objective completion/failure paths
Ensure reinforcements trigger correctly
Test edge cases (e.g., all units destroyed, buildings sold)
Verify translations display correctly
The campaign system is highly modular - most changes are isolated to individual mission files, with shared utilities providing common functionality across all missions.
