Task
What is the process to add a new Red Alert campaign?

## Complete conversation

38.9k / 200.0k

Roo said
I'll help you understand the process of adding a new Red Alert campaign to OpenRA. Let me explore the existing campaign structure to provide you with accurate information.


Roo wants to read this file

Error
Error reading file mods:
Path provided was not a file!


API Request
$0.0000

Roo wants to view the top level files in this directory

API Request
$0.0000

Roo wants to view the top level files in this directory

API Request
$0.0000

Roo wants to read this file

API Request
$0.0000

Roo wants to view the top level files in this directory

API Request
$0.0000

Roo wants to read multiple files

Error
Error reading file mods/ra/chrome/mainmenu.yaml:
ENOENT: no such file or directory, stat 'c:\Dev\github\OpenRA\mods\ra\chrome\mainmenu.yaml'


API Request
$0.0000

Roo wants to read multiple files

Error
Error reading file mods/ra/scripts/utils.lua:
File not found: c:\Dev\github\OpenRA\mods\ra\scripts\utils.lua


API Request
$0.0000

Task Completed
Process to Add a New Red Alert Campaign to OpenRA
Based on the OpenRA Red Alert mod structure, here's the comprehensive process for adding a new campaign:

1. Define Campaign in missions.yaml
Add your campaign entry to mods/ra/missions.yaml:

Your Campaign Name:
    mission-01
    mission-02
    mission-03

This file lists all campaigns and their mission identifiers in order.

2. Create Map Directory Structure
For each mission, create a directory under mods/ra/maps/ with the mission identifier name (e.g., mission-01/). Each mission directory must contain:

map.yaml - Main map configuration file
mission-script.lua - Mission-specific Lua scripting
rules.yaml - Mission-specific rule modifications (optional)
weapons.yaml - Mission-specific weapon modifications (optional)
3. Configure map.yaml
The map.yaml file defines:

MapFormat: 12
RequiresMod: ra
Title: 01: Your Mission Title
Author: Your Name
Tileset: SNOW  # or TEMPERAT, DESERT, INTERIOR
MapSize: 128,128
Bounds: 49,45,30,36
Visibility: MissionSelector
Categories: Campaign
LockPreview: True

Players:
    PlayerReference@Player1:
        Name: Greece
        Playable: True
        Required: True
        # ... player configuration
    PlayerReference@Enemy:
        Name: USSR
        Bot: campaign
        # ... AI configuration

Actors:
    # Define all map actors with locations and properties

Rules: ra|rules/campaign-rules.yaml, ra|rules/campaign-tooltips.yaml, ra|rules/campaign-palettes.yaml, rules.yaml
Weapons: weapons.yaml
FluentMessages: ra|fluent/lua.ftl, ra|fluent/campaign.ftl

4. Create Mission Script
Write your mission logic in Lua (e.g., allies01.lua). The script must include:

-- Define mission variables
-- Reinforcement types, paths, objectives, etc.

WorldLoaded = function()
    -- Initialize players
    Player1 = Player.GetPlayer("Greece")
    Enemy = Player.GetPlayer("USSR")
    
    -- Initialize objectives
    InitObjectives(Player1)
    Objective1 = AddPrimaryObjective(Player1, "objective-key")
    
    -- Set up triggers and events
    -- Trigger.OnKilled, Trigger.OnAllKilled, etc.
end

Tick = function()
    -- Per-tick logic (if needed)
end

The script has access to common campaign functions from campaign.lua:

InitObjectives(player) - Set up objective system
InitializeAttackAircraft(aircraft, enemyPlayer) - AI aircraft behavior
ChooseRandomTarget(unit, enemyPlayer) - Random target selection
OnAnyDamaged(actors, func) - Group damage triggers
5. Configure rules.yaml (Mission-Specific)
The mission's rules.yaml overrides game rules for that specific mission:

World:
    LuaScript:
        Scripts: campaign.lua, utils.lua, mission-script.lua
    MissionData:
        Briefing: Your mission briefing text
        BackgroundVideo: intro.vqa
        BriefingVideo: briefing.vqa
        StartVideo: start.vqa
        WinVideo: victory.vqa
        LossVideo: defeat.vqa

# Actor modifications for this mission
ACTOR_TYPE:
    Trait:
        Property: Value

6. Key Integration Points
Campaign rules: Referenced via ra|rules/campaign-rules.yaml in map.yaml
Fluent messages: Campaign text stored in ra|fluent/campaign.ftl and ra|fluent/lua.ftl
Common scripts: campaign.lua provides shared campaign functionality
Mission selection: Controlled by Visibility: MissionSelector and Categories: Campaign in map.yaml
7. Testing
After creating all mission files:

Launch OpenRA with the RA mod
Navigate to "Singleplayer" → "Missions"
Your campaign should appear in the mission browser
Select and play each mission to test
Summary
The essential steps are:

Add campaign to missions.yaml
Create mission directories under mods/ra/maps/
Configure map.yaml for each mission
Write mission Lua scripts
Add mission-specific rules if needed
Test through the mission browser
All missions follow this standardized structure, making it straightforward to add new campaigns by following the existing patterns in the Allied and Soviet campaigns.
