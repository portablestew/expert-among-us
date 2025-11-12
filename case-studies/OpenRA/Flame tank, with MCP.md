Task
How can I define another vehicle like the flame tank? Ask the expert first, then investigate deeper

57.7k / 200.0k

## Complete conversation

API Request
$0.0000

Roo said
I'll help you understand how to define a vehicle like the flame tank in OpenRA. Let me start by consulting the expert about this topic.


Roo wants to use a tool on the expert-among-us MCP server
expert-among-us

prompt
[Tool string omitted to avoid biasing the summary analysis]
{
  "expert_name": "OpenRA",
  "prompt": "How do I define a new vehicle unit like the flame tank? What files and traits do I need to configure?"
}
[See embedded MCP response below]


API Request
$0.0000

Roo said
I'll investigate the actual implementation to show you concrete examples from the OpenRA codebase.


Roo wants to read this file

API Request
$0.0000

Roo said
I'll also read the weapons file to show you the complete picture of defining a vehicle like the flame tank.


Roo wants to read this file

Error
Error reading file mods/cnc/rules/weapons.yaml:
ENOENT: no such file or directory, stat 'c:\Dev\github\OpenRA\mods\cnc\rules\weapons.yaml'


API Request
$0.0000

Roo wants to view the top level files in this directory

API Request
$0.0000

Roo wants to read multiple files

Error
Error reading file mods/cnc/weapons.yaml:
ENOENT: no such file or directory, stat 'c:\Dev\github\OpenRA\mods\cnc\weapons.yaml'


API Request
$0.0000

Roo wants to view the top level files in this directory

API Request
$0.0000

Roo wants to read this file

Error
Error reading file mods/cnc/weapons/misc.yaml:
ENOENT: no such file or directory, stat 'c:\Dev\github\OpenRA\mods\cnc\weapons\misc.yaml'


API Request
$0.0000

Roo wants to view the top level files in this directory

API Request
$0.0000

Roo wants to read this file

API Request
$0.0000

Task Completed
Based on the expert consultation and codebase investigation, here's how to define a vehicle like the flame tank in OpenRA:

Key Files You Need to Modify
1. Unit Definition (mods/cnc/rules/vehicles.yaml)
Define your vehicle by inheriting from base vehicle types and configuring traits:

FTNK:
  Inherits: ^Tank
  Inherits@EXPERIENCE: ^GainsExperience
  Inherits@CLOAK: ^AcceptsCloakCrate
  Inherits@AUTOTARGET: ^AutoTargetGroundAssaultMove
  Valued:
    Cost: 600
  Tooltip:
    Name: actor-ftnk.name
  UpdatesPlayerStatistics:
    AddToArmyValue: true
  Buildable:
    BuildPaletteOrder: 50
    Prerequisites: anyhq, ~techlevel.medium
    Queue: Vehicle.Nod
    Description: actor-ftnk.description
  Mobile:
    TurnSpeed: 28
    Speed: 92
  Health:
    HP: 27000
  Repairable:
    HpPerStep: 2046
  Armor:
    Type: Heavy
  RevealsShroud:
    Range: 6c0
  Armament:
    Weapon: BigFlamer
    LocalOffset: 512,128,42, 512,-128,42
    MuzzleSequence: muzzle
  Encyclopedia:
    Description: actor-ftnk.encyclopedia
    Order: 120
    Category: Vehicles
    Scale: 3
    PreviewOwner: NodUnits
  AttackFrontal:
    Voice: Attack
    FacingTolerance: 0
  AttackMove:
    Voice: Attack
  WithMuzzleOverlay:
  FireWarheadsOnDeath:
    Weapon: FlametankExplode
    EmptyWeapon: FlametankExplode
  SpawnActorOnDeath:
    Actor: FTNK.Husk
    OwnerType: InternalName
    EffectiveOwnerFromOwner: true


2. Weapon Definition (mods/cnc/weapons/other.yaml)
Define the weapon your vehicle uses:

BigFlamer:
  Inherits: ^FlameWeapon
  ReloadDelay: 65
  Range: 3c512
  Projectile: Bullet
    Speed: 341
  Burst: 2
  BurstDelays: 10
  Warhead@1Dam: SpreadDamage
    Spread: 400
    Damage: 10000
    Versus:
      None: 82
      Wood: 75
      Light: 75
      Heavy: 18
      Concrete: 75
  Warhead@3Eff: CreateEffect
    Explosions: med_napalm

3. Base Templates (mods/cnc/rules/defaults.yaml)
Leverage inheritance from base types:

^Vehicle - Basic vehicle traits (Mobile, Targetable, HiddenUnderFog, etc.)
^Tank - Tracked locomotor and tank-specific behavior
^GainsExperience - Veterancy system with rank bonuses
^AutoTargetGroundAssaultMove - Automatic targeting behavior
Essential Traits
Trait	Purpose	Example
Mobile	Movement speed, turn rate, locomotor type	Speed: 92, TurnSpeed: 28
Health	Hit points	HP: 27000
Armor	Damage resistance type	Type: Heavy
Armament	Weapon attachment with offsets	Weapon: BigFlamer
AttackFrontal/AttackTurreted	Combat behavior	FacingTolerance: 0
Buildable	Production queue, cost, prerequisites	Cost: 600, Queue: Vehicle.Nod
FireWarheadsOnDeath	Explosion on death	Weapon: FlametankExplode
SpawnActorOnDeath	Creates wreckage husk	Actor: FTNK.Husk
Key Patterns
Inheritance - Use Inherits: to get traits from base templates (reduces duplication)
Locomotor - Set in Mobile trait: tracked for tanks, wheeled for light vehicles
Weapon Offsets - LocalOffset positions weapon firing points (important for multi-barrel weapons)
Prerequisites - Tech tree requirements like anyhq, ~techlevel.medium
Death Effects - Both weapons (FireWarheadsOnDeath) and actors (SpawnActorOnDeath)
Steps to Create a New Vehicle
Choose appropriate base template (^Tank, ^Vehicle, etc.)
Define unit in rules/vehicles.yaml with traits
Create weapon definition in weapons/ directory
Add animation sequences (if new sprites)
Configure buildable queue and prerequisites
Define husk actor for wreckage
Test weapon balance and movement parameters
The flame tank is an excellent reference because it shows: tracked locomotor, frontal attack (must face target), flame weapons with area damage, heavy armor, and explosion effects on death.


## Embedded MCP response

Based on the commit history, here's how to define a new vehicle unit like the flame tank:

Key Files to Modify
rules/vehicles.yaml - Main unit definition
weapons.yaml - Weapon definitions
sequences/vehicles.xml - Animation sequences
Built-in trait requirements - The game engine handles most functionality
Essential Components
1. Unit Definition (vehicles.yaml)
FTNK:
	Inherits: ^Tank
	Inherits@EXPERIENCE: ^GainsExperience
	Inherits@CLOAK: ^AcceptsCloakCrate
	Inherits@AUTOTARGET: ^AutoTargetGroundAssaultMove
	Valued:
		Cost: 600
	Tooltip:
		Name: Flame Tank
	Buildable:
		BuildPaletteOrder: 50
		Prerequisites: afld, hq
		Queue: Vehicle.Nod
		Description: Anti-infantry vehicle
	Mobile:
		TurnSpeed: 28
		Speed: 92
	Health:
		HP: 27000
	Armor:
		Type: Heavy
	Armament:
		Weapon: BigFlamer
		LocalOffset: 512,128,42, 512,-128,42
		MuzzleSequence: muzzle
	AttackFrontal:
		Voice: Attack
		FacingTolerance: 0
	FireWarheadsOnDeath:
		Weapon: FlametankExplode
		EmptyWeapon: FlametankExplode
	SpawnActorOnDeath:
		Actor: FTNK.Husk


2. Weapon Definition (weapons.yaml)
BigFlamer:
	Damage: 50
	ROF: 50
	Range: 2
	Projectile: Invisible
	Speed: 100
	Warhead: Fire
	Report: FLAMER2

FlametankExplode:
	Warhead:
		Damage: 100		
		Spread: 24		
		Explosion: big_napalm
		InfDeath: 5
		ImpactSound: xplobig6

3. Animation Sequences (sequences/vehicles.xml)
<unit name="ftnk">
	<sequence name="idle" start="0" facings="32" />
	<sequence name="muzzle" start="32" length="4" />
</unit>

Key Traits from Commits
Core Vehicle Traits
Mobile - Movement (speed, turn rate, locomotor type)
Health - Hit points
Armor - Damage resistance type
Armament - Weapon attachment with offsets
AttackFrontal/AttackTurreted - Combat behavior
Common Patterns
Inherits: ^Tank - Gets tracked locomotor and base vehicle traits
Buildable - Production queue, cost, prerequisites
FireWarheadsOnDeath - Explosion on death
SpawnActorOnDeath - Creates husk wreckage
WithMuzzleOverlay - Weapon firing animation
Historical Context from Commits
The "Facing -> Unit trait" commit shows the architecture evolved to:

Store Facing in a Unit trait (not per-vehicle)
Use traits.Get<Unit>() to access facing/altitude
Separate rendering from unit state
The "Load unit data from ra.yaml" commit introduced:

YAML-based unit definitions replacing INI files
Trait inheritance via Inherits:
Trait parameters in nested dictionaries
Testing Recommendations from History
Weapon balance - The commits show multiple iterations adjusting damage, ROF, and range
Explosion effects - Several commits tweaked death weapons (FlametankExplode vs UnitExplode)
Husk behavior - The "husk experiment" commit added wreckage with particle effects
Prerequisite chains - Ensure tech tree requirements work (e.g., afld, hq)
Side Effects to Watch
Flame weapons can damage the unit itself if too close to walls
Heavy armor affects crush behavior and repair costs
Speed values interact with terrain modifiers
LocalOffset weapon positioning requires testing across all facings
The most relevant existing file for reference is mods/cnc/rules/vehicles.yaml which shows the full flame tank implementation with all traits properly configured.
