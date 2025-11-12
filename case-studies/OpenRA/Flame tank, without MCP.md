Task
How can I define another vehicle like the flame tank?

52.2k / 200.0k

## Complete conversation

Roo said
I'll help you understand how to define a new vehicle similar to the flame tank in OpenRA. Let me first explore the codebase to understand the structure and find existing vehicle definitions.


Roo wants to search this directory for flame.*tank|flametank

API Request
$0.0000

Roo said
Good! I can see there's a flame tank referenced as "FTNK" in the rules.yaml. Let me search for more vehicle definitions and find the actual unit definitions.


Roo wants to search this directory for ^FTNK:|^FTNK\s*:

API Request
$0.0000

Roo said
Perfect! I found the main flame tank definition in mods/cnc/rules/vehicles.yaml. Let me examine this file to understand the complete structure of the flame tank definition.


Roo wants to read this file

API Request
$0.0000

Roo said
Great! Now let me also check the base templates and weapon definitions to understand the complete structure. Let me look at the base templates and weapon definitions that the flame tank uses.


Roo wants to search this directory for ^.*Tank.*:|^.*Vehicle.*:

API Request
$0.0000

Roo said
Let me examine the base templates and also check the weapon definitions to understand the complete picture.


Roo wants to read multiple files

Error
Error reading file mods/cnc/rules/weapons.yaml:
File not found: c:\Dev\github\OpenRA\mods\cnc\rules\weapons.yaml


API Request
$0.0000

Roo said
Let me find the weapons file in the correct location:


Roo wants to search this directory for weapons\.yaml

API Request
$0.0000

Roo said
I can see the weapons are in subdirectories. Let me check the weapons directory structure and find the BigFlamer weapon that the flame tank uses:


Roo wants to recursively view all files in this directory

API Request
$0.0000

Roo said
Let me search for the BigFlamer weapon definition:


Roo wants to search this directory for BigFlamer:

API Request
$0.0000

Roo said
Perfect! Now let me examine the weapon definitions to understand the complete structure:


Roo wants to read this file

API Request
$0.0000

Roo said
Now let me also check the explosion definitions that the flame tank uses:


Roo wants to search this directory for FlametankExplode:

API Request
$0.0000

Roo wants to read this file

API Request
$0.0000

Roo said
Perfect! Now I have a complete understanding of how vehicles are defined in OpenRA. Let me also check if there are any husk definitions needed:


Roo wants to search this directory for FTNK\.Husk:|^FTNK\.Husk\s*:

API Request
$0.0000

Roo wants to read this file

API Request
$0.0000

Roo said
Now I have a complete understanding of how vehicles are defined in OpenRA. Let me provide you with a comprehensive guide on how to create a new vehicle similar to the flame tank.

How to Define a New Vehicle Like the Flame Tank in OpenRA
Based on my analysis of the OpenRA codebase, here's a complete guide to creating a new vehicle similar to the flame tank:

1. Understanding the Flame Tank Structure
The flame tank (FTNK) in OpenRA has this structure:

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


2. Key Components to Define
A. Vehicle Definition (in mods/cnc/rules/vehicles.yaml)
Every vehicle needs these essential components:

Inheritance Chain:

^Tank (inherits from ^Vehicle → ^SpriteActor → ^ExistsInWorld)
Optional mixins: ^GainsExperience, ^AcceptsCloakCrate, ^AutoTargetGroundAssaultMove
Core Properties:

Valued: Cost and build requirements
Buildable: Queue placement, prerequisites, build order
Mobile: Speed and turn rate
Health: Hit points
Armor: Damage resistance type
Armament: Weapon assignment
Optional Features:

Repairable: How it gets repaired
RevealsShroud: Vision range
Encyclopedia: In-game documentation
FireWarheadsOnDeath: Explosion on death
SpawnActorOnDeath: Husk creation
B. Weapon Definition (in mods/cnc/weapons/other.yaml)
Example weapon structure:

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

C. Explosion Definition (in mods/cnc/weapons/explosions.yaml)
FlametankExplode:
	Inherits: ^DamagingExplosion
	Warhead@1Dam: SpreadDamage
		Spread: 1c0
		Damage: 11500
		DamageTypes: Prone50Percent, TriggerProne, FireDeath
	Warhead@2Eff: CreateEffect
		Explosions: big_napalm
		ImpactSounds: flamer2.aud
	Warhead@3Smu: LeaveSmudge
		SmudgeType: Scorch

D. Husk Definition (in mods/cnc/rules/husks.yaml)
NEWVEH.Husk:
	Inherits: ^Husk
	Tooltip:
		Name: actor-newveh-husk-name
	TransformOnCapture:
		IntoActor: newveh
	RenderSprites:
		Image: newveh.destroyed

3. Creating Your New Vehicle
Here's a template for creating a new flame tank variant:

# In mods/cnc/rules/vehicles.yaml
NEWVEH:  # Your new vehicle ID
	Inherits: ^Tank
	Inherits@EXPERIENCE: ^GainsExperience
	Inherits@CLOAK: ^AcceptsCloakCrate
	Inherits@AUTOTARGET: ^AutoTargetGroundAssaultMove
	Valued:
		Cost: 700  # Adjust cost
	Tooltip:
		Name: actor-newveh.name  # Add to localization files
	UpdatesPlayerStatistics:
		AddToArmyValue: true
	Buildable:
		BuildPaletteOrder: 55  # Position in build queue
		Prerequisites: anyhq, ~techlevel.medium
		Queue: Vehicle.Nod  # Or Vehicle.GDI
		Description: actor-newveh.description
	Mobile:
		TurnSpeed: 25  # Adjust mobility
		Speed: 85
	Health:
		HP: 30000  # Adjust hit points
	Repairable:
		HpPerStep: 2200
	Armor:
		Type: Heavy
	RevealsShroud:
		Range: 6c0
	Armament:
		Weapon: BigFlamer  # Or create new weapon
		LocalOffset: 512,128,42, 512,-128,42
		MuzzleSequence: muzzle
	Encyclopedia:
		Description: actor-newveh.encyclopedia
		Order: 125  # Encyclopedia order
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
		Weapon: FlametankExplode  # Or create new explosion
		EmptyWeapon: FlametankExplode
	SpawnActorOnDeath:
		Actor: NEWVEH.Husk
		OwnerType: InternalName
		EffectiveOwnerFromOwner: true


4. Additional Files to Update
A. Localization (add to mods/cnc/localization/*.yaml)
actor-newveh.name: "New Vehicle"
actor-newveh.description: "A powerful flame tank variant"
actor-newveh.encyclopedia: "This advanced flame tank features..."
actor-newveh-husk-name: "New Vehicle Wreck"

B. Prerequisites (in mods/cnc/rules/prerequisites.yaml)
Add any new building requirements if needed.

C. Build Queues (in mods/cnc/rules/build-queues.yaml)
Ensure your vehicle queue exists.

5. Key Design Considerations
Balance: Adjust cost, HP, damage, and speed relative to existing units
Role: Define clear tactical purpose (anti-infantry, area denial, etc.)
Synergy: Consider how it fits with existing army compositions
Visuals: Ensure sprite/animation files exist for your vehicle ID
Audio: Add sound references if using custom weapons
6. Testing Your Vehicle
Syntax Check: Validate YAML syntax
Game Testing: Test in various scenarios
Balance Testing: Compare against existing units
Multiplayer Testing: Ensure network synchronization works
7. Advanced Features
You can add more complex behaviors:

Multiple Weapons: Use Armament@PRIMARY and Armament@SECONDARY
Special Abilities: Add traits like Cloak, AmmoPool, Transforms
Experience System: Customize with GainsExperience parameters
Custom AI: Add specific targeting behaviors
This structure gives you complete control over creating new vehicles while maintaining compatibility with OpenRA's existing systems. The inheritance system allows you to build upon proven templates while customizing specific aspects to create unique units.

New Vehicle Definition

Inherits from ^Tank

Base Vehicle Properties

Weapon Assignment

Build Requirements

Husk Creation

Weapon Definition

Warhead Effects

Damage Calculation

Visual Effects

Prerequisites

Build Queue

Husk Template

Capture Behavior

This diagram shows the key components and their relationships in vehicle creation.
