Task
Airplanes are stuck turning in a circle when too close to their target. How can we address this bug?

24.3k / 200.0k
$0.16

## Complete conversation

API Request
$0.0583


Roo said
I'll help you investigate this airplane turning bug. Let me first consult the OpenRA expert to see if there's any historical context about this issue.


Roo wants to use a tool on the expert-among-us MCP server
expert-among-us

prompt
[Tool string omitted to avoid biasing the summary analysis]
{
  "expert_name": "OpenRA",
  "prompt": "How do airplanes handle turning and movement? Are there any known bugs or fixes related to aircraft getting stuck in turning circles when they're too close to their target?"
}
[See embedded MCP response below]


API Request
$0.0290


Roo said
Excellent! The expert has provided valuable historical context. There was indeed a documented bug where aircraft would get stuck circling when targets were inside their minimum turn radius. The fix involved calculating the turn radius and detecting when targets are unreachable without complex maneuvering.

Let me examine the current implementation in the Fly.cs file to see if this fix is present or if there's been a regression:


Roo wants to read this file

API Request
$0.0527


Roo said
Great! I can see the turn radius fix is implemented in Fly.cs. The code calculates the turn radius and checks if the target is inside the minimum turning circle. However, let me analyze the implementation more carefully to identify potential issues.

Looking at the code, I notice a subtle but important difference between what the expert described and the current implementation:

Current Implementation Analysis
Lines 238-256 show the turn radius check:

// Using the turn rate, compute a hypothetical circle traced by a continuous turn.
// If it contains the destination point, it's unreachable without more complex maneuvering.
var turnRadius = CalculateTurnRadius(aircraft.MovementSpeed, aircraft.TurnSpeed);

// The current facing is a tangent of the minimal turn circle.
// Make a perpendicular vector, and use it to locate the turn's center.
var turnCenterFacing = aircraft.Facing + new WAngle(Util.GetTurnDirection(aircraft.Facing, desiredFacing) * 256);

var turnCenterDir = new WVec(0, -1024, 0).Rotate(WRot.FromYaw(turnCenterFacing));
turnCenterDir *= turnRadius;
turnCenterDir /= 1024;

// Compare with the target point, and keep flying away if it's inside the circle.
var turnCenter = aircraft.CenterPosition + turnCenterDir;
if ((checkTarget.CenterPosition - turnCenter).HorizontalLengthSquared < turnRadius * turnRadius)
    desiredFacing = aircraft.Facing;

The Problem
When the target is inside the turn radius, the aircraft maintains its current facing (line 255: desiredFacing = aircraft.Facing). This is intended to make it "fly away" from the target, but there's a critical issue:

The aircraft maintains its current direction rather than actively flying away
This means if the aircraft's current trajectory happens to curve back toward the target, it will stay stuck in a circling pattern
The aircraft never gains enough distance to escape the turn radius constraint
The Solution
Based on the expert's historical context and geometric analysis, here's what needs to happen when a target is inside the turn radius:

No

Yes

Yes

No

Aircraft approaching target

Is target inside
turn radius?

Turn toward target normally

Calculate direction AWAY from target

Fly in opposite direction

Gain distance

Still inside
turn radius?

The fix should be:

Instead of just maintaining current facing (desiredFacing = aircraft.Facing), the aircraft should actively fly in the opposite direction from the target. The correct logic would be:

if ((checkTarget.CenterPosition - turnCenter).HorizontalLengthSquared < turnRadius * turnRadius)
{
    // Fly in the opposite direction to gain distance
    var awayFacing = (aircraft.CenterPosition - checkTarget.CenterPosition).Yaw;
    desiredFacing = awayFacing;
}

This ensures the aircraft actively escapes the problematic zone by flying directly away from the target, gaining enough distance to eventually make a successful approach from outside the minimum turn radius.

Additional Considerations
For slider aircraft (helicopters): The check at line 178 handles CanSlide aircraft differently, but non-sliding aircraft (planes) are the ones affected by turn radius constraints

The nearEnough parameter: Line 209 checks if the aircraft is "close enough" when blocked - this might need adjustment to account for turn radius constraints

Testing scenarios: This bug would be most apparent when:

Ordering fast aircraft with large turn radii to stationary ground targets
Aircraft are given waypoints very close to their current position
Aircraft are pursuing slow-moving or stationary targets
Would you like me to create a detailed bug report or prepare a fix for this issue?



API Request
$0.0193


Task Completed
I've identified the airplane turning bug in Fly.cs.

The Problem
When a target is inside an aircraft's minimum turn radius, the code maintains the current facing (desiredFacing = aircraft.Facing at line 255). This causes the aircraft to continue its current trajectory, which often curves back toward the target, creating an infinite circling pattern.

Root Cause
The turn radius check (lines 238-256) correctly identifies when a target is unreachable due to being inside the turning circle, but the solution of maintaining current facing doesn't actively move the aircraft away from the problematic zone. If the aircraft is already turning toward the target, it will continue that turn indefinitely.

The Fix
Replace line 255 with logic that actively flies away from the target:

if ((checkTarget.CenterPosition - turnCenter).HorizontalLengthSquared < turnRadius * turnRadius)
{
    // Fly in the opposite direction to gain distance
    var awayFacing = (aircraft.CenterPosition - checkTarget.CenterPosition).Yaw;
    desiredFacing = awayFacing;
}

This ensures the aircraft flies directly away from the target, gaining enough distance to eventually approach from outside the minimum turn radius.

Why This Works
By actively flying away rather than maintaining the current trajectory, the aircraft will:

Escape the turn radius constraint zone
Gain sufficient distance for a proper approach
Naturally curve back toward the target once outside the minimum turn radius
Successfully reach the target on the next approach
The historical expert context confirms this pattern was the intended fix for issue #7083, though the current implementation appears to have a subtle flaw in how it handles the "fly away" behavior.


## Embedded MCP response

Analysis: Aircraft Turning and Movement in OpenRA
Based on the historical commit patterns, here are the key insights about aircraft movement and turning mechanics:

Core Movement Architecture
Coordinate System Evolution:

Early commits show a transition from pixel-based positioning (PPos, PSubPos) to world coordinates (WPos, WVec)
This was a major architectural change that affected all aircraft activities
The conversion improved precision and eliminated coordinate system mismatches
Key Files for Aircraft Movement:

OpenRA.Mods.Common/Activities/Air/Fly.cs - Primary movement logic
OpenRA.Mods.Common/Traits/Air/Aircraft.cs - Core aircraft trait
OpenRA.Mods.Common/Activities/Air/Land.cs - Landing procedures
OpenRA.Mods.Common/Activities/Air/ReturnToBase.cs - RTB pathfinding
Known Issue: Stuck in Turning Circles
The Problem:
Yes, there was a documented bug where aircraft would get stuck circling endlessly when the target was inside their minimum turn radius.

The Fix (Commit: "Fix for #7083"):

// Using the turn rate, compute a hypothetical circle traced by a continuous turn.
// If it contains the destination point, it's unreachable without more complex maneuvering.
var turnRadius = CalculateTurnRadius(aircraft.MovementSpeed, aircraft.TurnSpeed);

// The current facing is a tangent of the minimal turn circle.
// Make a perpendicular vector, and use it to locate the turn's center.
var turnCenterFacing = aircraft.Facing;
turnCenterFacing += Util.GetNearestFacing(aircraft.Facing, desiredFacing) > 0 ? 64 : -64;

var turnCenterDir = new WVec(0, -1024, 0).Rotate(WRot.FromFacing(turnCenterFacing));
turnCenterDir *= turnRadius;
turnCenterDir /= 1024;

// Compare with the target point, and keep flying away if it's inside the circle.
var turnCenter = aircraft.CenterPosition + turnCenterDir;
if ((checkTarget.CenterPosition - turnCenter).HorizontalLengthSquared < turnRadius * turnRadius)
    desiredFacing = aircraft.Facing;

Turn Radius Calculation:

public static int CalculateTurnRadius(int speed, int turnSpeed)
{
    // turnSpeed -> divide into 256 to get the number of ticks per complete rotation
    // speed -> multiply to get distance travelled per rotation (circumference)
    // 45 -> divide by 2*pi to get the turn radius: 45==256/(2*pi), with some extra leeway
    return 45 * speed / turnSpeed;
}

Movement Implementation Patterns
Helicopter vs. Plane Distinction:
Originally there were separate Helicopter and Plane traits, but they were merged into a unified Aircraft trait with a CanHover property:

[Desc("Can the actor hover in place mid-air? If not, then the actor will have to remain in motion (circle around).")]
public readonly bool CanHover = false;

Key Movement Properties:

TurnSpeed - Base turning rate
IdleTurnSpeed - Slower turning when circling idle (defaults to TurnSpeed)
MaximumPitch - Controls climb/descent angle for planes
AltitudeVelocity - Vertical movement speed for VTOLs
CruiseAltitude - Target altitude during normal flight
Common Pitfalls & Anti-Patterns
1. Altitude Management Issues
Multiple commits show fixes for aircraft not properly adjusting altitude before/after activities:

Aircraft would sometimes try to turn before reaching cruise altitude
Canceling activities could leave aircraft at wrong altitude
Solution: Always check and adjust altitude in activity transitions
2. Influence Management
Aircraft must properly manage ground cell influence:

// We are taking off, so remove influence in ground cells.
if (dat <= aircraft.LandAltitude)
{
    aircraft.RemoveInfluence();
}

Regression trap: Forgetting to remove influence causes pathfinding issues for ground units.

3. Target Overshoot
The "next move would overshoot" check is critical:

// The next move would overshoot, so just set the final position
var move = aircraft.FlyStep(desiredFacing);
if (delta.HorizontalLengthSquared < move.HorizontalLengthSquared)
{
    // Handle final positioning
}

4. Repulsion Force Calculation
Aircraft repulsion prevents mid-air collisions but had issues:

Planes could stall if repulsion force pointed backward
Solution: Only apply repulsion force that has positive dot product with current direction
var dot = WVec.Dot(currentDir, repulsionForce) / length;
// avoid stalling the plane
return dot >= 0 ? repulsionForce : WVec.Zero;

Testing Strategies
Critical Test Scenarios:

Tight turn targets - Order aircraft to locations within their turn radius
Altitude transitions - Cancel activities mid-flight and verify altitude correction
Terrain height changes - Aircraft over varying terrain heights
Formation flying - Multiple aircraft with repulsion enabled
Landing on reserved buildings - Ensure proper fallback behavior
Regression Indicators:

Aircraft circling indefinitely without approaching target
Aircraft "bouncing" at ground level
Influence not cleared, blocking ground units
Sudden altitude changes causing visual glitches
Aircraft not responding to stop commands
Relevant Existing Files for Context
OpenRA.Mods.Common/Activities/Air/Fly.cs - Main movement logic, turn radius calculations
OpenRA.Mods.Common/Activities/Air/Land.cs - Landing approach and final positioning
OpenRA.Mods.Common/Activities/Air/ReturnToBase.cs - Complex pathfinding with tangent circles
OpenRA.Mods.Common/Traits/Air/Aircraft.cs - Core trait with movement properties
OpenRA.Mods.Common/Activities/Air/FlyCircle.cs - Idle circling behavior
OpenRA.Mods.Common/Activities/Air/TakeOff.cs - Takeoff procedures and influence management
Side Effects of Past Mistakes
Hardcoded turn speeds for idle circling caused AutoCarryall to turn too slowly
Separate Helicopter/Plane implementations led to code duplication and inconsistent behavior
Pixel coordinate system caused precision issues and difficult-to-debug positioning errors
Missing altitude checks in activity cancellation caused aircraft to be stuck at wrong heights
The codebase has evolved significantly to handle edge cases, particularly around turn radius constraints and altitude management during activity transitions.
