Task
How does the voxel renderer work?

44.0k / 200.0k
$0.21

## Complete conversation

API Request
$0.0193


Roo said
I'll help you understand how the voxel renderer works in OpenRA. Let me start by consulting the repository's history to understand the implementation context and design decisions.



Roo wants to use a tool on the expert-among-us MCP server
expert-among-us

prompt
[Tool string omitted to avoid biasing the summary analysis]
{
  "expert_name": "OpenRA",
  "prompt": "How does the voxel renderer work? Explain the implementation, architecture, and key design decisions for rendering voxel models in the game."
}
[See embedded MCP response below]


API Request
$0.0273


Roo said
Now let me examine the actual implementation to provide you with a complete understanding. I'll read the key files involved in voxel rendering.


Roo wants to read multiple files

Error
Error reading file OpenRA.Game/Graphics/ModelRenderer.cs:
ENOENT: no such file or directory, stat 'c:\Dev\github\OpenRA\OpenRA.Game\Graphics\ModelRenderer.cs'


API Request
$0.0518


Roo said
Let me search for the ModelRenderer file:


Roo wants to search this directory for ModelRenderer

API Request
$0.0157


Roo wants to read multiple files

Error
Error reading file OpenRA.Game/Graphics/IModel.cs:
ENOENT: no such file or directory, stat 'c:\Dev\github\OpenRA\OpenRA.Game\Graphics\IModel.cs'


API Request
$0.0994


Roo said
OpenRA Voxel Renderer Architecture
Based on the repository history and implementation, here's how the voxel rendering system works in OpenRA:

Overview
The voxel renderer converts 3D voxel models (used in Command & Conquer: Tiberian Sun and Red Alert 2) into 2D sprites that can be efficiently rendered in the isometric game engine. It uses a two-phase rendering pipeline that separates model transformation from actual drawing.

Core Architecture
VXL/HVA Files

VoxelLoader

Voxel Model

ModelRenderData

Vertex Buffer

Actor Traits

RenderVoxels

ModelAnimation

ModelRenderable

PrepareRender Phase

ModelRenderer.RenderAsync

Render to Framebuffer

Generate Sprite

Render Phase

Draw Sprite to Screen

GPU Shaders

model.vert

model.frag

Key Components
1. VoxelLoader (OpenRA.Mods.Cnc/Graphics/VoxelLoader.cs)
Converts voxel file data into GPU-ready geometry:

Reads .vxl files: Contains the actual voxel data (3D grid of colored cubes)
Reads .hva files: Contains hierarchical animation data (transformation matrices per limb per frame)
Generates slice planes: Creates 2D quads for each visible face of the voxel grid (lines 100-169)
Culls hidden faces: Only generates geometry for voxel faces that border empty space (lines 118-147)
Creates vertex buffer: Stores all geometry in a GPU vertex buffer (lines 196-210)
Key optimization: The loader generates 6 vertices per visible face (2 triangles) rather than individual voxels, dramatically reducing geometry complexity.

2. Voxel (OpenRA.Mods.Cnc/Graphics/Voxel.cs)
Represents a loaded voxel model:

Stores per-limb data (scale, bounds, render data)
Provides transformation matrices for each limb at each animation frame (lines 60-81)
Combines limb transformations with HVA animation data
Calculates aggregate bounds for culling (lines 133-159)
3. ModelRenderer (OpenRA.Mods.Cnc/Traits/World/ModelRenderer.cs)
The core rendering engine that renders voxels to sprite sheets:

BeginFrame/EndFrame Pattern (lines 298-359)
BeginFrame()  // Start of frame
   ↓
RenderAsync() // Queue render operations (called for each actor)
   ↓
EndFrame()    // Execute all queued renders to framebuffers

This batching ensures expensive voxel→sprite conversion happens once per frame before the main draw loop.

RenderAsync Process (lines 95-245)
Calculate bounds: Determine sprite size needed for the model
Allocate sprites: Reserve space on sprite sheets for both model and shadow
Queue render function: Store a lambda that will render the voxel to the framebuffer
Return proxy: Provides the sprite handles for later drawing
The actual rendering happens in EndFrame(), which:

Binds framebuffers (render-to-texture)
Enables depth buffer for proper 3D ordering
Executes all queued render functions via GPU
Generates both color and shadow sprites
4. ModelRenderable (OpenRA.Mods.Cnc/Graphics/ModelRenderable.cs)
Implements the IRenderable interface:

PrepareRender() (line 117): Calls ModelRenderer.RenderAsync() to generate sprites
Render() (line 140): Draws the cached sprite to screen with proper positioning and shadows
5. Shadow Rendering
Shadows use a separate rendering pass (lines 232-236 in ModelRenderer):

Rendered at 2x resolution to reduce artifacts
Uses flat lighting (ambient=1, diffuse=0) for silhouette effect
Projected onto ground plane using light direction
Ground height calculated from terrain (lines 143-144 in ModelRenderable)
6. GPU Shaders
Vertex Shader (glsl/model.vert)
Applies transformation matrix to position voxels in 3D space
Passes texture coordinates and channel masks to fragment shader
Fragment Shader (glsl/model.frag)
Palette-based coloring: Looks up colors from indexed palette (line 21)
Normal map lighting: Applies ambient + diffuse lighting based on voxel normals (lines 25-28)
Alpha testing: Discards fully transparent pixels (line 23)
The lighting calculation: intensity = AmbientLight + DiffuseLight * max(dot(normal, LightDirection), 0.0)

Rendering Pipeline
Phase 1: Preparation (before drawing)
Actor → RenderVoxels.Render() → ModelRenderable
                                     ↓
                            PrepareRender() 
                                     ↓
                          ModelRenderer.RenderAsync()
                                     ↓
                        [Queued for later execution]

Phase 2: Execution (in ModelRenderer.EndFrame)
For each queued render:
  1. Bind framebuffer (render-to-texture)
  2. Enable depth buffer
  3. For each limb:
     - Apply transformation matrix
     - Calculate lighting direction
     - Render geometry with GPU shader
  4. Generate shadow with flat lighting
  5. Unbind framebuffer
     ↓
  Result: Sprite in sprite sheet

Phase 3: Display (actual frame rendering)
WorldRenderer draws scene:
  - Draw shadow sprite projected on ground
  - Draw model sprite at actor position
  - Apply terrain lighting tint if enabled

Key Design Decisions
Sprite Sheet Caching: Voxels are rendered to reusable sprite sheets rather than directly to screen, allowing the expensive 3D→2D conversion to be cached

Async Pattern: The separation between RenderAsync() (queuing) and EndFrame() (execution) prevents redundant renders

Depth Buffer: Proper 3D ordering within voxel models using GPU depth testing (enabled at line 316)

Immutable Structs: The IFinalizedRenderable interface prevents struct mutation issues during enumeration

Transform Hierarchy: Limb transformations are applied in order: scale → rotation → offset → camera projection

Performance Optimizations
Face culling: Only visible voxel faces generate geometry (VoxelLoader lines 131-144)
Batched rendering: All voxels rendered together in BeginFrame/EndFrame bracket
Framebuffer reuse: Sprite sheets are cached and reused across frames (lines 361-376)
Vertex buffer sharing: All voxel models share a single vertex buffer (VoxelLoader line 198)
This architecture elegantly balances 3D model fidelity with 2D rendering performance, making it possible to display complex animated units in an isometric RTS game.


# Embedded MCP response

Voxel Rendering Architecture in OpenRA
Based on the commit history, here's how the voxel rendering system works:

Core Architecture
The voxel rendering system follows a two-phase rendering pipeline:

Preparation Phase (PrepareRender): Transforms voxel data into GPU-ready sprites
Rendering Phase (Render): Actually draws the prepared sprites to screen
This separation was formalized in the commit "Split IFinalizedRenderable from Renderable to remove mutable structs," which introduced IFinalizedRenderable to cleanly separate these concerns.

Key Components
1. ModelRenderer (formerly VoxelRenderer)
Asynchronously renders voxel models to sprite sheets
Handles lighting calculations (ambient + diffuse)
Generates both color sprites and shadow sprites
Uses matrix transformations for camera projection and lighting
2. VoxelRenderable
Implements IRenderable and contains IFinalizedRenderable
Stores voxel animation data, camera angles, lighting parameters
The PrepareRender() method calls ModelRenderer.RenderAsync() to generate sprites
Returns a FinalizedVoxelRenderable struct that holds the rendered proxy
3. VoxelRenderProxy (now ModelRenderProxy)
Contains the final rendered sprite and shadow sprite
Stores projected shadow bounds for ground rendering
Acts as a cached result of the expensive voxel→sprite conversion
4. Trait System
RenderVoxels: Base trait that manages voxel rendering for actors
WithVoxelBody/Turret/Barrel: Specific body part renderers
IRenderActorPreviewVoxelsInfo: Interface for preview rendering
Critical Design Decisions
Async Rendering Pattern
The commit "Render voxels before BeginFrame" moved voxel preparation to happen before the main render loop:

PrepareRenderables() → calls PrepareRender() on all renderables
   ↓
ModelRenderer.BeginFrame() / EndFrame() bracket
   ↓
Draw() → just calls Render() on finalized renderables

This ensures expensive voxel→sprite conversion happens once per frame, not during the draw call.

Shadow Rendering
Shadows use a separate rendering pass with:

Identity ambient light (no color variation)
Zero diffuse light (flat appearance)
Projected onto ground plane using light direction
Rendered at 2x resolution to reduce artifacts (per "Improve voxel shadow rendering")
Depth Handling
Multiple commits addressed z-ordering issues:

"Fix voxel z offsets": Corrected screen space transformations to properly account for height
"Render voxels with approximately-correct z-offset": Added depth buffer support
"Fix voxel shadow geometry position": Calculated ground position correctly using terrain height
The key insight: y = TileSize.Height * (vec.Y - vec.Z) / 1024f combines Y position and height into screen Y coordinate.

GPU Shader Pipeline
Converted from QuadList to TriangleList ("Convert VoxelRenderer to TriangleList"), generating 6 vertices per face instead of 4. The shader (model.vert/model.frag) handles:

Palette-based coloring
Normal map lighting
Depth buffer writes
Actor Preview Support
The first commit added preview support by:

Implementing IRenderActorPreviewInfo on RenderVoxelsInfo
Each body part trait provides RenderPreviewVoxels() method
Constructs static voxel animations with initial facings/orientations
Returns VoxelPreview implementing IActorPreview
Movement to Common Mods
The commit "move generic Voxel render traits to Mods.Common" recognized these traits are useful beyond TS/RA2, making them available to all mod types.

Performance Considerations
Caching: VoxelRenderProxy caches the expensive voxel→sprite conversion
Batching: All voxels prepared together in BeginFrame/EndFrame bracket
Sprite sheets: Rendered voxels stored in dynamically-allocated sprite sheets
Culling: Only on-screen actors have their voxels prepared
Relevant Files for Context
OpenRA.Game/Graphics/ModelRenderer.cs - Core rendering engine
OpenRA.Mods.Common/Graphics/VoxelRenderable.cs - Renderable implementation
OpenRA.Mods.Common/Traits/Render/RenderVoxels.cs - Actor trait
OpenRA.Game/Graphics/WorldRenderer.cs - Integration with world rendering
glsl/model.vert and glsl/model.frag - GPU shaders
Common Pitfalls
Mutable structs: The IFinalizedRenderable split specifically avoided issues with struct mutation during enumeration
Z-fighting: Depth calculations require careful coordination between world space and screen space
Shadow positioning: Ground height must be calculated from terrain, not assumed flat
Barrel offsets: Turret rotation must be applied before body rotation (see "Fix voxel barrel rendering")
The architecture elegantly separates concerns while maintaining performance through caching and batching strategies.
