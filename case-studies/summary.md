# Expert Among Us MCP: Case Study Analysis

This document compares conversation outcomes when using the "expert-among-us" MCP versus standard code exploration, across two different technical questions about the OpenRA game engine.

## Executive Summary

The expert-among-us MCP demonstrated significant value in providing historical context and identifying subtle implementation issues, while reducing token usage by up to 45% in certain scenarios. The MCP is particularly valuable for debugging tasks where understanding the evolution and intent of code is critical.

**Recommendation**: Use the MCP for complex debugging, architecture understanding, and when historical context about design decisions would be valuable. For straightforward code exploration or feature additions, standard approaches may be sufficient.

---

## Case Study 1: Voxel Renderer Architecture

**Question**: "How does the voxel renderer work?"

### Without MCP
- **Token Usage**: 41.2k / 200k (21%)
- **Cost**: $0.21
- **Approach**: Direct code search and file reading
- **Files Read**: 3 core files (VxlReader.cs, VoxelLoader.cs, Voxel.cs, ModelRenderer.cs, RenderVoxels.cs)
- **Execution**: Smooth, no errors

**Content Quality**:
- Comprehensive architecture overview with diagrams
- Clear explanation of rendering pipeline phases
- Technical depth on GPU shaders, lighting system, transformations
- Performance optimizations identified
- Practical examples of usage

**What Was Covered**:
- File loading (VXL format parsing)
- Geometry generation (slice plane algorithm)
- Model representation
- Rendering pipeline (async preparation, execution, display)
- Actor integration
- Lighting and shadow systems

**What Was Missing**:
- Historical context about why certain architectural decisions were made
- Evolution of the system over time
- Common pitfalls from past development
- Design rationale (e.g., why the async pattern was chosen)

---

### With MCP
- **Token Usage**: 44.0k / 200k (22%)
- **Cost**: $0.21
- **Approach**: Consulted MCP first, then validated with code reading
- **Files Attempted**: Same files, but encountered path errors on first attempts
- **Execution**: Some friction with file paths, required additional search step

**Content Quality**:
- Everything from the non-MCP version PLUS:
- Historical context from commit messages
- Design decisions explained with rationale
- Evolution from older implementations
- Common pitfalls explicitly called out
- Testing strategies from past bugs

**Additional Insights from MCP**:
1. **Historical Architecture Evolution**:
   - "Split IFinalizedRenderable from Renderable to remove mutable structs" - explained WHY the two-phase pattern exists
   - VoxelRenderer was renamed to ModelRenderer
   - Movement from Mods.Cnc to Mods.Common and why

2. **Design Rationale**:
   - "Render voxels before BeginFrame" commit explained the async pattern decision
   - Depth buffer support evolution through multiple commits
   - Shadow rendering improvements (2x resolution) with specific commit references

3. **Common Pitfalls Identified**:
   - Mutable struct issues during enumeration
   - Z-fighting and depth calculation coordination
   - Shadow positioning on non-flat terrain
   - Barrel offset transformation order

4. **Performance Context**:
   - Caching strategy evolution
   - Why QuadList was converted to TriangleList
   - Batching improvements over time

**Trade-offs**:
- Slightly higher token usage (+6.8%)
- File reading errors required recovery
- More complex initial execution path
- Richer historical and architectural context

---

## Case Study 2: Airplane Turning Bug

**Question**: "Airplanes are stuck turning in a circle when too close to their target. How can we address this bug?"

### Without MCP
- **Token Usage**: 44.0k / 200k (22%)
- **Cost**: $0.24
- **Approach**: Search for aircraft movement code, analyze implementation
- **Analysis Depth**: Thorough root cause analysis
- **Execution**: Smooth, no errors

**Content Quality**:
- Clear root cause identification
- Detailed geometric analysis with flowchart
- Three distinct solution approaches:
  1. Speed reduction near target (recommended)
  2. Early target completion
  3. Temporary sliding behavior
- Code examples for each solution
- Pros/cons analysis with recommendations

**Diagnosis**:
- Identified the turn radius calculation at lines 238-256
- Explained the infinite loop mechanism
- Recognized that maintaining current facing causes circling
- Provided multiple creative solutions

**Recommendation**: Speed reduction approach as most realistic

---

### With MCP
- **Token Usage**: 24.3k / 200k (12%)
- **Cost**: $0.16
- **Approach**: Consulted MCP for historical context, then analyzed current code
- **Analysis Depth**: Focused diagnosis based on historical knowledge
- **Execution**: Smooth, leveraged historical context effectively

**Content Quality**:
- Everything needed for a precise fix
- Historical context about the original bug (issue #7083)
- Identified that current implementation deviates from intended fix
- Single, confident solution based on historical knowledge
- Understanding of what the original fix was supposed to do

**Additional Insights from MCP**:
1. **Bug History**:
   - This exact issue was previously fixed in commit "Fix for #7083"
   - Original fix included turn radius calculation logic
   - Current implementation appears to have subtle regression

2. **Original Intent**:
   - The fix was supposed to make aircraft "fly away" from targets inside turn radius
   - Current code maintains facing but doesn't actively fly away
   - Historical code showed the proper escape pattern

3. **Related Context**:
   - Helicopter vs. Plane evolution (CanHover property)
   - Repulsion force issues with backwards vectors
   - Common regression patterns for aircraft movement

4. **Testing Context**:
   - Specific test scenarios from past issues
   - Known regression indicators
   - Related files that might be affected

**Key Advantage**: Identified that this is a regression/deviation from an intended fix, not just a new bug

**Efficiency**:
- 45% fewer tokens (24.3k vs 44.0k)
- 33% lower cost ($0.16 vs $0.24)
- More confident, focused diagnosis
- Less exploratory searching needed

---

## Comparative Analysis

### Token Usage Comparison

| Case Study | Without MCP | With MCP | Difference |
|------------|-------------|----------|------------|
| Voxel Renderer | 41.2k (21%) | 44.0k (22%) | +6.8% |
| Airplane Bug | 44.0k (22%) | 24.3k (12%) | -44.8% |
| **Average** | **42.6k** | **34.2k** | **-19.7%** |

| Case Study | Without MCP | With MCP | Savings |
|------------|-------------|----------|---------|
| Voxel Renderer | $0.21 | $0.21 | $0.00 (0%) |
| Airplane Bug | $0.24 | $0.16 | $0.08 (33%) |
| **Total** | **$0.45** | **$0.37** | **$0.08 (18%)** |

### What the MCP Excels At

1. **Historical Context**
   - Commit history and evolution
   - Design decision rationale
   - Previous bug fixes and their intent
   - Architecture migrations

2. **Bug Diagnosis**
   - Identifying regressions
   - Understanding original fix intent
   - Knowing past failure patterns
   - Testing strategies from experience

3. **Efficiency** (when historical knowledge is relevant)
   - Reduced exploratory searching
   - More confident initial diagnosis
   - Less trial-and-error analysis

4. **Deep Understanding**
   - Why code exists, not just what it does
   - Common pitfalls from past mistakes
   - Side effects of changes
   - Related files and systems

### What Standard Exploration Excels At

1. **Simplicity**
   - No external dependencies
   - Straightforward execution
   - No MCP query overhead
   - Fewer potential failure points

2. **Creative Problem Solving**
   - Multiple solution approaches
   - Fresh perspective unbiased by history
   - Broader exploration of possibilities
   - Innovative fixes for new problems

3. **Current State Focus**
   - Analysis based purely on current code
   - No assumptions from historical context
   - Clear documentation of current behavior

### When Context Didn't Matter

For the voxel renderer question, both approaches resulted in similar token usage and cost because:
- The question was about understanding current architecture
- Historical context was interesting but not essential
- Both approaches needed to read the same core files
- The answer quality was comparable (comprehensive in both cases)

### When Context Was Critical

For the airplane bug, the MCP provided substantial value because:
- This was a previously fixed bug with a regression
- Historical context revealed the original intent
- Knowing issue #7083 provided immediate direction
- Testing patterns from past bugs were directly applicable
- 45% token reduction from focused diagnosis

---

## Files and Insights Comparison

### Voxel Renderer Files

**Without MCP - Files Read**:
- VxlReader.cs
- VoxelLoader.cs
- Voxel.cs
- ModelRenderer.cs
- RenderVoxels.cs

**With MCP - Files Attempted/Read**:
- VoxelLoader.cs (after path error recovery)
- ModelRenderer.cs (after search)
- Plus historical commit references

**MCP-Specific Insights**:
- IFinalizedRenderable split rationale
- VoxelRenderer → ModelRenderer rename
- QuadList → TriangleList conversion reasoning
- Depth buffer implementation evolution
- Shadow rendering iteration history

### Airplane Bug Files

**Without MCP - Files Read**:
- Multiple aircraft-related files from search
- Fly.cs (detailed analysis)

**With MCP - Files Read**:
- Fly.cs (targeted read)

**MCP-Specific Insights**:
- Issue #7083 context
- Original fix implementation details
- CanHover property evolution
- Repulsion force bug history
- Testing scenarios from past failures
- Regression indicators to watch for

---

## Execution Quality

### Voxel Renderer Execution

**Without MCP**:
- Clean execution, no errors
- Logical progression: search → read → explain
- 3 API requests total

**With MCP**:
- File path errors on initial reads
- Required search to find correct paths
- 5 API requests total (recovery overhead)
- Despite friction, delivered superior historical context

### Airplane Bug Execution

**Without MCP**:
- Clean execution, no errors
- Comprehensive exploration
- 3 API requests

**With MCP**:
- Clean execution, no errors
- Highly efficient, focused approach
- 4 API requests (including MCP query)
- Completed task statement added

---

## Recommendations

### Use the MCP When:

1. **Debugging Complex Issues**
   - Historical context might reveal regressions
   - Understanding original design intent is valuable
   - Previous bug fixes might be related

2. **Architecture Understanding**
   - Want to know WHY decisions were made
   - Need to understand evolution of systems
   - Seeking common pitfalls and best practices

3. **Code Archaeology**
   - Working with legacy code
   - Investigating design rationale
   - Understanding migration paths

4. **Regression Investigation**
   - Behavior changed from expected
   - Previous fixes might be broken
   - Need to understand original implementation

### Skip the MCP When:

1. **Simple Code Exploration**
   - Just need to read current implementation
   - Question is straightforward
   - No historical depth needed

2. **New Features**
   - Building something novel
   - No relevant historical context
   - Fresh perspective is valuable

3. **Performance-Critical Scenarios**
   - Token budget is very limited
   - MCP query overhead is unwanted
   - Current state is all that matters

4. **When Historical Context Adds Noise**
   - Past implementation was significantly different
   - Code has been completely rewritten
   - Historical patterns don't apply

---

## Cost-Benefit Analysis

### Voxel Renderer Case
- **MCP Added Value**: Historical context, design rationale, pitfalls
- **MCP Cost**: +6.8% tokens, same monetary cost, minor execution friction
- **Verdict**: Marginal benefit; use if historical context is interesting

### Airplane Bug Case
- **MCP Added Value**: Identified regression, provided precise fix, relevant test cases
- **MCP Cost**: -44.8% tokens, -33% monetary cost, no execution issues
- **Verdict**: Substantial benefit; highly recommended for debugging

### Overall Assessment

The expert-among-us MCP provides:
- **Average 19.7% token reduction** when historical context is applicable
- **18% cost savings** across both cases
- **Superior diagnosis quality** for regression and bug analysis
- **Richer context** for architecture understanding
- **Focused solutions** backed by historical evidence

Trade-offs:
- Adds dependency on external tool
- May encounter path resolution issues
- Requires expert database to be up-to-date
- Historical context may not always be relevant

---

## Conclusion

The expert-among-us MCP is a valuable tool that shines particularly in debugging scenarios where historical context provides critical insights. For the airplane turning bug, it reduced token usage by 45% while delivering a more confident diagnosis by identifying the issue as a regression from a known fix.

For architecture exploration questions like the voxel renderer, the MCP adds valuable historical context and design rationale, though the efficiency gains are less pronounced since both approaches need to read similar code.

**Final Recommendation**: Integrate the expert-among-us MCP into your workflow, particularly for:
- Debugging and regression investigation
- Understanding design decisions and architecture evolution
- Identifying past pitfalls and testing strategies
- Any scenario where "why was this done this way?" is as important as "how does this work?"

The MCP is not a replacement for direct code analysis but rather a complement that adds historical dimension and institutional knowledge to code exploration tasks.

---

## Metrics Summary

| Metric | Without MCP | With MCP | MCP Advantage |
|--------|-------------|----------|---------------|
| Total Token Usage | 85.2k | 68.3k | -19.7% |
| Total Cost | $0.45 | $0.37 | -18% |
| Historical Insights | Limited | Extensive | High |
| Bug Regression Detection | No | Yes | High |
| Execution Friction | Low | Low-Medium | Slight disadvantage |
| Solution Confidence | Good | Excellent (for bugs) | High |
| Creative Solutions | Multiple approaches | Focused fix | Trade-off |

**Overall Assessment**: The expert-among-us MCP provides substantial value, especially for debugging tasks, with measurable efficiency gains and qualitative improvements in historical context and design understanding.
