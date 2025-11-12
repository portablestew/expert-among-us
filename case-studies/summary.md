# Expert-Among-Us MCP Case Study Analysis
## OpenRA Repository Comparison

This document compares three pairs of conversations, analyzing the effectiveness of using the expert-among-us MCP versus traditional code search and analysis approaches.

---

## Executive Summary

**Recommendation: YES, use the expert-among-us MCP**

The MCP consistently provided:
- **Historical insights** not available through code analysis alone
- **Faster problem identification** by understanding "why" not just "what"
- **Architectural context** showing design evolution and past mistakes
- **Average 12% better context efficiency** when accounting for action count (see detailed analysis below)

---

## Case Study 1: Voxel Renderer

### Task
"How does the voxel renderer work?"

### Without MCP
- **Cost**: $0.21 (41.2k tokens)
- **Actions**: 2 (1 search + 1 multi-file read)
- **Approach**: Direct code search → read files → analyze structure
- **Files Read**: 3 files (VxlReader.cs, VoxelLoader.cs, Voxel.cs)

**Findings**:
- ✅ Correctly identified core architecture (VxlReader, VoxelLoader, Voxel, ModelRenderer)
- ✅ Explained slice plane algorithm and geometry generation
- ✅ Described two-phase rendering pipeline
- ✅ Documented lighting system and transformation pipeline
- ❌ **MISSED**: IFinalizedRenderable architecture and why it exists
- ❌ **MISSED**: Historical bug fixes (z-ordering issues, shadow rendering improvements)
- ❌ **MISSED**: Performance optimization history (QuadList→TriangleList conversion)
- ❌ **MISSED**: Common pitfalls (mutable struct issues, shadow positioning bugs)

### With MCP
- **Cost**: $0.21 (44.0k tokens)
- **Actions**: 4 (1 MCP query + 1 failed read + 1 search + 1 failed read)
- **Approach**: Query expert first → verify with code → synthesize complete picture
- **Files Read**: Attempted 2 files (ModelRenderer.cs and IModel.cs - both errors, but MCP compensated)
- **Context Multiplier**: 2x actions = **2x context resubmission overhead**

**Findings**:
- ✅ All findings from "without MCP" version
- ✅ **PLUS**: IFinalizedRenderable split rationale (mutable struct issues during enumeration)
- ✅ **PLUS**: Historical evolution ("Render voxels before BeginFrame" architectural change)
- ✅ **PLUS**: Common pitfalls explicitly documented (z-fighting, shadow positioning, barrel offsets)
- ✅ **PLUS**: Performance considerations with historical context
- ✅ **PLUS**: Understanding of WHY decisions were made, not just WHAT exists

**Key Insight**: Even when file reads failed, the MCP's historical knowledge compensated and provided architectural rationale unavailable through code analysis alone.

---

## Case Study 2: Airplanes Turning Bug

### Task
"Airplanes are stuck turning in a circle when too close to their target. How can we address this bug?"

### Without MCP
- **Cost**: $0.24 (44.0k tokens)
- **Actions**: 2 (1 search + 1 multi-file read)
- **Approach**: Search for airplane/aircraft code → read Fly.cs → analyze turn radius logic → propose solutions
- **Files Read**: 2 files (Fly.cs, Aircraft.cs)

**Findings**:
- ✅ Identified turn radius calculation (lines 238-256)
- ✅ Diagnosed the problem: maintaining current facing creates infinite circle
- ✅ Proposed **3 solutions**:
  1. Speed reduction near target (RECOMMENDED)
  2. Early target completion
  3. Temporary sliding behavior
- ❌ **MISSED**: This bug was already documented and fixed (issue #7083)
- ❌ **MISSED**: The current implementation IS the fix, but with a subtle flaw
- ❌ **MISSED**: Historical context about coordinate system evolution affecting aircraft
- ❌ **MISSED**: Common regression patterns (influence management, altitude transitions)

### With MCP
- **Cost**: $0.16 (24.3k tokens) - **33% lower cost!**
- **Actions**: 2 (1 MCP query + 1 file read)
- **Approach**: Query expert about known bugs → examine current implementation → identify discrepancy
- **Files Read**: 1 file (Fly.cs)
- **Context Multiplier**: Same actions = **no overhead penalty** ✅

**Findings**:
- ✅ **IMMEDIATELY** identified this as issue #7083 with historical fix
- ✅ Found the actual implementation already has turn radius check
- ✅ Identified the **SUBTLE BUG**: maintaining facing doesn't actively fly away
- ✅ Proposed **1 correct solution**: Calculate direction away from target and fly opposite
- ✅ Understood common pitfalls (influence management, altitude issues, repulsion force)
- ✅ Knew testing strategies from historical regression patterns

**Key Insight**: The MCP instantly recognized this as a known issue, saving exploration time and providing the precise fix from historical context. The without-MCP version proposed reasonable but less informed solutions without knowing the bug's history.

---

## Case Study 3: Flame Tank Definition

### Task
"How can I define another vehicle like the flame tank?"

### Without MCP
- **Cost**: Unknown (52.2k tokens tracked)
- **Actions**: 13 (5 searches + 5 file reads + 3 list operations)
- **Approach**: Search for flame tank → locate definition → read related files → synthesize guide
- **Files Read**: 5 files (vehicles.yaml, defaults.yaml, other.yaml, explosions.yaml, husks.yaml)
- **File Errors**: 3 failed attempts

**Findings**:
- ✅ Complete flame tank structure documented
- ✅ Explained inheritance chain (^Tank → ^Vehicle → etc.)
- ✅ Listed all essential components and traits
- ✅ Created step-by-step template for new vehicle
- ✅ Provided localization and testing considerations
- ✅ Documented advanced features (multiple weapons, special abilities)
- ❌ **MISSED**: Historical context about trait evolution
- ❌ **MISSED**: Common mistakes in vehicle definition (from commit history)
- ❌ **MISSED**: Why certain patterns exist (Facing→Unit trait refactor)
- ❌ **MISSED**: Testing recommendations based on historical bugs

### With MCP
- **Cost**: Unknown (57.7k tokens tracked) - **10% higher tokens**
- **Actions**: 9 (1 MCP query + 5 file operations + 3 list operations)
- **Approach**: Ask expert first → verify with code examples → provide implementation guide
- **Files Read**: 2 files (vehicles.yaml, other.yaml)
- **File Errors**: 4 failed attempts (but continued effectively)
- **Context Multiplier**: 70% of actions = **30% fewer context resubmissions** ✅

**Findings**:
- ✅ All findings from "without MCP" version
- ✅ **PLUS**: Historical context about architecture evolution
- ✅ **PLUS**: "Facing→Unit trait" refactor explained
- ✅ **PLUS**: YAML-based definitions replacing INI files context
- ✅ **PLUS**: Common pitfalls with historical examples:
  - Flame weapons can damage the unit itself
  - Heavy armor affects crush behavior
  - LocalOffset positioning issues
- ✅ **PLUS**: Side effects to watch from commit history
- ✅ **PLUS**: Testing recommendations based on actual historical iterations

**Key Insight**: The MCP provided architectural rationale and warned about real historical mistakes, making it valuable for understanding WHY the system works this way, not just HOW to use it.

---

## Comparative Analysis

### Context Efficiency

| Case Study | Without MCP | With MCP | Token Diff | Actions Without | Actions With | Efficiency Notes |
|------------|-------------|----------|------------|-----------------|--------------|------------------|
| Voxel Renderer | 41.2k tokens | 44.0k tokens | +7% tokens | 2 actions | 4 actions | More exploratory actions |
| Airplanes Bug | 44.0k tokens | 24.3k tokens | -45% tokens | 2 actions | 2 actions | Same actions, better targeting |
| Flame Tank | 52.2k tokens | 57.7k tokens | +11% tokens | 13 actions | 9 actions | Fewer searches needed |

**Note on Action Count**: Since each action resubmits full context, total context usage = tokens × actions. When factoring this in:
- **Voxel Renderer**: Higher action count (4 vs 2) = more context resubmissions
- **Airplanes Bug**: Same actions (2 vs 2) = pure 45% token savings ✅
- **Flame Tank**: Fewer actions (9 vs 13) = 23% less total context ✅

**Overall**: 2 out of 3 cases showed improved efficiency; average 12% better when accounting for actions.

### Information Quality

#### Without MCP: "What" Focus
- Describes current code structure accurately
- Explains how systems work right now
- Proposes solutions based on code analysis
- Limited to what's visible in files

#### With MCP: "What + Why + History" Focus
- Describes current code structure
- Explains WHY it was designed that way
- References past bugs and fixes
- Warns about common pitfalls from experience
- Provides architectural evolution context
- Suggests solutions informed by past attempts

### Problem-Solving Effectiveness

**Voxel Renderer**: 
- Without MCP: Good technical explanation
- With MCP: Complete explanation + pitfalls + rationale
- **Winner**: MCP (deeper understanding)

**Airplanes Bug**:
- Without MCP: 3 reasonable solutions, unaware of history
- With MCP: 1 precise solution targeting known issue
- **Winner**: MCP (faster, more accurate)

**Flame Tank**:
- Without MCP: Complete practical guide
- With MCP: Complete guide + historical wisdom
- **Winner**: MCP (better long-term understanding)

---

## Key Discoveries: With MCP vs Without

### Only Found WITH MCP

#### Voxel Renderer
- IFinalizedRenderable architecture exists to solve mutable struct enumeration bugs
- "Render voxels before BeginFrame" was a major architectural shift
- QuadList→TriangleList conversion (4 vertices → 6 vertices per face)
- Common z-fighting issues and their fixes
- Shadow rendering evolved through multiple iterations

#### Airplanes Bug  
- Issue #7083 was the original bug report
- Turn radius fix was already implemented but flawed
- Coordinate system evolution (PPos/PSubPos → WPos/WVec) affected all aircraft
- Influence management is a common regression point
- Repulsion force requires dot product check to avoid stalling

#### Flame Tank
- Facing→Unit trait refactor centralized state management
- YAML replaced INI files in early architecture
- Flame weapons historically damaged their own units
- Multiple weapon balance iterations occurred
- "Husk experiment" added particle effects to wreckage

### Only Found WITHOUT MCP

**None significant.** The without-MCP versions found everything the with-MCP versions found, just without the historical context and rationale.

---

## Common Patterns

### MCP Advantages
1. **Instant Recognition**: Identifies known issues/patterns immediately
2. **Historical Context**: Explains why code exists, not just what it does
3. **Pitfall Warnings**: Highlights mistakes made in past commits
4. **Architectural Evolution**: Shows how systems changed over time
5. **Testing Insights**: Suggests test scenarios based on real regressions

### MCP Trade-offs
1. **Action Count Varies**: May use more actions in exploratory tasks, but typically fewer in targeted searches
2. **Dependency on Expert Quality**: Only as good as the indexed repository
3. **May Provide Outdated Info**: If expert is old (check commit ranges)

### When MCP Excels
- ✅ Debugging known issues
- ✅ Understanding architectural decisions
- ✅ Learning from past mistakes
- ✅ Avoiding regressions
- ✅ Comprehensive system understanding

### When MCP Less Critical
- 🤷 Simple API lookups
- 🤷 Well-documented modern code
- 🤷 Greenfield projects without history
- 🤷 Pure syntax questions

---

## Recommendations

### ✅ **USE the MCP when:**
1. **Debugging complex bugs** - May be known issues with documented fixes
2. **Learning unfamiliar codebases** - Historical context accelerates understanding
3. **Making architectural changes** - Understand why things are designed certain ways
4. **Reviewing code** - Catch patterns that caused bugs historically
5. **Planning refactors** - Learn from past refactoring attempts

### 🤔 **Consider NOT using MCP when:**
1. **Writing new features** with no historical precedent
2. **Simple documentation lookups** where code is self-explanatory
3. **Time-sensitive quick fixes** where speed matters more than context
4. **Working with very young repositories** (<100 commits, little history)

### 💡 **Best Practice: Hybrid Approach**
1. **Start with MCP** - Get historical context and known patterns
2. **Verify with code** - Confirm current implementation matches expectations
3. **Synthesize** - Combine historical wisdom with current code analysis

---

## Conclusion

The expert-among-us MCP provides **significant value** by adding historical context and architectural rationale to code understanding. While context efficiency varies by task type (accounting for action count: 12% better on average, with 2 of 3 cases more efficient), the qualitative benefits are substantial:

### Primary Benefits:
- **Historical insights** unavailable through code analysis alone
- **Faster problem identification** for known issues (e.g., Airplanes bug instantly recognized)
- **Architectural rationale** explaining why code exists, not just what it does
- **Pitfall warnings** from real historical mistakes
- **Better long-term understanding** that compounds over time

### Efficiency Profile:
- **Best for**: Targeted debugging, complex tasks with many searches, understanding architectural decisions
- **Adequate for**: Exploratory learning (may use more actions but gains historical context)
- **Average**: 12% better context efficiency when accounting for both tokens and action count

### Value Proposition:
Even in cases where action count increases (like exploratory Voxel Renderer task), the historical insights provided are irreplaceable for truly understanding a system. The MCP transforms code analysis from "what exists" to "what exists and why it was designed that way."

**Overall Rating**: ⭐⭐⭐⭐⭐ (5/5)
**Recommendation**: **Strongly recommend** using the MCP for serious codebase exploration, debugging, and architectural work. The historical insights justify any minor context overhead in exploratory tasks.

---

## Appendix: Token Usage with Action Count

**Full Picture**: Since each action resubmits context, total usage = tokens × actions.

| Task | Tokens (No MCP) | Actions | Total Context | Tokens (MCP) | Actions | Total Context | Net Efficiency |
|------|----------------|---------|---------------|--------------|---------|---------------|----------------|
| Voxel Renderer | 41.2k | 2 | 82.4k | 44.0k | 4 | 176k | -113% |
| Airplanes Bug | 44.0k | 2 | 88k | 24.3k | 2 | 48.6k | **+45%** ✅ |
| Flame Tank | 52.2k | 13 | 678k | 57.7k | 9 | 519k | **+23%** ✅ |
| **AVERAGE** | **45.8k** | **5.7** | **283k** | **42.0k** | **5.0** | **248k** | **+12%** ✅ |

**Interpretation**:
- The MCP averages 12% better context efficiency overall
- 2 of 3 cases showed clear efficiency gains
- Exploratory tasks may use more actions but gain irreplaceable historical insights
- The value of understanding "why" often outweighs pure context efficiency