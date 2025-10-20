# Requirements Document: Parameter Processing Infrastructure

## Introduction

This specification defines the test infrastructure for pre and post processing of structured parameter data in TFMPE. The system needs to support hierarchical parameter structures with varying dimensionalities, functional parameters (time-varying or spatially-indexed data, referred to as "indices" in legacy code), masking for missing/padded data, and efficient batching within JIT-compiled functions. The design draws on patterns from `sfmpe_legacy` but creates a unified interface for parameter data that consolidates all metadata (masks, functional inputs, slice data) in one place, replacing the legacy approach of passing multiple separate objects.

**Terminology Note:** Throughout this document and the codebase, we use "functional parameters" and "functional inputs" to refer to auxiliary data that varies with the parameter structure (e.g., observation times, spatial coordinates). The legacy code refers to these as "indices" or "indexed parameters".

## Alignment with Package Vision

This feature directly supports the package's data-preprocessing goal: "Hands off processing of structured parameter flattening, functional-inputs, masking...". By establishing comprehensive tests for parameter processing, we ensure that:

1. **Sample Efficiency**: Proper parameter structuring enables efficient hierarchical model training
2. **Speed**: Validated flattening/restructuring operations ensure minimal overhead in JIT contexts
3. **Scalability**: Tested batching and masking strategies confirm low memory usage on large hierarchical problems

## Requirements

### Requirement 1: Unified Parameter Data Interface

**User Story:** As a researcher implementing hierarchical models, I want a single unified object that encapsulates all parameter data, metadata, and processing operations, so that I don't need to juggle multiple dictionaries, arrays, and slice objects.

#### Acceptance Criteria

1. WHEN creating parameter data THEN the system SHALL provide a single interface object that contains flattened data, labels, masks, functional inputs, and slice metadata
2. WHEN passing parameter data to functions THEN the system SHALL require only the unified interface object, not multiple separate arguments
3. WHEN accessing parameter components THEN the interface SHALL provide clear accessor methods for data, masks, functional inputs, and metadata
4. WHEN the interface is serialized THEN it SHALL preserve all metadata required for reconstruction
5. WHEN comparing with legacy code THEN the interface SHALL support all use cases from `hierarchical_gaussian.py`, `hierarchical_brownian.py`, and `seir.py`

### Requirement 2: Structured Parameter Flattening and Reconstruction

**User Story:** As a researcher implementing hierarchical models, I want to convert structured parameter dictionaries (PyTrees) to flat arrays and back through the unified interface, so that I can interface between hierarchical priors and flat neural network inputs.

#### Acceptance Criteria

1. WHEN a PyTree with nested parameters is flattened THEN the system SHALL produce a 2D array with shape `(batch, total_params)` where `total_params` is the sum of flattened parameter dimensions
2. WHEN a flattened array is reconstructed THEN the system SHALL recover the original PyTree structure with correct shapes and parameter names
3. WHEN parameters have different batch dimensions THEN the system SHALL pad to maximum batch size and track padding locations
4. WHEN parameters have event dimensions (e.g., `(n_sites, n_features)`) THEN the system SHALL flatten event dimensions while preserving batch dimensions
5. WHEN slice metadata is stored in the interface THEN reconstruction SHALL use it to split the flat array into original blocks

### Requirement 3: Functional Input Processing

**User Story:** As a researcher with functional parameters (e.g., observation times, spatial coordinates), I want to associate functional inputs with parameters and observations through the unified interface, so that my model can condition on these auxiliary data.

#### Acceptance Criteria

1. WHEN functional inputs are provided as a PyTree THEN the system SHALL flatten them consistently with parameter/observation structure and store them in the unified interface
2. WHEN functional inputs have different dimensions than parameters THEN the system SHALL pad with sentinel values (e.g., `-1e8`)
3. WHEN sampling with functional inputs THEN the system SHALL broadcast them to match sample batch dimensions
4. WHEN functional inputs are optional THEN the system SHALL support `None` values without errors
5. WHEN functional inputs vary per sample THEN the interface SHALL support storing them as part of batch data

### Requirement 4: Data Generation Strategies

**User Story:** As a researcher training models, I want to choose between memory-efficient data generators and static datasets, so that I can optimize for either non-jitted or jitted training loops.

#### Acceptance Criteria

1. WHEN using a data generator approach THEN the system SHALL provide an iterator that yields batches on-demand without pre-allocating full datasets
2. WHEN using static dataset generation THEN the system SHALL provide a function that creates the full dataset upfront for JIT compilation
3. WHEN switching between strategies THEN both SHALL produce equivalent outputs through the unified interface
4. WHEN using the generator in non-jitted loops THEN it SHALL minimize memory footprint by streaming data
5. WHEN using static datasets in jitted loops THEN all array shapes SHALL be known at compile time

### Requirement 5: Attention Mask Generation for Independence Structure

**User Story:** As a researcher specifying conditional independence, I want the unified interface to generate attention masks from independence declarations, so that my transformer can enforce structural constraints.

#### Acceptance Criteria

1. WHEN `local` independence is specified (e.g., `['theta', 'obs']`) THEN the interface SHALL create self-attention masks with zeros on the diagonal blocks
2. WHEN `cross` independence is specified (e.g., `[('mu', 'obs')]`) THEN the interface SHALL zero out cross-attention between those blocks
3. WHEN `cross_local` independence with functional input mapping is specified (e.g., `[('theta', 'obs', (0, 0))]`) THEN the interface SHALL enable attention only between matching indices along specified dimensions
4. WHEN no independence is specified THEN the interface SHALL create all-ones masks (full attention)
5. WHEN combining multiple independence rules THEN the interface SHALL apply all rules correctly

### Requirement 6: Padding Mask Generation for Variable-Length Data

**User Story:** As a researcher with variable-length sequences, I want the unified interface to generate padding masks, so that my model ignores padding during attention and aggregation.

#### Acceptance Criteria

1. WHEN parameters have different event shapes within a batch THEN the interface SHALL create padding masks with 1 for valid tokens and 0 for padding
2. WHEN padding masks are used with attention THEN padded tokens SHALL NOT contribute to attention computations
3. WHEN decoding flattened parameters THEN the interface SHALL remove padding based on stored slice metadata
4. WHEN masks are accessed THEN the interface SHALL provide them in formats compatible with transformer attention layers

### Requirement 7: Batching Support in JIT-Compiled Functions

**User Story:** As a researcher training models with JAX, I want parameter processing through the unified interface to work within JIT-compiled training loops, so that I can achieve maximum performance.

#### Acceptance Criteria

1. WHEN processing operations are JIT-compiled THEN they SHALL execute without tracer leaks or dynamic shape errors
2. WHEN batch dimensions vary THEN the system SHALL use static padding strategies compatible with JIT
3. WHEN vmap is applied THEN processing operations SHALL correctly vectorize over sample dimensions
4. WHEN the unified interface is passed to jitted functions THEN all contained arrays SHALL have static shapes

### Requirement 8: Label Assignment for Multi-Block Structures

**User Story:** As a researcher with multiple parameter types and observation types, I want the unified interface to assign labels to each token, so that my model can use learned embeddings per type.

#### Acceptance Criteria

1. WHEN parameters and observations are flattened THEN the interface SHALL assign integer labels to each token based on its block name
2. WHEN labels are broadcast THEN they SHALL match the shape of flattened data
3. WHEN combining datasets THEN the interface SHALL maintain consistent label assignments
4. WHEN accessing labels THEN the interface SHALL provide them aligned with the flattened data

## Non-Functional Requirements

### Performance

- Flattening and reconstruction operations MUST complete in O(total_params) time
- Mask generation MUST be computed once and reused across batches
- All operations MUST support JAX JIT compilation without fallback to Python
- Memory overhead for padding MUST NOT exceed 2x the actual parameter size
- Data generator approach MUST NOT pre-allocate arrays larger than batch_size

### Usability

- API MUST differ from legacy patterns by providing a unified interface instead of multiple separate objects
- Error messages MUST clearly indicate dimension mismatches or invalid independence specifications
- Documentation MUST include examples for common hierarchical structures from legacy examples
- Migration guide MUST explain how to convert from legacy multi-object approach to unified interface

### Maintainability

- Tests MUST cover all use cases from `sfmpe_legacy/examples/` (hierarchical_gaussian, hierarchical_brownian, seir)
- Tests MUST parametrize over different hierarchical structures (2-level, 3-level, etc.)
- Tests MUST verify both data generation strategies (generator and static)
- Code MUST follow project conventions (jaxtyping annotations, numpy-style docs, 80-char lines)
- Tests MUST run in <5 seconds (not marked as slow) for standard problem sizes
