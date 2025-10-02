# Implementation Plan: Parameter Processing Infrastructure

## Task Overview

This implementation creates a unified `Tokens` interface for TFMPE that replaces the legacy multi-object approach for parameter processing. The implementation follows a test-driven development approach: write tests first, then implement functionality to pass those tests.

The work is organized into three phases:
1. **Core Infrastructure** (Tasks 1-3): Low-level flattening, masking, and reconstruction utilities with tests
2. **Unified Interface** (Tasks 4-7): `Tokens` and `TokenView` classes with comprehensive tests
3. **Data Generation & Integration** (Tasks 8-9): Generator, static dataset, and integration tests

## Steering Document Compliance

**Project Structure (structure.md)**:
- All source code goes in `tfmpe/preprocessing/`
- All tests go in `test/test_preprocessing/`
- Tests mirror source file organization
- Update `__init__.py` files to expose public API

**Technical Standards (tech.md)**:
- Use `jaxtyping.Array` for type annotations
- Numpy-style docstrings for all public functions
- 80-character line limit
- Test with pytest, parametrize where possible
- Mark slow tests (>5s) with `@pytest.mark.slow`

## Atomic Task Requirements

Each task:
- Touches 1-3 files maximum
- Completable in 15-30 minutes
- Has one clear testable outcome
- Specifies exact file paths
- References specific requirements and leverage points

## Tasks

### Phase 1: Core Infrastructure

- [x] 1.0 Implement flattening utilities with tests
  - Purpose: Create and test PyTree flattening to unified token arrays
  - _Requirements: 2.1, 2.2_

  - [x] 1.1 Create flattening utility tests in test/test_preprocessing/test_flatten.py
    - File: test/test_preprocessing/test_flatten.py
    - Test `flatten_pytree()` with simple PyTree (2-3 keys)
    - Test padding with different batch dimensions
    - Test round-trip: flatten � reconstruct using slices
    - Test `update_flat_array()` updates correct offsets
    - Use `@pytest.mark.parametrize` for different structures
    - _Leverage: sfmpe_legacy/examples/hierarchical_gaussian.py (lines 48-52)_
    - _Requirements: 2.1, 2.2_

  - [x] 1.2 Create flattening utilities in tfmpe/preprocessing/flatten.py
    - File: tfmpe/preprocessing/flatten.py
    - Port `flatten_blocks()` from legacy, rename to `flatten_pytree()`
    - Single unified flattening (not separate theta/y)
    - Return tuple of (flat_array, slices_dict)
    - Add `update_flat_array()` for value override support
    - Add type annotations using `jaxtyping.Array`
    - Update tfmpe/preprocessing/__init__.py to export functions
    - _Leverage: sfmpe_legacy/sfmpe/util/dataloader.py (lines 140-223)_
    - _Requirements: 2.1, 2.2_

- [x] 2.0 Implement reconstruction utilities with tests
  - Purpose: Create and test PyTree reconstruction from flat arrays
  - _Requirements: 2.2, 2.5_

  - [x] 2.1 Create reconstruction tests in test/test_preprocessing/test_reconstruct.py
    - File: test/test_preprocessing/test_reconstruct.py
    - Test full PyTree reconstruction
    - Test selective key reconstruction with `decode_pytree_keys()`
    - Test shape preservation with different batch dims
    - Verify error on invalid slice metadata
    - Add fixtures to test/conftest.py as needed
    - _Leverage: test/conftest.py for fixtures_
    - _Requirements: 2.2, 2.5_

  - [x] 2.2 Create reconstruction utilities in tfmpe/preprocessing/reconstruct.py
    - File: tfmpe/preprocessing/reconstruct.py
    - Port `decode_theta()` from legacy, rename to `decode_pytree()`
    - Add `decode_pytree_keys()` for selective key reconstruction
    - Support optional `keys` parameter to decode subset
    - Add type annotations
    - Update tfmpe/preprocessing/__init__.py to export functions
    - _Leverage: sfmpe_legacy/sfmpe/util/dataloader.py (lines 570-609)_
    - _Requirements: 2.2, 2.5_

- [x] 3.0 Implement mask generation utilities with tests
  - Purpose: Create and test attention/padding mask generation
  - _Requirements: 3.1-3.5, 4.1-4.3_

  - [x] 3.1 Create mask generation tests in test/test_preprocessing/test_masks.py
    - File: test/test_preprocessing/test_masks.py
    - Test self-attention mask with `local` independence
    - Test cross-attention mask with `cross` independence
    - Test cross-local mask with functional input mapping
    - Test padding mask with variable event shapes
    - Test subset masking with `selected_keys` parameter
    - Add hierarchical_gaussian independence spec fixture to test/conftest.py
    - _Leverage: sfmpe_legacy/examples/hierarchical_gaussian.py (lines 48-52)_
    - _Requirements: 3.1-3.5, 4.1-4.3_

  - [x] 3.2 Create mask generation utilities in tfmpe/preprocessing/masks.py
    - File: tfmpe/preprocessing/masks.py
    - Port `build_self_attention_mask()` from legacy
    - Port `build_cross_attention_mask()` from legacy
    - Port `build_padding_mask()` from legacy
    - Add `selected_keys` parameter to all functions for subset masking
    - Add type annotations
    - Update tfmpe/preprocessing/__init__.py to export functions
    - _Leverage: sfmpe_legacy/sfmpe/util/dataloader.py (lines 225-432)_
    - _Requirements: 3.1-3.5, 4.1-4.3_

- [x] 4.0 Implement functional input utilities with tests
  - Purpose: Create and test functional input flattening
  - _Requirements: 3.1-3.4_

  - [x] 4.1 Create functional input tests in test/test_preprocessing/test_functional_inputs.py
    - File: test/test_preprocessing/test_functional_inputs.py
    - Test flattening with matching shapes
    - Test padding with sentinel value (-1e8)
    - Test `None` input returns `None`
    - Test alignment with token slices
    - Add fixtures to test/conftest.py as needed
    - _Leverage: sfmpe_legacy/examples/hierarchical_brownian.py (lines 104-112)_
    - _Requirements: 3.1-3.4_

  - [x] 4.2 Create functional input utilities in tfmpe/preprocessing/functional_inputs.py
    - File: tfmpe/preprocessing/functional_inputs.py
    - Port `_flatten_index()` from legacy, rename to `flatten_functional_inputs()`
    - Adapt to work with unified slices dict (not separate theta/y)
    - Support `None` functional inputs (return None)
    - Add type annotations
    - Update tfmpe/preprocessing/__init__.py to export functions
    - _Leverage: sfmpe_legacy/sfmpe/util/dataloader.py (lines 632-647)_
    - _Requirements: 3.1-3.4_

### Phase 2: Unified Interface

- [x] 5.0 Implement Tokens class basic functionality with tests
  - Purpose: Create core Tokens class with creation and decoding
  - _Requirements: 1.1-1.5, 2.1-2.5_

  - [x] 5.1 Create Tokens basic tests in test/test_preprocessing/test_tokens_basic.py
    - File: test/test_preprocessing/test_tokens_basic.py
    - Test `from_pytree()` with simple hierarchical structure
    - Test all fields populated correctly (data, labels, masks, slices)
    - Test `decode()` round-trip
    - Test `decode_keys()` with subset of keys
    - Use parametrize for 2-level vs 3-level structures
    - Add fixtures to test/conftest.py as needed
    - _Leverage: test/conftest.py for fixtures_
    - _Requirements: 1.1-1.5, 2.1-2.5_

  - [x] 5.2 Create Tokens class in tfmpe/preprocessing/tokens.py
    - File: tfmpe/preprocessing/tokens.py
    - Define `Tokens` dataclass with all fields from design
    - Implement `from_pytree()` classmethod
    - Implement `decode()` and `decode_keys()` methods
    - Add docstrings with numpy style
    - Update tfmpe/preprocessing/__init__.py to export Tokens
    - _Leverage: tfmpe/preprocessing/flatten.py, tfmpe/preprocessing/reconstruct.py, tfmpe/preprocessing/masks.py_
    - _Requirements: 1.1-1.5, 2.1-2.5_

- [ ] 6.0 Implement Tokens dynamic functionality with tests
  - Purpose: Add dynamic slicing and value override methods
  - _Requirements: 1.1-1.3, 3.1-3.5_

  - [ ] 6.1 Create Tokens dynamic tests in test/test_preprocessing/test_tokens_dynamic.py
    - File: test/test_preprocessing/test_tokens_dynamic.py
    - Test `select_tokens()` returns correct subset
    - Test `TokenView` properties are lazily evaluated
    - Verify no data copying (check array base or memory address)
    - Test `cross_attention_mask()` with different key sets
    - Test `with_values()` updates correct offsets
    - Test `with_values()` round-trip: override � decode � verify
    - Add fixtures to test/conftest.py as needed
    - _Leverage: test/conftest.py for fixtures_
    - _Requirements: 1.1-1.3, 3.1-3.5_

  - [ ] 6.2 Add dynamic methods to Tokens class in tfmpe/preprocessing/tokens.py
    - File: tfmpe/preprocessing/tokens.py (extend from task 5.2)
    - Implement `select_tokens()` method returning `TokenView`
    - Implement `cross_attention_mask()` method
    - Implement `with_values()` method for value override
    - _Leverage: tfmpe/preprocessing/masks.py, tfmpe/preprocessing/flatten.py_
    - _Requirements: 1.1-1.3, 3.1-3.5_

- [ ] 7.0 Implement TokenView class with tests
  - Purpose: Create zero-copy view into Tokens
  - _Requirements: 1.1-1.3_

  - [ ] 7.1 Add TokenView tests to test/test_preprocessing/test_tokens_dynamic.py
    - File: test/test_preprocessing/test_tokens_dynamic.py (extend from task 6.1)
    - Test all TokenView properties (data, labels, masks, slices)
    - Test lazy evaluation (properties not computed until accessed)
    - Test caching (accessing twice doesn't recompute)
    - Verify slices are re-indexed to start at 0
    - _Leverage: existing fixtures from task 6.1_
    - _Requirements: 1.1-1.3_

  - [ ] 7.2 Create TokenView class in tfmpe/preprocessing/token_view.py
    - File: tfmpe/preprocessing/token_view.py
    - Define `TokenView` dataclass
    - Implement all properties (data, labels, masks, etc.) with lazy evaluation
    - Add caching for computed properties using functools or manual flags
    - Ensure no data copying (only index computation)
    - Update tfmpe/preprocessing/__init__.py to export TokenView
    - _Leverage: tfmpe/preprocessing/tokens.py_
    - _Requirements: 1.1-1.3_

### Phase 3: Data Generation & Integration

- [ ] 8.0 Implement data generation with tests
  - Purpose: Create generator and static dataset functions
  - _Requirements: 4.1-4.5, 5.1-5.4_

  - [ ] 8.1 Create data generation tests in test/test_preprocessing/test_generation.py
    - File: test/test_preprocessing/test_generation.py
    - Test `TokenGenerator` yields correct number of batches
    - Test batch shapes are consistent
    - Test `generate_static_dataset()` creates static shapes
    - Test equivalence between generator and static for same seed
    - Verify generator uses less memory than static (qualitative check)
    - Add mock prior_fn and simulator_fn fixtures to test/conftest.py
    - _Leverage: test/conftest.py for mock prior/simulator_
    - _Requirements: 4.1-4.5, 5.1-5.4_

  - [ ] 8.2 Create TokenGenerator class in tfmpe/preprocessing/generator.py
    - File: tfmpe/preprocessing/generator.py
    - Implement `TokenGenerator` with `__init__`, `__iter__`, `__len__`
    - Generate batches on-demand (no pre-allocation)
    - Yield `Tokens` objects
    - Add type annotations
    - Update tfmpe/preprocessing/__init__.py to export TokenGenerator
    - _Leverage: tfmpe/preprocessing/tokens.py_
    - _Requirements: 4.1-4.5_

  - [ ] 8.3 Create static dataset function in tfmpe/preprocessing/dataset.py
    - File: tfmpe/preprocessing/dataset.py
    - Implement `generate_static_dataset()` function
    - Use `vmap` for batch generation
    - Return single `Tokens` object with static shapes
    - Add type annotations and docstring
    - Update tfmpe/preprocessing/__init__.py to export function
    - _Leverage: tfmpe/preprocessing/tokens.py_
    - _Requirements: 4.1-4.5, 5.1-5.4_

- [ ] 9.0 Create integration and JIT tests
  - Purpose: Validate legacy compatibility and JAX JIT compliance
  - _Requirements: All requirements (validation)_

  - [ ] 9.1 Create integration tests in test/test_preprocessing/test_integration.py
    - File: test/test_preprocessing/test_integration.py
    - Test hierarchical_gaussian use case (load spec, create Tokens, verify masks match legacy)
    - Test hierarchical_brownian use case with functional inputs
    - Test bottom_up.py use case 1: key swapping
    - Test bottom_up.py use case 2: selective key extraction
    - Test bottom_up.py use case 3: value override
    - Test bottom_up.py use case 4: dynamic restructuring
    - Mark as `@pytest.mark.slow` if tests take >5 seconds
    - Add fixtures to test/conftest.py as needed
    - _Leverage: sfmpe_legacy/examples/, test/conftest.py_
    - _Requirements: 1.1-1.5 (all requirements validated)_

  - [ ] 9.2 Create JIT compatibility tests in test/test_preprocessing/test_jit.py
    - File: test/test_preprocessing/test_jit.py
    - JIT-compile function that processes `Tokens`
    - Verify no tracer leaks or dynamic shape errors
    - Test with `vmap` over batch dimension
    - Test that all array accesses use static shapes
    - Add fixtures to test/conftest.py as needed
    - _Leverage: tfmpe/preprocessing/tokens.py_
    - _Requirements: 5.1-5.4_

## Notes

- **TDD Approach**: Write tests before implementation for each component
- **Incremental Development**: Each phase builds on the previous
- **Legacy Compatibility**: Integration tests verify equivalence with legacy behavior
- **Performance**: Generator approach enables memory-efficient training
- **Flexibility**: `select_tokens()` and `with_values()` enable bottom_up.py patterns
- **Fixtures**: Test fixtures added to `test/conftest.py` incrementally as needed
- **Exports**: Each implementation task updates `tfmpe/preprocessing/__init__.py`
