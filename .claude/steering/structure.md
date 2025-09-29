# Project Structure

## Directory Organization

### Source Code (`tfmpe/`)
```
tfmpe/
├── metrics              # Metrics for analysis of parameter inference
├── estimators           # Parameter estimators
├── bijectors            # Bijectors for structured parameter sets
├── preprocessing        # Pipelines for processing datasets for use with the estimators
├── sampling             # Sampling algorithms for training estimators
└── nn                   # Neural networkss
    └── transformer      # Transformer model
```

### Generated Structure (`.claude/`)
```
.claude/
├── steering/          # Persistent package context
│   ├── package.md     # Package vision
│   ├── tech.md        # Technical standards
│   └── structure.md   # Project conventions
├── specs/             # Feature specifications
│   └── {feature}/     # Per-feature directory
│       ├── requirements.md
│       ├── design.md
│       └── tasks.md
└── templates/         # Document templates
```

## Import Patterns

  * Always use named dependencies
  * Break out long imports onto multiple lines with parentheses
  * Imports are always at the top of the file

### Standard Order

  1. Built-in
  2. External dependencies (e.g. jax, flax, matplotlib,...)
  3. Internal modules
  4. Types
  5. Constants

## Testing

  - `pytest` is used for testing
  - Test files (in `test/`) match source file structure
  - Test names explaining expected behaviour behavior
  - Use pytest.mark.parameterize decorator for input testing
  - Mark slow tests as "slow"

### Benchmarking

  - `pytest-benchmark` is used for benchmarking
  - Benchmarks are in `test/benchmark`
  - Benchmarks are marked as `speed` (for standard datasets) and `scale` (for large datasets)
  - Estimator sampling and estimator training must have `speed` and `scale` benchmarks

## Configuration

### Root Level
  - `pyproject.toml`: Project metadata
  - `.gitignore`: Version control exclusions
  - `mkdocs.yml`: Documentation config
  - `env`: Virtual environment

## Documentation
  - README and LICENSE
  - NEWS.md
  - `docs/` MkDocs markdown
