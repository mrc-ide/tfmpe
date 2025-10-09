# Technology Stack

## Project Type

Scientific computing package

## Core Technologies

### Primary Language(s)

  - **Language**: Python 3
  - **Language-specific tools**: pip

### Key Dependencies/Libraries

  - **jax**: Numerical computing
  - **flax**: Neural networks
  - **matplotlib**: plotting
  - **seaborn**: plotting
  - **hydra**: experiment configuration

## Development Environment

### Build & Development Tools

  - **Build System**: pip
  - **Virtual environment**: `ROOT/env`

### Code Quality Tools

  - **Static Analysis**: pyright, ruff
  - **Testing Framework**: pytest
  - **Documentation**: mkdocs

### Version Control & Collaboration
  - **VCS**: git
  - **Branching Strategy**: git flow
  - **Agent concurrency**: git worktree

## Technical Requirements & Constraints

### Performance Requirements

  - All not `slow` tests pass
  - No `pyright` issues
  - No `ruff check` issues
  - No degredation in benchmarks
  - No dead or duplicated code
