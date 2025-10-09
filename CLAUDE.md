# CLAUDE.md

# Development Commands

- `python -m pytest test/` - Run test suite
- `pyright` - Type checking
- `ruff check` - Linting

The virtualenv is in the `env` directory at the root of the repo

## Testing

Tests use pytest and are located in `test/`.

Tests use custom markers to categorize different types of tests:

- **Default tests**: Core functionality tests that run quickly and should always pass
- **Slow tests**: Tests that take longer to run (marked with `@pytest.mark.slow`)

By default, slow and tests are excluded to keep the development feedback loop fast. Use these commands to run specific test categories:

```bash
# Run all tests including slow ones
python -m pytest test/ -m "slow or not slow"

# Run only slow tests
python -m pytest test/ -m "slow"

# Run all tests (including slow)
python -m pytest test/ -m ""
```

## Experimentation

I will ask you to make debugging code to log outputs or make plots. Experiments are throwaway code, do not save code to disk, just run the code and have the outputs saved to disk.

## Coding style

 * Please add type annotations to your function signatures
 * Use jax types from `jaxtyping`. e.g. `Array` not `jnp.ndarray`
 * Use numpy style function documentation

## Planning

 * Please write tests *before* you make implementations. As in TDD
 * Once I have confirmed the tests, do not change them.
 * Stop implementing if you come up against an inconsistency
