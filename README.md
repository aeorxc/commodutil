# commodutil
Common commodity and oil analytics utilities.

## Dependency modes
Azure Pipelines uses locked project mode: `pyproject.toml` declares dependencies and `uv.lock` fixes the exact CI solution.
The pipeline installs the PEP 735 `test` group on Python 3.11 and 3.12, then tests the exact wheel built once upstream.
Requirements mode installs `requirements*.txt` directly and remains only as a legacy escape hatch for older consumers.
The requirements files stay aligned with the project declarations but are not used by this repository's Azure pipeline.
After changing dependencies, regenerate with uv 0.11.28 against Oil_Feed and run `uv lock --check`.
The lock may contain the credential-free Oil_Feed URL; never commit usernames, tokens, or other registries.
