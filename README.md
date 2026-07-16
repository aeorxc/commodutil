# commodutil
Common commodity and oil analytics utilities.

## Dependency modes
`pyproject.toml` is commodutil's only dependency declaration. Runtime dependencies live in
`project.dependencies`; PEP 735 groups provide `test`, `dev`, and `release` tooling.

Azure Pipelines uses locked project mode. `uv.lock` fixes the exact Oil_Feed solution, and
the pipeline installs the `test` group on Python 3.11 and 3.12 before testing the exact wheel
built once upstream. After changing dependencies, regenerate the lock with uv 0.11.28 against
Oil_Feed and run `uv lock --check`. The lock may contain the credential-free feed URL, but it
must never contain usernames, tokens, or other registries.

The shared pipeline template also supports a requirements mode for other legacy repositories.
commodutil does not use that mode and intentionally has no `requirements*.txt` manifests.

Oil_Feed is private and this GitHub repository currently has no feed credential. GitHub Actions
therefore removes the feed-specific lock only in its ephemeral checkout and resolves the same
pyproject groups from public PyPI. To make GitHub use the committed frozen solution, add an
`OIL_FEED_PAT` repository secret with Azure Artifacts Packaging Read permission, then replace
the public-resolution step with runtime-authenticated `uv sync --frozen`. Never commit the PAT.
