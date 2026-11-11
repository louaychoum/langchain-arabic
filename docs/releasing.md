# Releasing

Release preparation does not authorize publication. Do not push a version tag
until the maintainer approves the release: that push triggers PyPI publishing.

## One-time maintainer setup

Configure a PyPI Trusted Publisher for the existing `langchain-arabic` project:

- Owner: `louaychoum`
- Repository: `langchain-arabic`
- Workflow filename: `release.yml`
- Environment: `pypi`

Create the matching GitHub `pypi` environment and configure any desired release
reviewers. No PyPI token should be stored in the repository. This external account
configuration cannot be established by committing the workflow.

## Before tagging

1. Review the diff and changelog, and finish CI on the intended commit.
2. Confirm `_version.py` and the intended tag match.
3. Run the checks in CONTRIBUTING.md, then build/check fresh artifacts:

```bash
python -m build
python -m twine check --strict dist/*
```

Use a clean `dist/` for each version. Inspect wheel/sdist contents for `py.typed`,
license, correct version and source. Install the wheel into a fresh environment
and run an LCEL example outside the source tree.

## After explicit release approval

If the branch has not been pushed yet:

```bash
git push -u origin refs/heads/v0.3.1
```

After the intended release commit is approved and merged, fetch and inspect the
default branch. Tag the exact verified commit; do not use the ambiguous branch
and tag name `v0.3.1` as a release target:

```bash
git fetch origin master
git show --stat refs/remotes/origin/master
# Confirm this is the reviewed 0.4.0 commit and its CI has passed.
git tag -a v0.4.0 -m "Release 0.4.0" refs/remotes/origin/master
git push origin refs/tags/v0.4.0
```

The release workflow reruns CI, verifies tag/version agreement, builds a wheel and
sdist, checks their metadata and publishes those artifacts through PyPI OIDC.
Environment protection may require a maintainer approval before publishing.
Verify the workflow and actual PyPI artifacts after publication. Do not claim a
release is published merely because a tag push or upload command succeeded.

The historical `v0.2.0` and `v0.3.0` tags both point to `5edc2c3`, whose version
is 0.2.0. Leave them untouched. The changelog follows published artifact versions
and relevant commits rather than treating that tag range as a release diff.
