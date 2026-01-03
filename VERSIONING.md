# Versioning Policy

## Semantic Versioning
The **x.titan** project follows [Semantic Versioning 2.0.0](https://semver.org/).

Given a version number **MAJOR.MINOR.PATCH**, increment the:
- **MAJOR** version when you make incompatible API changes,
- **MINOR** version when you add functionality in a backwards compatible manner, and
- **PATCH** version when you make backwards compatible bug fixes.

## Release Process
1. **Prepare**: Ensure all tests pass and documentation is updated.
2. **Tag**: Create a git tag following the `vX.Y.Z` format.
   ```bash
   git tag -a v1.0.0 -m "Release v1.0.0"
   git push origin v1.0.0
   ```
3. **Release**: The CI/CD pipeline will automatically create a GitHub Release when a tag is pushed.
4. **Changelog**: Maintain a clear changelog in the release notes.
