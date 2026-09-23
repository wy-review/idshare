# Licensing and attribution

Original IDShare contributions are licensed under Apache-2.0 (see `LICENSE`).
This does not replace the licenses of bundled third-party components.

## Preserved notices

- The TAAC runtime includes FuxiCTR-derived code. The license and component
  notices carried by the frozen package are preserved verbatim in
  `third_party/FuxiCTR-LICENSE` (Apache License 2.0 plus its bundled notices).
- Existing copyright/license headers in copied source files are retained.
- Frozen copies may include project-specific modifications. Per-file source
  and byte changes, including path sanitation, are recorded in
  `provenance/SOURCES.json`; original package hashes are in
  `provenance/SNAPSHOTS.json`. Internal source-tree locations are omitted from
  the anonymous release. `release_modified` identifies release-only changes.
- RankMixer, TokenMixer-Large, DIN, and AdamAR name architectural or algorithmic
  references. Their inclusion here does not assert that every implementation
  was copied from, or is endorsed by, the corresponding authors.
- TAAC and KuaiRand data are not bundled. Their owners' terms apply separately.

## Release Modifications

The TAAC integration adds shared identity lookup, quantization, and matched
regularization to the FuxiCTR training components. Public entry points relocate
data and output paths; the source index records changes from archived copies.
Unrelated model registrations and unused adapters are excluded. Third-party
copyright headers and the bundled FuxiCTR license are retained unchanged.

FuxiCTR upstream: https://github.com/reczoo/FuxiCTR
