# Changelog

<!-- scriv-insert-here -->

<a id='changelog-0.2.1'></a>
# 0.2.1 — 2026-11-20

## Fixed

- Fix compatibility with inline-snapshot 0.37.0 after its internal API changed by
  using the public `inline_snapshot.is_snapshot` API to detect snapshots.
  Requires inline-snapshot 0.37.0 or newer.

<a id='changelog-0.2.0'></a>
# 0.2.0 — 2026-10-27

## Added

- Added support for storing NumPy arrays in external `.npy` snapshots, with
  readable array previews and colorized text diffs.
