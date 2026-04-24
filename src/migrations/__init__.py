"""One-off migration scripts for the proposal-realignment change.

Each migration is reversible-ish: original files are backed up to a
`.orig` sibling before mutation, original checksums are preserved in
the migrated payload as `orig_checksum` for audit trail.
"""
