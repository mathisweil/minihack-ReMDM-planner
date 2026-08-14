#!/bin/sh
# PreToolUse hook: block automated edits to canonical experiment configs.
# final_* / *final* YAMLs under a configs/ directory define published
# scientific settings (see CLAUDE.md); hardware accommodations must not
# silently change them. Exit 2 blocks the tool call.
input=$(cat)
path=$(printf '%s' "$input" | sed -n 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
case "$path" in
  *configs/*final*.yml|*configs/*final*.yaml)
    printf 'Blocked: %s is a canonical experiment config; scientific settings must not be edited to fit local hardware. Ask the user for explicit approval (see CLAUDE.md).\n' "$path" >&2
    exit 2 ;;
esac
exit 0
