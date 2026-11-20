#!/usr/bin/env bash
# Push the current branch. The default gh account lacks `workflow` scope, so use the account that has it.
set -euo pipefail
tok=$(gh auth token -u "${JSR_PUSH_ACCOUNT:-jepsontaylor}")
hdr="AUTHORIZATION: basic $(printf 'x-access-token:%s' "$tok" | base64 -w0)"
git -c credential.helper= -c "http.extraheader=$hdr" push "$@"
