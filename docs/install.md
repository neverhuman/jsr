# Install jankurai

```sh
bash -o pipefail -c 'curl --proto "=https" --tlsv1.2 -fsSL https://raw.githubusercontent.com/neverhuman/jankurai/v1.7.0/jankurai-installer.sh | bash -s -- --tag v1.7.0'
export PATH="$HOME/.local/bin:$PATH"
jankurai init --yes
jankurai hooks install --yes
```

Pre-commit then runs `jankurai diff-audit --base-ref HEAD` on staged and worktree files. That is a developer check, not a README score. Run `jankurai audit .` (full tree) in CI or before merge for the SVG badge. Upgrade with `jankurai upgrade` (same signed installer).
