# Contributing to AnyStemDeck

Thanks for your interest in AnyStemDeck - free, local stem separation for musicians. Contributions of
all kinds are welcome, whether you write code or not.

By participating you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Report a bug** - open a [Bug report](https://github.com/edperch/anystemdeck/issues/new/choose).
  Include your OS, the AnyStemDeck version (Help icon -> About), and clear steps to reproduce.
- **Suggest a feature** - open a [Feature request](https://github.com/edperch/anystemdeck/issues/new/choose).
- **Improve the docs** - fixes and clarifications to the README or these guides are always useful.
- **Write code** - bug fixes and features. For anything large, please open an issue or a
  [Discussion](https://github.com/edperch/anystemdeck/discussions) first so we can agree on the
  approach before you invest time.
- **Found a security issue?** Do not open a public issue - see [SECURITY.md](SECURITY.md).

## Project layout

- `app/` - FastAPI backend (the audio pipeline, job registry, and HTTP API).
- `static/` - the web UI: plain ES-module JavaScript, HTML, and CSS. No build step, no bundler.
- `desktop/` - the Tauri v2 desktop shell (Rust) that wraps the backend + UI.
- `scripts/` - packaging scripts (macOS app/dmg, Windows portable, runtime pack).
- `tests/` - the pytest suite for the backend.

## Development setup

You need [`uv`](https://docs.astral.sh/uv/) and `ffmpeg` on your PATH.

```bash
uv sync --python 3.12        # install the Python environment
./run.sh start               # start the backend at http://localhost:8000
```

Other helpers:

```bash
./run.sh restart             # restart after backend changes
./run.sh stop                # stop the server
./run.sh status              # is it running?
```

Open `http://localhost:8000` in your browser to use the app. Frontend changes are picked up on
reload (no build step).

### Checkout inside a cloud-sync folder (OneDrive, Dropbox, Google Drive...)

If your clone lives inside a folder one of these tools syncs, `dist/`, `.venv/`,
`desktop/src-tauri/target/`, `desktop/src-tauri/gen/`, and `desktop/node_modules/` all get
synced too -- these tools sync whatever's physically on disk, `.gitignore` doesn't apply to
them. That's thousands of files and multiple GB of churn for folders that are pure build
output, and it doesn't stop there -- `.git/` itself lives inside the same sync scope, so any
large git operation (a rebase, `git gc`, cloning the full history) is exposed to the same
churn. NTFS junctions redirecting just the build-output folders were tried first and didn't
hold up -- the cloud-sync client followed the junctions and synced through them anyway, at
least on the version tested. The reliable fix is to keep your clone outside any cloud-sync
folder entirely; there's no per-folder workaround that's actually held up in practice.

## Before you open a pull request

Please run the same checks CI runs:

```bash
uv run ruff check app/ tests/        # lint
uv run ruff format --check app/ tests/   # formatting
node --check static/js/<changed>.js  # syntax-check any JS you touched
uv run pytest tests/ -q              # backend tests
```

For Rust changes in `desktop/`, also run `cargo fmt --check` and `cargo clippy -- -D warnings`.

## Pull request guidelines

- Branch off `main`.
- Use [Conventional Commits](https://www.conventionalcommits.org/) for messages
  (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `ci:`).
- Open the PR as a **draft** until it is ready for review.
- Keep PRs focused - one logical change per PR makes review faster.
- Describe what changed and why, and how you tested it.

## Tests

New API endpoints and pipeline stages should come with tests under `tests/`. The suite uses
`pytest` with `httpx.AsyncClient`; see the existing `tests/test_*.py` files for patterns.

## License

AnyStemDeck is licensed under the [Apache License 2.0](LICENSE). By contributing, you agree that your
contributions are licensed under the same terms.

## Questions

Open a [Discussion](https://github.com/edperch/anystemdeck/discussions) or join the
[Discord](https://discord.gg/YhCKsjhcwB). Thanks for helping make AnyStemDeck better.
