AnyStemDeck for macOS
==================

Install:

1. Open the AnyStemDeck DMG.
2. Drag AnyStemDeck.app to Applications.
3. Open AnyStemDeck from Applications.

First launch:

- AnyStemDeck is a thin native app. It downloads a pinned, checksummed AnyStemDeck
  runtime pack on first launch.
- The runtime installs to:
  ~/Library/Application Support/AnyStemDeck/runtime
- FFmpeg and ffprobe install to:
  ~/Library/Application Support/AnyStemDeck/ffmpeg
- Demucs model weights download on first use and are cached under:
  ~/Library/Application Support/AnyStemDeck/models

Uninstall:

1. Delete /Applications/AnyStemDeck.app.
2. To remove runtime files, jobs, caches, models, and logs, delete:
   ~/Library/Application Support/AnyStemDeck

Notes:

- Internet access is required for first-run setup.
- Public releases should be signed and notarized.
- Unsigned local builds are for development and internal testing only.
