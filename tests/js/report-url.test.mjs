// The pre-filled bug-report URL (notification centre → "Report on GitHub").
//
// Two classes of bug live here and neither is visible by eye:
//
//  1. GitHub issue-form dropdowns prefill only on an EXACT match with an option
//     string. "macOS" instead of "macOS (Apple Silicon)" is silently dropped --
//     the form opens with an empty OS field and nobody notices until a reporter
//     leaves it blank. These strings must track .github/ISSUE_TEMPLATE/
//     bug_report.yml.
//  2. The report must never carry the track title or source URL. That is a
//     product decision (issues are public), and a regression would leak it
//     quietly, once per report, forever.
//
// Run:  node tests/js/report-url.test.mjs

import {
  buildReportUrl,
  buildReportText,
  osOption,
  installOption,
  SAFE_URL_CHARS,
} from "../../static/js/notifications.js";

let pass = 0,
  fail = 0;
const check = (name, cond, detail = "") => {
  if (cond) {
    pass++;
    console.log(`PASS  ${name}`);
  } else {
    fail++;
    console.log(`FAIL  ${name}${detail ? "  -- " + detail : ""}`);
  }
};

const params = (url) => new URL(url).searchParams;
// Substring checks against the raw URL miss content URLSearchParams encoded
// (a space becomes "+", which plain decodeURIComponent does NOT reverse --
// that's an application/x-www-form-urlencoded convention) -- decode properly
// so a check for e.g. "line 399" actually means what it says.
const decoded = (url) => decodeURIComponent(url.replace(/\+/g, " "));

// The exact option strings from bug_report.yml. If the template changes, this
// list changes with it -- that is the point.
const OS_OPTIONS = ["macOS (Apple Silicon)", "macOS (Intel)", "Windows", "Linux", "Other"];
const INSTALL_OPTIONS = ["macOS DMG", "Windows ZIP", "Linux tar.gz", "From source", "Docker / self-hosted"];

// ─── dropdown mapping ───

for (const [target, expected] of [
  [{ os: "macos", arch: "arm64" }, "macOS (Apple Silicon)"],
  [{ os: "macos", arch: "x64" }, "macOS (Intel)"],
  [{ os: "windows", arch: "x64" }, "Windows"],
  [{ os: "linux", arch: "x64" }, "Linux"],
  [{ os: "freebsd", arch: "x64" }, "Other"],
  [null, "Other"],
]) {
  const got = osOption(target);
  check(`os: ${JSON.stringify(target)} -> ${expected}`, got === expected, got);
  check(`os option is one the form offers: ${got}`, OS_OPTIONS.includes(got));
}

for (const [target, desktop, expected] of [
  [{ os: "macos", arch: "arm64" }, true, "macOS DMG"],
  [{ os: "windows", arch: "x64" }, true, "Windows ZIP"],
  [{ os: "linux", arch: "x64" }, true, "Linux tar.gz"],
  [{ os: "freebsd", arch: "x64" }, true, "From source"],
  // No Tauri means the UI is served by a backend the user runs themselves.
  [{ os: "linux", arch: "x64" }, false, "Docker / self-hosted"],
]) {
  const got = installOption(target, desktop);
  check(`install: ${target.os}/${desktop ? "desktop" : "served"} -> ${expected}`, got === expected, got);
  check(`install option is one the form offers: ${got}`, INSTALL_OPTIONS.includes(got));
}

// ─── a realistic report ───

const RECORD = {
  kind: "import",
  message: "Audio processing failed. Please try another video.",
  detail: "out-of-memory — torch.OutOfMemoryError: CUDA out of memory.",
  cause: "out-of-memory",
  context: {
    jobId: "abcdefabcdef",
    stage: "Error: Processing failed",
    device: "cuda",
    gpuFallback: true,
    timings: '{"download": 4.2, "separate": 61.0}',
  },
  tail: ["torch.OutOfMemoryError: CUDA out of memory.", "Tried to allocate 2.40 GiB"],
};

const DIAG = {
  version: "0.9.1",
  model: "htdemucs_6s",
  ffmpegConfigured: true,
  buildTarget: { os: "windows", arch: "x64", gpu: "nvidia" },
  isDesktop: true,
};

const url = buildReportUrl(RECORD, DIAG);
const q = params(url);

check("targets the bug form", q.get("template") === "bug_report.yml");
check(
  "blank issues are disabled, so template must be present",
  url.startsWith("https://github.com/stemdeckapp/stemdeck/issues/new?"),
);
check("title leads with the cause", q.get("title") === "[Bug]: Import failed - out-of-memory");
check("version is prefixed with v", q.get("version") === "v0.9.1");
check("os matches the dropdown", q.get("os") === "Windows");
check("install matches the dropdown", q.get("install") === "Windows ZIP");
check("what carries StemDeck's own message", q.get("what").includes("Audio processing failed"));
check("what carries the classified detail", q.get("what").includes("out-of-memory"));
check("steps names the stage it died at", q.get("steps").includes("Error: Processing failed"));

// A short tail like this one fits comfortably inside the safe URL budget, so
// it's filled directly into the "Logs / screenshots" field -- no paste
// needed for the common case (see the length-ceiling section below for what
// happens when it doesn't fit).
check("extra carries the short stderr tail directly", decoded(url).includes("Tried to allocate 2.40 GiB"));
check(
  "extra is not just a clipboard pointer when the content already fit",
  !q.get("extra").startsWith("Copied to your clipboard"),
);

const text = buildReportText(RECORD, DIAG);
check("report text reports the device and the CPU fallback", text.includes("cuda (fell back to CPU)"));
check("report text reports the model", text.includes("htdemucs_6s"));
check("report text has no traceback section when none was fetched", !text.includes("Traceback:"));

// ─── traceback ───
//
// get_failure() now also returns the full backend traceback (already
// home-directory-redacted server-side), fetched into record.traceback the
// same way record.tail is. It must show up in the clipboard text, and -- when
// short enough, as it is here -- directly in the URL too, preferred over the
// stderr tail when both are present.

const WITH_TRACEBACK = {
  ...RECORD,
  traceback: [
    "Traceback (most recent call last):",
    '  File "<home>/AppData/Local/Programs/Python/Python312/Lib/asyncio/threads.py", line 25, in to_thread',
    "torch.OutOfMemoryError: CUDA out of memory.",
  ],
};
const tracebackText = buildReportText(WITH_TRACEBACK, DIAG);
check("report text carries the traceback", tracebackText.includes("CUDA out of memory."));
check("report text labels the traceback section", tracebackText.includes("Traceback:"));
check(
  "the traceback stays fenced as code, same as the stderr tail",
  tracebackText.split("```").length - 1 === 4,
  `${tracebackText.split("```").length - 1} fence markers`,
);
const tracebackUrl = buildReportUrl(WITH_TRACEBACK, DIAG);
const tracebackDecoded = decoded(tracebackUrl);
check("a short traceback is filled directly into the URL too", tracebackDecoded.includes("threads.py"));
check(
  "the traceback is preferred over the stderr tail when both are present",
  !tracebackDecoded.includes("Tried to allocate"),
);
check("report text carries the stderr tail", text.includes("Tried to allocate 2.40 GiB"));
check("tail is fenced so GitHub renders it as code", text.includes("```"));

// ─── opt-in "include recent logs" ───
//
// Unlike tail/traceback, record.logs is never fetched automatically -- only
// present once the user clicks the button in notifications.js. An empty
// string for a view (fetch failed, or nothing in the window) must not render
// an empty section.

const WITH_LOGS = {
  ...RECORD,
  logs: {
    backend: "2026-08-17 16:50:02 E stemdeck backend crashed",
    application: "",
    // setup intentionally absent -- same as "not fetched"
  },
};
const logsText = buildReportText(WITH_LOGS, DIAG);
check("report text carries the backend log when present", logsText.includes("backend crashed"));
check("report text labels which log a section came from", logsText.includes("Recent backend log:"));
check("an empty log view produces no section", !logsText.includes("Recent application log:"));
check("an unfetched log view produces no section", !logsText.includes("Recent setup log:"));
check("no logs section at all when record.logs was never set", !text.includes("Recent backend log:"));
const logsUrl = buildReportUrl(WITH_LOGS, DIAG);
check("recent logs never reach the URL, even when the rest of the content fit", !decoded(logsUrl).includes("crashed"));
check(
  "a clipboard note is added when logs were fetched, even though the short tail fit on its own",
  decoded(logsUrl).includes("clipboard"),
);

// preflight is a checkboxes field: GitHub cannot prefill it, and we must not
// pretend otherwise -- the user ticking it is the "I searched for duplicates"
// promise.
check("preflight is not faked", q.get("preflight") === null);

// ─── privacy ───

const PRIVATE = {
  ...RECORD,
  context: {
    ...RECORD.context,
    title: "Someone's Private Demo Take 3",
    sourceUrl: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  },
};
const privateUrl = buildReportUrl(PRIVATE, DIAG);
check("never carries a track title", !privateUrl.includes("Private%20Demo") && !privateUrl.includes("Private Demo"));
check("never carries a source URL", !privateUrl.includes("youtube.com") && !privateUrl.includes("dQw4w9WgXcQ"));

// ─── length ceiling ───
//
// Windows opens this URL via explorer.exe, which silently opens a plain File
// Explorer window instead of erroring past roughly 2000 characters -- the bug
// this file exists to guard against. A short trace is filled directly into
// the URL (see above); an oversized one must still keep the whole URL under
// that ceiling, filling in as much of the END of the tail as fits (that's
// where the actual error is) and pointing at the clipboard for the rest.

const HUGE = {
  ...RECORD,
  tail: Array.from({ length: 400 }, (_, i) => `line ${i}: ${"x".repeat(120)}`),
};
const hugeUrl = buildReportUrl(HUGE, DIAG);
check(
  "URL stays well under the platform ceiling regardless of tail size",
  hugeUrl.length <= SAFE_URL_CHARS,
  `${hugeUrl.length} chars (ceiling ${SAFE_URL_CHARS})`,
);
const hugeDecoded = decoded(hugeUrl);
check("truncation fills in as much of the END of the tail as fits", hugeDecoded.includes("line 399"));
check("truncation drops the START of an oversized tail", !hugeDecoded.includes("line 0:"));
check("truncation points at the clipboard for what didn't fit", hugeDecoded.includes("clipboard"));
const hugeText = buildReportText(HUGE, DIAG);
check("the clipboard text keeps the full, untruncated tail", hugeText.includes("line 399"));
check("the clipboard text keeps the start of the tail too", hugeText.includes("line 0:"));

// ─── degenerate input ───

const bare = buildReportUrl({ kind: "playback", message: "This track's audio could not be loaded." }, {});
check("survives no diagnostics at all", bare.includes("template=bug_report.yml"));
check("unknown OS falls back to Other", params(bare).get("os") === "Other");
check("no version is empty, not 'vundefined'", params(bare).get("version") === "");

console.log(`\n${pass}/${pass + fail} checks passed`);
process.exit(fail ? 1 : 0);
