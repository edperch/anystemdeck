// Small UI-chrome handlers extracted from inline index.html scripts / onclick
// attributes so the Content-Security-Policy can forbid inline script (#171).
// Loaded as a module (deferred), so the DOM is parsed before this runs.

import { storeGet } from "./utils.js";
import { getBuildTarget } from "./catalog.js";
import { t, onLanguageChange } from "./i18n.js";

// Upload button → trigger the hidden file input.
document.getElementById("uploadFileBtn")?.addEventListener("click", () => {
  document.getElementById("fileInput")?.click();
});

// Notification panel: toggle / close / close-on-outside-click.
const notifBtn = document.getElementById("notifBtn");
const notifWrap = notifBtn?.closest(".daw-notif-wrap");

function setNotifOpen(open) {
  notifWrap?.classList.toggle("open", open);
  notifBtn?.setAttribute("aria-expanded", String(open));
}

notifBtn?.addEventListener("click", () => {
  setNotifOpen(!notifWrap?.classList.contains("open"));
});

document
  .querySelector(".daw-notif-close")
  ?.addEventListener("click", () => setNotifOpen(false));

document.addEventListener("click", (e) => {
  if (notifWrap?.classList.contains("open") && !notifWrap.contains(e.target)) {
    setNotifOpen(false);
  }
});

// Persistent GPU/acceleration status: a dot on the Settings rail button
// (green once a real GPU -- native CUDA or AMD-via-WSL2+ROCm alike -- is
// confirmed, grey for CPU) plus a fuller device-name line pinned at the top
// of the notification panel. Both read the same live source of truth the
// Settings device dropdown itself uses -- GET /api/settings's
// demucs_device_resolved -- so this always matches what will actually run,
// not whatever onboarding decided at launch (which can go stale the moment
// someone changes the device in Settings, with no restart involved).
//
// On desktop, demucs_device_resolved alone can't tell a real NVIDIA card
// apart from AMD-via-WSL2+ROCm -- both report "cuda" once ROCm is presenting
// to PyTorch as plain torch.cuda -- so disambiguating needs the same
// wsl2BackendEnabled Tauri-store flag the Settings toggle itself reads (see
// wireWsl2Setting in catalog.js). getBuildTarget()'s browser/server fallback
// means this is skipped harmlessly outside Windows desktop.
const settingsBtn = document.getElementById("settingsBtn");
const gpuStatusDot = document.getElementById("gpuStatusDot");
const notifDevice = document.getElementById("notifDevice");
const notifDeviceLabel = document.getElementById("notifDeviceLabel");

function gpuStatusLabel(device, wsl2Enabled) {
  switch (device) {
    case "cuda":
      return t("gpu.status.accelerated", {
        device: wsl2Enabled ? "AMD GPU (WSL2 + ROCm)" : "NVIDIA GPU (CUDA)",
      });
    case "mps":
      return t("gpu.status.accelerated", { device: "Apple Silicon (MPS)" });
    case "dml":
      return t("gpu.status.accelerated", {
        device: "DirectML (AMD/Intel/NVIDIA)",
      });
    default:
      return t("gpu.status.cpu");
  }
}

async function refreshGpuStatus() {
  if (!settingsBtn) return;
  try {
    const [settingsRes, target] = await Promise.all([
      fetch("/api/settings", { cache: "no-store" }),
      getBuildTarget(),
    ]);
    if (!settingsRes.ok) return;
    const settings = await settingsRes.json();
    const device = settings.demucs_device_resolved;
    const wsl2Enabled =
      target.os === "windows" && window.__TAURI__?.core?.invoke
        ? (await storeGet("wsl2BackendEnabled", false)) === true
        : false;
    const accelerated = Boolean(device) && device !== "cpu";
    const label = gpuStatusLabel(device, wsl2Enabled);

    settingsBtn.title = label;
    // Always shown, grey (default) for CPU, green once a real GPU -- native
    // CUDA or AMD-via-WSL2+ROCm alike -- is confirmed.
    gpuStatusDot?.classList.toggle("gpu-status-dot-accelerated", accelerated);
    if (notifDevice && notifDeviceLabel) {
      notifDeviceLabel.textContent = label;
      notifDevice.classList.toggle("daw-notif-device-accelerated", accelerated);
      notifDevice.classList.remove("hidden");
    }
  } catch (e) {
    console.warn("[ui-chrome] GPU status refresh failed:", e);
  }
}

refreshGpuStatus();
// The device select in Settings applies live with no restart, and the panel
// re-checks each time it opens rather than polling, so a change made a
// moment ago is never stale by the time someone actually looks.
notifBtn?.addEventListener("click", () => {
  if (!notifWrap?.classList.contains("open")) refreshGpuStatus();
});
onLanguageChange(() => refreshGpuStatus());
