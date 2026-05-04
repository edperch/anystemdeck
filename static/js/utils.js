export function fmtTime(s) {
  if (!isFinite(s) || s < 0) return "00:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60).toString().padStart(2, "0");
  return `${m.toString().padStart(2, "0")}:${sec}`;
}

// Ruler tick: M:SS with no leading zero on the minutes digit ("0:30", "1:00", "12:30").
export function fmtTickLabel(s) {
  if (!isFinite(s) || s < 0) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60).toString().padStart(2, "0");
  return `${m}:${sec}`;
}

export const $ = (id) => document.getElementById(id);