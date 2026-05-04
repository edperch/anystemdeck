export const STEM_NAMES = ["vocals", "drums", "bass", "guitar", "piano", "other"];

// All track names the studio knows about, including the synthetic
// "original" track (the full song, served alongside the extracted
// stems whenever the user picked a strict subset). Used for mixer
// column / mixer state / VU iteration. The import-page stem selector
// still uses STEM_NAMES because "original" isn't a separable stem.
export const TRACK_NAMES = ["original", ...STEM_NAMES];

export const STEM_DISPLAY = {
  vocals: "Vocals",
  drums: "Drums",
  bass: "Bass",
  guitar: "Guitar",
  piano: "Piano",
  other: "Other",
  original: "Original",
};

// FL Studio-style channel palette: saturated but slightly dusty, designed
// to read well on a dark background.
export const STEM_COLORS = {
  vocals: "#e85f6f",
  drums: "#e89048",
  bass: "#e8b848",
  guitar: "#88d878",
  piano: "#b88fe8",
  other: "#88a8c8",
  original: "#a8b0bd",
};

export const PROGRESS_COLOR = "#3a3a3a";

export const LOOP_DEFAULT_START_FRAC = 0.25;
export const LOOP_DEFAULT_END_FRAC = 0.5;

export const LANE_VOLUME_MAX = 2;