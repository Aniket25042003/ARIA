export const colors = {
  // Core backgrounds
  background: "#F8F9FC",
  surface: "#FFFFFF",
  surfaceLight: "#F1F3F8",
  surfaceSecondary: "#EEF0F6",
  border: "#E2E5ED",
  borderLight: "#D1D5E0",

  // Primary - vibrant purple/indigo
  primary: "#6C5CE7",
  primaryDark: "#5A4BD1",
  primaryLight: "#A29BFE",
  primarySoft: "#EDE9FF",
  primaryGlow: "rgba(108, 92, 231, 0.12)",

  // Secondary accent
  accent: "#00B894",
  accentDark: "#00A884",
  accentLight: "#55EFC4",
  accentSoft: "#E0FFF5",

  // Status
  success: "#00B894",
  successDark: "#009B7D",
  successBorder: "#00B894",
  successSoft: "#E0FFF5",

  danger: "#E84848",
  dangerDark: "#D63030",
  dangerLight: "#FF6B6B",
  dangerSoft: "#FFF0F0",

  warning: "#FDCB6E",
  warningDark: "#E5A922",
  warningSoft: "#FFF8E7",

  // Text
  text: "#1A1D2E",
  textSecondary: "#545972",
  textMuted: "#8B90A5",
  textDim: "#AEB3C6",
  textPlaceholder: "#C8CCDA",

  // Shadows
  shadowColor: "#1A1D2E",

  // Overlays
  overlay: "rgba(26, 29, 46, 0.5)",
  overlayLight: "rgba(26, 29, 46, 0.08)",

  // Emotion colors (vibrant for both badges and accents)
  emotion: {
    neutral: "#8B90A5",
    happy: "#FDCB6E",
    sad: "#74B9FF",
    fear: "#E84848",
    angry: "#E17055",
    surprise: "#A29BFE",
    disgust: "#00B894",
    sos: "#E84848",
  } as Record<string, string>,
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  xxl: 24,
  xxxl: 32,
  xxxxl: 40,
};

export const borderRadius = {
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  xxl: 24,
  round: 50,
  full: 9999,
};

export const fontSize = {
  xs: 11,
  sm: 12,
  md: 14,
  lg: 16,
  xl: 17,
  xxl: 20,
  xxxl: 24,
  title: 30,
  display: 48,
};

export const shadows = {
  sm: {
    shadowColor: colors.shadowColor,
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 4,
    elevation: 2,
  },
  md: {
    shadowColor: colors.shadowColor,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.08,
    shadowRadius: 12,
    elevation: 4,
  },
  lg: {
    shadowColor: colors.shadowColor,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.12,
    shadowRadius: 24,
    elevation: 8,
  },
  glow: (color: string) => ({
    shadowColor: color,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 12,
    elevation: 6,
  }),
};
