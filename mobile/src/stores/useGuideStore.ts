import { create } from "zustand";

interface GuideState {
  isScanning: boolean;
  lastWarning: string;
  lastSeverity: "clear" | "caution" | "danger";

  setIsScanning: (scanning: boolean) => void;
  setLastWarning: (warning: string, severity: "clear" | "caution" | "danger") => void;
}

export const useGuideStore = create<GuideState>((set) => ({
  isScanning: false,
  lastWarning: "Tap 'Start Scanning' to begin",
  lastSeverity: "clear",

  setIsScanning: (scanning) => set({ isScanning: scanning }),
  setLastWarning: (warning, severity) =>
    set({ lastWarning: warning, lastSeverity: severity }),
}));
