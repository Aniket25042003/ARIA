import { create } from "zustand";
import { BACKEND_URL as DEFAULT_BACKEND_URL } from "../constants/config";
import {
  saveBackendUrl,
  getBackendUrl,
  removeBackendUrl,
} from "../utils/storage";

interface ConnectionState {
  backendUrl: string;
  backendConnected: boolean;
  backendChecked: boolean;
  signWsConnected: boolean;
  guideWsConnected: boolean;

  setBackendUrl: (url: string) => Promise<void>;
  resetBackendUrl: () => Promise<void>;
  loadBackendUrl: () => Promise<void>;
  setBackendConnected: (connected: boolean) => void;
  setSignWsConnected: (connected: boolean) => void;
  setGuideWsConnected: (connected: boolean) => void;
}

export const useConnectionStore = create<ConnectionState>((set) => ({
  backendUrl: DEFAULT_BACKEND_URL,
  backendConnected: false,
  backendChecked: false,
  signWsConnected: false,
  guideWsConnected: false,

  setBackendUrl: async (url: string) => {
    // Normalize: remove trailing slash
    const normalized = url.replace(/\/+$/, "");
    await saveBackendUrl(normalized);
    set({ backendUrl: normalized, backendConnected: false, backendChecked: false });
  },

  resetBackendUrl: async () => {
    await removeBackendUrl();
    set({
      backendUrl: DEFAULT_BACKEND_URL,
      backendConnected: false,
      backendChecked: false,
    });
  },

  loadBackendUrl: async () => {
    const saved = await getBackendUrl();
    if (saved) {
      // Clear stale USB-only URL if default has changed
      if (saved === "http://192.168.55.1:8000" && DEFAULT_BACKEND_URL !== "http://192.168.55.1:8000") {
        await removeBackendUrl();
      } else {
        set({ backendUrl: saved });
      }
    }
  },

  setBackendConnected: (connected) =>
    set({ backendConnected: connected, backendChecked: true }),
  setSignWsConnected: (connected) => set({ signWsConnected: connected }),
  setGuideWsConnected: (connected) => set({ guideWsConnected: connected }),
}));
