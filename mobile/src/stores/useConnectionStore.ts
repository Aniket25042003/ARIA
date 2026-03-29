import { create } from "zustand";

interface ConnectionState {
  backendConnected: boolean;
  backendChecked: boolean;
  signWsConnected: boolean;
  guideWsConnected: boolean;

  setBackendConnected: (connected: boolean) => void;
  setSignWsConnected: (connected: boolean) => void;
  setGuideWsConnected: (connected: boolean) => void;
}

export const useConnectionStore = create<ConnectionState>((set) => ({
  backendConnected: false,
  backendChecked: false,
  signWsConnected: false,
  guideWsConnected: false,

  setBackendConnected: (connected) =>
    set({ backendConnected: connected, backendChecked: true }),
  setSignWsConnected: (connected) => set({ signWsConnected: connected }),
  setGuideWsConnected: (connected) => set({ guideWsConnected: connected }),
}));
