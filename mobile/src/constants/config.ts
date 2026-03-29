// Backend URL — update this to your Jetson Nano's IP address
// On the Jetson, run `hostname -I` to find the IP
export const BACKEND_URL =
  process.env.EXPO_PUBLIC_BACKEND_URL || "http://10.7.112.227:8000";

export const WS_URL = BACKEND_URL.replace("http", "ws");

export const OBSTACLE_SCAN_INTERVAL_MS = 1500;
export const HEALTH_CHECK_INTERVAL_MS = 30000; // 30s — no need to poll aggressively
export const WS_RECONNECT_INTERVAL_MS = 3000;

export const SUPPORTED_LANGUAGES = [
  { code: "en", label: "English" },
  { code: "es", label: "Spanish" },
  { code: "fr", label: "French" },
  { code: "hi", label: "Hindi" },
] as const;
