import api from "./api";
import type { SpeakResponse } from "../types/api";

export async function triggerSOS(location?: {
  latitude: number;
  longitude: number;
}): Promise<SpeakResponse> {
  const payload = location
    ? { location: { lat: location.latitude, lng: location.longitude } }
    : {};
  const response = await api.post<SpeakResponse>("/sign/sos", payload);
  return response.data;
}

export async function manualSpeak(
  text: string,
  emotion: string = "neutral",
  language: string = "en"
): Promise<SpeakResponse> {
  const response = await api.post<SpeakResponse>("/sign/speak", {
    text,
    emotion,
    language,
  });
  return response.data;
}
