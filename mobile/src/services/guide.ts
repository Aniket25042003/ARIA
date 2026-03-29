import api from "./api";
import type { ObstacleResponse } from "../types/api";

export async function sendObstacleFrame(
  frameBase64: string
): Promise<ObstacleResponse> {
  const response = await api.post<ObstacleResponse>("/guide/obstacle", {
    frame: frameBase64,
  });
  return response.data;
}
