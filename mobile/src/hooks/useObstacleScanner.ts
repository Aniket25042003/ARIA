import { useCallback, useRef } from "react";
import { CameraView } from "expo-camera";
import { sendObstacleFrame } from "../services/guide";
import { playAudioFromUrl } from "../services/audio";
import { hapticLight, hapticHeavy } from "../utils/haptics";
import { useGuideStore } from "../stores/useGuideStore";
import { OBSTACLE_SCAN_INTERVAL_MS } from "../constants/config";

export function useObstacleScanner(cameraRef: React.RefObject<CameraView | null>) {
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const { setIsScanning, setLastWarning } = useGuideStore();

  const startScanning = useCallback(() => {
    setIsScanning(true);

    intervalRef.current = setInterval(async () => {
      try {
        if (!cameraRef.current) return;

        const photo = await cameraRef.current.takePictureAsync({
          base64: true,
          quality: 0.3,
          skipProcessing: true,
        });

        if (!photo?.base64) return;

        const response = await sendObstacleFrame(photo.base64);
        setLastWarning(response.warning, response.severity);

        // Play audio warning only if there is one (clear paths skip TTS)
        if (response.audio_url) {
          await playAudioFromUrl(response.audio_url);
        }

        // Haptic feedback based on severity
        if (response.severity === "danger") {
          await hapticHeavy();
        } else if (response.severity === "caution") {
          await hapticLight();
        }
      } catch {
        // Continue scanning even if one frame fails
      }
    }, OBSTACLE_SCAN_INTERVAL_MS);
  }, [cameraRef, setIsScanning, setLastWarning]);

  const stopScanning = useCallback(() => {
    setIsScanning(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, [setIsScanning]);

  return { startScanning, stopScanning };
}
