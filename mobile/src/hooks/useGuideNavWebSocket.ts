import { useCallback, useEffect, useRef } from "react";
import * as Location from "expo-location";
import { WSManager } from "../services/websocket";
import { playAudioFromUrl } from "../services/audio";
import { hapticSuccess } from "../utils/haptics";
import { useGuideStore } from "../stores/useGuideStore";
import { useConnectionStore } from "../stores/useConnectionStore";
import type { WSGuideServerMessage } from "../types/guide";

export function useGuideNavWebSocket() {
  const wsRef = useRef<WSManager | null>(null);
  const locationWatchRef = useRef<Location.LocationSubscription | null>(null);
  const { setCurrentStep, clearNavigation } = useGuideStore();
  const setGuideWsConnected = useConnectionStore((s) => s.setGuideWsConnected);

  const handleMessage = useCallback(
    (data: WSGuideServerMessage) => {
      switch (data.type) {
        case "next_step":
          setCurrentStep(data.step_index);
          if (data.audio_url) {
            playAudioFromUrl(data.audio_url);
          }
          hapticSuccess();
          break;

        case "off_route":
          // Could show a warning or speak the off-route message
          break;

        case "arrived":
          if ("audio_url" in data && data.audio_url) {
            playAudioFromUrl(data.audio_url as string);
          }
          hapticSuccess();
          break;
      }
    },
    [setCurrentStep]
  );

  const startNavTracking = useCallback(
    async (routeId: string) => {
      if (wsRef.current) {
        wsRef.current.disconnect();
      }

      wsRef.current = new WSManager(
        "/guide/ws/nav",
        handleMessage,
        setGuideWsConnected
      );
      await wsRef.current.connect({ route_id: routeId });

      // Watch location and send updates
      locationWatchRef.current = await Location.watchPositionAsync(
        {
          accuracy: Location.Accuracy.High,
          distanceInterval: 10, // Update every 10 meters
        },
        (location) => {
          wsRef.current?.send({
            type: "location_update",
            lat: location.coords.latitude,
            lng: location.coords.longitude,
          });
        }
      );
    },
    [handleMessage, setGuideWsConnected]
  );

  const stopNavTracking = useCallback(() => {
    locationWatchRef.current?.remove();
    locationWatchRef.current = null;
    wsRef.current?.disconnect();
    wsRef.current = null;
    setGuideWsConnected(false);
    clearNavigation();
  }, [setGuideWsConnected, clearNavigation]);

  useEffect(() => {
    return () => {
      locationWatchRef.current?.remove();
      wsRef.current?.disconnect();
    };
  }, []);

  return { startNavTracking, stopNavTracking };
}
