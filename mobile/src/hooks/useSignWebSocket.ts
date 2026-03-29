import { useCallback, useEffect, useRef } from "react";
import { WSManager } from "../services/websocket";
import { playAudioFromUrl } from "../services/audio";
import { hapticSuccess } from "../utils/haptics";
import { useSignStore } from "../stores/useSignStore";
import { useConnectionStore } from "../stores/useConnectionStore";
import type { WSServerMessage, TranscriptEntry } from "../types/sign";

export function useSignWebSocket() {
  const wsRef = useRef<WSManager | null>(null);
  const {
    setCurrentEmotion,
    setCurrentBuffer,
    addTranscriptEntry,
    isListening,
    setIsListening,
  } = useSignStore();
  const setSignWsConnected = useConnectionStore((s) => s.setSignWsConnected);

  const handleMessage = useCallback(
    (data: WSServerMessage) => {
      switch (data.type) {
        case "letter":
          setCurrentBuffer(data.buffer ?? "");
          break;

        case "signs_detected":
          setCurrentBuffer(data.signs ?? "");
          if (data.emotion) setCurrentEmotion(data.emotion);
          break;

        case "status":
          // Informational status messages from backend
          break;

        case "emotion":
          setCurrentEmotion(data.emotion);
          break;

        case "sentence": {
          const entry: TranscriptEntry = {
            id: Date.now().toString(),
            text: data.text,
            emotion: data.emotion,
            timestamp: new Date().toLocaleTimeString(),
          };
          addTranscriptEntry(entry);
          playAudioFromUrl(data.audio_url);
          hapticSuccess();
          break;
        }

        case "sos_triggered": {
          const sosEntry: TranscriptEntry = {
            id: Date.now().toString(),
            text: data.text,
            emotion: "sos",
            timestamp: new Date().toLocaleTimeString(),
          };
          addTranscriptEntry(sosEntry);
          playAudioFromUrl(data.audio_url);
          break;
        }

        case "error":
          console.error("Sign WS error:", data.message);
          break;
      }
    },
    [setCurrentBuffer, setCurrentEmotion, addTranscriptEntry]
  );

  const connect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.disconnect();
    }
    wsRef.current = new WSManager("/sign/ws", handleMessage, setSignWsConnected);
    wsRef.current.connect();
    setIsListening(true);
  }, [handleMessage, setSignWsConnected, setIsListening]);

  const disconnect = useCallback(() => {
    wsRef.current?.disconnect();
    wsRef.current = null;
    setIsListening(false);
    setSignWsConnected(false);
  }, [setIsListening, setSignWsConnected]);

  const sendFrame = useCallback((frameBase64: string) => {
    wsRef.current?.send({
      type: "frame",
      frame: frameBase64,
      timestamp: Date.now() / 1000,
    });
  }, []);

  const clearBuffer = useCallback(() => {
    wsRef.current?.send({ type: "clear_buffer" });
  }, []);

  const setLanguage = useCallback((language: string) => {
    wsRef.current?.send({ type: "set_language", language });
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      wsRef.current?.disconnect();
    };
  }, []);

  return {
    connect,
    disconnect,
    sendFrame,
    clearBuffer,
    setLanguage,
    isListening,
  };
}
