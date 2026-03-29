import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import { useSignStore } from "../../stores/useSignStore";
import { useSignWebSocket } from "../../hooks/useSignWebSocket";
import { triggerSOS } from "../../services/sign";
import { playAudioFromUrl } from "../../services/audio";
import EmotionBadge from "../../components/sign/EmotionBadge";
import TranscriptFeed from "../../components/sign/TranscriptFeed";
import SOSButton from "../../components/sign/SOSButton";
import LanguageSelector from "../../components/sign/LanguageSelector";
import { colors, spacing, borderRadius, fontSize } from "../../constants/theme";
import { a11y } from "../../constants/accessibility";

const FRAME_INTERVAL_MS = 200; // 5 fps — higher rate for better stability detection

export default function SignScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const {
    transcript,
    currentEmotion,
    currentBuffer,
    selectedLanguage,
    sosActive,
    setSelectedLanguage,
    setSosActive,
    addTranscriptEntry,
  } = useSignStore();

  const {
    connect,
    disconnect,
    sendFrame,
    clearBuffer,
    setLanguage,
    isListening,
  } = useSignWebSocket();

  useEffect(() => {
    requestPermission();
  }, []);

  const startListening = useCallback(async () => {
    if (!permission?.granted) {
      Alert.alert("Camera permission required for SIGN mode");
      return;
    }

    connect();

    // Start capturing and sending frames
    frameIntervalRef.current = setInterval(async () => {
      try {
        if (!cameraRef.current) return;
        const photo = await cameraRef.current.takePictureAsync({
          base64: true,
          quality: 0.5,
          skipProcessing: true,
        });
        if (photo?.base64) {
          sendFrame(photo.base64);
        }
      } catch {
        // Camera may not be ready yet
      }
    }, FRAME_INTERVAL_MS);
  }, [permission, connect, sendFrame]);

  const stopListening = useCallback(() => {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    disconnect();
  }, [disconnect]);

  const handleSOS = useCallback(async () => {
    setSosActive(true);
    try {
      const response = await triggerSOS();
      addTranscriptEntry({
        id: Date.now().toString(),
        text: response.sentence,
        emotion: "sos",
        timestamp: new Date().toLocaleTimeString(),
      });
      await playAudioFromUrl(response.audio_url);
    } catch {
      Alert.alert("SOS Error", "Could not reach backend. Shout for help!");
    }
    setTimeout(() => setSosActive(false), 3000);
  }, [setSosActive, addTranscriptEntry]);

  const handleLanguageChange = useCallback(
    (code: string) => {
      setSelectedLanguage(code);
      setLanguage(code);
    },
    [setSelectedLanguage, setLanguage]
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
    };
  }, []);

  if (!permission) return <View style={styles.container} />;

  if (!permission.granted) {
    return (
      <View style={styles.permContainer}>
        <Text style={styles.permText}>
          Camera access is needed for SIGN mode to detect your hand signs
        </Text>
        <TouchableOpacity style={styles.permBtn} onPress={requestPermission}>
          <Text style={styles.permBtnText}>Grant Camera Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Camera preview (small, top) */}
      {isListening && (
        <View style={styles.cameraContainer}>
          <CameraView
            ref={cameraRef}
            style={styles.camera}
            facing="front"
          />
          {currentBuffer ? (
            <View style={styles.bufferOverlay}>
              <Text style={styles.bufferText}>{currentBuffer}</Text>
            </View>
          ) : null}
        </View>
      )}

      {/* Emotion badge */}
      <EmotionBadge emotion={currentEmotion} />

      {/* Language selector */}
      <LanguageSelector
        selected={selectedLanguage}
        onSelect={handleLanguageChange}
      />

      {/* Transcript feed */}
      <TranscriptFeed entries={transcript} />

      {/* Controls */}
      <View style={styles.controls}>
        <TouchableOpacity
          style={[styles.listenBtn, isListening && styles.listenBtnActive]}
          onPress={isListening ? stopListening : startListening}
          accessibilityLabel={
            isListening
              ? a11y.sign.stopListening.label
              : a11y.sign.startListening.label
          }
          accessibilityHint={
            isListening
              ? a11y.sign.stopListening.hint
              : a11y.sign.startListening.hint
          }
          accessibilityRole="button"
        >
          <Text style={styles.listenBtnText}>
            {isListening ? "Stop Listening" : "Start Listening"}
          </Text>
        </TouchableOpacity>

        <SOSButton onPress={handleSOS} active={sosActive} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
    padding: spacing.lg,
  },
  permContainer: {
    flex: 1,
    backgroundColor: colors.background,
    justifyContent: "center",
    alignItems: "center",
    padding: spacing.xxl,
  },
  permText: {
    color: colors.text,
    textAlign: "center",
    marginBottom: spacing.xl,
    fontSize: fontSize.xl,
    lineHeight: 24,
  },
  permBtn: {
    backgroundColor: colors.primary,
    borderRadius: borderRadius.md,
    padding: spacing.lg,
    paddingHorizontal: spacing.xxxl,
  },
  permBtnText: {
    color: colors.text,
    fontWeight: "600",
    fontSize: fontSize.xl,
  },
  cameraContainer: {
    height: 150,
    borderRadius: borderRadius.md,
    overflow: "hidden",
    marginBottom: spacing.md,
    position: "relative",
  },
  camera: {
    flex: 1,
  },
  bufferOverlay: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: "rgba(0,0,0,0.7)",
    padding: spacing.sm,
  },
  bufferText: {
    color: colors.primary,
    fontSize: fontSize.xxl,
    fontWeight: "700",
    textAlign: "center",
    letterSpacing: 2,
  },
  controls: {
    flexDirection: "row",
    gap: spacing.md,
  },
  listenBtn: {
    flex: 1,
    backgroundColor: colors.primaryDark,
    borderWidth: 1,
    borderColor: colors.primary,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    alignItems: "center",
  },
  listenBtnActive: {
    backgroundColor: colors.primary,
  },
  listenBtnText: {
    color: colors.text,
    fontSize: fontSize.xl,
    fontWeight: "600",
  },
});
