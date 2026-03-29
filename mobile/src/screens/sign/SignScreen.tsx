import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  Animated,
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { CameraView, useCameraPermissions } from "expo-camera";
import { useSignStore } from "../../stores/useSignStore";
import { useSignWebSocket } from "../../hooks/useSignWebSocket";
import { triggerSOS } from "../../services/sign";
import { playAudioFromUrl } from "../../services/audio";
import EmotionBadge from "../../components/sign/EmotionBadge";
import TranscriptFeed from "../../components/sign/TranscriptFeed";
import SOSButton from "../../components/sign/SOSButton";
import LanguageSelector from "../../components/sign/LanguageSelector";
import { colors, spacing, borderRadius, fontSize, shadows } from "../../constants/theme";
import { a11y } from "../../constants/accessibility";

const FRAME_INTERVAL_MS = 200;

export default function SignScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing] = useState<"front" | "back">("front");
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

  // Animations
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const cameraScale = useRef(new Animated.Value(0.95)).current;
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const listenGlow = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    requestPermission();
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 400,
      useNativeDriver: true,
    }).start();
  }, []);

  // Listening pulse animation
  useEffect(() => {
    if (isListening) {
      Animated.spring(cameraScale, {
        toValue: 1,
        tension: 50,
        friction: 7,
        useNativeDriver: true,
      }).start();

      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, { toValue: 1.05, duration: 1200, useNativeDriver: true }),
          Animated.timing(pulseAnim, { toValue: 1, duration: 1200, useNativeDriver: true }),
        ])
      ).start();

      Animated.timing(listenGlow, { toValue: 1, duration: 300, useNativeDriver: true }).start();
    } else {
      Animated.timing(cameraScale, {
        toValue: 0.95,
        duration: 200,
        useNativeDriver: true,
      }).start();
      pulseAnim.setValue(1);
      Animated.timing(listenGlow, { toValue: 0, duration: 200, useNativeDriver: true }).start();
    }
  }, [isListening]);

  const startListening = useCallback(async () => {
    if (!permission?.granted) {
      Alert.alert("Camera permission required for SIGN mode");
      return;
    }
    connect();
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
        // Camera may not be ready
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

  const toggleCamera = useCallback(() => {
    setFacing((prev) => (prev === "front" ? "back" : "front"));
  }, []);

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

  useEffect(() => {
    return () => {
      if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
    };
  }, []);

  if (!permission) return <View style={styles.container} />;

  if (!permission.granted) {
    return (
      <View style={styles.permContainer}>
        <View style={styles.permCard}>
          <Text style={styles.permIcon}>S</Text>
          <Text style={styles.permTitle}>Camera Access Needed</Text>
          <Text style={styles.permText}>
            SIGN mode needs your camera to detect hand signs and facial expressions
          </Text>
          <TouchableOpacity style={styles.permBtn} onPress={requestPermission} activeOpacity={0.8}>
            <LinearGradient
              colors={[colors.primary, colors.primaryDark]}
              style={styles.permBtnGradient}
            >
              <Text style={styles.permBtnText}>Grant Permission</Text>
            </LinearGradient>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  return (
    <Animated.View style={[styles.container, { opacity: fadeAnim }]}>
      {/* Camera preview */}
      {isListening && (
        <Animated.View
          style={[
            styles.cameraContainer,
            {
              transform: [{ scale: Animated.multiply(cameraScale, pulseAnim) }],
            },
          ]}
        >
          <CameraView
            ref={cameraRef}
            style={styles.camera}
            facing={facing}
          />
          {currentBuffer ? (
            <View style={styles.bufferOverlay}>
              <Text style={styles.bufferText}>{currentBuffer}</Text>
            </View>
          ) : null}
          <TouchableOpacity
            style={styles.flipBtn}
            onPress={toggleCamera}
            activeOpacity={0.7}
            accessibilityLabel={`Switch to ${facing === "front" ? "back" : "front"} camera`}
            accessibilityRole="button"
          >
            <Text style={styles.flipBtnText}>
              {facing === "front" ? "BACK" : "FRONT"}
            </Text>
          </TouchableOpacity>
          {/* Live indicator */}
          <View style={styles.liveIndicator}>
            <View style={styles.liveDot} />
            <Text style={styles.liveText}>LIVE</Text>
          </View>
        </Animated.View>
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
          activeOpacity={0.8}
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
          {isListening ? (
            <View style={styles.listenBtnInner}>
              <View style={styles.stopIcon} />
              <Text style={styles.listenBtnTextActive}>Stop</Text>
            </View>
          ) : (
            <LinearGradient
              colors={[colors.primary, colors.primaryDark]}
              style={styles.listenBtnGradient}
            >
              <Text style={styles.listenBtnText}>Start Listening</Text>
            </LinearGradient>
          )}
        </TouchableOpacity>

        <SOSButton onPress={handleSOS} active={sosActive} />
      </View>
    </Animated.View>
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
  permCard: {
    backgroundColor: colors.surface,
    borderRadius: borderRadius.xxl,
    padding: spacing.xxxl,
    alignItems: "center",
    width: "100%",
    ...shadows.lg,
  },
  permIcon: {
    fontSize: 40,
    fontWeight: "800",
    color: colors.primary,
    marginBottom: spacing.lg,
  },
  permTitle: {
    fontSize: fontSize.xxl,
    fontWeight: "700",
    color: colors.text,
    marginBottom: spacing.sm,
  },
  permText: {
    color: colors.textSecondary,
    textAlign: "center",
    marginBottom: spacing.xxl,
    fontSize: fontSize.md,
    lineHeight: 22,
  },
  permBtn: {
    borderRadius: borderRadius.lg,
    overflow: "hidden",
    width: "100%",
    ...shadows.md,
  },
  permBtnGradient: {
    padding: spacing.lg,
    alignItems: "center",
    borderRadius: borderRadius.lg,
  },
  permBtnText: {
    color: "#fff",
    fontWeight: "700",
    fontSize: fontSize.xl,
  },
  cameraContainer: {
    height: 220,
    borderRadius: borderRadius.xl,
    overflow: "hidden",
    marginBottom: spacing.md,
    position: "relative",
    ...shadows.lg,
  },
  camera: {
    flex: 1,
  },
  bufferOverlay: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: "rgba(255,255,255,0.92)",
    padding: spacing.sm + 2,
  },
  bufferText: {
    color: colors.primary,
    fontSize: fontSize.xxl,
    fontWeight: "800",
    textAlign: "center",
    letterSpacing: 3,
  },
  flipBtn: {
    position: "absolute",
    top: spacing.sm,
    right: spacing.sm,
    backgroundColor: "rgba(255,255,255,0.9)",
    borderRadius: borderRadius.round,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs + 2,
    ...shadows.sm,
  },
  flipBtnText: {
    color: colors.text,
    fontSize: fontSize.xs,
    fontWeight: "700",
    letterSpacing: 1,
  },
  liveIndicator: {
    position: "absolute",
    top: spacing.sm,
    left: spacing.sm,
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(232, 72, 72, 0.9)",
    borderRadius: borderRadius.round,
    paddingHorizontal: spacing.sm + 2,
    paddingVertical: spacing.xs,
  },
  liveDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: "#fff",
    marginRight: spacing.xs,
  },
  liveText: {
    color: "#fff",
    fontSize: fontSize.xs - 1,
    fontWeight: "800",
    letterSpacing: 1,
  },
  controls: {
    flexDirection: "row",
    gap: spacing.md,
  },
  listenBtn: {
    flex: 1,
    borderRadius: borderRadius.lg,
    overflow: "hidden",
  },
  listenBtnActive: {
    backgroundColor: colors.surface,
    borderWidth: 2,
    borderColor: colors.danger,
    borderRadius: borderRadius.lg,
  },
  listenBtnInner: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    padding: spacing.lg,
    gap: spacing.sm,
  },
  stopIcon: {
    width: 16,
    height: 16,
    borderRadius: 3,
    backgroundColor: colors.danger,
  },
  listenBtnGradient: {
    padding: spacing.lg,
    alignItems: "center",
    borderRadius: borderRadius.lg,
    ...shadows.md,
  },
  listenBtnText: {
    color: "#fff",
    fontSize: fontSize.xl,
    fontWeight: "700",
  },
  listenBtnTextActive: {
    color: colors.danger,
    fontSize: fontSize.xl,
    fontWeight: "700",
  },
});
