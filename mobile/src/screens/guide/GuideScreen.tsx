import React, { useEffect, useRef } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Animated,
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { CameraView, useCameraPermissions } from "expo-camera";
import { useGuideStore } from "../../stores/useGuideStore";
import { useObstacleScanner } from "../../hooks/useObstacleScanner";
import ObstacleOverlay from "../../components/guide/ObstacleOverlay";
import { colors, spacing, borderRadius, fontSize, shadows } from "../../constants/theme";
import { a11y } from "../../constants/accessibility";

export default function GuideScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);

  const { isScanning, lastWarning, lastSeverity } = useGuideStore();
  const { startScanning, stopScanning } = useObstacleScanner(cameraRef);

  // Animations
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const scanPulse = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    requestPermission();
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 400,
      useNativeDriver: true,
    }).start();
  }, []);

  useEffect(() => {
    if (isScanning) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(scanPulse, { toValue: 1.02, duration: 800, useNativeDriver: true }),
          Animated.timing(scanPulse, { toValue: 1, duration: 800, useNativeDriver: true }),
        ])
      ).start();
    } else {
      scanPulse.setValue(1);
    }
  }, [isScanning]);

  if (!permission) return <View style={styles.container} />;

  if (!permission.granted) {
    return (
      <View style={styles.permContainer}>
        <View style={styles.permCard}>
          <Text style={styles.permIcon}>G</Text>
          <Text style={styles.permTitle}>Camera Access Needed</Text>
          <Text style={styles.permText}>
            GUIDE mode needs your camera to detect obstacles in your path and keep you safe
          </Text>
          <TouchableOpacity style={styles.permBtn} onPress={requestPermission} activeOpacity={0.8}>
            <LinearGradient
              colors={[colors.accent, colors.accentDark]}
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
      {/* Camera with obstacle overlay */}
      <Animated.View style={[styles.cameraWrapper, { transform: [{ scale: scanPulse }] }]}>
        <CameraView ref={cameraRef} style={styles.camera} facing="back">
          <ObstacleOverlay
            warning={lastWarning}
            severity={lastSeverity}
            isScanning={isScanning}
          />
        </CameraView>
      </Animated.View>

      {/* Scan toggle */}
      <View style={styles.controls}>
        <TouchableOpacity
          style={[styles.scanBtn, isScanning && styles.scanBtnActive]}
          onPress={isScanning ? stopScanning : startScanning}
          activeOpacity={0.8}
          accessibilityLabel={
            isScanning
              ? a11y.guide.stopScanning.label
              : a11y.guide.startScanning.label
          }
          accessibilityHint={
            isScanning
              ? a11y.guide.stopScanning.hint
              : a11y.guide.startScanning.hint
          }
          accessibilityRole="button"
        >
          {isScanning ? (
            <View style={styles.scanBtnInner}>
              <View style={styles.stopIcon} />
              <Text style={styles.scanBtnTextActive}>Stop Scanning</Text>
            </View>
          ) : (
            <LinearGradient
              colors={[colors.accent, colors.accentDark]}
              style={styles.scanBtnGradient}
            >
              <Text style={styles.scanBtnText}>Start Scanning</Text>
            </LinearGradient>
          )}
        </TouchableOpacity>
      </View>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
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
    color: colors.accent,
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
  cameraWrapper: {
    flex: 1,
    margin: spacing.md,
    borderRadius: borderRadius.xl,
    overflow: "hidden",
    ...shadows.lg,
  },
  camera: {
    flex: 1,
  },
  controls: {
    backgroundColor: colors.background,
    padding: spacing.lg,
  },
  scanBtn: {
    borderRadius: borderRadius.lg,
    overflow: "hidden",
  },
  scanBtnActive: {
    backgroundColor: colors.surface,
    borderWidth: 2,
    borderColor: colors.danger,
    borderRadius: borderRadius.lg,
  },
  scanBtnInner: {
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
  scanBtnGradient: {
    padding: spacing.lg,
    alignItems: "center",
    borderRadius: borderRadius.lg,
    ...shadows.md,
  },
  scanBtnText: {
    color: "#fff",
    fontSize: fontSize.xl,
    fontWeight: "700",
  },
  scanBtnTextActive: {
    color: colors.danger,
    fontSize: fontSize.xl,
    fontWeight: "700",
  },
});
