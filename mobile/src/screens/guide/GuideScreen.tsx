import React, { useEffect, useRef } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import { useGuideStore } from "../../stores/useGuideStore";
import { useObstacleScanner } from "../../hooks/useObstacleScanner";
import ObstacleOverlay from "../../components/guide/ObstacleOverlay";
import { colors, spacing, borderRadius, fontSize } from "../../constants/theme";
import { a11y } from "../../constants/accessibility";

export default function GuideScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);

  const {
    isScanning,
    lastWarning,
    lastSeverity,
  } = useGuideStore();

  const { startScanning, stopScanning } = useObstacleScanner(cameraRef);

  useEffect(() => {
    requestPermission();
  }, []);

  if (!permission) return <View style={styles.container} />;

  if (!permission.granted) {
    return (
      <View style={styles.permContainer}>
        <Text style={styles.permText}>
          Camera access is needed for GUIDE mode to detect obstacles in your path
        </Text>
        <TouchableOpacity style={styles.permBtn} onPress={requestPermission}>
          <Text style={styles.permBtnText}>Grant Camera Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Camera with obstacle overlay */}
      <CameraView ref={cameraRef} style={styles.camera} facing="back">
        <ObstacleOverlay
          warning={lastWarning}
          severity={lastSeverity}
          isScanning={isScanning}
        />
      </CameraView>

      {/* Scan toggle */}
      <View style={styles.controls}>
        <TouchableOpacity
          style={[styles.scanBtn, isScanning && styles.scanBtnActive]}
          onPress={isScanning ? stopScanning : startScanning}
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
          <Text style={styles.scanBtnText}>
            {isScanning ? "Stop Scanning" : "Start Scanning"}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
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
  permText: {
    color: colors.text,
    textAlign: "center",
    marginBottom: spacing.xl,
    fontSize: fontSize.xl,
    lineHeight: 24,
  },
  permBtn: {
    backgroundColor: colors.successBorder,
    borderRadius: borderRadius.md,
    padding: spacing.lg,
    paddingHorizontal: spacing.xxxl,
  },
  permBtnText: {
    color: colors.text,
    fontWeight: "600",
    fontSize: fontSize.xl,
  },
  camera: {
    flex: 1,
  },
  controls: {
    backgroundColor: colors.background,
    padding: spacing.lg,
    gap: spacing.md,
  },
  scanBtn: {
    backgroundColor: "#0a1a2e",
    borderWidth: 1,
    borderColor: "#2563eb",
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    alignItems: "center",
  },
  scanBtnActive: {
    backgroundColor: "#2563eb",
  },
  scanBtnText: {
    color: colors.text,
    fontSize: fontSize.xl,
    fontWeight: "600",
  },
});
