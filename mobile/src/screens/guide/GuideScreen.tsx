import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as Location from "expo-location";
import { useGuideStore } from "../../stores/useGuideStore";
import { useObstacleScanner } from "../../hooks/useObstacleScanner";
import { useGuideNavWebSocket } from "../../hooks/useGuideNavWebSocket";
import { getNavigation } from "../../services/guide";
import { playAudioFromUrl } from "../../services/audio";
import { manualSpeak } from "../../services/sign";
import ObstacleOverlay from "../../components/guide/ObstacleOverlay";
import NavigationStepCard from "../../components/guide/NavigationStepCard";
import DestinationInput from "../../components/guide/DestinationInput";
import { colors, spacing, borderRadius, fontSize } from "../../constants/theme";
import { a11y } from "../../constants/accessibility";

export default function GuideScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [locationPermission, requestLocationPermission] =
    Location.useForegroundPermissions();
  const [destination, setDestination] = useState("");
  const cameraRef = useRef<CameraView>(null);

  const {
    isScanning,
    lastWarning,
    lastSeverity,
    isNavigating,
    navSteps,
    currentStep,
    totalDistance,
    totalDuration,
    setNavigation,
  } = useGuideStore();

  const { startScanning, stopScanning } = useObstacleScanner(cameraRef);
  const { startNavTracking, stopNavTracking } = useGuideNavWebSocket();

  useEffect(() => {
    requestPermission();
    requestLocationPermission();
  }, []);

  const handleStartNavigation = useCallback(async () => {
    if (!locationPermission?.granted) {
      Alert.alert("Location permission required for navigation");
      return;
    }
    if (!destination.trim()) {
      Alert.alert("Please enter a destination");
      return;
    }

    try {
      const loc = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.High,
      });

      const data = await getNavigation(
        loc.coords.latitude,
        loc.coords.longitude,
        destination.trim()
      );

      if (!data.steps.length) {
        Alert.alert("Navigation error", "Could not find a route to that destination.");
        return;
      }

      setNavigation(
        data.route_id,
        data.steps,
        data.total_distance,
        data.total_duration
      );

      // Speak first instruction
      const firstStep = data.steps[0];
      try {
        const res = await manualSpeak(firstStep.instruction, "neutral");
        await playAudioFromUrl(res.audio_url);
      } catch {
        // TTS failed, but navigation still works
      }

      // Start location tracking for step advancement
      await startNavTracking(data.route_id);
    } catch (err) {
      Alert.alert("Navigation error", "Could not calculate route. Check destination.");
    }
  }, [
    locationPermission,
    destination,
    setNavigation,
    startNavTracking,
  ]);

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

      {/* Controls */}
      <View style={styles.controls}>
        {/* Destination input */}
        <DestinationInput
          value={destination}
          onChangeText={setDestination}
          onSubmit={handleStartNavigation}
        />

        {/* Navigation info */}
        {isNavigating && navSteps[currentStep] && (
          <View>
            <NavigationStepCard
              step={navSteps[currentStep]}
              currentIndex={currentStep}
              totalSteps={navSteps.length}
            />
            <Text style={styles.routeInfo}>
              {totalDistance} · {totalDuration}
            </Text>
            <TouchableOpacity
              style={styles.stopNavBtn}
              onPress={stopNavTracking}
              accessibilityLabel="Stop navigation"
              accessibilityRole="button"
            >
              <Text style={styles.stopNavBtnText}>Stop Navigation</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Scan toggle */}
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
  routeInfo: {
    color: colors.textMuted,
    fontSize: fontSize.sm,
    textAlign: "center",
    marginTop: spacing.xs,
  },
  stopNavBtn: {
    backgroundColor: colors.dangerDark,
    borderWidth: 1,
    borderColor: colors.danger,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    alignItems: "center",
    marginTop: spacing.sm,
  },
  stopNavBtnText: {
    color: colors.danger,
    fontSize: fontSize.lg - 1,
    fontWeight: "600",
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
