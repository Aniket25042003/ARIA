import React, { useEffect, useRef } from "react";
import { View, Text, StyleSheet, Animated } from "react-native";
import { colors, spacing, borderRadius, fontSize } from "../../constants/theme";

interface Props {
  warning: string;
  severity: "clear" | "caution" | "danger";
  isScanning: boolean;
}

const SEVERITY_CONFIG = {
  clear: { color: colors.success, bg: "rgba(0, 184, 148, 0.15)", label: "CLEAR" },
  caution: { color: colors.warningDark, bg: "rgba(253, 203, 110, 0.2)", label: "CAUTION" },
  danger: { color: colors.danger, bg: "rgba(232, 72, 72, 0.2)", label: "DANGER" },
};

export default function ObstacleOverlay({ warning, severity, isScanning }: Props) {
  const config = SEVERITY_CONFIG[severity];
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const scanDotAnim = useRef(new Animated.Value(0.3)).current;

  useEffect(() => {
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 300,
      useNativeDriver: true,
    }).start();
  }, []);

  useEffect(() => {
    if (severity === "danger") {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, { toValue: 1.03, duration: 400, useNativeDriver: true }),
          Animated.timing(pulseAnim, { toValue: 1, duration: 400, useNativeDriver: true }),
        ])
      ).start();
    } else {
      pulseAnim.setValue(1);
    }
  }, [severity]);

  useEffect(() => {
    if (isScanning) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(scanDotAnim, { toValue: 1, duration: 600, useNativeDriver: true }),
          Animated.timing(scanDotAnim, { toValue: 0.3, duration: 600, useNativeDriver: true }),
        ])
      ).start();
    }
  }, [isScanning]);

  return (
    <Animated.View style={[styles.overlay, { opacity: fadeAnim }]}>
      <Animated.View
        style={[
          styles.warningBanner,
          { backgroundColor: config.bg, borderColor: config.color + "60" },
          { transform: [{ scale: pulseAnim }] },
        ]}
        accessibilityLabel={`Obstacle warning: ${warning}. Severity: ${severity}`}
        accessibilityLiveRegion="assertive"
      >
        <View style={[styles.severityBadge, { backgroundColor: config.color }]}>
          <Text style={styles.severityBadgeText}>{config.label}</Text>
        </View>
        <Text style={styles.warningText}>{warning}</Text>
      </Animated.View>

      {isScanning && (
        <View style={styles.scanIndicator}>
          <Animated.View style={[styles.scanDotOuter, { opacity: scanDotAnim }]}>
            <View style={styles.scanDotInner} />
          </Animated.View>
          <Text style={styles.scanLabel}>Scanning for obstacles</Text>
        </View>
      )}
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    justifyContent: "space-between",
    padding: spacing.lg,
  },
  warningBanner: {
    borderRadius: borderRadius.xl,
    padding: spacing.lg,
    marginTop: spacing.sm,
    borderWidth: 1.5,
  },
  severityBadge: {
    alignSelf: "flex-start",
    borderRadius: borderRadius.round,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    marginBottom: spacing.sm,
  },
  severityBadgeText: {
    fontSize: fontSize.xs,
    fontWeight: "800",
    color: "#fff",
    letterSpacing: 1.5,
  },
  warningText: {
    color: "#fff",
    fontSize: fontSize.xxl,
    fontWeight: "600",
    textAlign: "center",
    textShadowColor: "rgba(0,0,0,0.5)",
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 4,
  },
  scanIndicator: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.sm,
  },
  scanDotOuter: {
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: "rgba(232, 72, 72, 0.3)",
    justifyContent: "center",
    alignItems: "center",
  },
  scanDotInner: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: colors.danger,
  },
  scanLabel: {
    color: "#fff",
    fontSize: fontSize.md,
    fontWeight: "600",
    backgroundColor: "rgba(0,0,0,0.5)",
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.round,
    overflow: "hidden",
  },
});
