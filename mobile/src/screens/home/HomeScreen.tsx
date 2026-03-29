import React from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
} from "react-native";
import { useNavigation } from "@react-navigation/native";
import type { BottomTabNavigationProp } from "@react-navigation/bottom-tabs";
import { useBackendHealth } from "../../hooks/useBackendHealth";
import { useConnectionStore } from "../../stores/useConnectionStore";
import { useAuthStore } from "../../stores/useAuthStore";
import { colors, spacing, borderRadius, fontSize } from "../../constants/theme";
import { a11y } from "../../constants/accessibility";
import type { MainTabParamList } from "../../navigation/types";

type HomeNav = BottomTabNavigationProp<MainTabParamList, "Home">;

export default function HomeScreen() {
  const navigation = useNavigation<HomeNav>();
  const backendConnected = useConnectionStore((s) => s.backendConnected);
  const backendChecked = useConnectionStore((s) => s.backendChecked);
  const userName = useAuthStore((s) => s.userName);

  useBackendHealth();

  return (
    <View style={styles.container}>
      <Text style={styles.logo}>ARIA</Text>
      <Text style={styles.tagline}>
        Adaptive Real-time Intelligence Assistant
      </Text>
      {userName && (
        <Text style={styles.greeting}>Welcome, {userName}</Text>
      )}

      <View style={styles.statusRow}>
        {!backendChecked ? (
          <ActivityIndicator color={colors.primary} />
        ) : (
          <Text
            style={
              backendConnected ? styles.connected : styles.disconnected
            }
          >
            {backendConnected
              ? "Connected to Jetson"
              : "Backend offline - check Jetson IP"}
          </Text>
        )}
      </View>

      <TouchableOpacity
        style={[styles.modeBtn, styles.signBtn]}
        onPress={() => navigation.navigate("Sign")}
        accessibilityLabel={a11y.home.signButton.label}
        accessibilityHint={a11y.home.signButton.hint}
        accessibilityRole="button"
      >
        <Text style={styles.modeBtnTitle}>SIGN mode</Text>
        <Text style={styles.modeBtnSub}>For deaf and mute people</Text>
        <Text style={styles.modeBtnDesc}>
          Sign in ASL with your phone camera. ARIA speaks your words aloud
          with emotion matching your facial expression.
        </Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={[styles.modeBtn, styles.guideBtn]}
        onPress={() => navigation.navigate("Guide")}
        accessibilityLabel={a11y.home.guideButton.label}
        accessibilityHint={a11y.home.guideButton.hint}
        accessibilityRole="button"
      >
        <Text style={styles.modeBtnTitle}>GUIDE mode</Text>
        <Text style={styles.modeBtnSub}>For blind people</Text>
        <Text style={styles.modeBtnDesc}>
          Point your camera ahead. ARIA warns of obstacles and provides
          turn-by-turn voice navigation.
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
    padding: spacing.xxl,
    justifyContent: "center",
  },
  logo: {
    fontSize: fontSize.display,
    fontWeight: "800",
    color: colors.primary,
    textAlign: "center",
    letterSpacing: 4,
  },
  tagline: {
    fontSize: fontSize.md,
    color: colors.primary,
    textAlign: "center",
    marginTop: spacing.xs,
    letterSpacing: 1,
  },
  greeting: {
    fontSize: fontSize.lg,
    color: colors.textSecondary,
    textAlign: "center",
    marginTop: spacing.sm,
  },
  statusRow: {
    alignItems: "center",
    marginVertical: spacing.xl,
  },
  connected: {
    color: colors.success,
    fontSize: fontSize.md,
  },
  disconnected: {
    color: colors.dangerLight,
    fontSize: fontSize.md,
  },
  modeBtn: {
    borderRadius: borderRadius.xl,
    padding: spacing.xl,
    marginBottom: spacing.lg,
    borderWidth: 1,
  },
  signBtn: {
    backgroundColor: colors.primaryDark,
    borderColor: colors.primary,
  },
  guideBtn: {
    backgroundColor: colors.successDark,
    borderColor: colors.successBorder,
  },
  modeBtnTitle: {
    fontSize: fontSize.xxxl,
    fontWeight: "700",
    color: colors.text,
    marginBottom: spacing.xs,
  },
  modeBtnSub: {
    fontSize: fontSize.md,
    color: colors.textSecondary,
    marginBottom: spacing.sm,
  },
  modeBtnDesc: {
    fontSize: fontSize.lg - 1,
    color: "#ccc",
    lineHeight: 20,
  },
});
