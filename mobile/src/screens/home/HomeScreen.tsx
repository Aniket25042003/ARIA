import React, { useEffect, useRef } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Animated,
  ScrollView,
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { useNavigation } from "@react-navigation/native";
import type { BottomTabNavigationProp } from "@react-navigation/bottom-tabs";
import { useBackendHealth } from "../../hooks/useBackendHealth";
import { useConnectionStore } from "../../stores/useConnectionStore";
import { useAuthStore } from "../../stores/useAuthStore";
import { colors, spacing, borderRadius, fontSize, shadows } from "../../constants/theme";
import { a11y } from "../../constants/accessibility";
import type { MainTabParamList } from "../../navigation/types";

type HomeNav = BottomTabNavigationProp<MainTabParamList, "Home">;

export default function HomeScreen() {
  const navigation = useNavigation<HomeNav>();
  const backendConnected = useConnectionStore((s) => s.backendConnected);
  const backendChecked = useConnectionStore((s) => s.backendChecked);
  const userName = useAuthStore((s) => s.userName);

  useBackendHealth();

  // Staggered entrance animations
  const headerFade = useRef(new Animated.Value(0)).current;
  const headerSlide = useRef(new Animated.Value(-20)).current;
  const card1Fade = useRef(new Animated.Value(0)).current;
  const card1Slide = useRef(new Animated.Value(60)).current;
  const card2Fade = useRef(new Animated.Value(0)).current;
  const card2Slide = useRef(new Animated.Value(60)).current;
  const statusScale = useRef(new Animated.Value(0)).current;

  // Pulsing dot for connection status
  const pulseAnim = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    Animated.stagger(150, [
      Animated.parallel([
        Animated.timing(headerFade, { toValue: 1, duration: 500, useNativeDriver: true }),
        Animated.timing(headerSlide, { toValue: 0, duration: 500, useNativeDriver: true }),
      ]),
      Animated.spring(statusScale, { toValue: 1, tension: 60, friction: 8, useNativeDriver: true }),
      Animated.parallel([
        Animated.timing(card1Fade, { toValue: 1, duration: 500, useNativeDriver: true }),
        Animated.spring(card1Slide, { toValue: 0, tension: 50, friction: 9, useNativeDriver: true }),
      ]),
      Animated.parallel([
        Animated.timing(card2Fade, { toValue: 1, duration: 500, useNativeDriver: true }),
        Animated.spring(card2Slide, { toValue: 0, tension: 50, friction: 9, useNativeDriver: true }),
      ]),
    ]).start();

    // Pulse loop
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 0.4, duration: 1000, useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1, duration: 1000, useNativeDriver: true }),
      ])
    ).start();
  }, []);

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      showsVerticalScrollIndicator={false}
    >
      {/* Header */}
      <Animated.View
        style={[
          styles.headerSection,
          { opacity: headerFade, transform: [{ translateY: headerSlide }] },
        ]}
      >
        {userName && (
          <Text style={styles.greeting}>Welcome back,</Text>
        )}
        <Text style={styles.userName}>{userName || "Hello"}</Text>
      </Animated.View>

      {/* Connection Status */}
      <Animated.View
        style={[
          styles.statusCard,
          backendChecked && backendConnected && styles.statusCardConnected,
          backendChecked && !backendConnected && styles.statusCardDisconnected,
          { transform: [{ scale: statusScale }] },
        ]}
      >
        <View style={styles.statusRow}>
          <Animated.View
            style={[
              styles.statusDot,
              !backendChecked && styles.statusDotPending,
              backendChecked && backendConnected && styles.statusDotConnected,
              backendChecked && !backendConnected && styles.statusDotDisconnected,
              !backendChecked && { opacity: pulseAnim },
            ]}
          />
          <Text style={styles.statusText}>
            {!backendChecked
              ? "Connecting to Jetson..."
              : backendConnected
              ? "Connected to Jetson"
              : "Backend offline"}
          </Text>
        </View>
      </Animated.View>

      {/* SIGN Mode Card */}
      <Animated.View
        style={[
          { opacity: card1Fade, transform: [{ translateY: card1Slide }] },
        ]}
      >
        <TouchableOpacity
          style={styles.modeCard}
          onPress={() => navigation.navigate("Sign")}
          activeOpacity={0.9}
          accessibilityLabel={a11y.home.signButton.label}
          accessibilityHint={a11y.home.signButton.hint}
          accessibilityRole="button"
        >
          <LinearGradient
            colors={[colors.primary, "#8B5CF6"]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={styles.modeCardGradient}
          >
            <View style={styles.modeCardHeader}>
              <View style={styles.modeIconContainer}>
                <Text style={styles.modeIcon}>S</Text>
              </View>
              <View style={styles.modeBadge}>
                <Text style={styles.modeBadgeText}>For Deaf & Mute</Text>
              </View>
            </View>
            <Text style={styles.modeTitle}>SIGN Mode</Text>
            <Text style={styles.modeDesc}>
              Sign in ASL with your camera. ARIA speaks your words aloud with emotion matching your facial expression.
            </Text>
            <View style={styles.modeArrow}>
              <Text style={styles.modeArrowText}>Start Signing  ›</Text>
            </View>
          </LinearGradient>
        </TouchableOpacity>
      </Animated.View>

      {/* GUIDE Mode Card */}
      <Animated.View
        style={[
          { opacity: card2Fade, transform: [{ translateY: card2Slide }] },
        ]}
      >
        <TouchableOpacity
          style={styles.modeCard}
          onPress={() => navigation.navigate("Guide")}
          activeOpacity={0.9}
          accessibilityLabel={a11y.home.guideButton.label}
          accessibilityHint={a11y.home.guideButton.hint}
          accessibilityRole="button"
        >
          <LinearGradient
            colors={[colors.accent, "#2ED8A3"]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={styles.modeCardGradient}
          >
            <View style={styles.modeCardHeader}>
              <View style={styles.modeIconContainer}>
                <Text style={styles.modeIcon}>G</Text>
              </View>
              <View style={styles.modeBadge}>
                <Text style={styles.modeBadgeText}>For Blind</Text>
              </View>
            </View>
            <Text style={styles.modeTitle}>GUIDE Mode</Text>
            <Text style={styles.modeDesc}>
              Point your camera ahead. ARIA warns of obstacles and provides turn-by-turn voice navigation.
            </Text>
            <View style={styles.modeArrow}>
              <Text style={styles.modeArrowText}>Start Guiding  ›</Text>
            </View>
          </LinearGradient>
        </TouchableOpacity>
      </Animated.View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  content: {
    padding: spacing.xxl,
    paddingTop: spacing.lg,
  },
  headerSection: {
    marginBottom: spacing.xl,
  },
  greeting: {
    fontSize: fontSize.lg,
    color: colors.textMuted,
    marginBottom: spacing.xs - 2,
  },
  userName: {
    fontSize: fontSize.title,
    fontWeight: "800",
    color: colors.text,
  },
  statusCard: {
    backgroundColor: colors.surface,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.xxl,
    borderWidth: 1,
    borderColor: colors.border,
    ...shadows.sm,
  },
  statusCardConnected: {
    borderColor: colors.success + "40",
    backgroundColor: colors.successSoft,
  },
  statusCardDisconnected: {
    borderColor: colors.danger + "40",
    backgroundColor: colors.dangerSoft,
  },
  statusRow: {
    flexDirection: "row",
    alignItems: "center",
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: spacing.sm + 2,
  },
  statusDotPending: {
    backgroundColor: colors.warning,
  },
  statusDotConnected: {
    backgroundColor: colors.success,
  },
  statusDotDisconnected: {
    backgroundColor: colors.danger,
  },
  statusText: {
    fontSize: fontSize.md,
    fontWeight: "600",
    color: colors.textSecondary,
  },
  modeCard: {
    borderRadius: borderRadius.xxl,
    overflow: "hidden",
    marginBottom: spacing.lg,
    ...shadows.lg,
  },
  modeCardGradient: {
    padding: spacing.xxl,
    borderRadius: borderRadius.xxl,
  },
  modeCardHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: spacing.lg,
  },
  modeIconContainer: {
    width: 44,
    height: 44,
    borderRadius: 14,
    backgroundColor: "rgba(255,255,255,0.25)",
    justifyContent: "center",
    alignItems: "center",
  },
  modeIcon: {
    fontSize: 22,
    fontWeight: "800",
    color: "#fff",
  },
  modeBadge: {
    backgroundColor: "rgba(255,255,255,0.2)",
    borderRadius: borderRadius.full,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
  },
  modeBadgeText: {
    fontSize: fontSize.xs,
    fontWeight: "700",
    color: "rgba(255,255,255,0.9)",
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  modeTitle: {
    fontSize: fontSize.xxxl + 4,
    fontWeight: "800",
    color: "#fff",
    marginBottom: spacing.sm,
  },
  modeDesc: {
    fontSize: fontSize.md + 1,
    color: "rgba(255,255,255,0.85)",
    lineHeight: 22,
    marginBottom: spacing.lg,
  },
  modeArrow: {
    alignSelf: "flex-start",
    backgroundColor: "rgba(255,255,255,0.2)",
    borderRadius: borderRadius.full,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
  },
  modeArrowText: {
    color: "#fff",
    fontSize: fontSize.md,
    fontWeight: "700",
  },
});
