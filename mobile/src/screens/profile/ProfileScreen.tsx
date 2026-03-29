import React, { useState, useEffect, useRef } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  TextInput,
  StyleSheet,
  Alert,
  Animated,
  ScrollView,
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { useAuthStore } from "../../stores/useAuthStore";
import { useConnectionStore } from "../../stores/useConnectionStore";
import { BACKEND_URL as DEFAULT_BACKEND_URL } from "../../constants/config";
import { colors, spacing, borderRadius, fontSize, shadows } from "../../constants/theme";

export default function ProfileScreen() {
  const { userName, logout } = useAuthStore();
  const { backendUrl, backendConnected, backendChecked, setBackendUrl, resetBackendUrl } =
    useConnectionStore();
  const [urlInput, setUrlInput] = useState(backendUrl);
  const [saving, setSaving] = useState(false);
  const [urlFocused, setUrlFocused] = useState(false);

  const isCustomUrl = backendUrl !== DEFAULT_BACKEND_URL;

  // Animations
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(30)).current;
  const avatarScale = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, { toValue: 1, duration: 500, useNativeDriver: true }),
      Animated.timing(slideAnim, { toValue: 0, duration: 500, useNativeDriver: true }),
      Animated.spring(avatarScale, { toValue: 1, tension: 50, friction: 7, useNativeDriver: true }),
    ]).start();
  }, []);

  const handleSaveUrl = async () => {
    const trimmed = urlInput.trim();
    if (!trimmed) {
      Alert.alert("Invalid URL", "Please enter a valid server URL.");
      return;
    }
    if (!trimmed.startsWith("http://") && !trimmed.startsWith("https://")) {
      Alert.alert("Invalid URL", "URL must start with http:// or https://");
      return;
    }
    setSaving(true);
    await setBackendUrl(trimmed);
    setSaving(false);
  };

  const handleReset = async () => {
    await resetBackendUrl();
    setUrlInput(DEFAULT_BACKEND_URL);
  };

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      showsVerticalScrollIndicator={false}
    >
      {/* Profile Header */}
      <Animated.View
        style={[
          styles.header,
          { opacity: fadeAnim, transform: [{ translateY: slideAnim }] },
        ]}
      >
        <Animated.View style={[styles.avatarContainer, { transform: [{ scale: avatarScale }] }]}>
          <LinearGradient
            colors={[colors.primary, colors.primaryLight]}
            style={styles.avatar}
          >
            <Text style={styles.avatarText}>
              {userName?.charAt(0)?.toUpperCase() || "?"}
            </Text>
          </LinearGradient>
        </Animated.View>
        <Text style={styles.name}>{userName || "User"}</Text>
      </Animated.View>

      {/* Server Connection Card */}
      <Animated.View
        style={[
          styles.card,
          { opacity: fadeAnim, transform: [{ translateY: slideAnim }] },
        ]}
      >
        <Text style={styles.sectionTitle}>Server Connection</Text>
        <Text style={styles.hint}>
          Enter your backend URL or tunnel URL (e.g. ngrok, cloudflared) to connect across networks.
        </Text>

        <View style={[styles.inputContainer, urlFocused && styles.inputFocused]}>
          <Text style={styles.inputLabel}>BACKEND URL</Text>
          <TextInput
            style={styles.urlInput}
            value={urlInput}
            onChangeText={setUrlInput}
            onFocus={() => setUrlFocused(true)}
            onBlur={() => setUrlFocused(false)}
            placeholder="https://your-tunnel-url.trycloudflare.com"
            placeholderTextColor={colors.textPlaceholder}
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="url"
          />
        </View>

        <View style={styles.urlActions}>
          <TouchableOpacity
            style={[styles.saveBtn, saving && styles.btnDisabled]}
            onPress={handleSaveUrl}
            disabled={saving}
            activeOpacity={0.8}
            accessibilityRole="button"
            accessibilityLabel="Save server URL"
          >
            <LinearGradient
              colors={[colors.primary, colors.primaryDark]}
              style={styles.saveBtnGradient}
            >
              <Text style={styles.saveBtnText}>
                {saving ? "Saving..." : "Save"}
              </Text>
            </LinearGradient>
          </TouchableOpacity>

          {isCustomUrl && (
            <TouchableOpacity
              style={styles.resetBtn}
              onPress={handleReset}
              activeOpacity={0.7}
              accessibilityRole="button"
              accessibilityLabel="Reset to default URL"
            >
              <Text style={styles.resetBtnText}>Reset</Text>
            </TouchableOpacity>
          )}
        </View>

        {/* Status */}
        <View style={[
          styles.statusCard,
          backendChecked && backendConnected && styles.statusConnectedCard,
          backendChecked && !backendConnected && styles.statusDisconnectedCard,
        ]}>
          <View style={styles.statusRow}>
            <View
              style={[
                styles.statusDot,
                backendChecked
                  ? backendConnected
                    ? styles.statusConnected
                    : styles.statusDisconnected
                  : styles.statusPending,
              ]}
            />
            <Text style={styles.statusText}>
              {!backendChecked
                ? "Checking..."
                : backendConnected
                ? "Connected"
                : "Disconnected"}
            </Text>
            {isCustomUrl && (
              <View style={styles.customBadge}>
                <Text style={styles.customBadgeText}>TUNNEL</Text>
              </View>
            )}
          </View>
          <Text style={styles.currentUrl} numberOfLines={1}>
            {backendUrl}
          </Text>
        </View>
      </Animated.View>

      {/* Sign Out */}
      <TouchableOpacity
        style={styles.logoutBtn}
        onPress={logout}
        activeOpacity={0.8}
        accessibilityRole="button"
        accessibilityLabel="Sign out"
      >
        <Text style={styles.logoutBtnText}>Sign Out</Text>
      </TouchableOpacity>
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
    paddingBottom: spacing.xxxxl,
  },
  header: {
    alignItems: "center",
    marginTop: spacing.lg,
    marginBottom: spacing.xxl,
  },
  avatarContainer: {
    marginBottom: spacing.lg,
  },
  avatar: {
    width: 88,
    height: 88,
    borderRadius: 28,
    justifyContent: "center",
    alignItems: "center",
    ...shadows.lg,
  },
  avatarText: {
    fontSize: 36,
    fontWeight: "800",
    color: "#fff",
  },
  name: {
    fontSize: fontSize.xxxl,
    fontWeight: "700",
    color: colors.text,
  },
  card: {
    backgroundColor: colors.surface,
    borderRadius: borderRadius.xxl,
    padding: spacing.xxl,
    marginBottom: spacing.xxl,
    ...shadows.md,
  },
  sectionTitle: {
    fontSize: fontSize.xxl,
    fontWeight: "700",
    color: colors.text,
    marginBottom: spacing.sm,
  },
  hint: {
    fontSize: fontSize.md,
    color: colors.textMuted,
    marginBottom: spacing.xl,
    lineHeight: 20,
  },
  inputContainer: {
    backgroundColor: colors.surfaceLight,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.lg,
    borderWidth: 1.5,
    borderColor: colors.border,
  },
  inputFocused: {
    borderColor: colors.primary,
    backgroundColor: colors.primarySoft + "40",
  },
  inputLabel: {
    fontSize: fontSize.xs,
    fontWeight: "600",
    color: colors.textMuted,
    letterSpacing: 1,
    marginBottom: spacing.xs,
  },
  urlInput: {
    color: colors.text,
    fontSize: fontSize.md,
    padding: 0,
  },
  urlActions: {
    flexDirection: "row",
    gap: spacing.md,
    marginBottom: spacing.lg,
  },
  saveBtn: {
    flex: 1,
    borderRadius: borderRadius.lg,
    overflow: "hidden",
    ...shadows.sm,
  },
  saveBtnGradient: {
    padding: spacing.md + 2,
    alignItems: "center",
    borderRadius: borderRadius.lg,
  },
  saveBtnText: {
    color: "#fff",
    fontSize: fontSize.lg,
    fontWeight: "600",
  },
  resetBtn: {
    backgroundColor: colors.surfaceLight,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: borderRadius.lg,
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.md + 2,
    alignItems: "center",
  },
  resetBtnText: {
    color: colors.textSecondary,
    fontSize: fontSize.lg,
    fontWeight: "600",
  },
  btnDisabled: {
    opacity: 0.5,
  },
  statusCard: {
    backgroundColor: colors.surfaceLight,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border,
  },
  statusConnectedCard: {
    backgroundColor: colors.successSoft,
    borderColor: colors.success + "40",
  },
  statusDisconnectedCard: {
    backgroundColor: colors.dangerSoft,
    borderColor: colors.danger + "40",
  },
  statusRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: spacing.xs,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: spacing.sm,
  },
  statusConnected: {
    backgroundColor: colors.success,
  },
  statusDisconnected: {
    backgroundColor: colors.danger,
  },
  statusPending: {
    backgroundColor: colors.warning,
  },
  statusText: {
    fontSize: fontSize.lg,
    fontWeight: "600",
    color: colors.textSecondary,
  },
  customBadge: {
    marginLeft: spacing.sm,
    backgroundColor: colors.primarySoft,
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: borderRadius.sm,
  },
  customBadgeText: {
    fontSize: fontSize.xs,
    fontWeight: "700",
    color: colors.primary,
  },
  currentUrl: {
    fontSize: fontSize.sm,
    color: colors.textDim,
  },
  logoutBtn: {
    backgroundColor: colors.dangerSoft,
    borderWidth: 1.5,
    borderColor: colors.danger + "30",
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    alignItems: "center",
    ...shadows.sm,
  },
  logoutBtnText: {
    color: colors.danger,
    fontSize: fontSize.xl,
    fontWeight: "700",
  },
});
