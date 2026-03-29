import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  Animated,
  RefreshControl,
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { fetchDashboardInsights } from "../../services/dashboard";
import type { DashboardStats, DashboardInsights } from "../../types/dashboard";
import { colors, spacing, borderRadius, fontSize, shadows } from "../../constants/theme";

// ── Animated Stat Card ──
function StatCard({
  label,
  value,
  color,
  delay = 0,
}: {
  label: string;
  value: string | number;
  color: string;
  delay?: number;
}) {
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(24)).current;
  const scaleAnim = useRef(new Animated.Value(0.9)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 450,
        delay,
        useNativeDriver: true,
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 450,
        delay,
        useNativeDriver: true,
      }),
      Animated.spring(scaleAnim, {
        toValue: 1,
        tension: 60,
        friction: 8,
        delay,
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  return (
    <Animated.View
      style={[
        styles.statCard,
        {
          opacity: fadeAnim,
          transform: [{ translateY: slideAnim }, { scale: scaleAnim }],
        },
      ]}
    >
      <View style={[styles.statAccent, { backgroundColor: color }]} />
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </Animated.View>
  );
}

// ── Emotion Bar ──
function EmotionBar({
  emotion,
  count,
  maxCount,
  delay,
}: {
  emotion: string;
  count: number;
  maxCount: number;
  delay: number;
}) {
  const widthAnim = useRef(new Animated.Value(0)).current;
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const barColor = colors.emotion[emotion] || colors.textMuted;
  const pct = maxCount > 0 ? count / maxCount : 0;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, { toValue: 1, duration: 300, delay, useNativeDriver: false }),
      Animated.timing(widthAnim, { toValue: pct, duration: 600, delay, useNativeDriver: false }),
    ]).start();
  }, []);

  return (
    <Animated.View style={[styles.emotionRow, { opacity: fadeAnim }]}>
      <Text style={styles.emotionLabel}>{emotion}</Text>
      <View style={styles.emotionBarBg}>
        <Animated.View
          style={[
            styles.emotionBarFill,
            {
              backgroundColor: barColor,
              width: widthAnim.interpolate({
                inputRange: [0, 1],
                outputRange: ["0%", "100%"],
              }),
            },
          ]}
        />
      </View>
      <Text style={styles.emotionCount}>{count}</Text>
    </Animated.View>
  );
}

// ── Insight Card ──
function InsightCard({
  icon,
  title,
  text,
  gradientColors,
  delay = 0,
}: {
  icon: string;
  title: string;
  text: string;
  gradientColors: [string, string];
  delay?: number;
}) {
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(30)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, { toValue: 1, duration: 400, delay, useNativeDriver: true }),
      Animated.timing(slideAnim, { toValue: 0, duration: 400, delay, useNativeDriver: true }),
    ]).start();
  }, []);

  return (
    <Animated.View
      style={[
        styles.insightCard,
        { opacity: fadeAnim, transform: [{ translateY: slideAnim }] },
      ]}
    >
      <LinearGradient
        colors={gradientColors}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.insightIconBg}
      >
        <Text style={styles.insightIcon}>{icon}</Text>
      </LinearGradient>
      <View style={styles.insightContent}>
        <Text style={styles.insightTitle}>{title}</Text>
        <Text style={styles.insightText}>{text}</Text>
      </View>
    </Animated.View>
  );
}

// ── Main Dashboard Screen ──
export default function DashboardScreen() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [insights, setInsights] = useState<DashboardInsights | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  // Entrance animation
  const headerFade = useRef(new Animated.Value(0)).current;
  const headerSlide = useRef(new Animated.Value(-20)).current;

  const loadDashboard = useCallback(async (isRefresh = false) => {
    if (isRefresh) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }
    setError(null);
    try {
      const data = await fetchDashboardInsights();
      setStats(data.stats);
      setInsights(data.insights);
    } catch (err: any) {
      const msg =
        err.response?.data?.detail ||
        err.message ||
        "Failed to load dashboard";
      setError(msg);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    Animated.parallel([
      Animated.timing(headerFade, { toValue: 1, duration: 500, useNativeDriver: true }),
      Animated.timing(headerSlide, { toValue: 0, duration: 500, useNativeDriver: true }),
    ]).start();

    loadDashboard();
  }, []);

  // ── Empty / loading / error states ──
  if (loading && !stats) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color={colors.primary} />
        <Text style={styles.loadingText}>Analyzing your usage...</Text>
      </View>
    );
  }

  if (error && !stats) {
    return (
      <View style={styles.centered}>
        <View style={styles.errorCard}>
          <Text style={styles.errorIcon}>!</Text>
          <Text style={styles.errorTitle}>Could not load dashboard</Text>
          <Text style={styles.errorMsg}>{error}</Text>
          <TouchableOpacity
            style={styles.retryBtn}
            onPress={() => loadDashboard()}
            activeOpacity={0.8}
          >
            <LinearGradient
              colors={[colors.primary, colors.primaryDark]}
              style={styles.retryGradient}
            >
              <Text style={styles.retryText}>Retry</Text>
            </LinearGradient>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  if (!stats) return null;

  const emotionEntries = Object.entries(stats.sign_mode.emotion_distribution).sort(
    (a, b) => b[1] - a[1]
  );
  const maxEmotion = emotionEntries.length > 0 ? emotionEntries[0][1] : 1;

  const langEntries = Object.entries(stats.sign_mode.language_usage);
  const totalLang = langEntries.reduce((s, [, c]) => s + c, 0) || 1;

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      showsVerticalScrollIndicator={false}
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={() => loadDashboard(true)}
          tintColor={colors.primary}
          colors={[colors.primary]}
        />
      }
    >
      {/* Header */}
      <Animated.View
        style={[
          styles.header,
          { opacity: headerFade, transform: [{ translateY: headerSlide }] },
        ]}
      >
        {insights?.greeting ? (
          <Text style={styles.greeting}>{insights.greeting}</Text>
        ) : (
          <Text style={styles.greeting}>Your Dashboard</Text>
        )}
        {insights?.streak_text && (
          <View style={styles.streakBadge}>
            <Text style={styles.streakText}>{insights.streak_text}</Text>
          </View>
        )}
      </Animated.View>

      {/* AI Summary */}
      {insights?.summary && (
        <InsightCard
          icon="AI"
          title="AI Summary"
          text={insights.summary}
          gradientColors={[colors.primary, colors.primaryLight]}
          delay={100}
        />
      )}

      {/* Quick Stats */}
      <Text style={styles.sectionTitle}>Overview</Text>
      <View style={styles.statsGrid}>
        <StatCard
          label="Sign Sessions"
          value={stats.sign_mode.total_sessions}
          color={colors.primary}
          delay={150}
        />
        <StatCard
          label="Sentences"
          value={stats.sign_mode.total_sentences}
          color={colors.primaryLight}
          delay={200}
        />
        <StatCard
          label="Navigations"
          value={stats.guide_mode.total_navigations}
          color={colors.accent}
          delay={250}
        />
        <StatCard
          label="Obstacle Scans"
          value={stats.guide_mode.total_obstacle_scans}
          color={colors.accentLight}
          delay={300}
        />
        <StatCard
          label="SOS Events"
          value={stats.sos_events}
          color={stats.sos_events > 0 ? colors.danger : colors.textDim}
          delay={350}
        />
        <StatCard
          label="Avg Session"
          value={
            stats.sign_mode.avg_session_duration_seconds > 0
              ? `${Math.round(stats.sign_mode.avg_session_duration_seconds)}s`
              : "--"
          }
          color={colors.warning}
          delay={400}
        />
      </View>

      {/* SIGN Insight */}
      {insights?.sign_insight && (
        <InsightCard
          icon="S"
          title="Sign Mode"
          text={insights.sign_insight}
          gradientColors={[colors.primary, "#8B5CF6"]}
          delay={300}
        />
      )}

      {/* Emotion Distribution */}
      {emotionEntries.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Emotion Distribution</Text>
          <View style={styles.card}>
            {emotionEntries.map(([emotion, count], i) => (
              <EmotionBar
                key={emotion}
                emotion={emotion}
                count={count}
                maxCount={maxEmotion}
                delay={350 + i * 80}
              />
            ))}
          </View>
        </View>
      )}

      {/* Language Usage */}
      {langEntries.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Languages Used</Text>
          <View style={styles.langRow}>
            {langEntries.map(([lang, count]) => {
              const pct = Math.round((count / totalLang) * 100);
              return (
                <View key={lang} style={styles.langChip}>
                  <Text style={styles.langCode}>{lang.toUpperCase()}</Text>
                  <Text style={styles.langPct}>{pct}%</Text>
                </View>
              );
            })}
          </View>
        </View>
      )}

      {/* GUIDE Insight */}
      {insights?.guide_insight && (
        <InsightCard
          icon="G"
          title="Guide Mode"
          text={insights.guide_insight}
          gradientColors={[colors.accent, "#2ED8A3"]}
          delay={400}
        />
      )}

      {/* Recent Sentences */}
      {stats.sign_mode.recent_sentences.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Recent Translations</Text>
          {stats.sign_mode.recent_sentences.map((s, i) => {
            const emotionColor = colors.emotion[s.emotion] || colors.textMuted;
            return (
              <View key={i} style={styles.recentItem}>
                <View style={[styles.recentDot, { backgroundColor: emotionColor }]} />
                <View style={styles.recentContent}>
                  <Text style={styles.recentText} numberOfLines={2}>
                    {s.text}
                  </Text>
                  <Text style={[styles.recentEmotion, { color: emotionColor }]}>
                    {s.emotion}
                  </Text>
                </View>
              </View>
            );
          })}
        </View>
      )}

      {/* Tip */}
      {insights?.tip && (
        <View style={styles.tipCard}>
          <LinearGradient
            colors={[colors.warningSoft, "#FFF5DC"]}
            style={styles.tipGradient}
          >
            <Text style={styles.tipLabel}>Tip</Text>
            <Text style={styles.tipText}>{insights.tip}</Text>
          </LinearGradient>
        </View>
      )}

      {/* Footer refresh hint */}
      <Text style={styles.footerHint}>Pull down to refresh</Text>
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
  centered: {
    flex: 1,
    backgroundColor: colors.background,
    justifyContent: "center",
    alignItems: "center",
    padding: spacing.xxl,
  },
  loadingText: {
    marginTop: spacing.lg,
    fontSize: fontSize.lg,
    color: colors.textMuted,
    fontWeight: "500",
  },

  // Error
  errorCard: {
    backgroundColor: colors.surface,
    borderRadius: borderRadius.xxl,
    padding: spacing.xxxl,
    alignItems: "center",
    width: "100%",
    ...shadows.lg,
  },
  errorIcon: {
    fontSize: 36,
    fontWeight: "800",
    color: colors.danger,
    marginBottom: spacing.lg,
  },
  errorTitle: {
    fontSize: fontSize.xxl,
    fontWeight: "700",
    color: colors.text,
    marginBottom: spacing.sm,
  },
  errorMsg: {
    fontSize: fontSize.md,
    color: colors.textMuted,
    textAlign: "center",
    marginBottom: spacing.xxl,
  },
  retryBtn: {
    borderRadius: borderRadius.lg,
    overflow: "hidden",
    width: "100%",
  },
  retryGradient: {
    padding: spacing.lg,
    alignItems: "center",
    borderRadius: borderRadius.lg,
  },
  retryText: {
    color: "#fff",
    fontWeight: "700",
    fontSize: fontSize.xl,
  },

  // Header
  header: {
    marginBottom: spacing.xxl,
  },
  greeting: {
    fontSize: fontSize.title,
    fontWeight: "800",
    color: colors.text,
    marginBottom: spacing.sm,
  },
  streakBadge: {
    backgroundColor: colors.primarySoft,
    borderRadius: borderRadius.round,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    alignSelf: "flex-start",
  },
  streakText: {
    color: colors.primary,
    fontSize: fontSize.md,
    fontWeight: "700",
  },

  // Section
  section: {
    marginBottom: spacing.xl,
  },
  sectionTitle: {
    fontSize: fontSize.xxl,
    fontWeight: "700",
    color: colors.text,
    marginBottom: spacing.lg,
  },

  // Stats grid
  statsGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: spacing.md,
    marginBottom: spacing.xxl,
  },
  statCard: {
    backgroundColor: colors.surface,
    borderRadius: borderRadius.xl,
    padding: spacing.lg,
    width: "47.5%",
    position: "relative",
    overflow: "hidden",
    ...shadows.sm,
  },
  statAccent: {
    position: "absolute",
    top: 0,
    left: 0,
    width: 4,
    height: "100%",
    borderTopLeftRadius: borderRadius.xl,
    borderBottomLeftRadius: borderRadius.xl,
  },
  statValue: {
    fontSize: fontSize.title,
    fontWeight: "800",
    color: colors.text,
    marginBottom: spacing.xs - 2,
  },
  statLabel: {
    fontSize: fontSize.sm,
    fontWeight: "600",
    color: colors.textMuted,
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },

  // Insight card
  insightCard: {
    backgroundColor: colors.surface,
    borderRadius: borderRadius.xl,
    padding: spacing.xl,
    marginBottom: spacing.xl,
    flexDirection: "row",
    alignItems: "flex-start",
    ...shadows.md,
  },
  insightIconBg: {
    width: 42,
    height: 42,
    borderRadius: 13,
    justifyContent: "center",
    alignItems: "center",
    marginRight: spacing.lg,
  },
  insightIcon: {
    fontSize: 16,
    fontWeight: "800",
    color: "#fff",
  },
  insightContent: {
    flex: 1,
  },
  insightTitle: {
    fontSize: fontSize.lg,
    fontWeight: "700",
    color: colors.text,
    marginBottom: spacing.xs,
  },
  insightText: {
    fontSize: fontSize.md,
    color: colors.textSecondary,
    lineHeight: 21,
  },

  // Emotion bars
  card: {
    backgroundColor: colors.surface,
    borderRadius: borderRadius.xl,
    padding: spacing.xl,
    ...shadows.sm,
  },
  emotionRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: spacing.md,
  },
  emotionLabel: {
    width: 70,
    fontSize: fontSize.sm,
    fontWeight: "600",
    color: colors.textSecondary,
    textTransform: "capitalize",
  },
  emotionBarBg: {
    flex: 1,
    height: 10,
    borderRadius: 5,
    backgroundColor: colors.surfaceSecondary,
    marginHorizontal: spacing.sm,
    overflow: "hidden",
  },
  emotionBarFill: {
    height: "100%",
    borderRadius: 5,
  },
  emotionCount: {
    width: 30,
    fontSize: fontSize.sm,
    fontWeight: "700",
    color: colors.textSecondary,
    textAlign: "right",
  },

  // Language chips
  langRow: {
    flexDirection: "row",
    gap: spacing.sm,
    flexWrap: "wrap",
  },
  langChip: {
    backgroundColor: colors.surface,
    borderRadius: borderRadius.lg,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    alignItems: "center",
    borderWidth: 1,
    borderColor: colors.border,
    ...shadows.sm,
  },
  langCode: {
    fontSize: fontSize.lg,
    fontWeight: "800",
    color: colors.primary,
    marginBottom: spacing.xs - 2,
  },
  langPct: {
    fontSize: fontSize.sm,
    fontWeight: "600",
    color: colors.textMuted,
  },

  // Recent sentences
  recentItem: {
    flexDirection: "row",
    alignItems: "flex-start",
    marginBottom: spacing.md,
    backgroundColor: colors.surface,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    ...shadows.sm,
  },
  recentDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginTop: 6,
    marginRight: spacing.md,
  },
  recentContent: {
    flex: 1,
  },
  recentText: {
    fontSize: fontSize.md,
    color: colors.text,
    fontWeight: "500",
    lineHeight: 20,
    marginBottom: spacing.xs - 2,
  },
  recentEmotion: {
    fontSize: fontSize.xs,
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },

  // Tip card
  tipCard: {
    borderRadius: borderRadius.xl,
    overflow: "hidden",
    marginBottom: spacing.xl,
    ...shadows.sm,
  },
  tipGradient: {
    padding: spacing.xl,
    borderRadius: borderRadius.xl,
  },
  tipLabel: {
    fontSize: fontSize.xs,
    fontWeight: "800",
    color: colors.warningDark,
    textTransform: "uppercase",
    letterSpacing: 1,
    marginBottom: spacing.xs,
  },
  tipText: {
    fontSize: fontSize.md,
    color: colors.text,
    lineHeight: 21,
    fontWeight: "500",
  },

  // Footer
  footerHint: {
    textAlign: "center",
    fontSize: fontSize.sm,
    color: colors.textDim,
    marginTop: spacing.sm,
  },
});
