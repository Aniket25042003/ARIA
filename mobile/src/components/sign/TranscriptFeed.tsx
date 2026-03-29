import React, { useRef, useEffect } from "react";
import { View, Text, ScrollView, StyleSheet, Animated } from "react-native";
import { colors, spacing, borderRadius, fontSize, shadows } from "../../constants/theme";
import type { TranscriptEntry } from "../../types/sign";

interface Props {
  entries: TranscriptEntry[];
}

function TranscriptItem({ entry, index }: { entry: TranscriptEntry; index: number }) {
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(20)).current;

  const emotionColor = colors.emotion[entry.emotion] || colors.emotion.neutral;
  const isSOS = entry.emotion === "sos";

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 350,
        delay: index * 50,
        useNativeDriver: true,
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 350,
        delay: index * 50,
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  return (
    <Animated.View
      style={[
        styles.entry,
        isSOS && styles.entrySOSPulse,
        { opacity: fadeAnim, transform: [{ translateY: slideAnim }] },
      ]}
      accessibilityLabel={`${entry.text}. Emotion: ${entry.emotion}. Time: ${entry.timestamp}`}
    >
      <Text style={[styles.entryText, isSOS && styles.entryTextSOS]}>{entry.text}</Text>
      <View style={styles.entryMeta}>
        <View style={[styles.emotionTag, { backgroundColor: emotionColor + "18", borderColor: emotionColor + "30" }]}>
          <View style={[styles.emotionDot, { backgroundColor: emotionColor }]} />
          <Text style={[styles.entryEmotion, { color: emotionColor }]}>
            {entry.emotion}
          </Text>
        </View>
        <Text style={styles.entryTime}>{entry.timestamp}</Text>
      </View>
    </Animated.View>
  );
}

export default function TranscriptFeed({ entries }: Props) {
  const scrollRef = useRef<ScrollView>(null);

  useEffect(() => {
    scrollRef.current?.scrollToEnd({ animated: true });
  }, [entries]);

  return (
    <ScrollView ref={scrollRef} style={styles.container} showsVerticalScrollIndicator={false}>
      {entries.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyIcon}>S</Text>
          <Text style={styles.emptyTitle}>Ready to listen</Text>
          <Text style={styles.emptyHint}>
            Start signing in front of your camera. Your transcribed words will appear here.
          </Text>
        </View>
      ) : (
        entries.map((entry, index) => (
          <TranscriptItem key={entry.id} entry={entry} index={index} />
        ))
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    marginBottom: spacing.lg,
  },
  emptyContainer: {
    alignItems: "center",
    marginTop: 50,
    padding: spacing.xxl,
  },
  emptyIcon: {
    fontSize: 40,
    fontWeight: "800",
    color: colors.primaryLight,
    marginBottom: spacing.lg,
  },
  emptyTitle: {
    fontSize: fontSize.xxl,
    fontWeight: "700",
    color: colors.textDim,
    marginBottom: spacing.sm,
  },
  emptyHint: {
    color: colors.textPlaceholder,
    fontSize: fontSize.md,
    textAlign: "center",
    lineHeight: 22,
  },
  entry: {
    backgroundColor: colors.surface,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.sm + 2,
    borderWidth: 1,
    borderColor: colors.border,
    ...shadows.sm,
  },
  entrySOSPulse: {
    borderColor: colors.danger + "50",
    backgroundColor: colors.dangerSoft,
  },
  entryText: {
    color: colors.text,
    fontSize: fontSize.xxl,
    fontWeight: "600",
    marginBottom: spacing.sm,
    lineHeight: 28,
  },
  entryTextSOS: {
    color: colors.danger,
  },
  entryMeta: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  emotionTag: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: spacing.sm + 2,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.round,
    borderWidth: 1,
  },
  emotionDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    marginRight: spacing.xs,
  },
  entryEmotion: {
    fontSize: fontSize.sm,
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  entryTime: {
    color: colors.textDim,
    fontSize: fontSize.sm,
    fontWeight: "500",
  },
});
