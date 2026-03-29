import React, { useEffect, useRef } from "react";
import { View, Text, StyleSheet, Animated } from "react-native";
import { colors, spacing, borderRadius, fontSize, shadows } from "../../constants/theme";
import type { EmotionType } from "../../types/sign";

interface Props {
  emotion: EmotionType;
}

export default function EmotionBadge({ emotion }: Props) {
  const emotionColor = colors.emotion[emotion] || colors.emotion.neutral;
  const scaleAnim = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    Animated.sequence([
      Animated.timing(scaleAnim, { toValue: 1.15, duration: 150, useNativeDriver: true }),
      Animated.spring(scaleAnim, { toValue: 1, tension: 100, friction: 8, useNativeDriver: true }),
    ]).start();
  }, [emotion]);

  return (
    <View style={styles.row}>
      <Text style={styles.label}>Detected emotion</Text>
      <Animated.View
        style={[
          styles.badge,
          {
            backgroundColor: emotionColor + "18",
            borderColor: emotionColor + "40",
            transform: [{ scale: scaleAnim }],
          },
        ]}
        accessibilityLabel={`Current emotion: ${emotion}`}
      >
        <View style={[styles.dot, { backgroundColor: emotionColor }]} />
        <Text style={[styles.text, { color: emotionColor }]}>
          {emotion.toUpperCase()}
        </Text>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: spacing.md,
  },
  label: {
    color: colors.textMuted,
    fontSize: fontSize.md,
    marginRight: spacing.sm + 2,
    fontWeight: "500",
  },
  badge: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs + 2,
    borderRadius: borderRadius.round,
    borderWidth: 1,
    ...shadows.sm,
  },
  dot: {
    width: 7,
    height: 7,
    borderRadius: 4,
    marginRight: spacing.xs + 2,
  },
  text: {
    fontSize: fontSize.sm,
    fontWeight: "700",
    letterSpacing: 0.8,
  },
});
