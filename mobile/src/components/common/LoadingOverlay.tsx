import React, { useEffect, useRef } from "react";
import { View, ActivityIndicator, Text, StyleSheet, Animated } from "react-native";
import { colors, fontSize, spacing } from "../../constants/theme";

interface Props {
  message?: string;
}

export default function LoadingOverlay({ message }: Props) {
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.9)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, { toValue: 1, duration: 250, useNativeDriver: true }),
      Animated.spring(scaleAnim, { toValue: 1, tension: 60, friction: 8, useNativeDriver: true }),
    ]).start();
  }, []);

  return (
    <Animated.View style={[styles.overlay, { opacity: fadeAnim }]}>
      <Animated.View style={[styles.card, { transform: [{ scale: scaleAnim }] }]}>
        <ActivityIndicator size="large" color={colors.primary} />
        {message && <Text style={styles.text}>{message}</Text>}
      </Animated.View>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  overlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(248, 249, 252, 0.85)",
    justifyContent: "center",
    alignItems: "center",
    zIndex: 100,
  },
  card: {
    backgroundColor: colors.surface,
    borderRadius: 20,
    padding: spacing.xxxl,
    alignItems: "center",
    shadowColor: colors.shadowColor,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.12,
    shadowRadius: 24,
    elevation: 8,
  },
  text: {
    color: colors.textSecondary,
    fontSize: fontSize.lg,
    fontWeight: "500",
    marginTop: spacing.lg,
  },
});
