import React, { useEffect, useRef } from "react";
import { TouchableOpacity, Text, StyleSheet, Vibration, Animated } from "react-native";
import { colors, spacing, borderRadius, fontSize, shadows } from "../../constants/theme";
import { a11y } from "../../constants/accessibility";

interface Props {
  onPress: () => void;
  active: boolean;
}

export default function SOSButton({ onPress, active }: Props) {
  const scaleAnim = useRef(new Animated.Value(1)).current;
  const pulseAnim = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    if (active) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, { toValue: 1.08, duration: 300, useNativeDriver: true }),
          Animated.timing(pulseAnim, { toValue: 1, duration: 300, useNativeDriver: true }),
        ])
      ).start();
    } else {
      pulseAnim.setValue(1);
    }
  }, [active]);

  const handlePressIn = () => {
    Animated.spring(scaleAnim, {
      toValue: 0.92,
      tension: 100,
      friction: 10,
      useNativeDriver: true,
    }).start();
  };

  const handlePressOut = () => {
    Animated.spring(scaleAnim, {
      toValue: 1,
      tension: 100,
      friction: 10,
      useNativeDriver: true,
    }).start();
  };

  const handlePress = () => {
    Vibration.vibrate([0, 300, 200, 300, 200, 300]);
    onPress();
  };

  return (
    <Animated.View style={{ transform: [{ scale: Animated.multiply(scaleAnim, pulseAnim) }] }}>
      <TouchableOpacity
        style={[styles.btn, active && styles.btnActive]}
        onPress={handlePress}
        onPressIn={handlePressIn}
        onPressOut={handlePressOut}
        activeOpacity={0.9}
        accessibilityLabel={a11y.sign.sosButton.label}
        accessibilityHint={a11y.sign.sosButton.hint}
        accessibilityRole="button"
      >
        <Text style={[styles.text, active && styles.textActive]}>SOS</Text>
      </TouchableOpacity>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  btn: {
    backgroundColor: colors.dangerSoft,
    borderWidth: 2,
    borderColor: colors.danger,
    borderRadius: borderRadius.lg,
    paddingHorizontal: spacing.xxl,
    padding: spacing.lg,
    alignItems: "center",
    justifyContent: "center",
    ...shadows.sm,
  },
  btnActive: {
    backgroundColor: colors.danger,
    ...shadows.glow(colors.danger),
  },
  text: {
    color: colors.danger,
    fontSize: fontSize.xl,
    fontWeight: "800",
    letterSpacing: 2,
  },
  textActive: {
    color: "#fff",
  },
});
