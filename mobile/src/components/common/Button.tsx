import React, { useRef } from "react";
import {
  TouchableOpacity,
  Text,
  StyleSheet,
  ActivityIndicator,
  ViewStyle,
  TextStyle,
  Animated,
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { hapticLight } from "../../utils/haptics";
import { colors, spacing, borderRadius, fontSize, shadows } from "../../constants/theme";

interface Props {
  title: string;
  onPress: () => void;
  variant?: "primary" | "danger" | "success" | "outline";
  loading?: boolean;
  disabled?: boolean;
  accessibilityLabel?: string;
  accessibilityHint?: string;
  style?: ViewStyle;
  textStyle?: TextStyle;
}

const gradientMap = {
  primary: [colors.primary, colors.primaryDark] as [string, string],
  danger: [colors.danger, colors.dangerDark] as [string, string],
  success: [colors.success, colors.successDark] as [string, string],
  outline: [colors.surface, colors.surface] as [string, string],
};

export default function Button({
  title,
  onPress,
  variant = "primary",
  loading = false,
  disabled = false,
  accessibilityLabel,
  accessibilityHint,
  style,
  textStyle,
}: Props) {
  const scaleAnim = useRef(new Animated.Value(1)).current;

  const handlePressIn = () => {
    Animated.spring(scaleAnim, {
      toValue: 0.96,
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

  const handlePress = async () => {
    await hapticLight();
    onPress();
  };

  const isOutline = variant === "outline";
  const textColor = isOutline ? colors.primary : "#fff";

  return (
    <Animated.View style={[{ transform: [{ scale: scaleAnim }] }, style]}>
      <TouchableOpacity
        onPress={handlePress}
        onPressIn={handlePressIn}
        onPressOut={handlePressOut}
        disabled={disabled || loading}
        activeOpacity={0.9}
        accessibilityRole="button"
        accessibilityLabel={accessibilityLabel || title}
        accessibilityHint={accessibilityHint}
        accessibilityState={{ disabled: disabled || loading }}
        style={[disabled && styles.disabled]}
      >
        {isOutline ? (
          <LinearGradient
            colors={gradientMap[variant]}
            style={[styles.btn, styles.outlineBorder]}
          >
            {loading ? (
              <ActivityIndicator color={textColor} />
            ) : (
              <Text style={[styles.text, { color: textColor }, textStyle]}>
                {title}
              </Text>
            )}
          </LinearGradient>
        ) : (
          <LinearGradient
            colors={gradientMap[variant]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={[styles.btn, shadows.md]}
          >
            {loading ? (
              <ActivityIndicator color={textColor} />
            ) : (
              <Text style={[styles.text, { color: textColor }, textStyle]}>
                {title}
              </Text>
            )}
          </LinearGradient>
        )}
      </TouchableOpacity>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  btn: {
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    alignItems: "center",
    justifyContent: "center",
    minHeight: 52,
  },
  outlineBorder: {
    borderWidth: 1.5,
    borderColor: colors.primary,
  },
  text: {
    fontSize: fontSize.xl,
    fontWeight: "700",
  },
  disabled: {
    opacity: 0.5,
  },
});
