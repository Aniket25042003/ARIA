import React, { useRef } from "react";
import { View, TouchableOpacity, Text, StyleSheet, Animated } from "react-native";
import { SUPPORTED_LANGUAGES } from "../../constants/config";
import { colors, spacing, borderRadius, fontSize, shadows } from "../../constants/theme";

interface Props {
  selected: string;
  onSelect: (code: string) => void;
}

function LangButton({
  lang,
  isSelected,
  onSelect,
}: {
  lang: { code: string; label: string };
  isSelected: boolean;
  onSelect: (code: string) => void;
}) {
  const scaleAnim = useRef(new Animated.Value(1)).current;

  const handlePressIn = () => {
    Animated.spring(scaleAnim, {
      toValue: 0.93,
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

  return (
    <Animated.View style={{ transform: [{ scale: scaleAnim }] }}>
      <TouchableOpacity
        style={[styles.btn, isSelected && styles.btnActive]}
        onPress={() => onSelect(lang.code)}
        onPressIn={handlePressIn}
        onPressOut={handlePressOut}
        activeOpacity={0.9}
        accessibilityLabel={`${lang.label} language`}
        accessibilityState={{ selected: isSelected }}
        accessibilityRole="button"
      >
        <Text style={[styles.text, isSelected && styles.textActive]}>
          {lang.label}
        </Text>
      </TouchableOpacity>
    </Animated.View>
  );
}

export default function LanguageSelector({ selected, onSelect }: Props) {
  return (
    <View style={styles.row}>
      {SUPPORTED_LANGUAGES.map((lang) => (
        <LangButton
          key={lang.code}
          lang={lang}
          isSelected={selected === lang.code}
          onSelect={onSelect}
        />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: "row",
    gap: spacing.sm,
    marginBottom: spacing.lg,
  },
  btn: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.round,
    backgroundColor: colors.surface,
    borderWidth: 1.5,
    borderColor: colors.border,
    ...shadows.sm,
  },
  btnActive: {
    backgroundColor: colors.primarySoft,
    borderColor: colors.primary,
  },
  text: {
    color: colors.textMuted,
    fontSize: fontSize.md,
    fontWeight: "600",
  },
  textActive: {
    color: colors.primary,
    fontWeight: "700",
  },
});
