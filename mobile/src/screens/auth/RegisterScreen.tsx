import React, { useState, useEffect, useRef } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Animated,
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { useAuthStore } from "../../stores/useAuthStore";
import { colors, spacing, borderRadius, fontSize, shadows } from "../../constants/theme";

export default function RegisterScreen() {
  const register = useAuthStore((s) => s.register);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [focusedField, setFocusedField] = useState<string | null>(null);

  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(40)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 500,
        useNativeDriver: true,
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 500,
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  const handleRegister = async () => {
    if (!name.trim() || !email.trim() || !password.trim()) {
      Alert.alert("Error", "Please fill in all fields");
      return;
    }
    if (password.length < 6) {
      Alert.alert("Error", "Password must be at least 6 characters");
      return;
    }
    setLoading(true);
    try {
      await register(email.trim(), password, name.trim());
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      const status = err.response?.status;
      const code = err.code;
      const url = err.config?.baseURL;
      const msg = detail
        ? `${status}: ${detail}`
        : `${code || "UNKNOWN"}: ${err.message}\n\nURL: ${url}`;
      Alert.alert("Registration Failed", msg);
    } finally {
      setLoading(false);
    }
  };

  const renderInput = (
    label: string,
    field: string,
    value: string,
    onChangeText: (t: string) => void,
    placeholder: string,
    options: {
      keyboardType?: "email-address" | "default";
      autoCapitalize?: "none" | "words";
      secureTextEntry?: boolean;
    } = {}
  ) => (
    <View
      style={[
        styles.inputContainer,
        focusedField === field && styles.inputFocused,
      ]}
    >
      <Text style={styles.inputLabel}>{label}</Text>
      <TextInput
        style={styles.input}
        placeholder={placeholder}
        placeholderTextColor={colors.textPlaceholder}
        value={value}
        onChangeText={onChangeText}
        onFocus={() => setFocusedField(field)}
        onBlur={() => setFocusedField(null)}
        autoCorrect={false}
        accessibilityLabel={label}
        {...options}
      />
    </View>
  );

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === "ios" ? "padding" : "height"}
    >
      <Animated.View
        style={[
          styles.inner,
          {
            opacity: fadeAnim,
            transform: [{ translateY: slideAnim }],
          },
        ]}
      >
        <View style={styles.headerSection}>
          <LinearGradient
            colors={[colors.primary, colors.primaryLight]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={styles.headerIcon}
          >
            <Text style={styles.headerIconText}>+</Text>
          </LinearGradient>
          <Text style={styles.title}>Join ARIA</Text>
          <Text style={styles.subtitle}>Create your account to get started</Text>
        </View>

        {renderInput("Full Name", "name", name, setName, "John Doe", {
          autoCapitalize: "words",
        })}
        {renderInput("Email", "email", email, setEmail, "your@email.com", {
          keyboardType: "email-address",
          autoCapitalize: "none",
        })}
        {renderInput(
          "Password",
          "password",
          password,
          setPassword,
          "Min 6 characters",
          { secureTextEntry: true }
        )}

        <TouchableOpacity
          style={styles.registerBtn}
          onPress={handleRegister}
          disabled={loading}
          activeOpacity={0.8}
          accessibilityRole="button"
          accessibilityLabel="Create account"
        >
          <LinearGradient
            colors={[colors.primary, colors.primaryDark]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={styles.registerGradient}
          >
            {loading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.registerBtnText}>Create Account</Text>
            )}
          </LinearGradient>
        </TouchableOpacity>
      </Animated.View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  inner: {
    flex: 1,
    justifyContent: "center",
    padding: spacing.xxl,
  },
  headerSection: {
    alignItems: "center",
    marginBottom: spacing.xxxl,
  },
  headerIcon: {
    width: 56,
    height: 56,
    borderRadius: 16,
    justifyContent: "center",
    alignItems: "center",
    marginBottom: spacing.lg,
    ...shadows.md,
  },
  headerIconText: {
    fontSize: 28,
    fontWeight: "700",
    color: "#fff",
  },
  title: {
    fontSize: fontSize.title,
    fontWeight: "800",
    color: colors.text,
    textAlign: "center",
  },
  subtitle: {
    fontSize: fontSize.lg,
    color: colors.textMuted,
    textAlign: "center",
    marginTop: spacing.xs,
  },
  inputContainer: {
    backgroundColor: colors.surface,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.md,
    borderWidth: 1.5,
    borderColor: colors.border,
    ...shadows.sm,
  },
  inputFocused: {
    borderColor: colors.primary,
    backgroundColor: colors.primarySoft + "40",
  },
  inputLabel: {
    fontSize: fontSize.xs,
    fontWeight: "600",
    color: colors.textMuted,
    textTransform: "uppercase",
    letterSpacing: 1,
    marginBottom: spacing.xs,
  },
  input: {
    color: colors.text,
    fontSize: fontSize.lg,
    padding: 0,
  },
  registerBtn: {
    borderRadius: borderRadius.lg,
    overflow: "hidden",
    marginTop: spacing.lg,
    ...shadows.md,
  },
  registerGradient: {
    padding: spacing.lg + 2,
    alignItems: "center",
    borderRadius: borderRadius.lg,
  },
  registerBtnText: {
    color: "#fff",
    fontSize: fontSize.xl,
    fontWeight: "700",
    letterSpacing: 0.5,
  },
});
