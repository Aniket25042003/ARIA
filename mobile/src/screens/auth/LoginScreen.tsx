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
import { useNavigation } from "@react-navigation/native";
import type { NativeStackNavigationProp } from "@react-navigation/native-stack";
import { useAuthStore } from "../../stores/useAuthStore";
import { colors, spacing, borderRadius, fontSize, shadows } from "../../constants/theme";
import type { AuthStackParamList } from "../../navigation/types";

type LoginNav = NativeStackNavigationProp<AuthStackParamList, "Login">;

export default function LoginScreen() {
  const navigation = useNavigation<LoginNav>();
  const login = useAuthStore((s) => s.login);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [emailFocused, setEmailFocused] = useState(false);
  const [passFocused, setPassFocused] = useState(false);

  // Animations
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(30)).current;
  const logoScale = useRef(new Animated.Value(0.8)).current;
  const formSlide = useRef(new Animated.Value(50)).current;
  const formFade = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.sequence([
      Animated.parallel([
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 600,
          useNativeDriver: true,
        }),
        Animated.spring(logoScale, {
          toValue: 1,
          tension: 50,
          friction: 7,
          useNativeDriver: true,
        }),
        Animated.timing(slideAnim, {
          toValue: 0,
          duration: 600,
          useNativeDriver: true,
        }),
      ]),
      Animated.parallel([
        Animated.timing(formFade, {
          toValue: 1,
          duration: 400,
          useNativeDriver: true,
        }),
        Animated.timing(formSlide, {
          toValue: 0,
          duration: 400,
          useNativeDriver: true,
        }),
      ]),
    ]).start();
  }, []);

  const handleLogin = async () => {
    if (!email.trim() || !password.trim()) {
      Alert.alert("Error", "Please fill in all fields");
      return;
    }
    setLoading(true);
    try {
      await login(email.trim(), password);
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      const status = err.response?.status;
      const code = err.code;
      const msg = detail
        ? `${status}: ${detail}`
        : `${code || "UNKNOWN"}: ${err.message}`;
      Alert.alert("Login Failed", msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === "ios" ? "padding" : "height"}
    >
      <View style={styles.inner}>
        {/* Logo Section */}
        <Animated.View
          style={[
            styles.logoSection,
            {
              opacity: fadeAnim,
              transform: [{ translateY: slideAnim }, { scale: logoScale }],
            },
          ]}
        >
          <View style={styles.logoContainer}>
            <LinearGradient
              colors={[colors.primary, colors.primaryLight]}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={styles.logoBg}
            >
              <Text style={styles.logoIcon}>A</Text>
            </LinearGradient>
          </View>
          <Text style={styles.logo}>ARIA</Text>
          <Text style={styles.tagline}>
            Adaptive Real-time Intelligence Assistant
          </Text>
        </Animated.View>

        {/* Form Section */}
        <Animated.View
          style={[
            styles.formSection,
            {
              opacity: formFade,
              transform: [{ translateY: formSlide }],
            },
          ]}
        >
          <Text style={styles.subtitle}>Sign in to continue</Text>

          <View style={[styles.inputContainer, emailFocused && styles.inputFocused]}>
            <Text style={styles.inputLabel}>Email</Text>
            <TextInput
              style={styles.input}
              placeholder="your@email.com"
              placeholderTextColor={colors.textPlaceholder}
              value={email}
              onChangeText={setEmail}
              onFocus={() => setEmailFocused(true)}
              onBlur={() => setEmailFocused(false)}
              keyboardType="email-address"
              autoCapitalize="none"
              autoCorrect={false}
              accessibilityLabel="Email address"
            />
          </View>

          <View style={[styles.inputContainer, passFocused && styles.inputFocused]}>
            <Text style={styles.inputLabel}>Password</Text>
            <TextInput
              style={styles.input}
              placeholder="Enter your password"
              placeholderTextColor={colors.textPlaceholder}
              value={password}
              onChangeText={setPassword}
              onFocus={() => setPassFocused(true)}
              onBlur={() => setPassFocused(false)}
              secureTextEntry
              accessibilityLabel="Password"
            />
          </View>

          <TouchableOpacity
            style={styles.loginBtn}
            onPress={handleLogin}
            disabled={loading}
            activeOpacity={0.8}
            accessibilityRole="button"
            accessibilityLabel="Sign in"
          >
            <LinearGradient
              colors={[colors.primary, colors.primaryDark]}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={styles.loginGradient}
            >
              {loading ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.loginBtnText}>Sign In</Text>
              )}
            </LinearGradient>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.registerLink}
            onPress={() => navigation.navigate("Register")}
            accessibilityRole="link"
            accessibilityLabel="Create a new account"
          >
            <Text style={styles.registerText}>
              Don't have an account?{" "}
              <Text style={styles.registerTextBold}>Sign Up</Text>
            </Text>
          </TouchableOpacity>
        </Animated.View>
      </View>
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
  logoSection: {
    alignItems: "center",
    marginBottom: spacing.xxxxl,
  },
  logoContainer: {
    marginBottom: spacing.lg,
  },
  logoBg: {
    width: 72,
    height: 72,
    borderRadius: 22,
    justifyContent: "center",
    alignItems: "center",
    ...shadows.lg,
  },
  logoIcon: {
    fontSize: 36,
    fontWeight: "900",
    color: "#fff",
  },
  logo: {
    fontSize: fontSize.display,
    fontWeight: "800",
    color: colors.text,
    textAlign: "center",
    letterSpacing: 6,
  },
  tagline: {
    fontSize: fontSize.md,
    color: colors.textMuted,
    textAlign: "center",
    marginTop: spacing.xs,
    letterSpacing: 0.5,
  },
  formSection: {
    // Form area
  },
  subtitle: {
    fontSize: fontSize.lg,
    color: colors.textSecondary,
    textAlign: "center",
    marginBottom: spacing.xxl,
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
  loginBtn: {
    borderRadius: borderRadius.lg,
    overflow: "hidden",
    marginTop: spacing.lg,
    ...shadows.md,
  },
  loginGradient: {
    padding: spacing.lg + 2,
    alignItems: "center",
    borderRadius: borderRadius.lg,
  },
  loginBtnText: {
    color: "#fff",
    fontSize: fontSize.xl,
    fontWeight: "700",
    letterSpacing: 0.5,
  },
  registerLink: {
    marginTop: spacing.xl,
    alignItems: "center",
  },
  registerText: {
    color: colors.textMuted,
    fontSize: fontSize.lg,
  },
  registerTextBold: {
    color: colors.primary,
    fontWeight: "700",
  },
});
