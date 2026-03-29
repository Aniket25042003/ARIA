import React, { useEffect, useRef } from "react";
import { Animated, View, StyleSheet } from "react-native";
import { useAuthStore } from "../stores/useAuthStore";
import { useConnectionStore } from "../stores/useConnectionStore";
import AuthNavigator from "./AuthNavigator";
import MainNavigator from "./MainNavigator";
import { colors } from "../constants/theme";

export default function RootNavigator() {
  const { isAuthenticated, isLoading, initialize } = useAuthStore();
  const loadBackendUrl = useConnectionStore((s) => s.loadBackendUrl);
  const pulseAnim = useRef(new Animated.Value(0.3)).current;

  useEffect(() => {
    loadBackendUrl();
    initialize();
  }, [initialize, loadBackendUrl]);

  useEffect(() => {
    if (isLoading) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue: 1,
            duration: 800,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnim, {
            toValue: 0.3,
            duration: 800,
            useNativeDriver: true,
          }),
        ])
      ).start();
    }
  }, [isLoading]);

  if (isLoading) {
    return (
      <View style={styles.loader}>
        <Animated.Text style={[styles.loaderLogo, { opacity: pulseAnim }]}>
          ARIA
        </Animated.Text>
      </View>
    );
  }

  return isAuthenticated ? <MainNavigator /> : <AuthNavigator />;
}

const styles = StyleSheet.create({
  loader: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: colors.background,
  },
  loaderLogo: {
    fontSize: 52,
    fontWeight: "800",
    color: colors.primary,
    letterSpacing: 6,
  },
});
