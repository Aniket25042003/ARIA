import React from "react";
import { Text, StyleSheet, View, Platform } from "react-native";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import HomeScreen from "../screens/home/HomeScreen";
import SignScreen from "../screens/sign/SignScreen";
import GuideScreen from "../screens/guide/GuideScreen";
import DashboardScreen from "../screens/dashboard/DashboardScreen";
import ProfileScreen from "../screens/profile/ProfileScreen";
import { colors, shadows } from "../constants/theme";
import type { MainTabParamList } from "./types";

const Tab = createBottomTabNavigator<MainTabParamList>();

function TabIcon({ label, color, focused }: { label: string; color: string; focused: boolean }) {
  return (
    <View style={[styles.tabIcon, focused && styles.tabIconActive]}>
      <Text style={[styles.tabIconText, { color }]}>{label}</Text>
    </View>
  );
}

export default function MainNavigator() {
  return (
    <Tab.Navigator
      screenOptions={{
        tabBarStyle: {
          backgroundColor: colors.surface,
          borderTopColor: colors.border,
          borderTopWidth: 1,
          height: Platform.OS === "ios" ? 88 : 64,
          paddingTop: 8,
          ...shadows.sm,
        },
        tabBarActiveTintColor: colors.primary,
        tabBarInactiveTintColor: colors.textMuted,
        tabBarLabelStyle: {
          fontSize: 11,
          fontWeight: "600",
          marginTop: 2,
        },
        headerStyle: {
          backgroundColor: colors.background,
          ...shadows.sm,
        },
        headerTintColor: colors.text,
        headerShadowVisible: false,
        headerTitleStyle: {
          fontWeight: "700",
          fontSize: 18,
          color: colors.text,
        },
      }}
    >
      <Tab.Screen
        name="Home"
        component={HomeScreen}
        options={{
          tabBarIcon: ({ color, focused }) => (
            <TabIcon label="H" color={color} focused={focused} />
          ),
          title: "ARIA",
          headerTitleStyle: {
            fontWeight: "800",
            fontSize: 20,
            color: colors.primary,
            letterSpacing: 2,
          },
        }}
      />
      <Tab.Screen
        name="Sign"
        component={SignScreen}
        options={{
          tabBarIcon: ({ color, focused }) => (
            <TabIcon label="S" color={color} focused={focused} />
          ),
          title: "SIGN Mode",
        }}
      />
      <Tab.Screen
        name="Guide"
        component={GuideScreen}
        options={{
          tabBarIcon: ({ color, focused }) => (
            <TabIcon label="G" color={color} focused={focused} />
          ),
          title: "GUIDE Mode",
        }}
      />
      <Tab.Screen
        name="Dashboard"
        component={DashboardScreen}
        options={{
          tabBarIcon: ({ color, focused }) => (
            <TabIcon label="D" color={color} focused={focused} />
          ),
          title: "Dashboard",
        }}
      />
      <Tab.Screen
        name="Profile"
        component={ProfileScreen}
        options={{
          tabBarIcon: ({ color, focused }) => (
            <TabIcon label="P" color={color} focused={focused} />
          ),
          title: "Profile",
        }}
      />
    </Tab.Navigator>
  );
}

const styles = StyleSheet.create({
  tabIcon: {
    width: 32,
    height: 32,
    borderRadius: 10,
    justifyContent: "center",
    alignItems: "center",
  },
  tabIconActive: {
    backgroundColor: colors.primarySoft,
  },
  tabIconText: {
    fontSize: 15,
    fontWeight: "700",
  },
});
