// Learn more https://docs.expo.io/guides/customizing-metro
const { getDefaultConfig } = require('expo/metro-config');

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

// Disable package exports resolution — fixes @ungap/structured-clone
// internal CJS requires that Metro can't resolve through the exports map
config.resolver.unstable_enablePackageExports = false;

module.exports = config;
