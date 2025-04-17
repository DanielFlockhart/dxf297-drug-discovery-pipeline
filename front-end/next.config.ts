import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  images: {
    domains: [
      "via.placeholder.com", 
      "127.0.0.1", 
      "localhost",
      "firebasestorage.googleapis.com", // Allow Firebase Storage for images
    ],
  },
  eslint: {
    ignoreDuringBuilds: true, // Disables ESLint checks during the build
  },
  typescript: {
    ignoreBuildErrors: true, // Disables TypeScript errors during the build
  },
  async rewrites() {
    return [
      {
        source: "/proxy-storage/:path*", // Proxy endpoint
        destination: "https://firebasestorage.googleapis.com/v0/b/erudite-8d040.firebasestorage.app/o/:path*?alt=media", // Firebase Storage URL
      },
    ];
  },
};

export default nextConfig;
