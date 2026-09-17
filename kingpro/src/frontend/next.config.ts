import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Emit a self-contained production server for the multi-stage Docker image.
  // Static assets are copied beside server.js by docker/frontend.Dockerfile.
  output: "standalone",
  // The local judge profile opens the app via 127.0.0.1 instead of localhost.
  // Next 16 protects development chunks by origin, so explicitly allow both
  // loopback hostnames used by the demo and Chrome DevTools profile.
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  async headers() {
    return [{
      source: "/:path*",
      headers: [
        { key: "X-Content-Type-Options", value: "nosniff" },
        { key: "X-Frame-Options", value: "DENY" },
        { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
        { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
        { key: "Content-Security-Policy", value: "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'" },
      ],
    }];
  },
};

export default nextConfig;
