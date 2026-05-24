import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Vercel runtime deployment needs Next server functions (e.g. /api/upload)
  // so we must not use static export mode.
  trailingSlash: false,
};

export default nextConfig;
