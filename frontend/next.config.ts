import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",          // static export for FastAPI serving
  trailingSlash: true,       // /chat/ instead of /chat for static routing
  images: { unoptimized: true }, // no Next.js image server in static mode
};

export default nextConfig;
