import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  pageExtensions: ["ts", "tsx"],
  output: "standalone",
  transpilePackages: ["next-mdx-remote"],
  outputFileTracingRoot: path.join(process.cwd()),
  eslint: { ignoreDuringBuilds: false },
};

export default nextConfig;
