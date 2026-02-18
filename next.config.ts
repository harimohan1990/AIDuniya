import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  pageExtensions: ["ts", "tsx"],
  output: "standalone",
  transpilePackages: ["next-mdx-remote"],
};

export default nextConfig;
