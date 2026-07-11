/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone", // small runtime image for the Docker multi-stage build
  async rewrites() {
    // Proxy API calls to the FastAPI backend so the browser stays same-origin.
    const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
    return [
      { source: "/api/backend/:path*", destination: `${apiUrl}/api/:path*` },
      { source: "/api/health", destination: `${apiUrl}/health` },
    ];
  },
};

module.exports = nextConfig;
