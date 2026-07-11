# =============================================================================
# InsightPulse Frontend — Next.js production image (multi-stage)
# =============================================================================
# Stage 1: install dependencies + build (standalone output)
# Stage 2: minimal runtime — only the standalone server, static assets,
#          and a non-root user.
# =============================================================================

FROM node:22-alpine AS builder

WORKDIR /build

# Dependency layer (cached until package files change)
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

# Build layer
COPY tsconfig.json tailwind.config.ts postcss.config.js next.config.js .eslintrc.json ./
COPY src/ src/
RUN npm run build

# -----------------------------------------------------------------------------
FROM node:22-alpine AS runtime

WORKDIR /app
ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

RUN addgroup -S app && adduser -S app -G app

# Standalone server + static assets only (no node_modules tree)
COPY --from=builder --chown=app:app /build/.next/standalone ./
COPY --from=builder --chown=app:app /build/.next/static ./.next/static

USER app
EXPOSE 3000

# 127.0.0.1 explicitly: busybox wget resolves localhost to ::1 first, but
# the standalone server binds IPv4 (0.0.0.0) only.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -qO /dev/null http://127.0.0.1:3000/login || exit 1

CMD ["node", "server.js"]
