# syntax=docker/dockerfile:1.7
FROM node:24-alpine AS deps
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

FROM node:24-alpine AS builder
ENV NEXT_TELEMETRY_DISABLED=1
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY frontend/package.json frontend/package-lock.json ./
COPY frontend/next.config.ts frontend/tsconfig.json frontend/postcss.config.mjs ./
COPY frontend/components.json ./components.json
COPY frontend/app ./app
COPY frontend/components ./components
COPY frontend/lib ./lib
RUN npm run build

FROM node:24-alpine AS runtime
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    HOSTNAME=0.0.0.0 \
    PORT=3000 \
    KINGPRO_API_URL=http://backend:8080 \
    KINGPRO_DATA_ROOT=/app/project

WORKDIR /app

RUN addgroup --system --gid 10001 kingpro \
    && adduser --system --uid 10001 --ingroup kingpro kingpro

COPY --from=builder --chown=kingpro:kingpro /app/.next/standalone ./
COPY --from=builder --chown=kingpro:kingpro /app/.next/static ./.next/static

USER kingpro

EXPOSE 3000

CMD ["node", "server.js"]

