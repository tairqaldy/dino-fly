# apps/api image (build context = repository root):  docker build -f infra/api.Dockerfile .
FROM node:24-alpine
RUN corepack enable
WORKDIR /repo
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml tsconfig.base.json ./
COPY packages/dino-core packages/dino-core
COPY packages/protocol packages/protocol
COPY apps/api apps/api
RUN pnpm install --frozen-lockfile --filter @dino-fly/api...
WORKDIR /repo/apps/api
ENV NODE_ENV=production
EXPOSE 8787
# migrations run on every deploy, then the server starts
CMD ["sh", "-c", "pnpm migrate && pnpm start"]
