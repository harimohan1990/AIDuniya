<<<<<<< HEAD
# AIDuniya
=======
# AICourseMap

A community-style platform for AI courses and structured roadmaps. Role-based and skill-based learning paths with progress tracking and an AI Tutor.

## Tech Stack

- **Frontend:** Next.js 15 (App Router), TypeScript, TailwindCSS, shadcn/ui
- **Auth:** NextAuth (credentials + Google)
- **Backend:** Next.js Route Handlers (REST)
- **DB:** PostgreSQL with Prisma
- **Search:** PostgreSQL full-text (Meilisearch-ready)
- **Content:** MDX for guides
- **Payments:** Stripe (scaffolded for Pro plan)

## Quick Start

### Local Development

1. **Clone and install**

   ```bash
   cd AIroadmap
   npm install
   # or: pnpm install
   ```

2. **Start PostgreSQL** (Docker - ensure Docker is running)

   ```bash
   docker-compose -f docker-compose.dev.yml up -d
   ```

   Or use a local PostgreSQL instance and set `DATABASE_URL` accordingly.

3. **Configure environment**

   ```bash
   cp .env.example .env
   # Edit .env: set DATABASE_URL, NEXTAUTH_SECRET, etc.
   ```

4. **Database setup**

   ```bash
   npm run db:generate
   npm run db:push
   npm run db:seed
   ```

5. **Run dev server**

   ```bash
   npm run dev
   # or: pnpm dev
   ```

   Open [http://localhost:3000](http://localhost:3000).

### Docker (full stack)

```bash
docker-compose up -d
# App at http://localhost:3000
# Run migrations and seed inside the app container or locally with DATABASE_URL pointing to the container.
```

For first run with Docker, run migrations and seed from your host:

```bash
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/aicoursemap"
npm run db:push
npm run db:seed
```

## Scripts

| Command        | Description                    |
|----------------|--------------------------------|
| `npm run dev`  | Start dev server (Turbopack)   |
| `npm run build`| Production build              |
| `npm run start`| Start production server       |
| `npm run db:generate` | Generate Prisma client |
| `npm run db:push`      | Push schema to DB       |
| `npm run db:seed`      | Seed data               |
| `npm run db:studio`    | Prisma Studio           |
| `npm run test`        | Run tests               |

## Seed Data

- **Admin:** `admin@aicoursemap.com` / `admin123`
- 3 roadmaps (AI Engineer, Prompt Engineering, RAG)
- 20 courses
- 10 guides
- 10 project ideas

## Routes

| Path              | Description          |
|-------------------|----------------------|
| `/`               | Home                 |
| `/roadmaps`       | Roadmaps list        |
| `/roadmaps/[slug]`| Roadmap detail + graph |
| `/courses`        | Courses catalog      |
| `/courses/[id]`   | Course detail        |
| `/guides`         | Guides list          |
| `/guides/[slug]`  | Guide (MDX)          |
| `/projects`       | Project ideas        |
| `/tutor`          | AI Tutor chat        |
| `/dashboard`      | My progress/bookmarks|
| `/admin`          | Admin (protected)    |

## Environment Variables

See `.env.example`. Required:

- `DATABASE_URL` – PostgreSQL connection string
- `NEXTAUTH_URL` – App URL (e.g. `http://localhost:3000`)
- `NEXTAUTH_SECRET` – Random secret (`openssl rand -base64 32`)

Optional: `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` for Google sign-in.

## License

MIT
>>>>>>> master
