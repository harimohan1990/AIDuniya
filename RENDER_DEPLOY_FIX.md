# Fix Render Deployment for AIDuniya

## The Problem
- **"Empty build command; skipping build"** – Render isn't running your build
- **"Publish directory .next does not exist!"** – The service is configured as a **Static Site**, but Next.js needs a **Web Service** (Node.js server)

## Solution: Create a Web Service (Not Static Site)

### Option A: Use Blueprint (Recommended)

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **New** → **Blueprint**
3. Connect your repo: `https://github.com/harimohan1990/AIDuniya`
4. Render will read `render.yaml` and create the **aiduniya** Web Service with:
   - Build Command: `npm ci && npm run build`
   - Start Command: `npm run start`
5. Add environment variables: `DATABASE_URL`, `NEXTAUTH_URL`, `NEXTAUTH_SECRET`

### Option B: Manual Web Service

1. **Delete** the existing Static Site (or create a new service)
2. Click **New** → **Web Service** (NOT Static Site)
3. Connect repo: `https://github.com/harimohan1990/AIDuniya`
4. Configure:
   - **Name:** aiduniya
   - **Runtime:** Node
   - **Build Command:** `npm ci && npm run build`
   - **Start Command:** `npm run start`
   - **Publish Directory:** Leave **blank** (do not set `.next`)
5. Add env vars: `DATABASE_URL`, `NEXTAUTH_URL`, `NEXTAUTH_SECRET`

## Key Point
| Static Site | Web Service (what you need) |
|-------------|-----------------------------|
| Publish directory (e.g. `.next`) | No publish directory |
| Serves pre-built files | Runs Node server (`npm start`) |
| Build command optional | Build + Start commands required |

Next.js with App Router needs a **Web Service** because it runs server-side code.
