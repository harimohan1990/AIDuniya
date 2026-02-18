# Fix Render Deployment

## The Error
- **"Empty build command; skipping build"** – Render isn't running your build
- **"Publish directory .next does not exist!"** – Service is configured as **Static Site** (wrong type)

## Root Cause
Your Render service was created as a **Static Site**. Next.js needs a **Web Service** (Node.js server).

## Fix: Create a Web Service

### Option A: Use Blueprint (Recommended)

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. **Delete** the existing Static Site (or leave it)
3. Click **New** → **Blueprint**
4. Connect: `https://github.com/harimohan1990/AIDuniya`
5. Render will read `render.yaml` and create the **aiduniya** Web Service
6. Add env vars: `DATABASE_URL`, `NEXTAUTH_URL`, `NEXTAUTH_SECRET`

### Option B: Manual Web Service

1. **Delete** the existing Static Site
2. Click **New** → **Web Service** (NOT Static Site)
3. Connect: `https://github.com/harimohan1990/AIDuniya`
4. Set:
   - **Build Command:** `npm run build`
   - **Start Command:** `npm run start`
   - **Publish Directory:** Leave **blank**
5. Add env vars

## Important
| Static Site (wrong) | Web Service (correct) |
|--------------------|------------------------|
| Publish directory: `.next` | No publish directory |
| Serves static files only | Runs Node server |
| Build command optional | Build + Start required |
