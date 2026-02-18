# AICourseMap Routes

All routes and their status.

## Public Routes

| Route | Description | Status |
|-------|-------------|--------|
| `/` | Home - hero, featured content | ✅ |
| `/roadmaps` | Roadmaps list (role/skill filters) | ✅ |
| `/roadmaps/[slug]` | Roadmap detail with graph (e.g. `/roadmaps/ai-engineer`) | ✅ |
| `/courses` | Courses catalog with filters | ✅ |
| `/courses/[id]` | Course detail | ✅ |
| `/guides` | Guides list | ✅ |
| `/guides/[slug]` | Guide detail (MDX) | ✅ |
| `/projects` | Project ideas by level | ✅ |
| `/projects/[id]` | Project detail | ✅ |
| `/tutor` | AI Tutor chat | ✅ |
| `/search` | Global search (courses, roadmaps, guides, projects) | ✅ |
| `/about` | About page | ✅ |
| `/pricing` | Pricing tiers (Free vs Pro) | ✅ |
| `/faq` | FAQ accordion | ✅ |
| `/contact` | Contact info | ✅ |

## Auth Routes

| Route | Description | Status |
|-------|-------------|--------|
| `/auth/signin` | Sign in (credentials + Google) | ✅ |
| `/auth/signup` | Create account | ✅ |
| `/auth/forgot-password` | Password reset flow | ✅ |

## Protected Routes (requires sign-in)

| Route | Description | Status |
|-------|-------------|--------|
| `/dashboard` | My roadmaps, progress, bookmarks | ✅ |
| `/dashboard/profile` | Update profile (level, goals) | ✅ |
| `/dashboard/settings` | Account settings | ✅ |

## Admin Routes (requires ADMIN role)

| Route | Description | Status |
|-------|-------------|--------|
| `/admin` | Admin dashboard | ✅ |
| `/admin/courses` | Manage courses | ✅ |
| `/admin/roadmaps` | Manage roadmaps | ✅ |
| `/admin/guides` | Manage guides | ✅ |
| `/admin/projects` | Manage projects | ✅ |

## API Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/api/auth/[...nextauth]` | GET/POST | NextAuth handlers |
| `/api/auth/register` | POST | User registration |
| `/api/roadmaps/[id]/progress` | GET | User progress for roadmap |
| `/api/roadmaps/[id]/progress/summary` | GET | Progress summary |
| `/api/roadmaps/[id]/save` | POST/DELETE | Save/unsave roadmap |
| `/api/progress` | POST | Update node progress |
| `/api/bookmarks` | POST/DELETE | Add/remove bookmark |
| `/api/user/profile` | PATCH | Update user profile |

## Error Pages

| Route | Description |
|-------|-------------|
| `404` | Custom not-found page |
| `error.tsx` | Global error boundary |
| `loading.tsx` | Global loading state |
