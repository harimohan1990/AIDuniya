/**
 * Data layer: tries Prisma first, falls back to static data when DB is unavailable.
 */
import { prisma } from "@/lib/prisma";
import {
  fallbackRoadmaps,
  fallbackRoadmapDetail,
  fallbackCourses,
  fallbackGuides,
  fallbackProjects,
} from "./fallback-data";

let dbAvailable: boolean | null = null;

async function checkDb(): Promise<boolean> {
  if (dbAvailable !== null) return dbAvailable;
  try {
    await prisma.$queryRaw`SELECT 1`;
    dbAvailable = true;
  } catch {
    dbAvailable = false;
  }
  return dbAvailable;
}

function safeCheckDb(): Promise<boolean> {
  try {
    return checkDb();
  } catch {
    dbAvailable = false;
    return Promise.resolve(false);
  }
}

export async function getRoadmaps(type?: string) {
  const ok = await safeCheckDb();
  if (!ok) {
    const filtered = type
      ? fallbackRoadmaps.filter((r) => r.type === type.toUpperCase())
      : fallbackRoadmaps;
    return filtered.sort((a, b) => a.order - b.order);
  }
  try {
    const where = type ? { type: type.toUpperCase() as "ROLE" | "SKILL" } : {};
    return prisma.roadmap.findMany({
      where,
      orderBy: [{ type: "asc" }, { order: "asc" }],
      include: { _count: { select: { nodes: true } } },
    });
  } catch {
    const filtered = type
      ? fallbackRoadmaps.filter((r) => r.type === type.toUpperCase())
      : fallbackRoadmaps;
    return filtered.sort((a, b) => a.order - b.order);
  }
}

export async function getRoadmap(slug: string, userId?: string) {
  const ok = await safeCheckDb();
  if (!ok) {
    return fallbackRoadmapDetail[slug] ?? null;
  }
  try {
    return prisma.roadmap.findUnique({
      where: { slug },
      include: {
        nodes: { orderBy: { order: "asc" } },
        edges: true,
        savedBy: userId ? { where: { userId } } : false,
      },
    });
  } catch {
    return fallbackRoadmapDetail[slug] ?? null;
  }
}

export async function getFeaturedData() {
  const ok = await safeCheckDb();
  if (!ok) {
    return {
      roadmaps: fallbackRoadmaps.filter((r) => r.featured).slice(0, 6),
      courses: fallbackCourses.filter((c) => c.featured).slice(0, 6),
      guides: fallbackGuides.filter((g) => g.featured).slice(0, 4),
      projects: fallbackProjects.slice(0, 6),
    };
  }
  try {
    const [roadmaps, courses, guides, projects] = await Promise.all([
      prisma.roadmap.findMany({
        where: { featured: true },
        orderBy: { order: "asc" },
        take: 6,
      }),
      prisma.course.findMany({
        where: { featured: true },
        orderBy: { order: "asc" },
        take: 6,
      }),
      prisma.guide.findMany({
        where: { featured: true },
        orderBy: { order: "asc" },
        take: 4,
      }),
      prisma.projectIdea.findMany({
        orderBy: { order: "asc" },
        take: 6,
      }),
    ]);
    return { roadmaps, courses, guides, projects };
  } catch {
    return {
      roadmaps: fallbackRoadmaps.filter((r) => r.featured).slice(0, 6),
      courses: fallbackCourses.filter((c) => c.featured).slice(0, 6),
      guides: fallbackGuides.filter((g) => g.featured).slice(0, 4),
      projects: fallbackProjects.slice(0, 6),
    };
  }
}

export async function getCourses(
  search?: string,
  filters?: { level?: string; type?: string; provider?: string }
) {
  const ok = await safeCheckDb();
  if (!ok) {
    let list = [...fallbackCourses];
    if (search?.trim()) {
      const q = search.trim().toLowerCase();
      list = list.filter(
        (c) =>
          c.title.toLowerCase().includes(q) ||
          c.tags.some((t) => t.toLowerCase().includes(q))
      );
    }
    if (filters?.level) list = list.filter((c) => c.level === filters.level);
    if (filters?.type) list = list.filter((c) => c.type === filters.type);
    if (filters?.provider)
      list = list.filter((c) =>
        c.provider.toLowerCase().includes((filters.provider ?? "").toLowerCase())
      );
    return list.sort((a, b) => a.order - b.order);
  }
  try {
    const where: Record<string, unknown> = {};
    if (search?.trim()) {
      where.OR = [
        { title: { contains: search.trim(), mode: "insensitive" } },
        { tags: { has: search.trim() } },
      ];
    }
    if (filters?.level) where.level = filters.level;
    if (filters?.type) where.type = filters.type;
    if (filters?.provider)
      where.provider = { contains: filters.provider, mode: "insensitive" };
    return prisma.course.findMany({
      where,
      orderBy: [{ featured: "desc" }, { order: "asc" }, { title: "asc" }],
    });
  } catch {
    return fallbackCourses;
  }
}

export async function getCourse(id: string) {
  const ok = await safeCheckDb();
  if (!ok) {
    return fallbackCourses.find((c) => c.id === id) ?? null;
  }
  try {
    return prisma.course.findUnique({ where: { id } });
  } catch {
    return fallbackCourses.find((c) => c.id === id) ?? null;
  }
}

export async function getGuides(search?: string) {
  const ok = await safeCheckDb();
  if (!ok) {
    let list = [...fallbackGuides];
    if (search?.trim()) {
      const q = search.trim().toLowerCase();
      list = list.filter(
        (g) =>
          g.title.toLowerCase().includes(q) ||
          g.tags.some((t) => t.toLowerCase().includes(q))
      );
    }
    return list.sort((a, b) => a.order - b.order);
  }
  try {
    const where = search?.trim()
      ? {
          OR: [
            { title: { contains: search.trim(), mode: "insensitive" as const } },
            { tags: { has: search.trim() } },
          ],
        }
      : {};
    return prisma.guide.findMany({
      where,
      orderBy: [{ featured: "desc" }, { order: "asc" }, { title: "asc" }],
    });
  } catch {
    return fallbackGuides.sort((a, b) => a.order - b.order);
  }
}

export async function getGuide(slug: string) {
  const ok = await safeCheckDb();
  if (!ok) {
    return fallbackGuides.find((g) => g.slug === slug) ?? null;
  }
  try {
    return prisma.guide.findUnique({ where: { slug } });
  } catch {
    return fallbackGuides.find((g) => g.slug === slug) ?? null;
  }
}

export async function getProjects(level?: string) {
  const ok = await safeCheckDb();
  if (!ok) {
    const list = level
      ? fallbackProjects.filter((p) => p.level === level)
      : fallbackProjects;
    return list.sort((a, b) => a.order - b.order);
  }
  try {
    const where = level
      ? { level: level as "BEGINNER" | "INTERMEDIATE" | "ADVANCED" }
      : {};
    return prisma.projectIdea.findMany({
      where,
      orderBy: [{ level: "asc" }, { order: "asc" }],
    });
  } catch {
    return fallbackProjects.sort((a, b) => a.order - b.order);
  }
}

export async function getProject(id: string) {
  const ok = await safeCheckDb();
  if (!ok) {
    return fallbackProjects.find((p) => p.id === id) ?? null;
  }
  try {
    return prisma.projectIdea.findUnique({ where: { id } });
  } catch {
    return fallbackProjects.find((p) => p.id === id) ?? null;
  }
}

export async function searchAll(query: string) {
  const ok = await safeCheckDb();
  if (!ok) {
    const q = query.trim().toLowerCase();
    if (!q)
      return { courses: [], roadmaps: [], guides: [], projects: [] };
    return {
      courses: fallbackCourses.filter(
        (c) =>
          c.title.toLowerCase().includes(q) ||
          c.tags.some((t) => t.toLowerCase().includes(q)) ||
          c.provider.toLowerCase().includes(q)
      ),
      roadmaps: fallbackRoadmaps.filter(
        (r) =>
          r.title.toLowerCase().includes(q) ||
          (r.description ?? "").toLowerCase().includes(q)
      ),
      guides: fallbackGuides.filter(
        (g) =>
          g.title.toLowerCase().includes(q) ||
          g.tags.some((t) => t.toLowerCase().includes(q))
      ),
      projects: fallbackProjects.filter(
        (p) =>
          p.title.toLowerCase().includes(q) ||
          p.tags.some((t) => t.toLowerCase().includes(q))
      ),
    };
  }
  try {
    const q = query.trim().toLowerCase();
    if (!q)
      return { courses: [], roadmaps: [], guides: [], projects: [] };
    const [courses, roadmaps, guides, projects] = await Promise.all([
      prisma.course.findMany({
        where: {
          OR: [
            { title: { contains: q, mode: "insensitive" } },
            { tags: { has: q } },
            { provider: { contains: q, mode: "insensitive" } },
          ],
        },
        take: 10,
      }),
      prisma.roadmap.findMany({
        where: {
          OR: [
            { title: { contains: q, mode: "insensitive" } },
            { description: { contains: q, mode: "insensitive" } },
          ],
        },
        take: 10,
      }),
      prisma.guide.findMany({
        where: {
          OR: [
            { title: { contains: q, mode: "insensitive" } },
            { tags: { has: q } },
          ],
        },
        take: 10,
      }),
      prisma.projectIdea.findMany({
        where: {
          OR: [
            { title: { contains: q, mode: "insensitive" } },
            { tags: { has: q } },
          ],
        },
        take: 10,
      }),
    ]);
    return { courses, roadmaps, guides, projects };
  } catch {
    const q = query.trim().toLowerCase();
    if (!q)
      return { courses: [], roadmaps: [], guides: [], projects: [] };
    return {
      courses: fallbackCourses.filter(
        (c) =>
          c.title.toLowerCase().includes(q) ||
          c.tags.some((t) => t.toLowerCase().includes(q))
      ),
      roadmaps: fallbackRoadmaps.filter(
        (r) =>
          r.title.toLowerCase().includes(q) ||
          (r.description ?? "").toLowerCase().includes(q)
      ),
      guides: fallbackGuides.filter(
        (g) =>
          g.title.toLowerCase().includes(q) ||
          g.tags.some((t) => t.toLowerCase().includes(q))
      ),
      projects: fallbackProjects.filter(
        (p) =>
          p.title.toLowerCase().includes(q) ||
          p.tags.some((t) => t.toLowerCase().includes(q))
      ),
    };
  }
}
