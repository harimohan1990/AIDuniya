import { redirect } from "next/navigation";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import Link from "next/link";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Map as MapIcon, BookOpen, FileText } from "lucide-react";

async function getDashboardData(userId: string) {
  try {
    const [savedRoadmaps, progress, bookmarks] = await Promise.all([
    prisma.savedRoadmap.findMany({
      where: { userId },
      include: {
        roadmap: {
          include: {
            _count: { select: { nodes: true } },
          },
        },
      },
    }),
    prisma.progress.findMany({
      where: { userId, status: "COMPLETED" },
      include: { node: { include: { roadmap: true } } },
    }),
    prisma.bookmark.findMany({
      where: { userId },
    }),
  ]);

  const roadmapProgress = new Map<string, { completed: number; total: number }>();
  for (const p of progress) {
    const rid = p.node.roadmapId;
    if (!roadmapProgress.has(rid)) {
      const total = await prisma.roadmapNode.count({
        where: { roadmapId: rid },
      });
      roadmapProgress.set(rid, { completed: 0, total });
    }
    const entry = roadmapProgress.get(rid)!;
    entry.completed++;
  }

    return { savedRoadmaps, roadmapProgress, bookmarks };
  } catch {
    return {
      savedRoadmaps: [],
      roadmapProgress: new Map<string, { completed: number; total: number }>(),
      bookmarks: [],
    };
  }
}

export default async function DashboardPage() {
  const session = await getServerSession(authOptions);
  if (!session?.user) redirect("/auth/signin");

  const userId = (session.user as { id?: string }).id!;
  const { savedRoadmaps, roadmapProgress, bookmarks } = await getDashboardData(userId);

  return (
    <div className="container mx-auto px-4 py-12">
      <h1 className="text-3xl font-bold mb-8">Dashboard</h1>

      <section className="mb-12">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <MapIcon className="h-5 w-5" />
          My Roadmaps
        </h2>
        {savedRoadmaps.length === 0 ? (
          <p className="text-muted-foreground">
            No saved roadmaps. <Link href="/roadmaps" className="text-primary hover:underline">Browse roadmaps</Link> to save.
          </p>
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {savedRoadmaps.map((sr) => {
              const prog = roadmapProgress.get(sr.roadmapId) ?? {
                completed: 0,
                total: sr.roadmap._count.nodes,
              };
              const percent = prog.total > 0 ? (prog.completed / prog.total) * 100 : 0;
              return (
                <Link key={sr.id} href={`/roadmaps/${sr.roadmap.slug}`}>
                  <Card className="hover:border-primary/50 transition-colors">
                    <CardHeader>
                      <h3 className="font-semibold">{sr.roadmap.title}</h3>
                      <Progress value={percent} className="h-2" />
                      <p className="text-sm text-muted-foreground">
                        {prog.completed} / {prog.total} completed
                      </p>
                    </CardHeader>
                  </Card>
                </Link>
              );
            })}
          </div>
        )}
      </section>

      <section>
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <BookOpen className="h-5 w-5" />
          Bookmarks
        </h2>
        {bookmarks.length === 0 ? (
          <p className="text-muted-foreground">
            No bookmarks yet. Bookmark courses and guides from their detail pages.
          </p>
        ) : (
          <div className="space-y-2">
            {bookmarks.map((b) => (
              <Card key={b.id}>
                <CardContent className="py-3">
                  <span className="text-sm text-muted-foreground">{b.entityType}</span>{" "}
                  <Link
                    href={
                      b.entityType === "ROADMAP"
                        ? `/roadmaps/${b.entityId}`
                        : b.entityType === "COURSE"
                        ? `/courses/${b.entityId}`
                        : `/guides/${b.entityId}`
                    }
                    className="text-primary hover:underline"
                  >
                    View
                  </Link>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
