import { notFound } from "next/navigation";
import { getRoadmap } from "@/lib/data";

export const dynamic = "force-dynamic";
import { RoadmapGraph } from "@/components/roadmap/roadmap-graph";
import { RoadmapProgress } from "@/components/roadmap/roadmap-progress";
import { SaveRoadmapButton } from "@/components/roadmap/save-roadmap-button";
import { getSafeSession } from "@/lib/session";
import { Badge } from "@/components/ui/badge";
import { Clock, MapPin, BookOpen, Sparkles } from "lucide-react";

export default async function RoadmapDetailPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const session = await getSafeSession();
  const userId = (session?.user as { id?: string })?.id;
  const roadmap = await getRoadmap(slug, userId ?? undefined);
  if (!roadmap) notFound();

  const saved =
    "savedBy" in roadmap &&
    Array.isArray(roadmap.savedBy) &&
    roadmap.savedBy.length > 0;

  const nodeCount = roadmap.nodes.length;
  const estimatedMonths = slug === "ai-engineer" ? "6–12" : nodeCount <= 2 ? "1–2" : "2–4";

  return (
    <div className="min-h-screen">
      {/* Journey Hero */}
      <section className="border-b bg-gradient-to-b from-primary/5 via-primary/[0.02] to-background">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-10 sm:py-12 md:py-16 max-w-7xl 2xl:max-w-8xl">
          <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-6">
            <div className="max-w-2xl">
              <Badge variant="secondary" className="mb-3">
                {roadmap.type} Roadmap
              </Badge>
              <h1 className="text-3xl sm:text-4xl md:text-5xl font-bold tracking-tight">
                {roadmap.title}
              </h1>
              <p className="mt-4 text-lg text-muted-foreground leading-relaxed">
                {roadmap.description}
              </p>
              <div className="mt-6 flex flex-wrap gap-4 text-sm text-muted-foreground">
                <span className="flex items-center gap-2">
                  <MapPin className="h-4 w-4" />
                  {nodeCount} learning stages
                </span>
                <span className="flex items-center gap-2">
                  <Clock className="h-4 w-4" />
                  ~{estimatedMonths} months
                </span>
                <span className="flex items-center gap-2">
                  <BookOpen className="h-4 w-4" />
                  Curated resources
                </span>
              </div>
              <div className="mt-6 flex flex-wrap items-center gap-4">
                <RoadmapProgress roadmapId={roadmap.id} />
                <SaveRoadmapButton roadmapId={roadmap.id} initialSaved={saved} />
              </div>
            </div>
            <div className="hidden md:flex flex-col items-center justify-center p-6 rounded-xl bg-muted/50 border">
              <Sparkles className="h-12 w-12 text-primary/60 mb-2" />
              <p className="text-sm font-medium text-center">Your journey</p>
              <p className="text-xs text-muted-foreground text-center mt-1">
                Follow the path below
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Learning Path */}
      <section className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 max-w-7xl 2xl:max-w-8xl">
        <h2 className="text-xl sm:text-2xl font-semibold mb-4 sm:mb-6 flex items-center gap-2">
          <MapPin className="h-6 w-6" />
          Learning Path
        </h2>
        <RoadmapGraph
          roadmapId={roadmap.id}
          nodes={roadmap.nodes}
          edges={roadmap.edges}
        />
      </section>
    </div>
  );
}
