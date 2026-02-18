import Link from "next/link";
import { getRoadmaps } from "@/lib/data";

export const dynamic = "force-dynamic";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export default async function RoadmapsPage({
  searchParams,
}: {
  searchParams: Promise<{ type?: string }>;
}) {
  const { type } = await searchParams;
  const roadmaps = await getRoadmaps(type ?? undefined);

  const roleRoadmaps = roadmaps.filter((r) => r.type === "ROLE");
  const skillRoadmaps = roadmaps.filter((r) => r.type === "SKILL");

  return (
    <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 max-w-7xl 2xl:max-w-8xl">
      <div className="mb-6 sm:mb-8">
        <h1 className="text-2xl sm:text-3xl font-bold">Roadmaps</h1>
        <p className="mt-2 text-muted-foreground max-w-2xl">
          Structured learning paths for AI and ML. Role-based roadmaps (AI Engineer, ML Engineer) take you from zero to job-ready. Skill-based roadmaps (Prompt Engineering, RAG, LLMOps) focus on specific competencies. Each node includes curated resources and clear learning objectives.
        </p>
      </div>

      <div className="flex flex-wrap gap-2 mb-6 sm:mb-8">
        <Link href="/roadmaps">
          <Badge
            variant={!type ? "default" : "outline"}
            className="cursor-pointer px-4 py-2 text-sm"
          >
            All
          </Badge>
        </Link>
        <Link href="/roadmaps?type=role">
          <Badge
            variant={type === "role" ? "default" : "outline"}
            className="cursor-pointer px-4 py-2 text-sm"
          >
            Role-based
          </Badge>
        </Link>
        <Link href="/roadmaps?type=skill">
          <Badge
            variant={type === "skill" ? "default" : "outline"}
            className="cursor-pointer px-4 py-2 text-sm"
          >
            Skill-based
          </Badge>
        </Link>
      </div>

      {(!type || type === "role") && roleRoadmaps.length > 0 && (
        <section className="mb-12">
          <h2 className="text-xl font-semibold mb-4">Role-based Roadmaps</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4">
            {roleRoadmaps.map((r) => (
              <Link key={r.id} href={`/roadmaps/${r.slug}`}>
                <Card className="h-full hover:border-primary/50 transition-colors">
                  <CardHeader>
                    <Badge variant="secondary">ROLE</Badge>
                    <CardTitle className="text-lg">{r.title}</CardTitle>
                    <CardDescription className="line-clamp-2">{r.description}</CardDescription>
                    <p className="text-sm text-muted-foreground">{r._count.nodes} nodes</p>
                  </CardHeader>
                </Card>
              </Link>
            ))}
          </div>
        </section>
      )}

      {(!type || type === "skill") && skillRoadmaps.length > 0 && (
        <section>
          <h2 className="text-xl font-semibold mb-4">Skill-based Roadmaps</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4">
            {skillRoadmaps.map((r) => (
              <Link key={r.id} href={`/roadmaps/${r.slug}`}>
                <Card className="h-full hover:border-primary/50 transition-colors">
                  <CardHeader>
                    <Badge variant="secondary">SKILL</Badge>
                    <CardTitle className="text-lg">{r.title}</CardTitle>
                    <CardDescription className="line-clamp-2">{r.description}</CardDescription>
                    <p className="text-sm text-muted-foreground">{r._count.nodes} nodes</p>
                  </CardHeader>
                </Card>
              </Link>
            ))}
          </div>
        </section>
      )}

      {roadmaps.length === 0 && (
        <p className="text-muted-foreground">No roadmaps found. Run the seed script to add data.</p>
      )}
    </div>
  );
}
