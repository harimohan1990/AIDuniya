import Link from "next/link";
import { getProjects } from "@/lib/data";

export const dynamic = "force-dynamic";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { getLevelColor } from "@/lib/utils";

export default async function ProjectsPage({
  searchParams,
}: {
  searchParams: Promise<{ level?: string }>;
}) {
  const { level } = await searchParams;
  const projects = await getProjects(level ?? undefined);

  const beginner = projects.filter((p) => p.level === "BEGINNER");
  const intermediate = projects.filter((p) => p.level === "INTERMEDIATE");
  const advanced = projects.filter((p) => p.level === "ADVANCED");

  const Section = ({
    title,
    items,
  }: {
    title: string;
    items: Array<{ id: string; title: string; level: string; description: string; skillsLearned: string[]; datasetSuggestions: string[]; tags: string[] }>;
  }) => (
    <section className="mb-12">
      <h2 className="text-xl font-semibold mb-4">{title}</h2>
      <div className="grid md:grid-cols-2 gap-4">
        {items.map((p) => (
          <Link key={p.id} href={`/projects/${p.id}`}>
            <Card id={p.id} className="overflow-hidden hover:border-primary/50 transition-colors h-full">
            <CardHeader>
              <Badge className={getLevelColor(p.level.toLowerCase())}>{p.level}</Badge>
              <h3 className="font-semibold text-lg">{p.title}</h3>
              <p className="text-muted-foreground text-sm">{p.description}</p>
            </CardHeader>
            <CardContent className="space-y-2">
              {p.skillsLearned.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium">Skills learned</h4>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {p.skillsLearned.map((s) => (
                      <Badge key={s} variant="secondary" className="text-xs">
                        {s}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
              {p.datasetSuggestions.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium">Dataset suggestions</h4>
                  <ul className="text-sm text-muted-foreground list-disc list-inside">
                    {p.datasetSuggestions.map((d) => (
                      <li key={d}>{d}</li>
                    ))}
                  </ul>
                </div>
              )}
              {p.tags.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {p.tags.map((t) => (
                    <Badge key={t} variant="outline" className="text-xs">
                      {t}
                    </Badge>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
          </Link>
        ))}
      </div>
    </section>
  );

  return (
    <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 max-w-7xl 2xl:max-w-8xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Project Ideas</h1>
        <p className="mt-2 text-muted-foreground max-w-2xl">
          Hands-on AI projects from beginner to advanced. Each project includes a syllabus with learning modules, skills you&apos;ll gain, dataset suggestions, and tags. Build your portfolio and demonstrate practical skills to employers.
        </p>
      </div>

      <div className="flex gap-2 mb-8">
        <Link href="/projects">
          <Badge
            variant={!level ? "default" : "outline"}
            className="cursor-pointer px-4 py-2 text-sm"
          >
            All
          </Badge>
        </Link>
        <Link href="/projects?level=BEGINNER">
          <Badge
            variant={level === "BEGINNER" ? "default" : "outline"}
            className="cursor-pointer px-4 py-2 text-sm"
          >
            Beginner
          </Badge>
        </Link>
        <Link href="/projects?level=INTERMEDIATE">
          <Badge
            variant={level === "INTERMEDIATE" ? "default" : "outline"}
            className="cursor-pointer px-4 py-2 text-sm"
          >
            Intermediate
          </Badge>
        </Link>
        <Link href="/projects?level=ADVANCED">
          <Badge
            variant={level === "ADVANCED" ? "default" : "outline"}
            className="cursor-pointer px-4 py-2 text-sm"
          >
            Advanced
          </Badge>
        </Link>
      </div>

      {beginner.length > 0 && <Section title="Beginner" items={beginner} />}
      {intermediate.length > 0 && <Section title="Intermediate" items={intermediate} />}
      {advanced.length > 0 && <Section title="Advanced" items={advanced} />}

      {projects.length === 0 && (
        <p className="text-muted-foreground">No projects found. Run the seed script.</p>
      )}
    </div>
  );
}
