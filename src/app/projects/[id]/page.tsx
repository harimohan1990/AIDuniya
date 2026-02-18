import { notFound } from "next/navigation";
import Link from "next/link";
import { getProject } from "@/lib/data";

export const dynamic = "force-dynamic";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft } from "lucide-react";
import { getLevelColor } from "@/lib/utils";

export default async function ProjectDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const project = await getProject(id);
  if (!project) notFound();

  return (
    <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 max-w-3xl">
      <Link
        href="/projects"
        className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground mb-6"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to projects
      </Link>

      <Card>
        <CardHeader>
          <Badge className={getLevelColor(project.level.toLowerCase())}>
            {project.level}
          </Badge>
          <h1 className="text-2xl font-bold mt-2">{project.title}</h1>
          <p className="text-muted-foreground">{project.description}</p>
        </CardHeader>
        <CardContent className="space-y-6">
          {"syllabus" in project && Array.isArray(project.syllabus) && project.syllabus.length > 0 && (
            <div>
              <h3 className="font-semibold mb-2">Syllabus</h3>
              <ol className="list-decimal list-inside space-y-2 text-muted-foreground">
                {project.syllabus.map((item, i) => (
                  <li key={i} className="pl-2">{item}</li>
                ))}
              </ol>
            </div>
          )}
          {project.skillsLearned.length > 0 && (
            <div>
              <h3 className="font-semibold mb-2">Skills learned</h3>
              <div className="flex flex-wrap gap-2">
                {project.skillsLearned.map((s) => (
                  <Badge key={s} variant="secondary">
                    {s}
                  </Badge>
                ))}
              </div>
            </div>
          )}
          {project.datasetSuggestions.length > 0 && (
            <div>
              <h3 className="font-semibold mb-2">Dataset suggestions</h3>
              <ul className="list-disc list-inside text-muted-foreground space-y-1">
                {project.datasetSuggestions.map((d) => (
                  <li key={d}>{d}</li>
                ))}
              </ul>
            </div>
          )}
          {project.tags.length > 0 && (
            <div>
              <h3 className="font-semibold mb-2">Tags</h3>
              <div className="flex flex-wrap gap-2">
                {project.tags.map((t) => (
                  <Badge key={t} variant="outline">
                    {t}
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
