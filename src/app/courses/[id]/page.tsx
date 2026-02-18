import { notFound } from "next/navigation";
import Link from "next/link";
import { getCourse } from "@/lib/data";

export const dynamic = "force-dynamic";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ExternalLink, ArrowLeft } from "lucide-react";
import { getLevelColor } from "@/lib/utils";

export default async function CourseDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const course = await getCourse(id);
  if (!course) notFound();

  return (
    <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 max-w-4xl">
      <Link
        href="/courses"
        className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground mb-6"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to courses
      </Link>

      <Card>
        <CardHeader>
          <div className="flex flex-wrap gap-2 mb-2">
            <Badge className={getLevelColor(course.level.toLowerCase())}>{course.level}</Badge>
            <Badge variant="outline">{course.type.replace("_", " ")}</Badge>
            {course.tags.map((t) => (
              <Badge key={t} variant="secondary">
                {t}
              </Badge>
            ))}
          </div>
          <h1 className="text-2xl font-bold">{course.title}</h1>
          <p className="text-muted-foreground">
            {course.provider}
            {course.duration && ` • ${course.duration}`}
            {course.rating != null && ` • ⭐ ${course.rating}`}
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          {course.description && (
            <p className="text-muted-foreground">{course.description}</p>
          )}
          <Button asChild>
            <a href={course.url} target="_blank" rel="noopener noreferrer">
              View course
              <ExternalLink className="ml-2 h-4 w-4" />
            </a>
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
