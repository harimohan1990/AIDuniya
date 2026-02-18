import Link from "next/link";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { getLevelColor } from "@/lib/utils";

interface Course {
  id: string;
  title: string;
  provider: string;
  level: string;
  type: string;
  duration: string | null;
  tags: string[];
  rating: number | null;
}

export function CourseGrid({ courses }: { courses: Course[] }) {
  if (courses.length === 0) {
    return (
      <p className="text-muted-foreground text-center py-12">
        No courses found. Try adjusting your filters.
      </p>
    );
  }

  return (
    <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-4">
      {courses.map((c) => (
        <Link key={c.id} href={`/courses/${c.id}`}>
          <Card className="h-full hover:border-primary/50 transition-colors">
            <CardHeader>
              <div className="flex flex-wrap gap-1 mb-2">
                <Badge className={getLevelColor(c.level.toLowerCase())}>{c.level}</Badge>
                <Badge variant="outline">{c.type.replace("_", " ")}</Badge>
              </div>
              <h3 className="font-semibold line-clamp-2">{c.title}</h3>
              <p className="text-sm text-muted-foreground">
                {c.provider}
                {c.duration && ` • ${c.duration}`}
                {c.rating != null && ` • ⭐ ${c.rating}`}
              </p>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="flex flex-wrap gap-1">
                {c.tags.slice(0, 3).map((t) => (
                  <Badge key={t} variant="secondary" className="text-xs">
                    {t}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        </Link>
      ))}
    </div>
  );
}
