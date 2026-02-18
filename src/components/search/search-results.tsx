"use client";

import { memo } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Map, BookOpen, FileText, FolderKanban } from "lucide-react";
import { getLevelColor } from "@/lib/utils";

interface SearchResultsProps {
  query: string;
  courses: { id: string; title: string; provider: string; level: string; type: string }[];
  roadmaps: { id: string; slug: string; title: string; type: string }[];
  guides: { id: string; slug: string; title: string; tags: string[] }[];
  projects: { id: string; title: string; level: string; description: string }[];
}

function SearchResultsInner({
  query,
  courses,
  roadmaps,
  guides,
  projects,
}: SearchResultsProps) {
  const hasResults =
    courses.length > 0 ||
    roadmaps.length > 0 ||
    guides.length > 0 ||
    projects.length > 0;

  if (!query.trim()) {
    return (
      <div className="space-y-4">
        <p className="text-muted-foreground">
          Enter a search term to find courses, roadmaps, guides, and projects. Results are grouped by type.
        </p>
        <div>
          <p className="text-sm font-medium mb-2">Popular searches</p>
          <div className="flex flex-wrap gap-2">
            <a href="/search?q=LLM" className="text-sm px-3 py-1.5 rounded-md border hover:bg-muted transition-colors">LLM</a>
            <a href="/search?q=RAG" className="text-sm px-3 py-1.5 rounded-md border hover:bg-muted transition-colors">RAG</a>
            <a href="/search?q=deep+learning" className="text-sm px-3 py-1.5 rounded-md border hover:bg-muted transition-colors">Deep Learning</a>
            <a href="/search?q=beginner" className="text-sm px-3 py-1.5 rounded-md border hover:bg-muted transition-colors">Beginner</a>
          </div>
        </div>
      </div>
    );
  }

  if (!hasResults) {
    return (
      <p className="text-muted-foreground">
        No results found for &quot;{query}&quot;. Try different keywords.
      </p>
    );
  }

  return (
    <div className="space-y-10">
      {courses.length > 0 && (
        <section>
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <BookOpen className="h-5 w-5" />
            Courses ({courses.length})
          </h2>
          <div className="grid md:grid-cols-2 gap-4">
            {courses.map((c) => (
              <Link key={c.id} href={`/courses/${c.id}`}>
                <Card className="hover:border-primary/50 transition-colors h-full">
                  <CardHeader className="pb-2">
                    <Badge className={getLevelColor(c.level.toLowerCase())} variant="secondary">
                      {c.level}
                    </Badge>
                    <h3 className="font-semibold">{c.title}</h3>
                    <p className="text-sm text-muted-foreground">{c.provider}</p>
                  </CardHeader>
                </Card>
              </Link>
            ))}
          </div>
        </section>
      )}

      {roadmaps.length > 0 && (
        <section>
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Map className="h-5 w-5" />
            Roadmaps ({roadmaps.length})
          </h2>
          <div className="grid md:grid-cols-2 gap-4">
            {roadmaps.map((r) => (
              <Link key={r.id} href={`/roadmaps/${r.slug}`}>
                <Card className="hover:border-primary/50 transition-colors h-full">
                  <CardHeader className="pb-2">
                    <Badge variant="secondary">{r.type}</Badge>
                    <h3 className="font-semibold">{r.title}</h3>
                  </CardHeader>
                </Card>
              </Link>
            ))}
          </div>
        </section>
      )}

      {guides.length > 0 && (
        <section>
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Guides ({guides.length})
          </h2>
          <div className="grid md:grid-cols-2 gap-4">
            {guides.map((g) => (
              <Link key={g.id} href={`/guides/${g.slug}`}>
                <Card className="hover:border-primary/50 transition-colors h-full">
                  <CardHeader className="pb-2">
                    <h3 className="font-semibold">{g.title}</h3>
                    <div className="flex flex-wrap gap-1">
                      {g.tags.slice(0, 3).map((t) => (
                        <Badge key={t} variant="secondary" className="text-xs">
                          {t}
                        </Badge>
                      ))}
                    </div>
                  </CardHeader>
                </Card>
              </Link>
            ))}
          </div>
        </section>
      )}

      {projects.length > 0 && (
        <section>
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <FolderKanban className="h-5 w-5" />
            Projects ({projects.length})
          </h2>
          <div className="grid md:grid-cols-2 gap-4">
            {projects.map((p) => (
              <Link key={p.id} href={`/projects/${p.id}`}>
                <Card className="hover:border-primary/50 transition-colors h-full">
                  <CardHeader className="pb-2">
                    <Badge className={getLevelColor(p.level.toLowerCase())} variant="secondary">
                      {p.level}
                    </Badge>
                    <h3 className="font-semibold">{p.title}</h3>
                    <p className="text-sm text-muted-foreground line-clamp-2">{p.description}</p>
                  </CardHeader>
                </Card>
              </Link>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

export const SearchResults = memo(SearchResultsInner);
