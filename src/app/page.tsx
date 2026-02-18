import Link from "next/link";
import { Suspense } from "react";
import { getFeaturedData } from "@/lib/data";

export const dynamic = "force-dynamic";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { SearchBar } from "@/components/search-bar";
import { Map, BookOpen, FileText, FolderKanban } from "lucide-react";
import { getLevelColor } from "@/lib/utils";

export default async function HomePage() {
  const { roadmaps, courses, guides, projects } = await getFeaturedData();
  const hasData = roadmaps.length > 0 || courses.length > 0 || projects.length > 0;

  return (
    <div className="flex flex-col">
      {/* Hero */}
      <section className="relative overflow-hidden border-b gradient-mesh bg-gradient-to-b from-primary/[0.06] via-background to-background py-24 md:py-32">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 text-center max-w-7xl 2xl:max-w-8xl relative">
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight md:text-6xl lg:text-7xl text-foreground">
            Grow your AI career
          </h1>
          <p className="mt-6 text-lg sm:text-xl text-muted-foreground max-w-2xl mx-auto px-2 font-medium">
            Structured roadmaps, curated courses, and practical projects to master AI and ML.
            From prompt engineering to production MLOps.
          </p>
          <div className="mt-6 sm:mt-8 max-w-xl mx-auto px-2">
            <Suspense fallback={<div className="h-10 rounded-md border bg-muted animate-pulse" />}>
              <SearchBar placeholder="Search courses, roadmaps, guides..." searchPath="/search" />
            </Suspense>
          </div>
        </div>
      </section>

      {/* Categories */}
      <section className="py-8 sm:py-12 border-b">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl 2xl:max-w-8xl">
          <h2 className="text-xl sm:text-2xl font-semibold mb-4 sm:mb-6">Explore by category</h2>
          <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4">
            <Link href="/roadmaps?type=role">
              <Card className="h-full card-hover cursor-pointer border-0 shadow-sm">
                <CardHeader className="pb-2">
                  <Map className="h-8 w-8 text-primary mb-2" />
                  <CardTitle className="text-sm sm:text-base">Role-based Roadmaps</CardTitle>
                  <CardDescription className="text-xs sm:text-sm line-clamp-2">AI Engineer, ML Engineer, MLOps, Data Scientist</CardDescription>
                </CardHeader>
              </Card>
            </Link>
            <Link href="/roadmaps?type=skill">
              <Card className="h-full card-hover cursor-pointer border-0 shadow-sm">
                <CardHeader className="pb-2">
                  <BookOpen className="h-8 w-8 text-primary mb-2" />
                  <CardTitle className="text-sm sm:text-base">Skill-based Roadmaps</CardTitle>
                  <CardDescription className="text-xs sm:text-sm line-clamp-2">Prompt Eng, RAG, LLMOps, Fine-tuning, Agents</CardDescription>
                </CardHeader>
              </Card>
            </Link>
            <Link href="/projects">
              <Card className="h-full card-hover cursor-pointer border-0 shadow-sm">
                <CardHeader className="pb-2">
                  <FolderKanban className="h-8 w-8 text-primary mb-2" />
                  <CardTitle className="text-sm sm:text-base">Project Ideas</CardTitle>
                  <CardDescription className="text-xs sm:text-sm line-clamp-2">Hands-on projects from beginner to advanced</CardDescription>
                </CardHeader>
              </Card>
            </Link>
            <Link href="/guides">
              <Card className="h-full card-hover cursor-pointer border-0 shadow-sm">
                <CardHeader className="pb-2">
                  <FileText className="h-8 w-8 text-primary mb-2" />
                  <CardTitle className="text-sm sm:text-base">Guides</CardTitle>
                  <CardDescription className="text-xs sm:text-sm line-clamp-2">MDX articles, tutorials, and best practices</CardDescription>
                </CardHeader>
              </Card>
            </Link>
          </div>
        </div>
      </section>

      {/* Featured Roadmaps */}
      <section className="py-8 sm:py-12 border-b">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl 2xl:max-w-8xl">
          <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4 mb-4 sm:mb-6">
            <h2 className="text-xl sm:text-2xl font-semibold">Featured Roadmaps</h2>
            <Button variant="ghost" asChild>
              <Link href="/roadmaps">View all</Link>
            </Button>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {roadmaps.map((r) => (
              <Link key={r.id} href={`/roadmaps/${r.slug}`}>
                <Card className="h-full card-hover border-0 shadow-sm">
                  <CardHeader>
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary">{r.type}</Badge>
                    </div>
                    <CardTitle className="text-lg">{r.title}</CardTitle>
                    <CardDescription className="line-clamp-2">{r.description}</CardDescription>
                  </CardHeader>
                </Card>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Featured Courses */}
      <section className="py-8 sm:py-12 border-b">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl 2xl:max-w-8xl">
          <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4 mb-4 sm:mb-6">
            <h2 className="text-xl sm:text-2xl font-semibold">Featured Courses</h2>
            <Button variant="ghost" asChild>
              <Link href="/courses">View all</Link>
            </Button>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {courses.map((c) => (
              <Link key={c.id} href={`/courses/${c.id}`}>
                <Card className="h-full card-hover border-0 shadow-sm">
                  <CardHeader>
                    <div className="flex flex-wrap gap-1 mb-2">
                      <Badge className={getLevelColor((c.level ?? "beginner").toLowerCase())}>{c.level ?? "Beginner"}</Badge>
                      <Badge variant="outline">{(c.type ?? "full").replace("_", " ")}</Badge>
                    </div>
                    <CardTitle className="text-lg">{c.title}</CardTitle>
                    <CardDescription>
                      {c.provider} {c.duration && `• ${c.duration}`}
                    </CardDescription>
                  </CardHeader>
                </Card>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Project Ideas */}
      <section className="py-8 sm:py-12">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl 2xl:max-w-8xl">
          <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4 mb-4 sm:mb-6">
            <h2 className="text-xl sm:text-2xl font-semibold">Project Ideas</h2>
            <Button variant="ghost" asChild>
              <Link href="/projects">View all</Link>
            </Button>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {projects.map((p) => (
              <Link key={p.id} href={`/projects/${p.id}`}>
                <Card className="h-full card-hover border-0 shadow-sm">
                  <CardHeader>
                    <Badge className={getLevelColor((p.level ?? "beginner").toLowerCase())}>{p.level ?? "Beginner"}</Badge>
                    <CardTitle className="text-lg">{p.title}</CardTitle>
                    <CardDescription className="line-clamp-2">{p.description}</CardDescription>
                  </CardHeader>
                </Card>
              </Link>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
