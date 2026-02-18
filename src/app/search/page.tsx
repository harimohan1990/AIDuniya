import { Suspense } from "react";
import { searchAll } from "@/lib/data";
import { SearchResults } from "@/components/search/search-results";
import { SearchBar } from "@/components/search-bar";

export const dynamic = "force-dynamic";

export default async function SearchPage({
  searchParams,
}: {
  searchParams: Promise<{ q?: string }>;
}) {
  const { q } = await searchParams;
  const results = await searchAll(q ?? "");

  return (
    <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 max-w-7xl 2xl:max-w-8xl">
      <h1 className="text-2xl sm:text-3xl font-bold mb-4 sm:mb-6">Search</h1>
      <p className="text-muted-foreground mb-6 max-w-2xl">
        Search across courses, roadmaps, guides, and projects. Results are grouped by type. Try topics like &quot;RAG&quot;, &quot;deep learning&quot;, &quot;LLM&quot;, or &quot;MLOps&quot;.
      </p>
      <div className="flex flex-wrap gap-2 mb-6">
        <a href="/search?q=LLM" className="text-sm px-3 py-1.5 rounded-md bg-muted hover:bg-muted/80 transition-colors">LLM</a>
        <a href="/search?q=RAG" className="text-sm px-3 py-1.5 rounded-md bg-muted hover:bg-muted/80 transition-colors">RAG</a>
        <a href="/search?q=deep+learning" className="text-sm px-3 py-1.5 rounded-md bg-muted hover:bg-muted/80 transition-colors">Deep Learning</a>
        <a href="/search?q=MLOps" className="text-sm px-3 py-1.5 rounded-md bg-muted hover:bg-muted/80 transition-colors">MLOps</a>
        <a href="/search?q=prompt" className="text-sm px-3 py-1.5 rounded-md bg-muted hover:bg-muted/80 transition-colors">Prompt Engineering</a>
        <a href="/search?q=beginner" className="text-sm px-3 py-1.5 rounded-md bg-muted hover:bg-muted/80 transition-colors">Beginner</a>
      </div>

      <div className="max-w-2xl mb-8">
        <Suspense fallback={<div className="h-10 rounded-md border bg-muted animate-pulse" />}>
          <SearchBar
            placeholder="Search courses, roadmaps, guides..."
            defaultValue={q}
            searchPath="/search"
          />
        </Suspense>
      </div>

      <SearchResults
        query={q ?? ""}
        courses={results.courses}
        roadmaps={results.roadmaps}
        guides={results.guides}
        projects={results.projects}
      />
    </div>
  );
}
