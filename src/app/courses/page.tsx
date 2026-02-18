import { Suspense } from "react";
import { getCourses } from "@/lib/data";

export const dynamic = "force-dynamic";
import { CourseFilters } from "@/components/courses/course-filters";
import { CourseGrid } from "@/components/courses/course-grid";

export default async function CoursesPage({
  searchParams,
}: {
  searchParams: Promise<{ search?: string; level?: string; type?: string; provider?: string }>;
}) {
  const params = await searchParams;
  const courses = await getCourses(params.search, {
    level: params.level,
    type: params.type,
    provider: params.provider,
  });

  return (
    <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 max-w-7xl 2xl:max-w-8xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Courses</h1>
        <p className="mt-2 text-muted-foreground max-w-2xl">
          Curated AI and ML courses from Coursera, DeepLearning.AI, Fast.ai, Hugging Face, and more. Filter by level (beginner/intermediate/advanced), type (short/full), and provider. Each course includes duration and key topics.
        </p>
      </div>

      <div className="flex flex-col lg:flex-row gap-8">
        <aside className="lg:w-64 shrink-0">
          <Suspense fallback={<div className="h-64 bg-muted animate-pulse rounded-lg" />}>
            <CourseFilters />
          </Suspense>
        </aside>
        <div className="flex-1 min-w-0">
          <CourseGrid courses={courses} />
        </div>
      </div>
    </div>
  );
}
