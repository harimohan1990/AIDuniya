import Link from "next/link";
import { getBlogPosts } from "@/lib/data";

export const dynamic = "force-dynamic";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Newspaper } from "lucide-react";

export default async function BlogPage({
  searchParams,
}: {
  searchParams: Promise<{ search?: string }>;
}) {
  const { search } = await searchParams;
  const posts = await getBlogPosts(search ?? undefined);

  return (
    <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 max-w-7xl 2xl:max-w-8xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <Newspaper className="h-8 w-8 text-primary" />
          Blog
        </h1>
        <p className="mt-2 text-muted-foreground max-w-2xl">
          Updates, insights, and articles on AI learning, LLM trends, RAG, and building AI applications. Stay informed as the field evolves.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
        {posts.map((post) => (
          <Link key={post.id} href={`/blog/${post.slug}`}>
            <Card className="h-full card-hover transition-all duration-200 hover:shadow-lg hover:-translate-y-0.5">
              <CardHeader>
                <h3 className="font-semibold line-clamp-2 text-lg">{post.title}</h3>
                {post.excerpt && (
                  <p className="text-sm text-muted-foreground line-clamp-3 mt-1">
                    {post.excerpt}
                  </p>
                )}
                <div className="flex flex-wrap gap-1 mt-3">
                  {post.tags.map((t) => (
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

      {posts.length === 0 && (
        <p className="text-muted-foreground text-center py-12">
          No blog posts found. Run the seed script to add data.
        </p>
      )}
    </div>
  );
}
