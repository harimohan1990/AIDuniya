import { notFound } from "next/navigation";
import Link from "next/link";
import { getBlogPost } from "@/lib/data";

export const dynamic = "force-dynamic";
import { MDXContent } from "@/components/mdx-content";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, Calendar } from "lucide-react";

export default async function BlogPostPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const post = await getBlogPost(slug);
  if (!post) notFound();

  const publishedAt =
    post.publishedAt instanceof Date
      ? post.publishedAt
      : new Date(post.publishedAt as string);

  return (
    <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 max-w-3xl">
      <Link
        href="/blog"
        className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground mb-6"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to blog
      </Link>

      <article>
        <h1 className="text-3xl sm:text-4xl font-bold mb-4">{post.title}</h1>
        <div className="flex flex-wrap items-center gap-2 mb-6 text-sm text-muted-foreground">
          <span className="flex items-center gap-1">
            <Calendar className="h-4 w-4" />
            {publishedAt.toLocaleDateString("en-US", {
              year: "numeric",
              month: "long",
              day: "numeric",
            })}
          </span>
          <span className="text-muted-foreground/60">·</span>
          <div className="flex flex-wrap gap-1">
            {post.tags.map((t) => (
              <Badge key={t} variant="secondary" className="text-xs">
                {t}
              </Badge>
            ))}
          </div>
        </div>
        {post.excerpt && (
          <p className="text-lg text-muted-foreground mb-8 border-l-4 border-primary/30 pl-4">
            {post.excerpt}
          </p>
        )}
        <div className="prose prose-slate dark:prose-invert max-w-none">
          <MDXContent source={post.contentMDX} />
        </div>
      </article>
    </div>
  );
}
