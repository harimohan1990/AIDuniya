import { notFound } from "next/navigation";
import Link from "next/link";
import { getGuide } from "@/lib/data";

export const dynamic = "force-dynamic";
import { MDXContent } from "@/components/mdx-content";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft } from "lucide-react";

export default async function GuideDetailPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const guide = await getGuide(slug);
  if (!guide) notFound();

  return (
    <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 max-w-3xl">
      <Link
        href="/guides"
        className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground mb-6"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to guides
      </Link>

      <article>
        <h1 className="text-3xl font-bold mb-4">{guide.title}</h1>
        <div className="flex flex-wrap gap-2 mb-8">
          {guide.tags.map((t) => (
            <Badge key={t} variant="secondary">
              {t}
            </Badge>
          ))}
        </div>
        <div className="prose prose-slate dark:prose-invert max-w-none">
          <MDXContent source={guide.contentMDX} />
        </div>
      </article>
    </div>
  );
}
