import Link from "next/link";
import { getGuides } from "@/lib/data";

export const dynamic = "force-dynamic";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export default async function GuidesPage({
  searchParams,
}: {
  searchParams: Promise<{ search?: string }>;
}) {
  const { search } = await searchParams;
  const guides = await getGuides(search ?? undefined);

  return (
    <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 max-w-7xl 2xl:max-w-8xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Guides</h1>
        <p className="mt-2 text-muted-foreground max-w-2xl">
          Practical articles and tutorials on prompt engineering, RAG architecture, vector databases, fine-tuning vs RAG, and building AI agents. Written for practitioners who want to ship, not just learn theory.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4">
        {guides.map((g) => (
          <Link key={g.id} href={`/guides/${g.slug}`}>
            <Card className="h-full hover:border-primary/50 transition-colors">
              <CardHeader>
                <h3 className="font-semibold line-clamp-2">{g.title}</h3>
                <div className="flex flex-wrap gap-1">
                  {g.tags.map((t) => (
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

      {guides.length === 0 && (
        <p className="text-muted-foreground text-center py-12">
          No guides found. Run the seed script to add data.
        </p>
      )}
    </div>
  );
}
