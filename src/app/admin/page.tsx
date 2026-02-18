import { redirect } from "next/navigation";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import Link from "next/link";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { BookOpen, Map, FileText, FolderKanban } from "lucide-react";

export default async function AdminPage() {
  const session = await getServerSession(authOptions);
  if (!session?.user) redirect("/auth/signin");

  const role = (session.user as { role?: string }).role;
  if (role !== "ADMIN") redirect("/dashboard");

  return (
    <div className="container mx-auto px-4 py-12">
      <h1 className="text-3xl font-bold mb-8">Admin Dashboard</h1>

      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Link href="/admin/courses">
          <Card className="hover:border-primary/50 transition-colors h-full">
            <CardHeader>
              <BookOpen className="h-8 w-8 text-primary mb-2" />
              <h3 className="font-semibold">Courses</h3>
              <p className="text-sm text-muted-foreground">Add, edit, manage courses</p>
            </CardHeader>
          </Card>
        </Link>
        <Link href="/admin/roadmaps">
          <Card className="hover:border-primary/50 transition-colors h-full">
            <CardHeader>
              <Map className="h-8 w-8 text-primary mb-2" />
              <h3 className="font-semibold">Roadmaps</h3>
              <p className="text-sm text-muted-foreground">Manage nodes and edges</p>
            </CardHeader>
          </Card>
        </Link>
        <Link href="/admin/guides">
          <Card className="hover:border-primary/50 transition-colors h-full">
            <CardHeader>
              <FileText className="h-8 w-8 text-primary mb-2" />
              <h3 className="font-semibold">Guides</h3>
              <p className="text-sm text-muted-foreground">MDX content management</p>
            </CardHeader>
          </Card>
        </Link>
        <Link href="/admin/projects">
          <Card className="hover:border-primary/50 transition-colors h-full">
            <CardHeader>
              <FolderKanban className="h-8 w-8 text-primary mb-2" />
              <h3 className="font-semibold">Projects</h3>
              <p className="text-sm text-muted-foreground">Project ideas</p>
            </CardHeader>
          </Card>
        </Link>
      </div>

      <p className="mt-8 text-sm text-muted-foreground">
        Admin CRUD pages can be added. For now, use Prisma Studio (<code>npm run db:studio</code>) to manage data.
      </p>
    </div>
  );
}
