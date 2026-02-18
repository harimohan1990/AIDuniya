import { redirect } from "next/navigation";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";

export default async function AdminCoursesPage() {
  const session = await getServerSession(authOptions);
  if (!session?.user) redirect("/auth/signin");
  if ((session.user as { role?: string }).role !== "ADMIN") redirect("/dashboard");

  return (
    <div className="container mx-auto px-4 py-8">
      <Link
        href="/admin"
        className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground mb-6"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to Admin
      </Link>
      <h1 className="text-2xl font-bold mb-4">Manage Courses</h1>
      <p className="text-muted-foreground mb-4">
        Use Prisma Studio (<code>npm run db:studio</code>) to add and edit courses.
      </p>
      <Button asChild>
        <a href="http://localhost:5555" target="_blank" rel="noopener noreferrer">
          Open Prisma Studio
        </a>
      </Button>
    </div>
  );
}
