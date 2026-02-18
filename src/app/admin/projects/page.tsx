import { redirect } from "next/navigation";
import { getSafeSession } from "@/lib/session";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";

export default async function AdminProjectsPage() {
  const session = await getSafeSession();
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
      <h1 className="text-2xl font-bold mb-4">Manage Project Ideas</h1>
      <p className="text-muted-foreground">
        Use Prisma Studio to manage project ideas.
      </p>
    </div>
  );
}
