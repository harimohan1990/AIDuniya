import { redirect } from "next/navigation";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { SettingsForm } from "@/components/dashboard/settings-form";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";

export default async function SettingsPage() {
  const session = await getServerSession(authOptions);
  if (!session?.user) redirect("/auth/signin");

  return (
    <div className="container mx-auto px-4 py-8 max-w-2xl">
      <Link
        href="/dashboard"
        className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground mb-6"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to Dashboard
      </Link>
      <h1 className="text-2xl font-bold mb-6">Settings</h1>
      <SettingsForm />
    </div>
  );
}
