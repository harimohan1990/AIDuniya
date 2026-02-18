import { redirect } from "next/navigation";
import { getSafeSession } from "@/lib/session";
import { ProfileForm } from "@/components/dashboard/profile-form";

export default async function ProfilePage() {
  const session = await getSafeSession();
  if (!session?.user) redirect("/auth/signin");

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-6">Profile</h1>
      <ProfileForm />
    </div>
  );
}
