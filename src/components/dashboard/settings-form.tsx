"use client";

import { useSession } from "next-auth/react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

export function SettingsForm() {
  const { data: session, status } = useSession();
  const [saved, setSaved] = useState(false);

  if (status !== "authenticated") return null;

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const form = e.currentTarget;
    const formData = new FormData(form);
    const res = await fetch("/api/user/profile", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: formData.get("name"),
        goals: formData.get("goals") || undefined,
        preferredTrack: formData.get("preferredTrack") || undefined,
      }),
    });
    if (res.ok) setSaved(true);
  };

  return (
    <Card>
      <CardHeader>
        <h2 className="font-semibold">Account</h2>
        <p className="text-sm text-muted-foreground">
          Update your account settings.
        </p>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              value={session.user?.email ?? ""}
              disabled
              className="bg-muted"
            />
            <p className="text-xs text-muted-foreground">
              Email cannot be changed.
            </p>
          </div>
          <div className="space-y-2">
            <Label htmlFor="name">Name</Label>
            <Input
              id="name"
              name="name"
              defaultValue={session.user?.name ?? ""}
              placeholder="Your name"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="goals">Goals</Label>
            <Input
              id="goals"
              name="goals"
              placeholder="e.g. Get ML Engineer job, Build AI startup"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="preferredTrack">Preferred Track</Label>
            <Input
              id="preferredTrack"
              name="preferredTrack"
              placeholder="e.g. AI Engineer, GenAI"
            />
          </div>
          <Button type="submit">{saved ? "Saved!" : "Save changes"}</Button>
        </form>
      </CardContent>
    </Card>
  );
}
