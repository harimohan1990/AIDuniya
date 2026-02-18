"use client";

import { useSession } from "next-auth/react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

export function ProfileForm() {
  const { data: session, status } = useSession();
  const [saved, setSaved] = useState(false);
  const [level, setLevel] = useState("__none__");

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
        level: level && level !== "__none__" ? level : undefined,
        goals: formData.get("goals") || undefined,
        preferredTrack: formData.get("preferredTrack") || undefined,
      }),
    });
    if (res.ok) setSaved(true);
  };

  return (
    <Card>
      <CardHeader>
        <h2 className="font-semibold">Update your profile</h2>
        <p className="text-sm text-muted-foreground">
          Set your experience level and goals for better AI Tutor recommendations.
        </p>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
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
            <Label htmlFor="level">Experience Level</Label>
            <Select value={level} onValueChange={setLevel}>
              <SelectTrigger>
                <SelectValue placeholder="Select level" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__none__">Not set</SelectItem>
                <SelectItem value="BEGINNER">Beginner</SelectItem>
                <SelectItem value="INTERMEDIATE">Intermediate</SelectItem>
                <SelectItem value="ADVANCED">Advanced</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="goals">Goals</Label>
            <Input
              id="goals"
              name="goals"
              placeholder="e.g. Get a job as ML Engineer, Build a startup"
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
          <Button type="submit">{saved ? "Saved!" : "Save"}</Button>
        </form>
      </CardContent>
    </Card>
  );
}
