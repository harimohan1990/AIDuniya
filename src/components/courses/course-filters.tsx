"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useCallback } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

const LEVELS = ["BEGINNER", "INTERMEDIATE", "ADVANCED"];
const TYPES = ["SHORT", "FULL", "PRO_CERTIFICATE"];
const PROVIDERS = [
  "Coursera",
  "edX",
  "DeepLearning.AI",
  "Fast.ai",
  "Stanford",
  "MIT",
  "Google",
  "Microsoft",
  "Udacity",
  "YouTube",
];

export function CourseFilters() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const search = searchParams.get("search") ?? "";
  const level = searchParams.get("level") ?? "__all__";
  const type = searchParams.get("type") ?? "__all__";
  const provider = searchParams.get("provider") ?? "__all__";

  const updateFilters = useCallback((key: string, value: string) => {
    const params = new URLSearchParams(searchParams.toString());
    if (value && value !== "__all__") params.set(key, value);
    else params.delete(key);
    router.push(`/courses?${params.toString()}`);
  }, [searchParams, router]);

  const handleSearch = useCallback((e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const form = e.currentTarget;
    const input = form.querySelector<HTMLInputElement>('input[name="search"]');
    if (input) updateFilters("search", input.value);
  }, [updateFilters]);

  const clearFilters = useCallback(() => {
    router.push("/courses");
  }, [router]);

  return (
    <Card>
      <CardHeader>
        <h3 className="font-semibold">Filters</h3>
      </CardHeader>
      <CardContent className="space-y-4">
        <form onSubmit={handleSearch} className="space-y-2">
          <Label>Search</Label>
          <Input
            name="search"
            placeholder="Title or tags..."
            defaultValue={search}
            className="w-full"
          />
          <Button type="submit" size="sm" className="w-full">
            Search
          </Button>
        </form>

        <div className="space-y-2">
          <Label>Level</Label>
          <Select value={level} onValueChange={(v) => updateFilters("level", v)}>
            <SelectTrigger>
              <SelectValue placeholder="Any level" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__all__">Any level</SelectItem>
              {LEVELS.map((l) => (
                <SelectItem key={l} value={l}>
                  {l.charAt(0) + l.slice(1).toLowerCase()}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label>Type</Label>
          <Select value={type} onValueChange={(v) => updateFilters("type", v)}>
            <SelectTrigger>
              <SelectValue placeholder="Any type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__all__">Any type</SelectItem>
              {TYPES.map((t) => (
                <SelectItem key={t} value={t}>
                  {t.replace("_", " ")}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label>Provider</Label>
          <Select value={provider} onValueChange={(v) => updateFilters("provider", v)}>
            <SelectTrigger>
              <SelectValue placeholder="Any provider" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__all__">Any provider</SelectItem>
              {PROVIDERS.map((p) => (
                <SelectItem key={p} value={p}>
                  {p}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <Button variant="outline" size="sm" className="w-full" onClick={clearFilters}>
          Clear filters
        </Button>
      </CardContent>
    </Card>
  );
}
