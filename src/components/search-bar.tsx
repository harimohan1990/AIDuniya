"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useState, useTransition, useCallback } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Search } from "lucide-react";

interface SearchBarProps {
  placeholder?: string;
  defaultValue?: string;
  /** Path to navigate on search (default: /courses) */
  searchPath?: string;
  /** Query param name (default: search for courses, q for search page) */
  paramName?: string;
}

export function SearchBar({
  placeholder = "Search...",
  defaultValue,
  searchPath = "/courses",
  paramName,
}: SearchBarProps) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const param = paramName ?? (searchPath === "/search" ? "q" : "search");
  const [query, setQuery] = useState(
    defaultValue ?? searchParams.get(param) ?? searchParams.get("search") ?? ""
  );
  const [isPending, startTransition] = useTransition();

  const handleSearch = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    const params = new URLSearchParams();
    if (query.trim()) {
      params.set(param, query.trim());
    }
    startTransition(() => {
      router.push(`${searchPath}${params.toString() ? `?${params.toString()}` : ""}`);
    });
  }, [query, param, searchPath, router]);

  return (
    <form onSubmit={handleSearch} className="flex flex-col sm:flex-row gap-2 w-full">
      <div className="relative flex-1">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          type="search"
          placeholder={placeholder}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="pl-9"
        />
      </div>
      <Button type="submit" disabled={isPending} className="shrink-0">
        {isPending ? "Searching..." : "Search"}
      </Button>
    </form>
  );
}
