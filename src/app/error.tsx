"use client";

import { useEffect } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { AlertCircle } from "lucide-react";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Application error:", error);
  }, [error]);

  const isDev = process.env.NODE_ENV === "development";

  return (
    <div className="container mx-auto flex min-h-[60vh] flex-col items-center justify-center px-4 text-center">
      <AlertCircle className="h-16 w-16 text-destructive mb-4" />
      <h1 className="text-2xl font-bold">Something went wrong</h1>
      <p className="mt-2 text-muted-foreground max-w-md">
        An unexpected error occurred. Please try again.
      </p>
      {isDev && error?.message && (
        <pre className="mt-4 p-4 bg-muted rounded-lg text-left text-sm overflow-auto max-w-2xl max-h-40">
          {error.message}
        </pre>
      )}
      <div className="mt-6 flex gap-4">
        <Button onClick={reset}>Try again</Button>
        <Button variant="outline" asChild>
          <Link href="/">Go home</Link>
        </Button>
      </div>
    </div>
  );
}
