"use client";

import { useSession } from "next-auth/react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Bookmark, BookmarkCheck } from "lucide-react";

interface SaveRoadmapButtonProps {
  roadmapId: string;
  initialSaved?: boolean;
}

export function SaveRoadmapButton({ roadmapId, initialSaved = false }: SaveRoadmapButtonProps) {
  const { status } = useSession();
  const [saved, setSaved] = useState(initialSaved);
  const [loading, setLoading] = useState(false);

  if (status !== "authenticated") return null;

  const toggle = async () => {
    setLoading(true);
    try {
      if (saved) {
        await fetch(`/api/roadmaps/${roadmapId}/save`, { method: "DELETE" });
        setSaved(false);
      } else {
        await fetch(`/api/roadmaps/${roadmapId}/save`, { method: "POST" });
        setSaved(true);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <Button
      variant="outline"
      size="sm"
      onClick={toggle}
      disabled={loading}
    >
      {saved ? (
        <>
          <BookmarkCheck className="h-4 w-4 mr-2" />
          Saved
        </>
      ) : (
        <>
          <Bookmark className="h-4 w-4 mr-2" />
          Save to My Roadmaps
        </>
      )}
    </Button>
  );
}
