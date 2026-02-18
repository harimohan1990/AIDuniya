"use client";

import { useProgressSummary } from "@/hooks/use-progress";
import { Progress } from "@/components/ui/progress";

interface RoadmapProgressProps {
  roadmapId: string;
}

export function RoadmapProgress({ roadmapId }: RoadmapProgressProps) {
  const { completed, total, percent } = useProgressSummary(roadmapId);

  if (total === 0) return null;

  return (
    <div className="mt-4 flex items-center gap-4">
      <Progress value={percent} className="w-48 h-2" />
      <span className="text-sm text-muted-foreground">
        {completed} / {total} completed
      </span>
    </div>
  );
}
