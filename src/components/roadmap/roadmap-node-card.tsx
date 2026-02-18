"use client";

import { useState, useMemo, memo } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Checkbox } from "@/components/ui/checkbox";
import { ExternalLink, BookOpen, FileText, Video } from "lucide-react";
import { getLevelColor } from "@/lib/utils";
import { useProgress } from "@/hooks/use-progress";

function NodeContent({ content }: { content: string }) {
  const formatBold = (text: string) => {
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((p, i) =>
      p.startsWith("**") && p.endsWith("**") ? (
        <strong key={i} className="text-foreground font-medium">{p.slice(2, -2)}</strong>
      ) : (
        <span key={i}>{p}</span>
      )
    );
  };
  const blocks = content.split(/\n\n+/);
  return (
    <div className="space-y-4 text-sm">
      {blocks.map((block, i) => {
        const trimmed = block.trim();
        if (!trimmed) return null;
        const lines = trimmed.split("\n");
        const bullets = lines.filter((l) => /^[•\-]\s/.test(l));
        if (bullets.length > 0) {
          return (
            <div key={i} className="space-y-2">
              {lines[0] && !lines[0].startsWith("•") && !lines[0].startsWith("-") && (
                <p className="text-muted-foreground">{formatBold(lines[0])}</p>
              )}
              <ul className="list-disc list-inside space-y-1 text-muted-foreground ml-1">
                {bullets.map((b, j) => (
                  <li key={j}>{formatBold(b.replace(/^[•\-]\s*/, ""))}</li>
                ))}
              </ul>
            </div>
          );
        }
        return (
          <p key={i} className="text-muted-foreground leading-relaxed">
            {formatBold(trimmed)}
          </p>
        );
      })}
    </div>
  );
}

function ResourceIcon({ type }: { type?: string }) {
  switch (type) {
    case "course":
      return <BookOpen className="h-3.5 w-3.5 text-primary" />;
    case "article":
      return <FileText className="h-3.5 w-3.5 text-amber-600" />;
    case "video":
      return <Video className="h-3.5 w-3.5 text-red-500" />;
    default:
      return <ExternalLink className="h-3.5 w-3.5" />;
  }
}

interface RoadmapNodeCardProps {
  roadmapId: string;
  node: {
    id: string;
    title: string;
    content: string | null;
    difficulty: string | null;
    resources: string | null;
  };
  compact?: boolean;
}

function RoadmapNodeCardInner({ roadmapId, node, compact }: RoadmapNodeCardProps) {
  const [open, setOpen] = useState(false);
  const { status, toggleProgress, isLoading } = useProgress(roadmapId, node.id);

  const resources = useMemo(() => {
    try {
      return node.resources
        ? (JSON.parse(node.resources) as { type?: string; title?: string; url?: string }[])
        : [];
    } catch {
      return [];
    }
  }, [node.resources]);

  const isCompleted = status === "COMPLETED";

  return (
    <>
      <Card
        className={`cursor-pointer transition-all hover:shadow-md ${
          isCompleted ? "border-primary/50 bg-primary/5" : ""
        } ${compact ? "h-full" : ""}`}
        onClick={() => setOpen(true)}
      >
        <CardHeader className="p-4 pb-2">
          <div className="flex items-start justify-between gap-2">
            <div className="flex-1 min-w-0">
              <h3 className="font-semibold text-sm line-clamp-2">{node.title}</h3>
              <div className="flex flex-wrap gap-1 mt-1.5">
                {node.difficulty && (
                  <Badge className={`${getLevelColor(node.difficulty)} text-[10px] px-1.5 py-0`} variant="secondary">
                    {node.difficulty}
                  </Badge>
                )}
                {compact && resources.length > 0 && (
                  <span className="text-[10px] text-muted-foreground">
                    {resources.length} resources
                  </span>
                )}
              </div>
            </div>
            <div
              onClick={(e) => e.stopPropagation()}
              className="flex items-center shrink-0"
            >
              <Checkbox
                checked={isCompleted}
                onCheckedChange={() => toggleProgress()}
                disabled={isLoading}
                className="data-[state=checked]:bg-primary data-[state=checked]:border-primary"
              />
            </div>
          </div>
        </CardHeader>
        {!compact && resources.length > 0 && (
          <CardContent className="p-4 pt-0">
            <p className="text-xs text-muted-foreground flex items-center gap-1">
              <BookOpen className="h-3 w-3" />
              {resources.length} resource{resources.length !== 1 ? "s" : ""}
            </p>
          </CardContent>
        )}
      </Card>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-[95vw] sm:max-w-lg max-h-[85vh] sm:max-h-[80vh] overflow-y-auto w-full">
          <DialogHeader>
            <DialogTitle>{node.title}</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            {node.difficulty && (
              <Badge className={getLevelColor(node.difficulty)}>{node.difficulty}</Badge>
            )}
            {node.content && (
              <NodeContent content={node.content} />
            )}
            {resources.length > 0 && (
              <div>
                <h4 className="font-medium mb-3">Resources</h4>
                <ul className="space-y-3">
                  {resources.map((r, i) => (
                    <li key={i}>
                      <a
                        href={r.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-start gap-3 p-2 rounded-lg hover:bg-muted/50 transition-colors group"
                      >
                        <ResourceIcon type={r.type} />
                        <span className="flex-1 text-sm group-hover:text-primary">
                          {r.title ?? r.url}
                        </span>
                        <ExternalLink className="h-3 w-3 shrink-0 opacity-50 group-hover:opacity-100" />
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            <div className="flex items-center gap-2 pt-4 border-t">
              <Checkbox
                id={`complete-${node.id}`}
                checked={isCompleted}
                onCheckedChange={() => toggleProgress()}
                disabled={isLoading}
              />
              <label htmlFor={`complete-${node.id}`} className="text-sm cursor-pointer">
                Mark as complete
              </label>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}

export const RoadmapNodeCard = memo(RoadmapNodeCardInner);
