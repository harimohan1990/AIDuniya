"use client";

import { useSession } from "next-auth/react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

async function getProgress(roadmapId: string) {
  const res = await fetch(`/api/roadmaps/${roadmapId}/progress`);
  if (!res.ok) throw new Error("Failed to fetch progress");
  return res.json();
}

async function getProgressSummary(roadmapId: string) {
  const res = await fetch(`/api/roadmaps/${roadmapId}/progress/summary`);
  if (!res.ok) throw new Error("Failed to fetch progress");
  return res.json();
}

async function updateProgress(nodeId: string, status: string) {
  const res = await fetch(`/api/progress`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ nodeId, status }),
  });
  if (!res.ok) throw new Error("Failed to update progress");
  return res.json();
}

export function useProgress(roadmapId: string, nodeId: string) {
  const { data: session, status } = useSession();
  const queryClient = useQueryClient();

  const { data: progressMap } = useQuery({
    queryKey: ["progress", roadmapId],
    queryFn: () => getProgress(roadmapId),
    enabled: status === "authenticated",
  });

  const mutation = useMutation({
    mutationFn: ({ nodeId, status }: { nodeId: string; status: string }) =>
      updateProgress(nodeId, status),
    onSuccess: (_, { nodeId }) => {
      queryClient.invalidateQueries({ queryKey: ["progress", roadmapId] });
      queryClient.invalidateQueries({ queryKey: ["progress-summary", roadmapId] });
    },
  });

  const nodeStatus = progressMap?.[nodeId] ?? "NOT_STARTED";

  const toggleProgress = () => {
    if (status !== "authenticated") return;
    const next =
      nodeStatus === "COMPLETED" ? "NOT_STARTED" : "COMPLETED";
    mutation.mutate({ nodeId, status: next });
  };

  return {
    status: nodeStatus,
    toggleProgress,
    isLoading: mutation.isPending,
  };
}

export function useProgressSummary(roadmapId: string) {
  const { data: session, status } = useSession();

  const { data } = useQuery({
    queryKey: ["progress-summary", roadmapId],
    queryFn: () => getProgressSummary(roadmapId),
    enabled: status === "authenticated",
  });

  return {
    completed: data?.completed ?? 0,
    total: data?.total ?? 0,
    percent: data?.percent ?? 0,
  };
}
