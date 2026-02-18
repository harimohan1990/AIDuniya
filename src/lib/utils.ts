import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDuration(duration: string | null | undefined): string {
  if (!duration) return "—";
  return duration;
}

export function getLevelColor(level: string): string {
  switch (level?.toLowerCase()) {
    case "beginner":
      return "bg-emerald-500/20 text-emerald-600 dark:text-emerald-400";
    case "intermediate":
      return "bg-amber-500/20 text-amber-600 dark:text-amber-400";
    case "advanced":
      return "bg-rose-500/20 text-rose-600 dark:text-rose-400";
    default:
      return "bg-slate-500/20 text-slate-600 dark:text-slate-400";
  }
}
