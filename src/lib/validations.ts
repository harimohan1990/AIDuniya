import { z } from "zod";

export const roadmapNodeSchema = z.object({
  title: z.string().min(1).max(200),
  content: z.string().optional(),
  difficulty: z.enum(["beginner", "intermediate", "advanced"]).optional(),
  order: z.number().int().min(0).optional(),
  resources: z.string().optional(),
});

export const courseSchema = z.object({
  title: z.string().min(1).max(300),
  provider: z.string().min(1).max(100),
  url: z.string().url(),
  level: z.enum(["BEGINNER", "INTERMEDIATE", "ADVANCED"]),
  type: z.enum(["SHORT", "FULL", "PRO_CERTIFICATE"]),
  duration: z.string().max(50).optional(),
  description: z.string().optional(),
  rating: z.number().min(0).max(5).optional(),
  tags: z.array(z.string()).optional(),
});

export const guideSchema = z.object({
  title: z.string().min(1).max(300),
  contentMDX: z.string().min(1),
  tags: z.array(z.string()).optional(),
});

export const projectSchema = z.object({
  title: z.string().min(1).max(200),
  level: z.enum(["BEGINNER", "INTERMEDIATE", "ADVANCED"]),
  description: z.string().min(1),
  skillsLearned: z.array(z.string()).optional(),
  datasetSuggestions: z.array(z.string()).optional(),
  tags: z.array(z.string()).optional(),
});

export const progressSchema = z.object({
  nodeId: z.string().cuid(),
  status: z.enum(["NOT_STARTED", "IN_PROGRESS", "COMPLETED"]),
});

export const bookmarkSchema = z.object({
  entityType: z.enum(["ROADMAP", "COURSE", "GUIDE"]),
  entityId: z.string().min(1),
});

export const userProfileSchema = z.object({
  name: z.string().min(1).max(100).optional(),
  level: z.enum(["BEGINNER", "INTERMEDIATE", "ADVANCED"]).optional(),
  goals: z.string().max(500).optional(),
  preferredTrack: z.string().max(200).optional(),
});
