import { describe, it, expect } from "vitest";
import {
  progressSchema,
  bookmarkSchema,
  courseSchema,
  userProfileSchema,
} from "./validations";

describe("progressSchema", () => {
  it("accepts valid progress", () => {
    const result = progressSchema.safeParse({
      nodeId: "clxyz12345678901234567890",
      status: "COMPLETED",
    });
    expect(result.success).toBe(true);
  });

  it("rejects invalid status", () => {
    const result = progressSchema.safeParse({
      nodeId: "clxyz12345678901234567890",
      status: "INVALID",
    });
    expect(result.success).toBe(false);
  });
});

describe("bookmarkSchema", () => {
  it("accepts valid bookmark", () => {
    const result = bookmarkSchema.safeParse({
      entityType: "COURSE",
      entityId: "course-1",
    });
    expect(result.success).toBe(true);
  });

  it("rejects invalid entity type", () => {
    const result = bookmarkSchema.safeParse({
      entityType: "INVALID",
      entityId: "id",
    });
    expect(result.success).toBe(false);
  });
});

describe("courseSchema", () => {
  it("accepts valid course", () => {
    const result = courseSchema.safeParse({
      title: "ML Course",
      provider: "Coursera",
      url: "https://coursera.org/learn/ml",
      level: "BEGINNER",
      type: "FULL",
    });
    expect(result.success).toBe(true);
  });

  it("rejects invalid URL", () => {
    const result = courseSchema.safeParse({
      title: "ML",
      provider: "Coursera",
      url: "not-a-url",
      level: "BEGINNER",
      type: "FULL",
    });
    expect(result.success).toBe(false);
  });
});

describe("userProfileSchema", () => {
  it("accepts empty object", () => {
    const result = userProfileSchema.safeParse({});
    expect(result.success).toBe(true);
  });

  it("accepts valid profile", () => {
    const result = userProfileSchema.safeParse({
      name: "John",
      level: "INTERMEDIATE",
      goals: "Get a job",
    });
    expect(result.success).toBe(true);
  });
});
