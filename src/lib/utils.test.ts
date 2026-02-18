import { describe, it, expect } from "vitest";
import { cn, getLevelColor, formatDuration } from "./utils";

describe("cn", () => {
  it("merges class names", () => {
    expect(cn("foo", "bar")).toBe("foo bar");
  });

  it("handles conditional classes", () => {
    expect(cn("base", false && "hidden", "visible")).toBe("base visible");
  });

  it("merges tailwind classes correctly", () => {
    expect(cn("px-2", "px-4")).toBe("px-4");
  });
});

describe("getLevelColor", () => {
  it("returns beginner color", () => {
    expect(getLevelColor("beginner")).toContain("emerald");
  });

  it("returns intermediate color", () => {
    expect(getLevelColor("intermediate")).toContain("amber");
  });

  it("returns advanced color", () => {
    expect(getLevelColor("advanced")).toContain("rose");
  });

  it("returns default for unknown", () => {
    expect(getLevelColor("")).toContain("slate");
  });
});

describe("formatDuration", () => {
  it("returns duration as-is", () => {
    expect(formatDuration("10 hours")).toBe("10 hours");
  });

  it("returns dash for null/undefined", () => {
    expect(formatDuration(null)).toBe("—");
    expect(formatDuration(undefined)).toBe("—");
  });
});
