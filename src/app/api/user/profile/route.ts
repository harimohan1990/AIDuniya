import { NextResponse } from "next/server";
import { getSafeSession } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import { z } from "zod";
import { userProfileSchema } from "@/lib/validations";

export async function PATCH(req: Request) {
  const session = await getSafeSession();
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const body = await req.json();
    const data = userProfileSchema.parse(body);
    const userId = (session.user as { id?: string }).id!;

    await prisma.user.update({
      where: { id: userId },
      data: {
        name: data.name,
        level: data.level as "BEGINNER" | "INTERMEDIATE" | "ADVANCED" | undefined,
        goals: data.goals,
        preferredTrack: data.preferredTrack,
      },
    });

    return NextResponse.json({ success: true });
  } catch (e) {
    if (e instanceof z.ZodError) {
      return NextResponse.json(
        { error: e.errors.map((x) => x.message).join(", ") },
        { status: 400 }
      );
    }
    return NextResponse.json(
      { error: "Failed to update profile" },
      { status: 500 }
    );
  }
}
