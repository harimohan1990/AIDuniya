import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import { z } from "zod";
import { progressSchema } from "@/lib/validations";

export async function POST(req: Request) {
  const session = await getServerSession(authOptions);
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const body = await req.json();
    const { nodeId, status } = progressSchema.parse(body);

    const userId = (session.user as { id?: string }).id!;

    await prisma.progress.upsert({
      where: {
        userId_nodeId: { userId, nodeId },
      },
      create: { userId, nodeId, status },
      update: { status },
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
      { error: "Failed to update progress" },
      { status: 500 }
    );
  }
}
