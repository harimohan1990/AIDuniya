import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export async function POST(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const session = await getServerSession(authOptions);
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id: roadmapId } = await params;
  const userId = (session.user as { id?: string }).id!;

  const roadmap = await prisma.roadmap.findUnique({
    where: { id: roadmapId },
  });
  if (!roadmap) {
    return NextResponse.json({ error: "Roadmap not found" }, { status: 404 });
  }

  await prisma.savedRoadmap.upsert({
    where: {
      userId_roadmapId: { userId, roadmapId },
    },
    create: { userId, roadmapId },
    update: {},
  });

  return NextResponse.json({ success: true });
}

export async function DELETE(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const session = await getServerSession(authOptions);
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id: roadmapId } = await params;
  const userId = (session.user as { id?: string }).id!;

  await prisma.savedRoadmap.deleteMany({
    where: { userId, roadmapId },
  });

  return NextResponse.json({ success: true });
}
