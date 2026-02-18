import { NextResponse } from "next/server";
import { getSafeSession } from "@/lib/session";
import { prisma } from "@/lib/prisma";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const session = await getSafeSession();
  if (!session?.user) {
    return NextResponse.json({ completed: 0, total: 0, percent: 0 });
  }

  const { id: roadmapId } = await params;

  const [total, completed] = await Promise.all([
    prisma.roadmapNode.count({ where: { roadmapId } }),
    prisma.progress.count({
      where: {
        userId: (session.user as { id?: string }).id!,
        status: "COMPLETED",
        node: { roadmapId },
      },
    }),
  ]);

  const percent = total > 0 ? Math.round((completed / total) * 100) : 0;

  return NextResponse.json({ completed, total, percent });
}
