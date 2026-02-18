import { NextResponse } from "next/server";
import { getSafeSession } from "@/lib/session";
import { prisma } from "@/lib/prisma";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const session = await getSafeSession();
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id: roadmapId } = await params;

  const progress = await prisma.progress.findMany({
    where: {
      userId: (session.user as { id?: string }).id!,
      node: { roadmapId },
    },
  });

  const map: Record<string, string> = {};
  for (const p of progress) {
    map[p.nodeId] = p.status;
  }

  return NextResponse.json(map);
}
