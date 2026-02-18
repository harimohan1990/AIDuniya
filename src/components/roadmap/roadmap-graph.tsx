"use client";

import { useMemo } from "react";
import { RoadmapNodeCard } from "./roadmap-node-card";
import { ChevronDown } from "lucide-react";

const NODE_WIDTH = 280;
const NODE_HEIGHT = 120;
const HORIZONTAL_GAP = 40;
const VERTICAL_GAP = 60;

interface Node {
  id: string;
  title: string;
  content: string | null;
  difficulty: string | null;
  order: number;
  resources: string | null;
}

interface Edge {
  id: string;
  fromNodeId: string;
  toNodeId: string;
}

interface RoadmapGraphProps {
  roadmapId: string;
  nodes: Node[];
  edges: Edge[];
}

export function RoadmapGraph({ roadmapId, nodes, edges }: RoadmapGraphProps) {
  const sortedNodes = useMemo(() => [...nodes].sort((a, b) => a.order - b.order), [nodes]);

  const graphData = useMemo(() => {
    const nodeMap = new Map(nodes.map((n) => [n.id, { node: n, index: 0 }]));
    const children = new Map<string, string[]>();
    const parents = new Map<string, string[]>();

    for (const e of edges) {
      if (!children.has(e.fromNodeId)) children.set(e.fromNodeId, []);
      children.get(e.fromNodeId)!.push(e.toNodeId);
      if (!parents.has(e.toNodeId)) parents.set(e.toNodeId, []);
      parents.get(e.toNodeId)!.push(e.fromNodeId);
    }

    const roots = nodes.filter((n) => !parents.has(n.id) || parents.get(n.id)!.length === 0);
    const visited = new Set<string>();
    const levels: string[][] = [];
    let queue = roots.map((r) => r.id);

    while (queue.length > 0) {
      const level: string[] = [];
      const nextQueue: string[] = [];
      for (const id of queue) {
        if (visited.has(id)) continue;
        visited.add(id);
        level.push(id);
        const kids = children.get(id) ?? [];
        nextQueue.push(...kids);
      }
      if (level.length > 0) levels.push(level);
      queue = nextQueue;
    }

    const orphanIds = nodes.filter((n) => !visited.has(n.id)).map((n) => n.id);
    if (orphanIds.length > 0) levels.push(orphanIds);

    const positions = new Map<string, { x: number; y: number }>();
    let maxCols = 0;
    for (let row = 0; row < levels.length; row++) {
      const levelNodes = levels[row];
      maxCols = Math.max(maxCols, levelNodes.length);
      for (let col = 0; col < levelNodes.length; col++) {
        const id = levelNodes[col];
        const x = col * (NODE_WIDTH + HORIZONTAL_GAP) + NODE_WIDTH / 2;
        const y = row * (NODE_HEIGHT + VERTICAL_GAP) + NODE_HEIGHT / 2;
        positions.set(id, { x, y });
      }
    }

    const svgPaths: { d: string; fromId: string; toId: string }[] = [];
    for (const e of edges) {
      const from = positions.get(e.fromNodeId);
      const to = positions.get(e.toNodeId);
      if (!from || !to) continue;
      const startX = from.x + NODE_WIDTH / 2;
      const startY = from.y + NODE_HEIGHT / 2;
      const endX = to.x - NODE_WIDTH / 2;
      const endY = to.y;
      const midX = (startX + endX) / 2;
      const d = `M ${startX} ${startY} C ${midX} ${startY}, ${midX} ${endY}, ${endX} ${endY}`;
      svgPaths.push({ d, fromId: e.fromNodeId, toId: e.toNodeId });
    }

    const width = maxCols * (NODE_WIDTH + HORIZONTAL_GAP) + HORIZONTAL_GAP;
    const height = levels.length * (NODE_HEIGHT + VERTICAL_GAP) + VERTICAL_GAP;

    return {
      positions,
      paths: svgPaths,
      width: Math.max(width, 800),
      height: Math.max(height, 400),
    };
  }, [nodes, edges]);

  const { positions, paths, width: graphWidth, height: graphHeight } = graphData;

  const nodeList = useMemo(() => nodes.map((n) => {
    const pos = positions.get(n.id);
    if (!pos) return null;
    return (
      <div
        key={n.id}
        className="absolute"
        style={{
          left: pos.x - NODE_WIDTH / 2,
          top: pos.y - NODE_HEIGHT / 2,
          width: NODE_WIDTH,
          height: NODE_HEIGHT,
        }}
      >
        <RoadmapNodeCard roadmapId={roadmapId} node={n} compact />
      </div>
    );
  }), [nodes, positions, roadmapId]);

  return (
    <>
      {/* Mobile & Tablet: Vertical list */}
      <div className="lg:hidden space-y-0">
        {sortedNodes.map((node, i) => (
          <div key={node.id} className="flex flex-col items-center">
            <div className="w-full max-w-sm sm:max-w-md mx-auto">
              <RoadmapNodeCard roadmapId={roadmapId} node={node} compact />
            </div>
            {i < sortedNodes.length - 1 && (
              <div className="flex flex-col items-center py-2">
                <div className="w-px h-4 bg-border" />
                <ChevronDown className="h-5 w-5 text-muted-foreground" />
                <div className="w-px h-4 bg-border" />
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Desktop & Big screen: Graph layout */}
      <div className="hidden lg:block overflow-auto rounded-lg border bg-muted/20 p-4 xl:p-6 2xl:p-8 min-h-[400px] relative">
        <div className="relative" style={{ minWidth: Math.min(graphWidth, 1200) }}>
          <svg
            className="absolute pointer-events-none"
            style={{
              width: graphWidth,
              height: graphHeight,
              left: 0,
              top: 0,
            }}
          >
            <defs>
              <marker
                id="arrowhead"
                markerWidth="10"
                markerHeight="7"
                refX="9"
                refY="3.5"
                orient="auto"
              >
                <polygon points="0 0, 10 3.5, 0 7" fill="hsl(var(--muted-foreground) / 0.5)" />
              </marker>
            </defs>
            {paths.map((p, i) => (
              <path
                key={i}
                d={p.d}
                fill="none"
                stroke="hsl(var(--muted-foreground) / 0.5)"
                strokeWidth="2"
                markerEnd="url(#arrowhead)"
              />
            ))}
          </svg>
          <div
            className="relative"
            style={{
              width: graphWidth,
              height: graphHeight,
            }}
          >
            {nodeList}
          </div>
        </div>
      </div>
    </>
  );
}
