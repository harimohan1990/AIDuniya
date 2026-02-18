import Link from "next/link";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { GraduationCap, Map, BookOpen, Bot, Target } from "lucide-react";

export default function AboutPage() {
  return (
    <div className="container mx-auto px-4 sm:px-6 py-8 sm:py-12 max-w-3xl">
      <div className="text-center mb-12">
        <GraduationCap className="h-16 w-16 text-primary mx-auto mb-4" />
        <h1 className="text-4xl font-bold">About AICourseMap</h1>
        <p className="mt-4 text-lg text-muted-foreground">
          Your structured path to mastering AI and ML.
        </p>
      </div>

      <div className="prose prose-slate dark:prose-invert max-w-none space-y-6">
        <p>
          AICourseMap is a community-style platform that helps you grow your AI career through
          curated roadmaps, courses, guides, and hands-on projects. Whether you&apos;re aiming to
          become an AI Engineer, ML Engineer, or specialize in GenAI and agents, we provide the
          structure to get there.
        </p>

        <h2 className="text-2xl font-semibold mt-8">What we offer</h2>
        <div className="grid gap-4 not-prose mt-4">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <Map className="h-5 w-5 text-primary" />
                <h3 className="font-semibold">Role & Skill Roadmaps</h3>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground text-sm">
                Step-by-step learning paths for AI Engineer, ML Engineer, MLOps, Data Scientist,
                GenAI Engineer, and skill-based tracks like Prompt Engineering, RAG, and LLMOps.
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <BookOpen className="h-5 w-5 text-primary" />
                <h3 className="font-semibold">Curated Courses</h3>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground text-sm">
                Hand-picked courses from Coursera, DeepLearning.AI, Fast.ai, and more. Filter by
                level, type, and provider.
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <Bot className="h-5 w-5 text-primary" />
                <h3 className="font-semibold">AI Tutor</h3>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground text-sm">
                Get personalized roadmap recommendations, weekly study plans, and project
                suggestions based on your goals.
              </p>
            </CardContent>
          </Card>
        </div>

        <h2 className="text-2xl font-semibold mt-8">Our mission</h2>
        <p>
          We believe everyone should have access to clear, structured paths for learning AI. Our
          goal is to reduce the overwhelm of choosing what to learn next and help you build
          real skills through practice.
        </p>

        <div className="flex gap-4 mt-8">
          <Button asChild>
            <Link href="/roadmaps">Explore Roadmaps</Link>
          </Button>
          <Button variant="outline" asChild>
            <Link href="/courses">Browse Courses</Link>
          </Button>
        </div>
      </div>
    </div>
  );
}
