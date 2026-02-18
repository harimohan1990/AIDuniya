import Link from "next/link";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Check, Sparkles } from "lucide-react";

export default function PricingPage() {
  return (
    <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 max-w-5xl">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold">Pricing</h1>
        <p className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto">
          Start free. Upgrade to Pro for advanced AI Tutor features and custom roadmaps.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
        <Card className="border-2">
          <CardHeader>
            <h2 className="text-2xl font-semibold">Free</h2>
            <p className="text-3xl font-bold mt-2">$0</p>
            <p className="text-muted-foreground text-sm">Forever free</p>
          </CardHeader>
          <CardContent className="space-y-4">
            <ul className="space-y-2 text-sm">
              <li className="flex items-center gap-2">
                <Check className="h-4 w-4 text-primary" />
                Browse all roadmaps, courses, guides
            </li>
              <li className="flex items-center gap-2">
                <Check className="h-4 w-4 text-primary" />
                Save roadmaps & track progress
            </li>
              <li className="flex items-center gap-2">
                <Check className="h-4 w-4 text-primary" />
                Bookmark courses & guides
            </li>
              <li className="flex items-center gap-2">
                <Check className="h-4 w-4 text-primary" />
                Basic AI Tutor (recommendations)
            </li>
            </ul>
            <Button variant="outline" className="w-full" asChild>
              <Link href="/auth/signup">Get started</Link>
            </Button>
          </CardContent>
        </Card>

        <Card className="border-2 border-primary relative">
          <div className="absolute -top-3 left-1/2 -translate-x-1/2">
            <span className="bg-primary text-primary-foreground text-xs font-medium px-3 py-1 rounded-full flex items-center gap-1">
              <Sparkles className="h-3 w-3" />
              Pro
            </span>
          </div>
          <CardHeader className="pt-6">
            <h2 className="text-2xl font-semibold">Pro</h2>
            <p className="text-3xl font-bold mt-2">$9<span className="text-lg font-normal text-muted-foreground">/mo</span></p>
            <p className="text-muted-foreground text-sm">Billed monthly</p>
          </CardHeader>
          <CardContent className="space-y-4">
            <ul className="space-y-2 text-sm">
              <li className="flex items-center gap-2">
                <Check className="h-4 w-4 text-primary" />
                Everything in Free
            </li>
              <li className="flex items-center gap-2">
                <Check className="h-4 w-4 text-primary" />
                Advanced AI Tutor (OpenAI/Gemini)
            </li>
              <li className="flex items-center gap-2">
                <Check className="h-4 w-4 text-primary" />
                Custom roadmaps
            </li>
              <li className="flex items-center gap-2">
                <Check className="h-4 w-4 text-primary" />
                Weekly study plan generation
            </li>
              <li className="flex items-center gap-2">
                <Check className="h-4 w-4 text-primary" />
                Priority support
            </li>
            </ul>
            <Button className="w-full" asChild>
              <Link href="/auth/signup?plan=pro">Upgrade to Pro</Link>
            </Button>
            <p className="text-xs text-muted-foreground text-center">
              Stripe integration coming soon
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
