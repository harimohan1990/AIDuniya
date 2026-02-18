import Link from "next/link";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Mail, MessageSquare, Github, Linkedin } from "lucide-react";

export default function ContactPage() {
  return (
    <div className="container mx-auto px-4 sm:px-6 py-8 sm:py-12 max-w-2xl">
      <div className="mb-10">
        <h1 className="text-4xl font-bold mb-2 tracking-tight">Contact</h1>
        <p className="text-muted-foreground text-lg">
          Get in touch with the AICourseMap team.
        </p>
      </div>

      <div className="space-y-4">
        <Card className="border-0 shadow-sm card-hover">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Mail className="h-5 w-5 text-primary" />
              <h3 className="font-semibold">Email</h3>
            </div>
          </CardHeader>
          <CardContent>
            <a
              href="mailto:harimohan.info@gmail.com"
              className="text-primary hover:underline"
            >
              harimohan.info@gmail.com
            </a>
            <p className="text-sm text-muted-foreground mt-1">
              For support, feedback, or partnership inquiries.
            </p>
          </CardContent>
        </Card>

        <Card className="border-0 shadow-sm card-hover">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Linkedin className="h-5 w-5 text-primary" />
              <h3 className="font-semibold">LinkedIn</h3>
            </div>
          </CardHeader>
          <CardContent>
            <a
              href="https://www.linkedin.com/in/hari-mohan-47299b54/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              linkedin.com/in/hari-mohan-47299b54
            </a>
            <p className="text-sm text-muted-foreground mt-1">
              Connect with Hari Mohan on LinkedIn.
            </p>
          </CardContent>
        </Card>

        <Card className="border-0 shadow-sm">
          <CardHeader>
            <div className="flex items-center gap-2">
              <MessageSquare className="h-5 w-5 text-primary" />
              <h3 className="font-semibold">Community</h3>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground text-sm">
              Join our community to share progress, ask questions, and connect with other learners.
              (Coming soon)
            </p>
          </CardContent>
        </Card>

        <Card className="border-0 shadow-sm">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Github className="h-5 w-5 text-primary" />
              <h3 className="font-semibold">Open Source</h3>
            </div>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground text-sm">
              AICourseMap is open source. Report issues or contribute on GitHub.
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="mt-8 text-center">
        <Link href="/" className="text-primary hover:underline text-sm">
          ← Back to home
        </Link>
      </div>
    </div>
  );
}
