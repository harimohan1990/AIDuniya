"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Bot, Send, User } from "lucide-react";
import { tutorRespond } from "@/lib/tutor";

interface Message {
  role: "user" | "assistant";
  content: string;
}

/** Renders tutor message with **bold** and [text](href) links */
function MessageContent({ content }: { content: string }) {
  const segments: React.ReactNode[] = [];
  const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
  let lastIndex = 0;
  let match;
  let key = 0;

  while ((match = linkRegex.exec(content)) !== null) {
    const textBefore = content.slice(lastIndex, match.index);
    segments.push(
      <span key={key++}>
        {textBefore.split(/(\*\*[^*]+\*\*)/g).map((part, j) =>
          part.startsWith("**") && part.endsWith("**") ? (
            <strong key={j}>{part.slice(2, -2)}</strong>
          ) : (
            part
          )
        )}
      </span>
    );
    segments.push(
      <Link key={key++} href={match[2]} className="underline hover:opacity-80">
        {match[1]}
      </Link>
    );
    lastIndex = match.index + match[0].length;
  }
  const textAfter = content.slice(lastIndex);
  segments.push(
    <span key={key++}>
      {textAfter.split(/(\*\*[^*]+\*\*)/g).map((part, j) =>
        part.startsWith("**") && part.endsWith("**") ? (
          <strong key={j}>{part.slice(2, -2)}</strong>
        ) : (
          part
        )
      )}
    </span>
  );

  return (
    <div className="whitespace-pre-wrap font-sans text-sm">
      {segments}
    </div>
  );
}

export default function TutorPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Hi! I'm your AI Tutor. I can help you:\n\n• **Recommend a roadmap** – Tell me your experience (beginner/intermediate/advanced) and goals (job/startup)\n• **Generate a study plan** – I'll create a weekly schedule for your chosen path\n• **Suggest projects** – Get project ideas matched to your roadmap or interests\n\nTry: \"Recommend a roadmap for beginners\" or \"Create a weekly study plan for AI Engineer\"",
    },
  ]);

  const suggestedPrompts = [
    "Recommend a roadmap for beginners",
    "Create a weekly study plan",
    "Suggest projects for RAG",
    "What should I learn for an AI Engineer job?",
  ];
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const submitMessage = async (userMessage: string) => {
    if (!userMessage.trim() || loading) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: userMessage }]);
    setLoading(true);
    try {
      const response = await tutorRespond(userMessage);
      setMessages((m) => [...m, { role: "assistant", content: response }]);
    } catch {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: "Sorry, I encountered an error. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    submitMessage(input.trim());
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-3xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">AI Tutor</h1>
        <p className="mt-2 text-muted-foreground max-w-2xl">
          Get personalized recommendations for roadmaps, study plans, and projects. Ask in natural language—no need to know exact commands. The tutor uses rule-based logic to guide you through the platform.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          {suggestedPrompts.map((prompt) => (
            <button
              key={prompt}
              type="button"
              onClick={() => submitMessage(prompt)}
              disabled={loading}
              className="text-sm px-3 py-1.5 rounded-md bg-muted hover:bg-muted/80 transition-colors disabled:opacity-50"
            >
              {prompt}
            </button>
          ))}
        </div>
      </div>

      <Card className="min-h-[500px] flex flex-col">
        <CardHeader className="border-b">
          <div className="flex items-center gap-2">
            <Bot className="h-6 w-6 text-primary" />
            <h2 className="font-semibold">Chat</h2>
          </div>
        </CardHeader>
        <CardContent className="flex-1 flex flex-col p-0">
          <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-[300px]">
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex gap-3 ${
                  msg.role === "user" ? "flex-row-reverse" : ""
                }`}
              >
                <div
                  className={`shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                    msg.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted"
                  }`}
                >
                  {msg.role === "user" ? (
                    <User className="h-4 w-4" />
                  ) : (
                    <Bot className="h-4 w-4" />
                  )}
                </div>
                <div
                  className={`rounded-lg px-4 py-2 max-w-[80%] ${
                    msg.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted"
                  }`}
                >
                  {msg.role === "assistant" ? (
                    <MessageContent content={msg.content} />
                  ) : (
                    <pre className="whitespace-pre-wrap font-sans text-sm">
                      {msg.content}
                    </pre>
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div className="flex gap-3">
                <div className="shrink-0 w-8 h-8 rounded-full bg-muted flex items-center justify-center">
                  <Bot className="h-4 w-4" />
                </div>
                <div className="rounded-lg px-4 py-2 bg-muted">
                  <span className="animate-pulse">Thinking...</span>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
          <form
            onSubmit={handleSubmit}
            className="flex gap-2 p-4 border-t"
          >
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about roadmaps, study plans, or projects..."
              disabled={loading}
              className="flex-1"
            />
            <Button type="submit" disabled={loading || !input.trim()}>
              <Send className="h-4 w-4" />
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
