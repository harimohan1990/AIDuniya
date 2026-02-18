import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

const faqs = [
  {
    q: "What is AICourseMap?",
    a: "AICourseMap is a platform that provides structured roadmaps, curated courses, guides, and project ideas for learning AI and ML. We help you navigate the overwhelming landscape of AI education with clear learning paths.",
  },
  {
    q: "Is it free?",
    a: "Yes! The core platform is free. You can browse roadmaps, courses, guides, save progress, and use the basic AI Tutor. Pro features (advanced AI Tutor, custom roadmaps) are available with a paid plan.",
  },
  {
    q: "How do roadmaps work?",
    a: "Roadmaps are structured learning paths with nodes connected by prerequisites. Each node covers a topic and links to resources. You can mark nodes complete and track your progress.",
  },
  {
    q: "What's the difference between role-based and skill-based roadmaps?",
    a: "Role-based roadmaps (e.g. AI Engineer, ML Engineer) cover the full path to a job role. Skill-based roadmaps (e.g. Prompt Engineering, RAG) focus on a specific skill you want to master.",
  },
  {
    q: "How does the AI Tutor work?",
    a: "The AI Tutor is a chat assistant that can recommend roadmaps based on your profile, generate weekly study plans, and suggest projects. The free version uses rule-based responses; Pro uses advanced AI models.",
  },
  {
    q: "Can I contribute courses or roadmaps?",
    a: "Admin users can add and edit content via the admin dashboard. Community contributions are on our roadmap.",
  },
];

export default function FAQPage() {
  return (
    <div className="container mx-auto px-4 sm:px-6 py-8 sm:py-12 max-w-3xl">
      <h1 className="text-4xl font-bold mb-2">FAQ</h1>
      <p className="text-muted-foreground mb-8">
        Frequently asked questions about AICourseMap.
      </p>

      <Accordion type="single" collapsible className="w-full">
        {faqs.map((faq, i) => (
          <AccordionItem key={i} value={`item-${i}`}>
            <AccordionTrigger>{faq.q}</AccordionTrigger>
            <AccordionContent>{faq.a}</AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>
    </div>
  );
}
