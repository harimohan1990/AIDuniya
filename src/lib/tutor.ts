/**
 * Rule-based AI Tutor - placeholder implementation.
 * Adapter interface for future OpenAI/Gemini integration.
 * Responses align with actual roadmaps: AI Engineer, Prompt Engineering, RAG Systems, ML Engineer, LLMOps.
 */

const ACTUAL_ROADMAPS = [
  { title: "AI Engineer", slug: "ai-engineer", type: "role" },
  { title: "Prompt Engineering", slug: "prompt-engineering", type: "skill" },
  { title: "RAG Systems", slug: "rag-systems", type: "skill" },
  { title: "ML Engineer", slug: "ml-engineer", type: "role" },
  { title: "LLMOps", slug: "llmops", type: "skill" },
];

function matchKeywords(text: string, keywords: string[]): boolean {
  const lower = text.toLowerCase();
  return keywords.some((k) => lower.includes(k.toLowerCase()));
}

export async function tutorRespond(userMessage: string): Promise<string> {
  const msg = userMessage.toLowerCase().trim();

  // Project queries first (before generic "suggest" matches roadmap)
  if (matchKeywords(msg, ["project", "projects", "build", "portfolio"])) {
    return `Great! Here are project ideas by level:\n\n` +
      `**Beginner:** Sentiment analysis, Image classifier, Chatbot\n` +
      `**Intermediate:** RAG system, Fine-tuned model, MLOps pipeline\n` +
      `**Advanced:** Multi-agent system, Custom LLM fine-tuning, Production RAG\n\n` +
      `Visit [roadmaps](/roadmaps) to pick a path, then check [projects](/projects) for the full list with descriptions and milestones.`;
  }

  // Study plan
  if (matchKeywords(msg, ["study plan", "weekly", "schedule", "plan"])) {
    return `Here's a sample weekly study plan:\n\n` +
      `**Week 1-2:** Fundamentals (Python, Linear Algebra, Statistics)\n` +
      `**Week 3-4:** ML basics (Supervised Learning, scikit-learn)\n` +
      `**Week 5-6:** Deep Learning (Neural Networks, PyTorch/TensorFlow)\n` +
      `**Week 7-8:** NLP or CV specialization\n` +
      `**Week 9+:** Projects and portfolio building\n\n` +
      `Spend 2-3 hours daily. Adjust based on your roadmap. Want a plan for a specific roadmap?`;
  }

  // "What should I learn for AI Engineer job?" and similar
  if (matchKeywords(msg, ["learn", "job", "career", "ai engineer", "ml engineer", "what should"])) {
    const roadmapList = ACTUAL_ROADMAPS.map((r) => `[${r.title}](/roadmaps/${r.slug})`).join(", ");
    return `For an AI Engineer role, I recommend the **[AI Engineer](/roadmaps/ai-engineer)** roadmap. It covers:\n\n` +
      `• Python & programming → Math → ML basics → Deep Learning → LLMs & Modern AI → MLOps\n\n` +
      `Our roadmaps: ${roadmapList}\n\n` +
      `Visit [roadmaps](/roadmaps) to explore. Tell me your experience (beginner/intermediate/advanced) for a tailored path.`;
  }

  // Roadmap recommendations
  if (matchKeywords(msg, ["recommend", "roadmap", "suggest", "which", "what roadmap", "beginner", "intermediate", "advanced"])) {
    const roleRoadmaps = ACTUAL_ROADMAPS.filter((r) => r.type === "role").map((r) => r.title).join(", ");
    const skillRoadmaps = ACTUAL_ROADMAPS.filter((r) => r.type === "skill").map((r) => r.title).join(", ");
    return `Based on your interest, I recommend exploring these roadmaps:\n\n` +
      `**Role-based:** ${roleRoadmaps}\n` +
      `**Skill-based:** ${skillRoadmaps}\n\n` +
      `Visit [roadmaps](/roadmaps) to browse all. Tell me your experience level and goals (job/startup) for a more personalized recommendation.`;
  }

  if (matchKeywords(msg, ["hello", "hi", "hey"])) {
    return `Hello! I'm your AI Tutor. I can help you with roadmap recommendations, weekly study plans, and project suggestions. What would you like to explore?`;
  }

  if (matchKeywords(msg, ["help", "what can you"])) {
    return `I can help you with:\n\n` +
      `1. **Roadmap recommendations** - Tell me your experience and goals\n` +
      `2. **Weekly study plans** - I'll create a structured learning schedule\n` +
      `3. **Project suggestions** - Based on your selected roadmap or interests\n\n` +
      `Just ask in natural language!`;
  }

  return `I can help with:\n\n` +
    `• **Roadmap recommendations** – e.g. "recommend a roadmap for beginners"\n` +
    `• **Weekly study plans** – e.g. "create a weekly study plan"\n` +
    `• **Project suggestions** – e.g. "suggest projects for RAG"\n` +
    `• **Career guidance** – e.g. "what should I learn for an AI Engineer job?"\n\n` +
    `Try one of these, or ask something similar!`;
}
