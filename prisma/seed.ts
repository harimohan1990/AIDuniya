import { PrismaClient } from "@prisma/client";
import bcrypt from "bcryptjs";

const prisma = new PrismaClient();

async function main() {
  // Clean seed data (keep users)
  await prisma.progress.deleteMany();
  await prisma.bookmark.deleteMany();
  await prisma.savedRoadmap.deleteMany();
  await prisma.roadmapEdge.deleteMany();
  await prisma.roadmapNode.deleteMany();
  await prisma.roadmap.deleteMany();
  await prisma.course.deleteMany();
  await prisma.guide.deleteMany();
  await prisma.projectIdea.deleteMany();

  const hashedPassword = await bcrypt.hash("admin123", 12);

  const admin = await prisma.user.upsert({
    where: { email: "admin@aicoursemap.com" },
    update: {},
    create: {
      email: "admin@aicoursemap.com",
      name: "Admin",
      password: hashedPassword,
      role: "ADMIN",
    },
  });

  console.log("Created admin:", admin.email);

  // Roadmaps
  const roadmap1 = await prisma.roadmap.upsert({
    where: { slug: "ai-engineer" },
    update: {},
    create: {
      slug: "ai-engineer",
      title: "AI Engineer",
      type: "ROLE",
      description: "A complete journey from zero to job-ready AI Engineer. Master Python, math, ML, deep learning, LLMs, and production deployment.",
      featured: true,
      order: 0,
    },
  });

  const roadmap2 = await prisma.roadmap.upsert({
    where: { slug: "prompt-engineering" },
    update: {},
    create: {
      slug: "prompt-engineering",
      title: "Prompt Engineering",
      type: "SKILL",
      description: "Master the art of crafting effective prompts for LLMs.",
      featured: true,
      order: 1,
    },
  });

  const roadmap3 = await prisma.roadmap.upsert({
    where: { slug: "rag-systems" },
    update: {},
    create: {
      slug: "rag-systems",
      title: "RAG Systems",
      type: "SKILL",
      description: "Build Retrieval-Augmented Generation systems for production.",
      featured: true,
      order: 2,
    },
  });

  const r1n1 = await prisma.roadmapNode.create({
    data: {
      roadmapId: roadmap1.id,
      title: "Python & Programming",
      content: "Your foundation for everything in AI. Every ML library—NumPy, PyTorch, scikit-learn—is built on Python. Without solid programming skills, you'll struggle to debug, experiment, and ship.\n\n**What you'll learn:**\n• Python syntax: variables, loops, conditionals, functions\n• Data structures: lists, dicts, sets, tuples—when to use each\n• Object-oriented programming: classes, inheritance, encapsulation\n• NumPy: arrays, broadcasting, indexing, vectorized operations\n• File I/O, error handling, and debugging with print/pdb\n\n**Week-by-week syllabus:**\n• Week 1–2: Python basics, data types, control flow, functions\n• Week 3: Data structures, list comprehensions, dict operations\n• Week 4: OOP, classes, modules, packages\n• Week 5: NumPy arrays, broadcasting, indexing, vectorization\n• Week 6: pandas basics, CSV/JSON I/O, data cleaning\n\n**Learning objectives:**\n• Write clean, readable Python code\n• Manipulate data with NumPy (no loops for array ops)\n• Solve LeetCode easy problems in under 30 min\n\n**Build:** A script that loads a CSV, cleans it with pandas, and computes basic statistics. Estimated time: 4–6 weeks.\n\n**Industry project:** Data pipeline for e-commerce analytics—ingest sales data, clean, aggregate by category, and generate reports.",
      difficulty: "beginner",
      order: 0,
      resources: JSON.stringify([
        { type: "course", title: "Python for Everybody", url: "https://www.coursera.org/specializations/python" },
        { type: "course", title: "CS50's Introduction to Programming with Python", url: "https://www.edx.org/learn/python/harvard-university-cs50-s-introduction-to-programming-with-python" },
        { type: "article", title: "NumPy Quickstart Tutorial", url: "https://numpy.org/doc/stable/user/quickstart.html" },
        { type: "video", title: "Python for Data Science (freeCodeCamp)", url: "https://www.youtube.com/watch?v=LHBE6Q9XlzI" },
      ]),
    },
  });

  const r1n2 = await prisma.roadmapNode.create({
    data: {
      roadmapId: roadmap1.id,
      title: "Math Foundations",
      content: "You don't need a PhD, but you need intuition. When you see 'embedding' or 'attention matrix,' you should picture vectors and dot products—not black boxes.\n\n**Linear algebra:**\n• Vectors: magnitude, direction, dot product (similarity)\n• Matrices: multiplication, transpose, eigenvalues (optional)\n• Why embeddings are vectors; why attention uses matrix multiplication\n\n**Calculus:**\n• Derivatives: rate of change\n• Chain rule: how backpropagation works\n• Gradients: direction of steepest ascent\n\n**Probability & statistics:**\n• Distributions: normal, Bernoulli, multinomial\n• Bayes' theorem: foundation of Naive Bayes, Bayesian ML\n• Metrics: mean, variance, correlation, p-values\n\n**Week-by-week syllabus:**\n• Week 1–2: Linear algebra—vectors, matrices, dot products, norms\n• Week 3: Calculus—derivatives, chain rule, partial derivatives\n• Week 4: Probability—distributions, Bayes, expectation\n• Week 5–6: Statistics—hypothesis testing, train/test, cross-validation\n• Week 7–8: Apply to ML—loss functions, gradients, bias-variance\n\n**Learning objectives:**\n• Interpret a loss curve (why it goes down)\n• Explain what a gradient is\n• Know when to use train/test split vs cross-validation\n\nEstimated time: 4–8 weeks (can overlap with Python).\n\n**Industry project:** Build a simple A/B test analyzer—compute confidence intervals, p-values, and effect size for conversion experiments.",
      difficulty: "beginner",
      order: 1,
      resources: JSON.stringify([
        { type: "course", title: "Khan Academy Linear Algebra", url: "https://www.khanacademy.org/math/linear-algebra" },
        { type: "course", title: "3Blue1Brown Essence of Linear Algebra", url: "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab" },
        { type: "course", title: "Mathematics for Machine Learning (Coursera)", url: "https://www.coursera.org/specializations/mathematics-machine-learning" },
        { type: "article", title: "The Matrix Calculus You Need For Deep Learning", url: "https://arxiv.org/abs/1802.01528" },
      ]),
    },
  });

  const r1n3 = await prisma.roadmapNode.create({
    data: {
      roadmapId: roadmap1.id,
      title: "Machine Learning Basics",
      content: "This is where AI becomes tangible. You'll train models that predict house prices, classify spam, and recommend products. Every advanced topic (deep learning, LLMs) builds on these ideas.\n\n**Supervised learning:**\n• Regression: predicting numbers (price, temperature)\n• Classification: predicting categories (spam/not, cat/dog)\n• Key algorithms: linear regression, logistic regression, decision trees, k-NN, SVM\n\n**Model evaluation:**\n• Train/test split, cross-validation\n• Metrics: MSE, MAE, accuracy, precision, recall, F1, ROC-AUC\n• Overfitting vs underfitting; bias-variance tradeoff\n\n**Feature engineering:**\n• Encoding categorical variables (one-hot, label)\n• Scaling (StandardScaler, MinMaxScaler)\n• Handling missing values, outliers\n\n**Week-by-week syllabus:**\n• Week 1–2: Linear/logistic regression, loss functions, metrics\n• Week 3: Decision trees, random forests, XGBoost\n• Week 4: Classification metrics, confusion matrix, ROC-AUC\n• Week 5: Feature engineering, scaling, encoding\n• Week 6: Cross-validation, hyperparameter tuning (GridSearchCV)\n• Week 7–8: End-to-end pipeline, model selection, deployment prep\n\n**Learning objectives:**\n• Train a model with scikit-learn in 10 lines\n• Interpret confusion matrix and classification report\n• Use GridSearchCV for hyperparameter tuning\n\n**Build:** Predict housing prices (regression) and classify Iris flowers (classification). Estimated time: 6–8 weeks.\n\n**Industry project:** Customer churn prediction—predict which users will cancel. Build feature pipeline, train classifier, deploy as API with business metrics.",
      difficulty: "intermediate",
      order: 2,
      resources: JSON.stringify([
        { type: "course", title: "Machine Learning by Andrew Ng", url: "https://www.coursera.org/learn/machine-learning" },
        { type: "course", title: "Hands-On ML with Scikit-Learn (book)", url: "https://www.oreilly.com/library/view/hands-on-machine-learning/1098960571/" },
        { type: "article", title: "Scikit-learn User Guide", url: "https://scikit-learn.org/stable/user_guide.html" },
        { type: "video", title: "Machine Learning Course (StatQuest)", url: "https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF" },
      ]),
    },
  });

  const r1n4 = await prisma.roadmapNode.create({
    data: {
      roadmapId: roadmap1.id,
      title: "Deep Learning",
      content: "Neural networks power image recognition, speech, and language models. Understanding how they work—not just calling APIs—is what separates engineers from users.\n\n**Neural network basics:**\n• Perceptron, multilayer perceptron (MLP)\n• Forward pass: input → hidden layers → output\n• Backpropagation: how gradients flow backward\n• Activation functions: ReLU, sigmoid, softmax\n• Loss functions: MSE, cross-entropy\n\n**Frameworks:**\n• PyTorch (recommended): dynamic graphs, research standard\n• TensorFlow: production, Keras for simplicity\n• Build a model from scratch with nn.Module\n\n**Architectures:**\n• CNNs: convolutions, pooling, for images (ResNet, VGG)\n• RNNs/LSTMs: sequences (optional, transformers replaced most)\n• Transformers: self-attention, the architecture behind GPT and BERT\n\n**Week-by-week syllabus:**\n• Week 1–2: MLP basics, PyTorch nn.Module, training loop\n• Week 3–4: CNNs, convolutions, pooling, image classification\n• Week 5–6: Transfer learning, fine-tuning ResNet/VGG\n• Week 7–8: NLP basics, tokenization, embeddings\n• Week 9–10: Transformers, BERT, fine-tuning for classification\n• Week 11–12: Production deployment, ONNX, inference optimization\n\n**Learning objectives:**\n• Implement a simple MLP in PyTorch\n• Train on GPU, use DataLoader\n• Fine-tune a pretrained model (e.g., ResNet)\n\n**Build:** Image classifier (CIFAR-10) and sentiment analyzer (BERT/finetune). Estimated time: 8–12 weeks.\n\n**Industry project:** Visual defect detection for manufacturing—train CNN to classify product images as pass/fail. Deploy to edge or cloud.",
      difficulty: "intermediate",
      order: 3,
      resources: JSON.stringify([
        { type: "course", title: "Deep Learning Specialization (Andrew Ng)", url: "https://www.coursera.org/specializations/deep-learning" },
        { type: "course", title: "Fast.ai Practical Deep Learning", url: "https://course.fast.ai/" },
        { type: "course", title: "PyTorch for Deep Learning (YouTube)", url: "https://www.youtube.com/playlist?list=PLZbbT5o_s2xrfNyHZsM6ufI0iZENK9xgG" },
        { type: "article", title: "Attention Is All You Need (Transformer paper)", url: "https://arxiv.org/abs/1706.03762" },
        { type: "video", title: "3Blue1Brown Neural Networks", url: "https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi" },
      ]),
    },
  });

  const r1n5 = await prisma.roadmapNode.create({
    data: {
      roadmapId: roadmap1.id,
      title: "LLMs & Modern AI",
      content: "LLMs have created a new category of AI engineering. Most roles today involve building applications on top of GPT, Claude, or open-source models—not training them from scratch.\n\n**Prompt engineering:**\n• Zero-shot, few-shot, chain-of-thought\n• System prompts, role-playing, output formatting\n• When prompts work vs when you need RAG/fine-tuning\n\n**RAG (Retrieval-Augmented Generation):**\n• Embeddings: turn text into vectors\n• Vector databases: Pinecone, Chroma, pgvector\n• Chunking strategies, retrieval, reranking\n• LangChain, LlamaIndex for orchestration\n\n**Fine-tuning:**\n• When to fine-tune vs use RAG\n• LoRA, QLoRA for efficient tuning\n• Datasets, training loops, evaluation\n\n**Agents:**\n• Tools and function calling\n• ReAct, plan-and-execute patterns\n• Building multi-step reasoning systems\n\n**Week-by-week syllabus:**\n• Week 1–2: Prompt patterns, API usage, streaming responses\n• Week 3: Embeddings, vector stores, similarity search\n• Week 4–5: RAG pipeline—chunking, retrieval, generation\n• Week 6: Fine-tuning with LoRA, dataset prep, evaluation\n• Week 7: Agents with tools, function calling, ReAct\n• Week 8: Production deployment, caching, cost optimization\n\n**Learning objectives:**\n• Build a RAG app that answers questions over your docs\n• Create an agent that uses search + calculator\n• Fine-tune a small model for a custom task\n\n**Build:** Document Q&A chatbot, coding assistant with tools, custom classifier via fine-tuning. Estimated time: 6–8 weeks.\n\n**Industry project:** Enterprise knowledge base—RAG over internal docs, FAQs, and policies. Deploy with auth, audit logging, and feedback loop.",
      difficulty: "intermediate",
      order: 4,
      resources: JSON.stringify([
        { type: "course", title: "LLM University (Cohere)", url: "https://cohere.com/llmu" },
        { type: "course", title: "Building Systems with ChatGPT API", url: "https://www.coursera.org/learn/build-chatgpt-api-systems" },
        { type: "course", title: "LangChain for LLM Application Development", url: "https://www.coursera.org/learn/langchain" },
        { type: "article", title: "Building RAG Applications (DeepLearning.AI)", url: "https://www.deeplearning.ai/short-courses/building-applications-with-vector-databases/" },
        { type: "video", title: "Prompt Engineering for Developers", url: "https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/" },
      ]),
    },
  });

  const r1n6 = await prisma.roadmapNode.create({
    data: {
      roadmapId: roadmap1.id,
      title: "MLOps & Production",
      content: "A model in a Jupyter notebook isn't a product. Production ML requires reproducibility, monitoring, and the ability to iterate safely.\n\n**Experiment tracking:**\n• MLflow, Weights & Biases: log params, metrics, artifacts\n• Why you need it: compare runs, reproduce results\n• Model registry: version models, promote to staging/prod\n\n**Reproducibility:**\n• Pin dependencies (requirements.txt, Poetry)\n• Docker: containerize your app and model\n• Data versioning: DVC, when it matters\n\n**Deployment:**\n• FastAPI: serve models as REST APIs\n• Batch vs real-time inference\n• Cloud: AWS SageMaker, GCP Vertex, Hugging Face Inference\n\n**CI/CD for ML:**\n• GitHub Actions: run tests, deploy on merge\n• Model validation: accuracy, latency, drift checks\n• Canary deployments, rollback strategies\n\n**Week-by-week syllabus:**\n• Week 1: MLflow setup, experiment tracking, model registry\n• Week 2: Docker for ML, FastAPI inference endpoints\n• Week 3: CI/CD with GitHub Actions, automated testing\n• Week 4: Monitoring—latency, errors, data drift\n• Week 5: A/B testing, canary deployment, rollback\n• Week 6: Cost optimization, scaling, documentation\n\n**Learning objectives:**\n• Deploy a model with FastAPI + Docker\n• Set up MLflow for experiment tracking\n• Have one project live on the internet\n\n**Build:** Deploy your RAG chatbot or image classifier. Add monitoring. Put it on your resume. Estimated time: 4–6 weeks.\n\n**Industry project:** End-to-end ML platform—ingest data, train models, deploy to staging/prod, monitor drift, retrain on schedule. Full CI/CD pipeline.",
      difficulty: "advanced",
      order: 5,
      resources: JSON.stringify([
        { type: "course", title: "MLOps Fundamentals (Google)", url: "https://www.coursera.org/learn/mlops-fundamentals" },
        { type: "course", title: "Machine Learning Engineering for Production", url: "https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops" },
        { type: "article", title: "MLflow Documentation", url: "https://mlflow.org/docs/latest/index.html" },
        { type: "video", title: "Full Stack ML (YouTube)", url: "https://www.youtube.com/playlist?list=PL8motc6AQftk1Bs42EW45kwqbyJvPTK1i" },
      ]),
    },
  });

  await prisma.roadmapEdge.createMany({
    data: [
      { roadmapId: roadmap1.id, fromNodeId: r1n1.id, toNodeId: r1n2.id },
      { roadmapId: roadmap1.id, fromNodeId: r1n2.id, toNodeId: r1n3.id },
      { roadmapId: roadmap1.id, fromNodeId: r1n3.id, toNodeId: r1n4.id },
      { roadmapId: roadmap1.id, fromNodeId: r1n4.id, toNodeId: r1n5.id },
      { roadmapId: roadmap1.id, fromNodeId: r1n5.id, toNodeId: r1n6.id },
    ],
    skipDuplicates: true,
  });

  const r2n1 = await prisma.roadmapNode.create({
    data: {
      roadmapId: roadmap2.id,
      title: "LLM Basics",
      content: "Before crafting prompts, understand what you're working with. LLMs are autoregressive models trained on vast text—they predict the next token. They have strengths (fluency, reasoning, following instructions) and limitations (hallucination, context limits, cost).\n\n**What you'll learn:** How tokenization works, context windows, temperature and sampling, common model families (GPT, Claude, Llama), and when to use API vs local models.\n\n**Week-by-week syllabus:**\n• Week 1: Tokenization, context windows, model families\n• Week 2: Temperature, top-p, sampling strategies\n• Week 3: API vs local models, cost and latency tradeoffs\n\n**Industry project:** Internal tool for summarizing meeting notes—extract action items, decisions, and owners from raw transcripts.",
      difficulty: "beginner",
      order: 0,
      resources: JSON.stringify([{ type: "course", title: "LLM University", url: "https://cohere.com/llmu" }, { type: "article", title: "What is a Large Language Model?", url: "https://www.ibm.com/topics/large-language-models" }]),
    },
  });

  const r2n2 = await prisma.roadmapNode.create({
    data: {
      roadmapId: roadmap2.id,
      title: "Prompt Patterns",
      content: "Effective prompts are structured. Learn the patterns that work across models.\n\n**Zero-shot:** Ask directly; good for simple tasks.\n**Few-shot:** Give 2–5 examples; improves consistency.\n**Chain-of-thought:** Ask the model to \"think step by step\"; boosts reasoning.\n**Role-playing:** \"You are an expert...\" sets tone and expertise.\n**Output format:** Specify JSON, markdown, or structure to parse reliably.\n\n**Week-by-week syllabus:**\n• Week 1: Zero-shot vs few-shot, when to use each\n• Week 2: Chain-of-thought, role-playing, output formatting\n• Week 3: Iteration, testing, prompt versioning\n\n**Build:** A prompt that extracts structured data from unstructured text. Estimated time: 2–3 weeks.\n\n**Industry project:** Customer support ticket classifier—route tickets to correct team using few-shot prompts. Integrate with ticketing system.",
      difficulty: "intermediate",
      order: 1,
      resources: JSON.stringify([{ type: "video", title: "Prompt Engineering for Developers", url: "https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/" }, { type: "article", title: "OpenAI Prompt Engineering Guide", url: "https://platform.openai.com/docs/guides/prompt-engineering" }]),
    },
  });

  const r2n3 = await prisma.roadmapNode.create({
    data: {
      roadmapId: roadmap2.id,
      title: "Advanced Prompting",
      content: "Take prompts to production. Handle edge cases, reduce cost, and improve reliability.\n\n**Structured outputs:** JSON mode, function calling, constrained generation\n**Prompt optimization:** Shorter prompts, caching, batching\n**Evaluation:** Automated evals for consistency and quality\n**Multi-turn:** Conversation design, context management, memory\n\n**Week-by-week syllabus:**\n• Week 1: JSON mode, structured extraction at scale\n• Week 2: Prompt caching, cost optimization, latency\n• Week 3: Evals—relevance, faithfulness, consistency\n• Week 4: Multi-turn flows, conversation design\n\n**Industry project:** Legal contract clause extractor—extract parties, dates, obligations into structured JSON. Handle variations across contract types.",
      difficulty: "intermediate",
      order: 2,
      resources: JSON.stringify([{ type: "article", title: "Structured Outputs (OpenAI)", url: "https://platform.openai.com/docs/guides/structured-outputs" }, { type: "article", title: "Prompt Caching", url: "https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching" }]),
    },
  });

  await prisma.roadmapEdge.createMany({
    data: [
      { roadmapId: roadmap2.id, fromNodeId: r2n1.id, toNodeId: r2n2.id },
      { roadmapId: roadmap2.id, fromNodeId: r2n2.id, toNodeId: r2n3.id },
    ],
    skipDuplicates: true,
  });

  const r3n1 = await prisma.roadmapNode.create({
    data: {
      roadmapId: roadmap3.id,
      title: "Embeddings",
      content: "Embeddings turn text into dense vectors that capture meaning. Similar concepts cluster together in vector space. This is the foundation of semantic search, RAG, and recommendation systems.\n\n**What you'll learn:** How embedding models work (e.g., OpenAI ada, sentence-transformers), choosing dimensions, when to use dense vs sparse. Batch encoding, caching, and cost optimization.\n\n**Week-by-week syllabus:**\n• Week 1: Embedding models, dimensions, similarity metrics\n• Week 2: Batch encoding, caching, cost optimization\n\n**Build:** A script that embeds a document collection and finds the most similar passage for a query. Estimated time: 1–2 weeks.\n\n**Industry project:** Semantic product search—embed product catalog, enable natural language search (e.g., \"comfortable running shoes under $100\").",
      difficulty: "intermediate",
      order: 0,
      resources: JSON.stringify([{ type: "course", title: "Building RAG with Vector DBs", url: "https://www.deeplearning.ai/short-courses/building-applications-with-vector-databases/" }, { type: "article", title: "Embeddings Guide (OpenAI)", url: "https://platform.openai.com/docs/guides/embeddings" }]),
    },
  });

  const r3n2 = await prisma.roadmapNode.create({
    data: {
      roadmapId: roadmap3.id,
      title: "Vector Databases",
      content: "Vector DBs store embeddings and support fast similarity search. You'll need one for any RAG system.\n\n**Options:** Pinecone (managed), Chroma (local/dev), Weaviate (open-source), pgvector (PostgreSQL extension).\n\n**Key concepts:** Index types (flat, IVF, HNSW), chunking strategies (size, overlap, semantic), metadata filtering, hybrid search (keyword + vector).\n\n**Week-by-week syllabus:**\n• Week 1: Chunking strategies—size, overlap, semantic splitting\n• Week 2: Vector DB setup, indexing, similarity search\n• Week 3: Hybrid search, reranking, metadata filters\n\n**Build:** A RAG pipeline: ingest docs → chunk → embed → store → retrieve → generate. Answer questions over your own documents. Estimated time: 2–3 weeks.\n\n**Industry project:** Internal documentation Q&A—RAG over company wikis, runbooks, and policies. Add auth and usage analytics.",
      difficulty: "intermediate",
      order: 1,
      resources: JSON.stringify([{ type: "course", title: "LangChain for LLM Apps", url: "https://www.coursera.org/learn/langchain" }, { type: "article", title: "Chroma Documentation", url: "https://docs.trychroma.com/" }]),
    },
  });

  const r3n3 = await prisma.roadmapNode.create({
    data: {
      roadmapId: roadmap3.id,
      title: "Production RAG",
      content: "Scale RAG for real users. Optimize latency, improve retrieval quality, and handle updates.\n\n**Retrieval optimization:** Reranking (Cohere, cross-encoder), query expansion, multi-query\n**Latency:** Caching, async indexing, batch inference\n**Evaluation:** Relevance, faithfulness, answer correctness\n**Updates:** Incremental indexing, re-indexing strategies\n\n**Week-by-week syllabus:**\n• Week 1: Reranking, query expansion, retrieval quality\n• Week 2: Caching, latency optimization, cost control\n• Week 3: Evals, A/B testing RAG variants\n• Week 4: Incremental updates, re-indexing pipelines\n\n**Industry project:** Healthcare FAQ bot—RAG over medical guidelines and policies. Ensure citations, handle sensitive data, add disclaimer.",
      difficulty: "advanced",
      order: 2,
      resources: JSON.stringify([{ type: "article", title: "Advanced RAG (LangChain)", url: "https://python.langchain.com/docs/use_cases/question_answering/" }, { type: "article", title: "RAG Evaluation", url: "https://www.anthropic.com/research/evals" }]),
    },
  });

  await prisma.roadmapEdge.createMany({
    data: [
      { roadmapId: roadmap3.id, fromNodeId: r3n1.id, toNodeId: r3n2.id },
      { roadmapId: roadmap3.id, fromNodeId: r3n2.id, toNodeId: r3n3.id },
    ],
    skipDuplicates: true,
  });

  const roadmap4 = await prisma.roadmap.upsert({
    where: { slug: "ml-engineer" },
    update: {},
    create: {
      slug: "ml-engineer",
      title: "ML Engineer",
      type: "ROLE",
      description: "Become an ML Engineer focused on building and deploying machine learning systems at scale.",
      featured: true,
      order: 3,
    },
  });

  const r4n1 = await prisma.roadmapNode.create({
    data: {
      roadmapId: roadmap4.id,
      title: "ML Foundations",
      content: "Core ML: supervised/unsupervised learning, model evaluation, feature engineering. Hands-on with scikit-learn.\n\n**Week-by-week syllabus:**\n• Week 1–2: Regression, classification, scikit-learn pipelines\n• Week 3: Feature engineering, scaling, encoding\n• Week 4: Model evaluation, cross-validation, hyperparameter tuning\n\n**Industry project:** Fraud detection—build classifier for transaction fraud. Handle imbalanced data, deploy with real-time inference.",
      difficulty: "intermediate",
      order: 0,
      resources: JSON.stringify([{ type: "course", title: "ML by Andrew Ng", url: "https://www.coursera.org/learn/machine-learning" }]),
    },
  });
  const r4n2 = await prisma.roadmapNode.create({
    data: {
      roadmapId: roadmap4.id,
      title: "Deep Learning",
      content: "Neural networks, PyTorch/TensorFlow, CNNs, transformers. Build and train models from scratch.\n\n**Week-by-week syllabus:**\n• Week 1–2: PyTorch basics, training loops, GPU\n• Week 3–4: CNNs, transfer learning, image tasks\n• Week 5–6: Transformers, fine-tuning for NLP\n\n**Industry project:** Demand forecasting—LSTM or transformer for time-series. Integrate with inventory system.",
      difficulty: "intermediate",
      order: 1,
      resources: JSON.stringify([{ type: "course", title: "Fast.ai Practical DL", url: "https://course.fast.ai/" }]),
    },
  });
  const r4n3 = await prisma.roadmapNode.create({
    data: {
      roadmapId: roadmap4.id,
      title: "MLOps Basics",
      content: "Experiment tracking (MLflow, W&B), model versioning, reproducibility with Docker.\n\n**Week-by-week syllabus:**\n• Week 1: MLflow, experiment tracking, model registry\n• Week 2: Docker for ML, dependency management\n• Week 3: Reproducible training pipelines\n\n**Industry project:** Recommendation system—collaborative filtering or two-tower model. A/B test in production.",
      difficulty: "intermediate",
      order: 2,
      resources: JSON.stringify([{ type: "course", title: "MLOps Fundamentals", url: "https://www.coursera.org/learn/mlops-fundamentals" }]),
    },
  });
  const r4n4 = await prisma.roadmapNode.create({
    data: {
      roadmapId: roadmap4.id,
      title: "Deployment & Serving",
      content: "REST APIs with FastAPI, batch vs real-time inference, cloud deployment (AWS, GCP).\n\n**Week-by-week syllabus:**\n• Week 1: FastAPI inference endpoints, batching\n• Week 2: Cloud deployment (SageMaker, Vertex)\n• Week 3: Load testing, scaling, cost optimization\n\n**Industry project:** Real-time ad CTR predictor—low-latency inference, feature store integration.",
      difficulty: "advanced",
      order: 3,
      resources: JSON.stringify([{ type: "article", title: "Deploying ML Models", url: "https://mlflow.org/docs/latest/models.html" }]),
    },
  });
  const r4n5 = await prisma.roadmapNode.create({
    data: {
      roadmapId: roadmap4.id,
      title: "Monitoring & Iteration",
      content: "Data drift, model performance monitoring, A/B testing, retraining pipelines.\n\n**Week-by-week syllabus:**\n• Week 1: Drift detection, performance dashboards\n• Week 2: Retraining triggers, automated pipelines\n• Week 3: A/B testing, rollback, incident response\n\n**Industry project:** Credit scoring model—monitor drift, retrain quarterly, comply with explainability requirements.",
      difficulty: "advanced",
      order: 4,
      resources: JSON.stringify([{ type: "course", title: "ML Engineering for Production", url: "https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops" }]),
    },
  });

  await prisma.roadmapEdge.createMany({
    data: [
      { roadmapId: roadmap4.id, fromNodeId: r4n1.id, toNodeId: r4n2.id },
      { roadmapId: roadmap4.id, fromNodeId: r4n2.id, toNodeId: r4n3.id },
      { roadmapId: roadmap4.id, fromNodeId: r4n3.id, toNodeId: r4n4.id },
      { roadmapId: roadmap4.id, fromNodeId: r4n4.id, toNodeId: r4n5.id },
    ],
    skipDuplicates: true,
  });

  const roadmap5 = await prisma.roadmap.upsert({
    where: { slug: "llmops" },
    update: {},
    create: {
      slug: "llmops",
      title: "LLMOps",
      type: "SKILL",
      description: "Operate and productionize Large Language Model applications with monitoring and evaluation.",
      featured: true,
      order: 4,
    },
  });

  const r5n1 = await prisma.roadmapNode.create({
    data: {
      roadmapId: roadmap5.id,
      title: "LLM Application Patterns",
      content: "Prompt engineering, RAG, agents. Building apps with OpenAI, Anthropic, open-source models.\n\n**Week-by-week syllabus:**\n• Week 1: API usage, streaming, error handling\n• Week 2: RAG patterns, retrieval, generation\n• Week 3: Agents, tools, function calling\n\n**Industry project:** Customer support triage bot—classify intent, route, suggest responses. Integrate with Zendesk/Intercom.",
      difficulty: "intermediate",
      order: 0,
      resources: JSON.stringify([{ type: "course", title: "Building Systems with ChatGPT API", url: "https://www.coursera.org/learn/build-chatgpt-api-systems" }]),
    },
  });
  const r5n2 = await prisma.roadmapNode.create({
    data: {
      roadmapId: roadmap5.id,
      title: "Prompt & Model Versioning",
      content: "Version prompts, A/B test prompts, track which model versions are in production.\n\n**Week-by-week syllabus:**\n• Week 1: Prompt registry, versioning, rollback\n• Week 2: A/B testing prompts and models\n• Week 3: Feature flags, gradual rollout\n\n**Industry project:** Multi-model routing—route by latency/cost/quality. Fallback chains, shadow mode.",
      difficulty: "intermediate",
      order: 1,
      resources: JSON.stringify([{ type: "article", title: "Prompt Versioning", url: "https://www.anthropic.com/news/prompt-engineering" }]),
    },
  });
  const r5n3 = await prisma.roadmapNode.create({
    data: {
      roadmapId: roadmap5.id,
      title: "Evaluation & Monitoring",
      content: "LLM evals: relevance, faithfulness, toxicity. Latency, cost, error rate monitoring.\n\n**Week-by-week syllabus:**\n• Week 1: Evals—relevance, faithfulness, toxicity\n• Week 2: Latency, cost, token tracking\n• Week 3: Dashboards, alerts, incident response\n\n**Industry project:** LLM observability platform—log requests, run evals, track drift. Slack alerts on degradation.",
      difficulty: "advanced",
      order: 2,
      resources: JSON.stringify([{ type: "article", title: "Evaluating LLMs", url: "https://www.deeplearning.ai/the-batch/evaluating-llm-applications/" }]),
    },
  });
  const r5n4 = await prisma.roadmapNode.create({
    data: {
      roadmapId: roadmap5.id,
      title: "Production Deployment",
      content: "Caching, batching, fallbacks. Deploying RAG and agent systems at scale.\n\n**Week-by-week syllabus:**\n• Week 1: Response caching, prompt caching\n• Week 2: Batching, async inference, queues\n• Week 3: Fallbacks, multi-region, SLA\n\n**Industry project:** Enterprise chatbot—auth, audit logs, rate limits, PII redaction. Deploy to Kubernetes.",
      difficulty: "advanced",
      order: 3,
      resources: JSON.stringify([{ type: "course", title: "Production RAG", url: "https://www.deeplearning.ai/short-courses/" }]),
    },
  });

  await prisma.roadmapEdge.createMany({
    data: [
      { roadmapId: roadmap5.id, fromNodeId: r5n1.id, toNodeId: r5n2.id },
      { roadmapId: roadmap5.id, fromNodeId: r5n2.id, toNodeId: r5n3.id },
      { roadmapId: roadmap5.id, fromNodeId: r5n3.id, toNodeId: r5n4.id },
    ],
    skipDuplicates: true,
  });

  // Courses
  const courses = [
    { title: "Machine Learning", provider: "Coursera", url: "https://www.coursera.org/learn/machine-learning", level: "BEGINNER" as const, type: "FULL" as const, duration: "60 hours", tags: ["ML", "supervised learning"], featured: true },
    { title: "Deep Learning Specialization", provider: "Coursera", url: "https://www.coursera.org/specializations/deep-learning", level: "INTERMEDIATE" as const, type: "FULL" as const, duration: "4 months", tags: ["deep learning", "neural networks"], featured: true },
    { title: "Fast.ai Practical Deep Learning", provider: "Fast.ai", url: "https://course.fast.ai/", level: "BEGINNER" as const, type: "FULL" as const, duration: "8 weeks", tags: ["deep learning", "pytorch"], featured: true },
    { title: "Stanford CS231n", provider: "Stanford", url: "https://cs231n.stanford.edu/", level: "ADVANCED" as const, type: "FULL" as const, duration: "10 weeks", tags: ["computer vision", "CNNs"] },
    { title: "LLM University", provider: "Cohere", url: "https://cohere.com/llmu", level: "BEGINNER" as const, type: "SHORT" as const, duration: "5 hours", tags: ["LLM", "NLP"] },
    { title: "LangChain for LLM Application Development", provider: "DeepLearning.AI", url: "https://www.coursera.org/learn/langchain", level: "INTERMEDIATE" as const, type: "SHORT" as const, duration: "1 week", tags: ["LangChain", "LLM", "RAG"] },
    { title: "Building Systems with the ChatGPT API", provider: "DeepLearning.AI", url: "https://www.coursera.org/learn/build-chatgpt-api-systems", level: "BEGINNER" as const, type: "SHORT" as const, duration: "1 week", tags: ["OpenAI", "API"] },
    { title: "Generative AI with LLMs", provider: "DeepLearning.AI", url: "https://www.coursera.org/learn/generative-ai-with-llms", level: "INTERMEDIATE" as const, type: "SHORT" as const, duration: "2 weeks", tags: ["LLM", "fine-tuning"] },
    { title: "MLOps Fundamentals", provider: "Google", url: "https://www.coursera.org/learn/mlops-fundamentals", level: "INTERMEDIATE" as const, type: "SHORT" as const, duration: "1 week", tags: ["MLOps", "deployment"] },
    { title: "Vector Databases", provider: "Pinecone", url: "https://www.pinecone.io/learn/", level: "INTERMEDIATE" as const, type: "SHORT" as const, duration: "2 hours", tags: ["vector DB", "embeddings"] },
    { title: "Transformers for NLP", provider: "Hugging Face", url: "https://huggingface.co/course", level: "INTERMEDIATE" as const, type: "FULL" as const, duration: "20 hours", tags: ["transformers", "NLP"] },
    { title: "Practical Data Science", provider: "AWS", url: "https://www.coursera.org/learn/aws-practical-data-science", level: "ADVANCED" as const, type: "FULL" as const, duration: "4 weeks", tags: ["MLOps", "AWS"] },
    { title: "AI For Everyone", provider: "Coursera", url: "https://www.coursera.org/learn/ai-for-everyone", level: "BEGINNER" as const, type: "SHORT" as const, duration: "6 hours", tags: ["AI overview"] },
    { title: "Introduction to TensorFlow", provider: "Coursera", url: "https://www.coursera.org/learn/introduction-tensorflow", level: "BEGINNER" as const, type: "SHORT" as const, duration: "1 week", tags: ["TensorFlow", "deep learning"] },
    { title: "PyTorch for Deep Learning", provider: "YouTube", url: "https://www.youtube.com/playlist?list=PLZbbT5o_s2xrfNyHZsM6ufI0iZENK9xgG", level: "BEGINNER" as const, type: "FULL" as const, duration: "60 hours", tags: ["PyTorch"] },
    { title: "Prompt Engineering for Developers", provider: "DeepLearning.AI", url: "https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/", level: "BEGINNER" as const, type: "SHORT" as const, duration: "1 hour", tags: ["prompt engineering"] },
    { title: "Building RAG Applications", provider: "DeepLearning.AI", url: "https://www.deeplearning.ai/short-courses/building-applications-with-vector-databases/", level: "INTERMEDIATE" as const, type: "SHORT" as const, duration: "1 hour", tags: ["RAG", "vector DB"] },
    { title: "AI Agents", provider: "DeepLearning.AI", url: "https://www.deeplearning.ai/short-courses/building-systems-with-llm-agents/", level: "INTERMEDIATE" as const, type: "SHORT" as const, duration: "1 hour", tags: ["agents", "LLM"] },
    { title: "Fine-tuning LLMs", provider: "DeepLearning.AI", url: "https://www.deeplearning.ai/short-courses/finetuning-large-language-models/", level: "ADVANCED" as const, type: "SHORT" as const, duration: "2 hours", tags: ["fine-tuning", "LLM"] },
    { title: "Google AI Essentials", provider: "Google", url: "https://www.coursera.org/learn/google-ai-essentials", level: "BEGINNER" as const, type: "SHORT" as const, duration: "4 hours", tags: ["AI", "Gemini"] },
  ];

  for (let i = 0; i < courses.length; i++) {
    await prisma.course.upsert({
      where: { id: `seed-course-${i}` },
      update: { ...courses[i], order: i },
      create: {
        id: `seed-course-${i}`,
        ...courses[i],
        order: i,
      },
    });
  }

  // Guides
  const guides = [
    { slug: "getting-started-ai", title: "Getting Started with AI", contentMDX: "# Getting Started with AI\n\nWelcome! This guide covers the basics of starting your AI journey.\n\n## Prerequisites\n- Basic programming (Python recommended)\n- High school math\n\n## First Steps\n1. Learn Python\n2. Take an intro ML course\n3. Build a simple project", tags: ["beginner", "overview"], featured: true },
    { slug: "prompt-engineering-tips", title: "Prompt Engineering Tips", contentMDX: "# Prompt Engineering Tips\n\nBest practices for working with LLMs.\n\n## Key Techniques\n- Be specific\n- Use few-shot examples\n- Chain-of-thought for reasoning\n- Iterate and refine", tags: ["LLM", "prompting"], featured: true },
    { slug: "rag-architecture", title: "RAG Architecture Guide", contentMDX: "# RAG Architecture\n\nHow to design Retrieval-Augmented Generation systems.\n\n## Components\n1. Document ingestion\n2. Chunking strategy\n3. Embedding model\n4. Vector store\n5. Retrieval\n6. Generation", tags: ["RAG", "architecture"] },
    { slug: "vector-db-comparison", title: "Vector Database Comparison", contentMDX: "# Vector DB Comparison\n\nPinecone vs Weaviate vs Chroma vs pgvector.\n\n| DB | Best for | Pricing |\n|---|---|---|\n| Pinecone | Production | Paid |\n| Chroma | Local dev | Free |\n| pgvector | PostgreSQL users | Free |", tags: ["vector DB", "comparison"] },
    { slug: "llmops-checklist", title: "LLMOps Checklist", contentMDX: "# LLMOps Checklist\n\n- [ ] Model versioning\n- [ ] Prompt versioning\n- [ ] Evaluation pipeline\n- [ ] Monitoring\n- [ ] Cost tracking", tags: ["LLMOps", "production"] },
    { slug: "fine-tuning-vs-rag", title: "Fine-tuning vs RAG", contentMDX: "# When to Fine-tune vs Use RAG\n\n**RAG** when:\n- Knowledge changes frequently\n- Domain is broad\n- Quick to implement\n\n**Fine-tuning** when:\n- Specific output format\n- Consistent style\n- Latency critical", tags: ["RAG", "fine-tuning"] },
    { slug: "transformer-explained", title: "Transformers Explained", contentMDX: "# Transformers Explained\n\nThe architecture behind GPT, BERT, and modern NLP.\n\n## Key Concepts\n- Self-attention\n- Positional encoding\n- Encoder vs Decoder", tags: ["transformers", "NLP"] },
    { slug: "ai-safety-basics", title: "AI Safety Basics", contentMDX: "# AI Safety Basics\n\n- Alignment\n- Robustness\n- Interpretability\n- Fairness", tags: ["safety", "ethics"] },
    { slug: "building-ai-agents", title: "Building AI Agents", contentMDX: "# Building AI Agents\n\nAgents use tools and reasoning to accomplish tasks.\n\n## Framework\n1. Plan\n2. Act (use tools)\n3. Observe\n4. Repeat", tags: ["agents", "LLM"] },
    { slug: "diffusion-models", title: "Diffusion Models", contentMDX: "# Diffusion Models\n\nHow DALL-E, Stable Diffusion work.\n\n- Forward process: add noise\n- Reverse process: denoise\n- U-Net architecture", tags: ["diffusion", "generative"] },
  ];

  for (let i = 0; i < guides.length; i++) {
    await prisma.guide.upsert({
      where: { slug: guides[i].slug },
      update: {},
      create: {
        ...guides[i],
        order: i,
      },
    });
  }

  // Projects
  const projects = [
    { title: "Sentiment Analyzer", level: "BEGINNER" as const, description: "Build a sentiment classifier for product reviews using BERT or a simple model.", skillsLearned: ["NLP", "classification"], datasetSuggestions: ["IMDB", "Amazon reviews"], syllabus: ["Introduction to NLP and sentiment analysis", "Text preprocessing: tokenization, stemming, stop words", "Exploratory data analysis on review datasets", "Building a baseline with TF-IDF + Logistic Regression", "Using pre-trained BERT for sentiment classification", "Evaluation: accuracy, F1, confusion matrix", "Deployment: FastAPI + simple web interface"], tags: ["NLP", "beginner"], order: 0 },
    { title: "Image Classifier", level: "BEGINNER" as const, description: "Train a CNN to classify images (e.g. CIFAR-10).", skillsLearned: ["computer vision", "CNNs"], datasetSuggestions: ["CIFAR-10", "MNIST"], syllabus: ["Introduction to computer vision and image classification", "Data loading: ImageFolder, transforms, DataLoader", "Image augmentation: rotation, flip, color jitter", "CNN basics: convolutions, pooling, fully connected layers", "Building a model with PyTorch or TensorFlow", "Training loop: loss, optimizer, epochs", "Evaluation: accuracy, confusion matrix, visualization"], tags: ["CV", "beginner"], order: 1 },
    { title: "Chatbot", level: "BEGINNER" as const, description: "Create a rule-based or LLM-powered chatbot.", skillsLearned: ["NLP", "conversational AI"], datasetSuggestions: [], syllabus: ["Defining use case and conversation flow", "Rule-based: keyword matching and state machines", "LLM approach: system prompts and conversation history", "Integrating OpenAI or Claude API", "Streaming responses for better UX", "Building a simple chat UI (React or Streamlit)", "Testing and handling edge cases"], tags: ["LLM", "beginner"], order: 2 },
    { title: "RAG Q&A System", level: "INTERMEDIATE" as const, description: "Build a question-answering system over documents using RAG.", skillsLearned: ["RAG", "embeddings", "vector DB"], datasetSuggestions: ["custom docs"], syllabus: ["Document loading and preprocessing", "Chunking strategies: size, overlap, semantic", "Embedding models and vector stores (Chroma, Pinecone)", "Retrieval: similarity search, hybrid search", "Reranking for better relevance", "Prompt design for Q&A with context", "End-to-end pipeline and evaluation"], tags: ["RAG", "intermediate"], order: 3 },
    { title: "Fine-tuned LLM", level: "INTERMEDIATE" as const, description: "Fine-tune an open-source LLM for a specific task.", skillsLearned: ["fine-tuning", "LoRA"], datasetSuggestions: ["custom dataset"], syllabus: ["Choosing base model and task (classification, generation)", "Dataset preparation: format, size, quality", "LoRA and QLoRA: parameters, rank, alpha", "Training setup: Hugging Face, PEFT, bitsandbytes", "Training loop: loss curves, checkpointing", "Evaluation on held-out data", "Inference and deployment"], tags: ["LLM", "intermediate"], order: 4 },
    { title: "MLOps Pipeline", level: "INTERMEDIATE" as const, description: "Set up CI/CD for ML models with experiment tracking.", skillsLearned: ["MLOps", "DVC", "MLflow"], datasetSuggestions: [], syllabus: ["Experiment tracking: log params, metrics, artifacts", "Model registry: versioning and staging", "Reproducibility: pin dependencies, Docker", "CI/CD with GitHub Actions: test, build, deploy", "Model validation: accuracy, latency checks", "Monitoring: drift, performance alerts", "Rollback and canary deployments"], tags: ["MLOps", "intermediate"], order: 5 },
    { title: "Multi-Agent System", level: "ADVANCED" as const, description: "Build a system of collaborating AI agents.", skillsLearned: ["agents", "orchestration"], datasetSuggestions: [], syllabus: ["Agent architecture: ReAct, plan-and-execute", "Tool design and function calling", "Multi-agent coordination patterns", "Orchestration with LangGraph or CrewAI", "Memory and context management", "Error handling and fallbacks", "Evaluation of agent systems"], tags: ["agents", "advanced"], order: 6 },
    { title: "Production RAG", level: "ADVANCED" as const, description: "Deploy a RAG system with monitoring and evaluation.", skillsLearned: ["RAG", "deployment", "evaluation"], datasetSuggestions: [], syllabus: ["Production chunking and indexing pipeline", "Latency optimization: caching, batching", "Evaluation framework: faithfulness, relevance", "A/B testing RAG variants", "Monitoring: latency, error rates, retrieval quality", "Handling updates and re-indexing", "Cost tracking and optimization"], tags: ["RAG", "advanced"], order: 7 },
    { title: "Custom Diffusion Model", level: "ADVANCED" as const, description: "Train a diffusion model on a custom dataset.", skillsLearned: ["diffusion", "generative"], datasetSuggestions: ["custom images"], syllabus: ["Diffusion theory: forward and reverse process", "U-Net architecture and conditioning", "Dataset preparation and preprocessing", "Training Stable Diffusion or similar", "LoRA for efficient fine-tuning", "Inference and sampling strategies", "Evaluation: FID, CLIP score"], tags: ["diffusion", "advanced"], order: 8 },
    { title: "Document Understanding Pipeline", level: "INTERMEDIATE" as const, description: "Extract structured data from PDFs using vision-language models.", skillsLearned: ["document AI", "VLM"], datasetSuggestions: ["custom PDFs"], syllabus: ["PDF parsing and layout analysis", "Vision-language models (GPT-4V, LLaVA)", "Structured extraction with prompts", "Table and form recognition", "Handling multi-page documents", "Validation and error correction", "Building an extraction API"], tags: ["VLM", "intermediate"], order: 9 },
    // Industry-level projects
    { title: "Customer Churn Prediction", level: "INTERMEDIATE" as const, description: "Predict which customers will cancel. Build feature pipeline, train classifier, deploy as API. Used by SaaS, telecom, and subscription businesses.", skillsLearned: ["classification", "feature engineering", "API deployment"], datasetSuggestions: ["Telco churn", "subscription data"], syllabus: ["Business context: churn definition, retention metrics", "Data exploration: tenure, usage, support tickets", "Feature engineering: recency, frequency, engagement scores", "Handling imbalanced data: SMOTE, class weights", "Model selection: logistic regression, XGBoost, ensemble", "Evaluation: precision, recall, AUC, business metrics", "Deployment: FastAPI, batch scoring, dashboard", "Monitoring: drift, model performance over time"], tags: ["industry", "ML", "churn"], order: 10 },
    { title: "Fraud Detection System", level: "INTERMEDIATE" as const, description: "Real-time fraud detection for transactions. Handle imbalanced data, low-latency inference, explainability for compliance.", skillsLearned: ["anomaly detection", "real-time inference", "explainability"], datasetSuggestions: ["Kaggle fraud", "synthetic transaction data"], syllabus: ["Fraud patterns: velocity, geography, amount anomalies", "Feature engineering: time windows, aggregations", "Imbalanced learning: undersampling, oversampling, focal loss", "Model: isolation forest, XGBoost, neural network", "Real-time inference: sub-100ms latency requirements", "Explainability: SHAP, LIME for regulatory compliance", "Deployment: streaming pipeline, rule engine hybrid", "Monitoring: false positive rate, fraud catch rate"], tags: ["industry", "finance", "fraud"], order: 11 },
    { title: "Enterprise Knowledge Base Q&A", level: "INTERMEDIATE" as const, description: "RAG over internal docs, wikis, runbooks. Add auth, audit logging, feedback loop. Used by support, sales, and engineering teams.", skillsLearned: ["RAG", "enterprise", "auth"], datasetSuggestions: ["company wikis", "Confluence", "Notion"], syllabus: ["Document ingestion: Confluence, Notion, SharePoint APIs", "Chunking: semantic splitting, metadata preservation", "Vector store: Pinecone/Weaviate with access control", "Retrieval: hybrid search, reranking, citation", "Auth: SSO, role-based document access", "Audit logging: queries, answers, user feedback", "Feedback loop: thumbs up/down, retrain embeddings", "Deployment: Kubernetes, rate limits, caching"], tags: ["industry", "RAG", "enterprise"], order: 12 },
    { title: "Manufacturing Defect Detection", level: "INTERMEDIATE" as const, description: "CNN to classify product images as pass/fail. Deploy to edge or cloud. Used in quality control for electronics, textiles, food.", skillsLearned: ["computer vision", "CNNs", "edge deployment"], datasetSuggestions: ["MVTec AD", "custom defect images"], syllabus: ["Defect types: scratches, cracks, misalignment", "Data collection: camera setup, labeling workflow", "Augmentation: rotation, noise, lighting variations", "Model: ResNet/EfficientNet fine-tuning", "Handling small datasets: transfer learning, few-shot", "Edge deployment: ONNX, TensorRT, Raspberry Pi", "Integration: PLC, conveyor belt triggers", "Monitoring: false reject rate, drift detection"], tags: ["industry", "CV", "manufacturing"], order: 13 },
    { title: "Legal Contract Clause Extractor", level: "INTERMEDIATE" as const, description: "Extract parties, dates, obligations from contracts into structured JSON. Handle variations across contract types. Used by legal ops.", skillsLearned: ["NLP", "structured extraction", "LLM"], datasetSuggestions: ["SEC filings", "NDA templates"], syllabus: ["Contract structure: parties, dates, obligations, termination", "Document parsing: PDF layout, table extraction", "Prompt design: few-shot for each clause type", "Structured output: JSON schema, validation", "Handling variations: different jurisdictions, formats", "Human-in-the-loop: review queue, corrections", "API: batch processing, webhook notifications", "Compliance: audit trail, versioning"], tags: ["industry", "legal", "NLP"], order: 14 },
    { title: "Customer Support Triage Bot", level: "INTERMEDIATE" as const, description: "Classify support tickets, route to correct team, suggest responses. Integrate with Zendesk, Intercom, or Freshdesk.", skillsLearned: ["classification", "LLM", "integrations"], datasetSuggestions: ["support ticket history"], syllabus: ["Ticket taxonomy: categories, subcategories, priority", "Few-shot classification with LLM", "Routing logic: urgency, team capacity, skills", "Response suggestion: RAG over knowledge base", "Integration: Zendesk/Intercom API, webhooks", "Escalation rules: confidence threshold, fallback", "Metrics: routing accuracy, resolution time", "Deployment: serverless, auto-scaling"], tags: ["industry", "support", "LLM"], order: 15 },
    { title: "Demand Forecasting Pipeline", level: "ADVANCED" as const, description: "LSTM or transformer for time-series. Integrate with inventory system. Used by retail, logistics, supply chain.", skillsLearned: ["time series", "LSTM", "MLOps"], datasetSuggestions: ["retail sales", "warehouse data"], syllabus: ["Time series basics: trend, seasonality, exogenous vars", "Feature engineering: lags, rolling stats, holidays", "Models: Prophet, LSTM, Transformer, N-BEATS", "Evaluation: MAPE, MAE, bias, forecast intervals", "Pipeline: retrain schedule, incremental updates", "Integration: ERP, inventory optimization", "Monitoring: forecast accuracy, bias drift", "A/B test: new model vs baseline"], tags: ["industry", "time-series", "retail"], order: 16 },
    { title: "E-commerce Recommendation Engine", level: "ADVANCED" as const, description: "Two-tower or collaborative filtering for product recommendations. A/B test in production. Used by marketplaces and retailers.", skillsLearned: ["recommendations", "two-tower", "A/B testing"], datasetSuggestions: ["Amazon reviews", "retail transactions"], syllabus: ["Recommendation types: collaborative, content-based, hybrid", "Two-tower model: user and item embeddings", "Training: contrastive loss, negative sampling", "Serving: approximate nearest neighbors (ANN)", "Cold start: content features, popularity fallback", "A/B testing: CTR, conversion, diversity metrics", "Real-time: update embeddings, personalization", "Deployment: low-latency retrieval, caching"], tags: ["industry", "recommendations", "e-commerce"], order: 17 },
    { title: "Healthcare FAQ Bot with RAG", level: "ADVANCED" as const, description: "RAG over medical guidelines and policies. Ensure citations, handle sensitive data, add disclaimer. Used by patient support.", skillsLearned: ["RAG", "healthcare", "compliance"], datasetSuggestions: ["medical guidelines", "FAQ documents"], syllabus: ["Domain: medical guidelines, policy documents", "Chunking: preserve context, avoid fragmentation", "Retrieval: strict citation, no hallucination", "Prompt: disclaimer, \"consult a doctor\"", "PII handling: redaction, audit logging", "Evaluation: medical accuracy, citation quality", "Compliance: HIPAA, audit trail", "Deployment: secure, access-controlled"], tags: ["industry", "healthcare", "RAG"], order: 18 },
  ];

  for (let i = 0; i < projects.length; i++) {
    await prisma.projectIdea.upsert({
      where: { id: `seed-project-${i}` },
      update: projects[i],
      create: {
        id: `seed-project-${i}`,
        ...projects[i],
      },
    });
  }

  console.log("Seed completed!");
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
