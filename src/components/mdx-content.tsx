import { MDXRemote } from "next-mdx-remote/rsc";

interface MDXContentProps {
  source: string;
}

export async function MDXContent({ source }: MDXContentProps) {
  try {
    return <MDXRemote source={source} />;
  } catch {
    return (
      <div className="prose dark:prose-invert max-w-none">
        <pre className="whitespace-pre-wrap">{source}</pre>
      </div>
    );
  }
}
