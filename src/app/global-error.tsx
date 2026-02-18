"use client";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <html lang="en">
      <body>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            minHeight: "100vh",
            padding: "2rem",
            fontFamily: "system-ui, sans-serif",
            textAlign: "center",
          }}
        >
          <h1 style={{ fontSize: "1.5rem", marginBottom: "0.5rem" }}>
            Something went wrong
          </h1>
          <p style={{ color: "#666", marginBottom: "1.5rem" }}>
            An unexpected error occurred. Please try again.
          </p>
          {process.env.NODE_ENV === "development" && error?.message && (
            <pre
              style={{
                padding: "1rem",
                background: "#f5f5f5",
                borderRadius: "0.5rem",
                fontSize: "0.875rem",
                maxWidth: "32rem",
                overflow: "auto",
                textAlign: "left",
                marginBottom: "1.5rem",
              }}
            >
              {error.message}
            </pre>
          )}
          <div style={{ display: "flex", gap: "1rem" }}>
            <button
              onClick={reset}
              style={{
                padding: "0.5rem 1rem",
                background: "#000",
                color: "#fff",
                border: "none",
                borderRadius: "0.25rem",
                cursor: "pointer",
              }}
            >
              Try again
            </button>
            <button
              type="button"
              onClick={() => (window.location.href = "/")}
              style={{
                padding: "0.5rem 1rem",
                border: "1px solid #ccc",
                borderRadius: "0.25rem",
                background: "transparent",
                cursor: "pointer",
                color: "#333",
              }}
            >
              Go home
            </button>
          </div>
        </div>
      </body>
    </html>
  );
}
