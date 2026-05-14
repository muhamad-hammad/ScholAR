"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { useAppState } from "../lib/useAppState";
import { summarize } from "../lib/api";
import Spinner from "./Spinner";

export default function SummaryView() {
  const {
    retrieverReady,
    usageMode,
    provider,
    modelId,
    apiKey,
    paperSummary,
    setPaperSummary,
  } = useAppState();

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isDisabled = !retrieverReady || usageMode === "browse";

  async function handleGenerate() {
    if (isDisabled || !usageMode) return;
    setError(null);
    setLoading(true);
    try {
      const result = await summarize({ provider, modelId, apiKey, usageMode });
      setPaperSummary(result.summary);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-4 overflow-y-auto">
      <div>
        <h2 className="text-xl font-semibold">Paper summary</h2>
        <p className="mt-1 text-sm text-[var(--app-muted)]">
          An AI-generated abstract built from the indexed paper. The summary is
          cached — click Regenerate to recompute.
        </p>
      </div>

      {!retrieverReady && (
        <div className="rounded-md border border-blue-500/30 bg-blue-500/10 px-4 py-3 text-sm text-blue-400">
          Index a document first to generate a summary.
        </div>
      )}

      {usageMode === "browse" && (
        <div className="rounded-md border border-yellow-500/30 bg-yellow-500/10 px-4 py-3 text-sm text-yellow-400">
          Browse-only mode is active. Switch startup mode to enable AI summary
          generation.
        </div>
      )}

      <button
        type="button"
        onClick={handleGenerate}
        disabled={isDisabled || loading}
        className="flex w-fit items-center gap-2 rounded bg-[var(--app-accent)] px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-[var(--app-accent-hover)] disabled:opacity-50"
      >
        {loading && <Spinner />}
        {paperSummary ? "Regenerate summary" : "Generate summary"}
      </button>

      {error && (
        <div className="rounded-md border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {paperSummary && (
        <div className="mt-2">
          <h3 className="mb-2 text-lg font-medium">Summary</h3>
          <div className="text-sm leading-relaxed text-[var(--app-text)] [&_h1]:mb-2 [&_h1]:text-xl [&_h1]:font-bold [&_h2]:mb-2 [&_h2]:text-lg [&_h2]:font-semibold [&_h3]:mb-1 [&_h3]:font-semibold [&_li]:ml-4 [&_li]:list-disc [&_p]:mb-2 [&_strong]:font-semibold">
            <ReactMarkdown>{paperSummary}</ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
}
