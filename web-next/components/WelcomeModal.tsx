"use client";

import { useState } from "react";
import { useAppState } from "../lib/useAppState";
import type { UsageMode } from "../lib/types";
import Spinner from "./Spinner";

interface Option {
  title: string;
  description: string;
  buttonLabel: string;
  mode: UsageMode;
}

const OPTIONS: Option[] = [
  {
    title: "Browse only",
    description:
      "Explore and retrieve passages from your documents. No AI answer generation — no API key needed.",
    buttonLabel: "Continue without a key",
    mode: "browse",
  },
  {
    title: "Enter my own key",
    description:
      "Paste a fresh API key directly in the sidebar. Useful when testing a different account or model.",
    buttonLabel: "I'll provide a key",
    mode: "own_key",
  },
];

export default function WelcomeModal() {
  const usageMode = useAppState((s) => s.usageMode);
  const setUsageMode = useAppState((s) => s.setUsageMode);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (usageMode !== null) return null;

  async function handleSelectMode(mode: UsageMode) {
    setLoading(true);
    setError(null);
    try {
      setUsageMode(mode);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60">
      <div className="w-full max-w-4xl rounded-2xl border border-[var(--app-border)] bg-[var(--app-bg)] p-8 shadow-2xl">
        <h1 className="text-3xl font-bold text-center mb-2 text-[var(--app-text)]">
          Welcome to ScholAR
        </h1>
        <p className="text-center mb-8 text-sm text-[var(--app-muted)]">
          How would you like to use the AI assistant?
        </p>

        <div className="flex flex-col md:flex-row gap-4">
          {OPTIONS.map((opt) => (
            <div
              key={opt.mode}
              className="flex-1 flex flex-col rounded-xl border border-[var(--app-border)] bg-[var(--app-secondary-bg)] p-6"
            >
              <h2 className="text-lg font-semibold mb-2 text-[var(--app-text)]">
                {opt.title}
              </h2>
              <p className="text-sm flex-1 mb-6 text-[var(--app-muted)]">
                {opt.description}
              </p>
              <button
                type="button"
                onClick={() => handleSelectMode(opt.mode)}
                disabled={loading}
                className="w-full rounded-lg py-2 px-4 text-sm font-medium text-white bg-[var(--app-accent)] hover:bg-[var(--app-accent-hover)] transition-colors disabled:cursor-not-allowed disabled:opacity-60 flex items-center justify-center gap-2"
              >
                {loading && <Spinner />}
                {opt.buttonLabel}
              </button>
            </div>
          ))}

        </div>

        {error && (
          <p className="mt-4 text-center text-sm text-red-400">{error}</p>
        )}
      </div>
    </div>
  );
}
