"use client";

import { useRef, useEffect, useState } from "react";
import { useAppState } from "../lib/useAppState";
import { queryRag } from "../lib/api";
import ChatMessage from "./ChatMessage";
import Spinner from "./Spinner";

export default function ChatView() {
  const {
    conversationHistory,
    setConversationHistory,
    retrieverReady,
    usageMode,
    provider,
    modelId,
    apiKey,
  } = useAppState();

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [warning, setWarning] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversationHistory]);

  async function handleAsk() {
    setWarning(null);
    setError(null);

    if (!input.trim()) {
      setWarning("Please enter a question.");
      return;
    }
    if (!retrieverReady) {
      setWarning("Please index a document first.");
      return;
    }
    if (usageMode === "browse") {
      setWarning("Browse-only mode is active. Switch startup mode to enable AI answers.");
      return;
    }
    if (!usageMode) return;

    setLoading(true);
    try {
      const result = await queryRag({
        question: input.trim(),
        conversationHistory,
        provider,
        modelId,
        apiKey,
        usageMode,
      });
      setConversationHistory(result.conversationHistory);
      setInput("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred.");
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" && !loading) handleAsk();
  }

  return (
    <div className="flex flex-col">
      <div className="py-2 pr-1">
        {conversationHistory.length === 0 ? (
          <p className="text-sm text-[var(--app-muted)] text-center mt-8">
            No messages yet. Ask a question to get started.
          </p>
        ) : (
          conversationHistory.map((msg, i) => (
            <ChatMessage key={i} message={msg} />
          ))
        )}
        <div ref={bottomRef} />
      </div>

      {warning && (
        <p className="text-sm text-yellow-400 mb-2">{warning}</p>
      )}
      {error && (
        <div className="mb-2 rounded-md border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      <div className="flex shrink-0 gap-2 border-t border-[var(--app-border)] pt-3">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Enter your question"
          disabled={loading}
          className="flex-1 rounded border border-[var(--app-border)] bg-[var(--app-secondary-bg)] px-3 py-2 text-sm text-[var(--app-text)] placeholder-[var(--app-muted)] focus:border-[var(--app-accent)] focus:outline-none disabled:opacity-50"
        />
        <button
          type="button"
          onClick={handleAsk}
          disabled={loading}
          className="flex items-center gap-2 rounded bg-[var(--app-accent)] px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-[var(--app-accent-hover)] disabled:opacity-50"
        >
          {loading && <Spinner />}
          Ask
        </button>
      </div>
    </div>
  );
}
