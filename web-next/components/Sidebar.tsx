"use client";

import { useState } from "react";
import { useAppState } from "../lib/useAppState";
import { ingestPdf } from "../lib/api";
import Spinner from "./Spinner";

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
}

const PROVIDER_DEFAULTS: Record<string, string> = {
  openai: "gpt-4o-mini",
  gemini: "gemini-2.0-flash",
  groq: "llama-3.3-70b-versatile",
  openrouter: "meta-llama/llama-3.3-8b-instruct:free",
  grok: "grok-3-mini",
};

const API_KEY_ENV: Record<string, string> = {
  openai: "OPENAI_API_KEY",
  gemini: "GEMINI_API_KEY",
  groq: "GROQ_API_KEY",
  openrouter: "OPENROUTER_API_KEY",
  grok: "XAI_API_KEY",
};

const PROVIDERS = ["openai", "gemini", "groq", "openrouter", "grok"];

const inputClass =
  "w-full rounded-lg border border-[var(--app-border)] bg-[var(--app-btn-bg)] py-1.5 px-2 text-sm text-[var(--app-text)] outline-none focus:ring-1 focus:ring-[var(--app-accent)]";

export default function Sidebar({ isOpen, onToggle }: SidebarProps) {
  const [ingesting, setIngesting] = useState(false);
  const [ingestStatus, setIngestStatus] = useState<{
    type: "error" | "info" | "success";
    text: string;
  } | null>(null);

  const usageMode = useAppState((s) => s.usageMode);
  const themeMode = useAppState((s) => s.themeMode);
  const pdfFile = useAppState((s) => s.pdfFile);
  const ingestedFile = useAppState((s) => s.ingestedFile);
  const provider = useAppState((s) => s.provider);
  const modelId = useAppState((s) => s.modelId);
  const apiKey = useAppState((s) => s.apiKey);

  const setThemeMode = useAppState((s) => s.setThemeMode);
  const setPdfFile = useAppState((s) => s.setPdfFile);
  const setIngestedFile = useAppState((s) => s.setIngestedFile);
  const setRetrieverReady = useAppState((s) => s.setRetrieverReady);
  const setProvider = useAppState((s) => s.setProvider);
  const setModelId = useAppState((s) => s.setModelId);
  const setApiKey = useAppState((s) => s.setApiKey);
  const resetStartup = useAppState((s) => s.resetStartup);
  const clearConversation = useAppState((s) => s.clearConversation);

  function handleThemeToggle() {
    const next = themeMode === "dark" ? "light" : "dark";
    setThemeMode(next);
    document.documentElement.dataset.theme = next;
  }

  function handleProviderChange(p: string) {
    setProvider(p);
    setModelId(PROVIDER_DEFAULTS[p] ?? "");
  }

  async function handleIngest() {
    if (!pdfFile) {
      setIngestStatus({ type: "error", text: "Please select a PDF first." });
      return;
    }
    if (ingestedFile === pdfFile.name) {
      setIngestStatus({ type: "info", text: "Already indexed — using cached retriever." });
      return;
    }
    setIngesting(true);
    setIngestStatus(null);
    try {
      await ingestPdf(pdfFile);
      setRetrieverReady(true);
      setIngestedFile(pdfFile.name);
      setIngestStatus({ type: "success", text: "Ingestion complete." });
    } catch (err) {
      setIngestStatus({
        type: "error",
        text: err instanceof Error ? err.message : "Ingestion failed.",
      });
    } finally {
      setIngesting(false);
    }
  }

  const envVarName = API_KEY_ENV[provider] ?? "";

  return (
    <>
      {/* Hamburger button — shown when sidebar is closed */}
      {!isOpen && (
        <button
          type="button"
          onClick={onToggle}
          className="fixed left-3 top-3 z-50 flex h-9 w-9 items-center justify-center rounded-lg border border-[var(--app-border)] bg-[var(--app-sidebar-bg)] text-[var(--app-text)]"
          aria-label="Open sidebar"
        >
          <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
      )}

      {/* Mobile backdrop — tap outside to close */}
      {isOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/50 md:hidden"
          onClick={onToggle}
        />
      )}

      {/* Sidebar panel */}
      <aside
        className={`fixed left-0 top-0 z-40 h-full w-[260px] border-r border-[var(--app-border)] transition-transform duration-300 ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex h-full flex-col overflow-y-auto bg-[var(--app-sidebar-bg)] text-[var(--app-text)]">

          {/* Header */}
          <div className="flex items-center justify-between px-5 pb-3 pt-5">
            <h1 className="text-2xl font-bold">ScholAR</h1>
            <button
              type="button"
              onClick={onToggle}
              className="flex h-7 w-7 items-center justify-center rounded-md text-[var(--app-muted)]"
              aria-label="Close sidebar"
            >
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Theme toggle */}
          <div className="flex items-center justify-between px-5 py-3">
            <span className="text-sm">Dark mode</span>
            <button
              type="button"
              role="switch"
              aria-checked={themeMode === "dark"}
              aria-label="Toggle dark mode"
              onClick={handleThemeToggle}
              className={`relative inline-flex h-5 w-9 flex-shrink-0 cursor-pointer items-center rounded-full transition-colors ${
                themeMode === "dark" ? "bg-[var(--app-accent)]" : "bg-[var(--app-border)]"
              }`}
            >
              <span
                className={`inline-block h-3.5 w-3.5 rounded-full bg-white shadow transition-transform ${
                  themeMode === "dark" ? "translate-x-[18px]" : "translate-x-1"
                }`}
              />
            </button>
          </div>

          <hr className="mx-5 border-[var(--app-border)]" />

          {/* Change startup mode */}
          <div className="px-5 py-3">
            <button
              type="button"
              onClick={resetStartup}
              className="w-full rounded-lg border border-[var(--app-border)] bg-[var(--app-btn-bg)] py-2 px-4 text-sm text-[var(--app-text)] transition-colors hover:bg-[var(--app-btn-hover-bg)]"
            >
              Change startup mode
            </button>
          </div>

          <hr className="mx-5 border-[var(--app-border)]" />

          {/* Document section */}
          <div className="px-5 py-3">
            <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-[var(--app-muted)]">
              Document
            </p>
            <label
              htmlFor="pdf-file-input"
              className="block w-full cursor-pointer rounded-lg border border-[var(--app-border)] bg-[var(--app-btn-bg)] py-2 px-3 text-center text-sm text-[var(--app-muted)] transition-colors hover:bg-[var(--app-btn-hover-bg)]"
            >
              Choose PDF
            </label>
            <input
              id="pdf-file-input"
              type="file"
              accept=".pdf"
              className="sr-only"
              onChange={(e) => {
                const file = e.target.files?.[0] ?? null;
                setPdfFile(file);
                setIngestStatus(null);
              }}
            />
            {pdfFile && (
              <p className="mt-2 truncate text-xs text-[var(--app-muted)]">{pdfFile.name}</p>
            )}
          </div>

          {/* AI model section */}
          <div className="px-5 py-3">
            <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-[var(--app-muted)]">
              AI model
            </p>

            <label htmlFor="provider-select" className="mb-1 block text-xs text-[var(--app-muted)]">Provider</label>
            <select
              id="provider-select"
              value={provider}
              onChange={(e) => handleProviderChange(e.target.value)}
              className={`${inputClass} mb-3 cursor-pointer`}
            >
              {PROVIDERS.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>

            <p className="mb-1 text-xs text-[var(--app-muted)]">Model</p>
            <p className="mb-3 rounded-lg border border-[var(--app-border)] bg-[var(--app-btn-bg)] py-1.5 px-2 text-sm text-[var(--app-text)] opacity-60 truncate">
              {modelId || "—"}
            </p>

            {/* API key / mode info */}
            {usageMode === "browse" ? (
              <div className="rounded-lg border border-blue-500/25 bg-blue-500/10 p-3 text-xs text-blue-400">
                Browse mode: AI answer generation is disabled.
              </div>
            ) : usageMode === "env_key" ? (
              <div className="rounded-lg border border-green-500/25 bg-green-500/10 p-3 text-xs text-green-400">
                Using API key from .env file ({envVarName}).
              </div>
            ) : usageMode === "own_key" ? (
              <>
                <label className="mb-1 block text-xs text-[var(--app-muted)]">{envVarName}</label>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  aria-label={envVarName}
                  className={inputClass}
                />
              </>
            ) : null}
          </div>

          {/* Run ingestion */}
          <div className="px-5 py-3">
            <button
              type="button"
              onClick={handleIngest}
              disabled={ingesting}
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-[var(--app-accent)] py-2 px-4 text-sm font-medium text-white transition-colors hover:bg-[var(--app-accent-hover)] disabled:cursor-not-allowed disabled:opacity-60"
            >
              {ingesting && <Spinner />}
              Run ingestion
            </button>
            {ingestStatus && (
              <p
                className={`mt-2 text-xs ${
                  ingestStatus.type === "error"
                    ? "text-red-400"
                    : ingestStatus.type === "success"
                    ? "text-green-400"
                    : "text-blue-400"
                }`}
              >
                {ingestStatus.text}
              </p>
            )}
          </div>

          <hr className="mx-5 border-[var(--app-border)]" />

          {/* Clear conversation */}
          <div className="px-5 py-3 pb-6">
            <button
              type="button"
              onClick={clearConversation}
              className="w-full rounded-lg border border-[var(--app-border)] bg-[var(--app-btn-bg)] py-2 px-4 text-sm text-[var(--app-text)] transition-colors hover:bg-[var(--app-btn-hover-bg)]"
            >
              Clear conversation
            </button>
          </div>

        </div>
      </aside>
    </>
  );
}
