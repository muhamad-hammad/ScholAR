"use client";

import { useAppState } from "../lib/useAppState";

type View = "chat" | "summary" | "visuals";

const TABS: { id: View; label: string }[] = [
  { id: "chat", label: "Chat" },
  { id: "summary", label: "Summary" },
  { id: "visuals", label: "Visualizations" },
];

export default function NavTabs() {
  const { activeView, setActiveView } = useAppState();

  return (
    <div className="flex gap-2 flex-wrap">
      {TABS.map((tab) => {
        const isActive = activeView === tab.id;
        return (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveView(tab.id)}
            className={`rounded px-4 py-2 text-sm font-medium transition-colors ${
              isActive
                ? "bg-[var(--app-accent)] text-white"
                : "border border-[var(--app-border)] bg-[var(--app-btn-bg)] text-[var(--app-text)] hover:bg-[var(--app-btn-hover-bg)]"
            }`}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}
