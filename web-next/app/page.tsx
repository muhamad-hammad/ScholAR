"use client";

import { useState, useEffect } from "react";
import WelcomeModal from "../components/WelcomeModal";
import Sidebar from "../components/Sidebar";
import NavTabs from "../components/NavTabs";
import ChatView from "../components/ChatView";
import SummaryView from "../components/SummaryView";
import VisualsView from "../components/VisualsView";
import { useAppState } from "../lib/useAppState";

const MODE_CAPTIONS: Record<string, string> = {
  browse: "Mode: Browse only (no AI generation)",
  own_key: "Mode: Using a custom API key",
};

export default function Home() {
  const { usageMode, retrieverReady, activeView } = useAppState();
  const [sidebarOpen, setSidebarOpen] = useState(true);

  useEffect(() => {
    setSidebarOpen(window.innerWidth >= 768);
  }, []);

  return (
    <main className="flex min-h-screen flex-col bg-[var(--app-bg)] text-[var(--app-text)]">
      <WelcomeModal />
      <div className="flex flex-1 min-h-0">
        <Sidebar isOpen={sidebarOpen} onToggle={() => setSidebarOpen((v) => !v)} />
        <div className={`flex flex-1 flex-col p-6 gap-3 transition-[margin] duration-300 ${sidebarOpen ? "md:ml-[260px]" : ""}`}>
          <h1 className="shrink-0 text-3xl font-bold">ScholAR</h1>

          {usageMode && (
            <p className="shrink-0 text-sm text-[var(--app-muted)]">
              {MODE_CAPTIONS[usageMode]}
            </p>
          )}

          <div className="shrink-0">
            {retrieverReady ? (
              <div className="rounded-md border border-green-500/30 bg-green-500/10 px-4 py-3 text-sm text-green-400">
                Document indexed and ready.
              </div>
            ) : (
              <div className="rounded-md border border-blue-500/30 bg-blue-500/10 px-4 py-3 text-sm text-blue-400">
                No document indexed yet. Upload a PDF and click Run ingestion.
              </div>
            )}
          </div>

          <div className="shrink-0">
            <NavTabs />
          </div>

          <hr className="shrink-0 border-[var(--app-border)]" />

          <div className="flex-1">
            {activeView === "chat" && <ChatView />}
            {activeView === "summary" && <SummaryView />}
            {activeView === "visuals" && <VisualsView />}
          </div>
        </div>
      </div>
    </main>
  );
}
