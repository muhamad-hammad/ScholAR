import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { UsageMode, ThemeMode, Message, PdfImage } from "./types";

interface AppState {
  usageMode: UsageMode | null;
  themeMode: ThemeMode;
  ingestedFile: string | null;
  retrieverReady: boolean;
  conversationHistory: Message[];
  compiledGraphReady: boolean;
  paperSummary: string | null;
  pdfFile: File | null;
  pdfImages: PdfImage[] | null;
  provider: string;
  modelId: string;
  apiKey: string;
  activeView: "chat" | "summary" | "visuals";


  setUsageMode(mode: UsageMode | null): void;
  setThemeMode(mode: ThemeMode): void;
  setIngestedFile(name: string | null): void;
  setRetrieverReady(ready: boolean): void;
  setConversationHistory(history: Message[]): void;
  setCompiledGraphReady(ready: boolean): void;
  setPaperSummary(summary: string | null): void;
  setPdfFile(file: File | null): void;
  setPdfImages(images: PdfImage[] | null): void;
  setProvider(p: string): void;
  setModelId(id: string): void;
  setApiKey(key: string): void;
  setActiveView(view: "chat" | "summary" | "visuals"): void;

  clearConversation(): void;
  resetStartup(): void;
}

export const useAppState = create<AppState>()(
  persist(
    (set) => ({
      usageMode: null,
      themeMode: "dark",
      ingestedFile: null,
      retrieverReady: false,
      conversationHistory: [],
      compiledGraphReady: false,
      paperSummary: null,
      pdfFile: null,
      pdfImages: null,
      provider: "openai",
      modelId: "gpt-4o-mini",
      apiKey: "",
      activeView: "chat",


      setUsageMode: (mode) => set({ usageMode: mode }),
      setThemeMode: (mode) => set({ themeMode: mode }),
      setIngestedFile: (name) => set({ ingestedFile: name }),
      setRetrieverReady: (ready) => set({ retrieverReady: ready }),
      setConversationHistory: (history) => set({ conversationHistory: history }),
      setCompiledGraphReady: (ready) => set({ compiledGraphReady: ready }),
      setPaperSummary: (summary) => set({ paperSummary: summary }),
      setPdfFile: (file) => set({ pdfFile: file }),
      setPdfImages: (images) => set({ pdfImages: images }),
      setProvider: (p) => set({ provider: p }),
      setModelId: (id) => set({ modelId: id }),
      setApiKey: (key) => set({ apiKey: key }),
      setActiveView: (view) => set({ activeView: view }),

      clearConversation: () =>
        set({ conversationHistory: [], compiledGraphReady: false }),
      resetStartup: () => set({ usageMode: null, compiledGraphReady: false }),
    }),
    {
      name: "scholar-app-state",
      partialize: (state) => ({
        themeMode: state.themeMode,
      }),
    }
  )
);
