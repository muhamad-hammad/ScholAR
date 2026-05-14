"use client";

import type { Message } from "../lib/types";

export default function ChatMessage({ message }: { message: Message }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3`}>
      <div
        className={`max-w-[75%] rounded-lg px-4 py-2 text-sm whitespace-pre-wrap break-words ${
          isUser
            ? "bg-[var(--app-accent)] text-white"
            : "bg-[var(--app-secondary-bg)] border border-[var(--app-border)] text-[var(--app-text)]"
        }`}
      >
        {message.content}
      </div>
    </div>
  );
}
