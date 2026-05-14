export type UsageMode = "browse" | "own_key" | "env_key";
export type ThemeMode = "dark" | "light";
export type Role = "user" | "assistant";

export interface Message {
  role: Role;
  content: string;
}

export interface PdfImage {
  page: number;
  ext: string;
  data: string;
}
