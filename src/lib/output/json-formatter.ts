import type { ChunkType } from "../store/types";

export interface JsonOutput {
  results?: ChunkType[];
  hits?: unknown[];
  tsv?: string;
  format?: string;
  metadata?: {
    count: number;
    query?: string;
  };
}

export function formatJson(data: JsonOutput): string {
  return JSON.stringify(data, null, 2);
}
