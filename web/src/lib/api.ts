import axios from "axios";

const api = axios.create({ baseURL: "/api" });

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

export interface Document {
  id: string;
  original_filename: string;
  status: string;
  document_type: string;
  ocr_engine: string | null;
  ocr_confidence: number | null;
  uploaded_at: string;
  extractions: ExtractionSummary[];
}

export interface ExtractionSummary {
  id: string;
  document_id: string;
  version: number;
  overall_confidence: number;
  confidence_tier: string;
  verdict: string | null;
  extracted_at: string;
}

export interface ExtractionDetail {
  extraction_id: string;
  version: number;
  overall_confidence: number;
  confidence_tier: string;
  data: Record<string, unknown>;
  verification: {
    verdict: string;
    overall_score: number;
    field_scores: Record<string, number>;
    rule_flags: string[];
  } | null;
}

export const auth = {
  signupTenant: (name: string) => api.post("/auth/signup/tenant", { name }),
  signupUser: (email: string, password: string) =>
    api.post("/auth/signup/user", { email, password }),
  login: async (email: string, password: string) => {
    const res = await api.post<{ access_token: string }>("/auth/login", { email, password });
    localStorage.setItem("token", res.data.access_token);
    return res.data;
  },
  logout: () => localStorage.removeItem("token"),
  isAuthenticated: () => !!localStorage.getItem("token"),
};

export const documents = {
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return api.post<{ document_id: string; status: string }>("/documents", form);
  },
  list: (statusFilter?: string) =>
    api.get<Document[]>("/documents", { params: statusFilter ? { status_filter: statusFilter } : {} }),
  get: (id: string) => api.get<Document>(`/documents/${id}`),
  getExtraction: (id: string) => api.get<ExtractionDetail>(`/documents/${id}/extraction`),
};

export const reviewQueue = {
  list: () => api.get<Document[]>("/review-queue"),
};
