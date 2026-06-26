"use client";

import { useEffect, useState } from "react";
import { documents, type Document } from "@/lib/api";
import { useAuthGuard } from "@/lib/useAuth";

const STATUS_BADGE: Record<string, string> = {
  pending:    "bg-gray-100 text-gray-700",
  processing: "bg-yellow-100 text-yellow-700",
  extracted:  "bg-blue-100 text-blue-700",
  verified:   "bg-green-100 text-green-700",
  flagged:    "bg-orange-100 text-orange-700",
  rejected:   "bg-red-100 text-red-700",
  failed:     "bg-red-100 text-red-700",
};

const TERMINAL = new Set(["verified", "flagged", "rejected", "failed"]);
const POLL_INTERVAL = 3000;

export default function Dashboard() {
  const isAuth = useAuthGuard();
  const [docs, setDocs] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!isAuth) return;

    let timer: ReturnType<typeof setTimeout>;

    const fetch = () => {
      documents.list()
        .then((r) => {
          setDocs(r.data);
          setLoading(false);
          const hasInFlight = r.data.some((d) => !TERMINAL.has(d.status));
          if (hasInFlight) timer = setTimeout(fetch, POLL_INTERVAL);
        })
        .catch(() => {
          setError("Could not load documents. Are you logged in?");
          setLoading(false);
        });
    };

    fetch();
    return () => clearTimeout(timer);
  }, [isAuth]);

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-semibold">Documents</h1>
          {docs.some((d) => !TERMINAL.has(d.status)) && (
            <span className="flex items-center gap-1.5 text-xs text-yellow-600 animate-pulse">
              <span className="w-2 h-2 bg-yellow-400 rounded-full inline-block"></span>
              Processing…
            </span>
          )}
        </div>
        <a
          href="/upload"
          className="bg-brand-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-brand-700"
        >
          + Upload
        </a>
      </div>


      {loading && <p className="text-gray-500">Loading…</p>}
      {error && <p className="text-red-500">{error}</p>}

      {!loading && !error && docs.length === 0 && (
        <div className="text-center py-20 text-gray-400">
          <p className="text-lg">No documents yet.</p>
          <a href="/upload" className="text-brand-600 underline mt-2 inline-block">Upload your first document</a>
        </div>
      )}

      {docs.length > 0 && (
        <table className="w-full bg-white rounded-xl shadow-sm overflow-hidden text-sm">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              <th className="text-left px-4 py-3 font-medium text-gray-600">File</th>
              <th className="text-left px-4 py-3 font-medium text-gray-600">Status</th>
              <th className="text-left px-4 py-3 font-medium text-gray-600">Confidence</th>
              <th className="text-left px-4 py-3 font-medium text-gray-600">Verdict</th>
              <th className="text-left px-4 py-3 font-medium text-gray-600">Uploaded</th>
              <th className="px-4 py-3"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {docs.map((doc) => {
              const latest = doc.extractions[0];
              return (
                <tr key={doc.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3 font-medium truncate max-w-xs">{doc.original_filename}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${STATUS_BADGE[doc.status] ?? "bg-gray-100"}`}>
                      {doc.status}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-gray-600">
                    {latest ? `${(latest.overall_confidence * 100).toFixed(0)}%` : "—"}
                  </td>
                  <td className="px-4 py-3 text-gray-600">{latest?.verdict ?? "—"}</td>
                  <td className="px-4 py-3 text-gray-500">
                    {new Date(doc.uploaded_at).toLocaleDateString()}
                  </td>
                  <td className="px-4 py-3">
                    <a href={`/documents/${doc.id}`} className="text-brand-600 hover:underline text-xs">
                      View
                    </a>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
