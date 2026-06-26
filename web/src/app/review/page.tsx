"use client";

import { useEffect, useState } from "react";
import { reviewQueue, documents, type Document, type ExtractionDetail } from "@/lib/api";
import { useAuthGuard } from "@/lib/useAuth";

export default function ReviewQueuePage() {
  useAuthGuard();
  const [docs, setDocs] = useState<Document[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [extraction, setExtraction] = useState<ExtractionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [extractionLoading, setExtractionLoading] = useState(false);

  useEffect(() => {
    reviewQueue.list()
      .then((r) => setDocs(r.data))
      .finally(() => setLoading(false));
  }, []);

  const loadExtraction = async (docId: string) => {
    setSelected(docId);
    setExtractionLoading(true);
    try {
      const res = await documents.getExtraction(docId);
      setExtraction(res.data);
    } finally {
      setExtractionLoading(false);
    }
  };

  const VERDICT_COLOR: Record<string, string> = {
    ACCEPT: "text-green-600",
    FLAG: "text-orange-600",
    REJECT: "text-red-600",
  };

  return (
    <div>
      <h1 className="text-2xl font-semibold mb-6">Review Queue</h1>
      <p className="text-sm text-gray-500 mb-4">
        Documents flagged or rejected by the verification layer. Click a row to inspect.
      </p>

      {loading && <p className="text-gray-400">Loading…</p>}

      {!loading && docs.length === 0 && (
        <div className="text-center py-20 text-gray-400">
          <p>No documents need review.</p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: list */}
        <div className="space-y-2">
          {docs.map((doc) => {
            const latest = doc.extractions[0];
            return (
              <div
                key={doc.id}
                onClick={() => loadExtraction(doc.id)}
                className={`p-4 bg-white rounded-xl border cursor-pointer hover:border-brand-400 transition-colors ${
                  selected === doc.id ? "border-brand-500 shadow-sm" : "border-gray-200"
                }`}
              >
                <p className="font-medium text-sm truncate">{doc.original_filename}</p>
                <div className="flex items-center gap-3 mt-1 text-xs text-gray-500">
                  <span className={`font-semibold ${doc.status === "rejected" ? "text-red-600" : "text-orange-600"}`}>
                    {doc.status.toUpperCase()}
                  </span>
                  {latest && (
                    <>
                      <span>Confidence: {(latest.overall_confidence * 100).toFixed(0)}%</span>
                      {latest.verdict && (
                        <span className={VERDICT_COLOR[latest.verdict] ?? ""}>
                          Verdict: {latest.verdict}
                        </span>
                      )}
                    </>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Right: extraction detail */}
        {selected && (
          <div className="bg-white rounded-xl border border-gray-200 p-5 overflow-auto max-h-[70vh]">
            {extractionLoading && <p className="text-gray-400 text-sm">Loading extraction…</p>}
            {extraction && !extractionLoading && (
              <div className="space-y-4 text-sm">
                <div className="flex items-center gap-3">
                  <span className="font-semibold">Confidence:</span>
                  <span>{(extraction.overall_confidence * 100).toFixed(0)}% ({extraction.confidence_tier})</span>
                </div>

                {extraction.verification && (
                  <div className="p-3 bg-gray-50 rounded-lg">
                    <p className="font-semibold mb-2">Verification</p>
                    <p>
                      Verdict:{" "}
                      <span className={`font-bold ${VERDICT_COLOR[extraction.verification.verdict] ?? ""}`}>
                        {extraction.verification.verdict}
                      </span>
                    </p>
                    <p>Score: {(extraction.verification.overall_score * 100).toFixed(0)}%</p>
                    {extraction.verification.rule_flags.length > 0 && (
                      <div className="mt-2">
                        <p className="font-medium text-red-600">Rule flags:</p>
                        <ul className="list-disc list-inside text-red-500">
                          {extraction.verification.rule_flags.map((f, i) => (
                            <li key={i}>{f}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}

                <div>
                  <p className="font-semibold mb-2">Extracted Data</p>
                  <pre className="text-xs bg-gray-50 p-3 rounded overflow-auto max-h-80">
                    {JSON.stringify(extraction.data, null, 2)}
                  </pre>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
