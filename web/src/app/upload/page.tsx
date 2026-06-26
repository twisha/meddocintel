"use client";

import { useState, useCallback } from "react";
import { documents } from "@/lib/api";

export default function UploadPage() {
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<"idle" | "uploading" | "done" | "error">("idle");
  const [docId, setDocId] = useState("");
  const [error, setError] = useState("");

  const handleFile = (f: File) => setFile(f);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, []);

  const upload = async () => {
    if (!file) return;
    setStatus("uploading");
    setError("");
    try {
      const res = await documents.upload(file);
      setDocId(res.data.document_id);
      setStatus("done");
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Upload failed";
      setError(msg);
      setStatus("error");
    }
  };

  return (
    <div className="max-w-xl mx-auto">
      <h1 className="text-2xl font-semibold mb-6">Upload Document</h1>

      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors ${
          dragging ? "border-brand-500 bg-brand-50" : "border-gray-300 hover:border-brand-400"
        }`}
        onClick={() => document.getElementById("file-input")?.click()}
      >
        <input
          id="file-input"
          type="file"
          className="hidden"
          accept=".pdf,.png,.jpg,.jpeg,.tiff,.txt"
          onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        />
        {file ? (
          <div>
            <p className="font-medium text-gray-800">{file.name}</p>
            <p className="text-sm text-gray-500 mt-1">{(file.size / 1024).toFixed(1)} KB</p>
          </div>
        ) : (
          <div>
            <p className="text-gray-500">Drag & drop or click to select</p>
            <p className="text-xs text-gray-400 mt-1">PDF, PNG, JPG, TIFF, TXT</p>
          </div>
        )}
      </div>

      {file && status === "idle" && (
        <button
          onClick={upload}
          className="mt-4 w-full bg-brand-600 text-white py-2.5 rounded-lg hover:bg-brand-700 font-medium"
        >
          Upload &amp; Process
        </button>
      )}

      {status === "uploading" && (
        <div className="mt-4 text-center text-gray-500 animate-pulse">Uploading…</div>
      )}

      {status === "done" && (
        <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
          <p className="text-green-700 font-medium">Document queued for processing</p>
          <p className="text-sm text-gray-500 mt-1">ID: {docId}</p>
          <a href="/" className="mt-2 inline-block text-brand-600 text-sm underline">
            Back to dashboard
          </a>
        </div>
      )}

      {status === "error" && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          {error}
        </div>
      )}
    </div>
  );
}
