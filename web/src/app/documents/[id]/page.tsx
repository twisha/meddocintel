"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { documents, type Document, type ExtractionDetail } from "@/lib/api";

const VERDICT_COLOR: Record<string, string> = {
  ACCEPT: "text-green-600 bg-green-50 border-green-200",
  FLAG:   "text-orange-600 bg-orange-50 border-orange-200",
  REJECT: "text-red-600 bg-red-50 border-red-200",
};

const STATUS_BADGE: Record<string, string> = {
  pending:    "bg-gray-100 text-gray-700",
  processing: "bg-yellow-100 text-yellow-700",
  extracted:  "bg-blue-100 text-blue-700",
  verified:   "bg-green-100 text-green-700",
  flagged:    "bg-orange-100 text-orange-700",
  rejected:   "bg-red-100 text-red-700",
  failed:     "bg-red-100 text-red-700",
};

function Field({ label, value, confidence, span }: {
  label: string;
  value: unknown;
  confidence?: number;
  span?: [number, number] | null;
}) {
  if (value === null || value === undefined) return null;
  const conf = confidence ?? 0;
  const confColor = conf > 0.85 ? "text-green-600" : conf > 0.7 ? "text-orange-500" : "text-red-500";
  return (
    <div className="flex items-start gap-2 py-1.5 border-b border-gray-100 last:border-0">
      <span className="w-44 shrink-0 text-xs text-gray-500 font-medium pt-0.5">{label}</span>
      <span className="text-sm text-gray-900 flex-1">{String(value)}</span>
      {confidence !== undefined && (
        <span className={`text-xs font-mono ${confColor}`}>{(conf * 100).toFixed(0)}%</span>
      )}
      {span && (
        <span className="text-xs text-gray-300 font-mono">[{span[0]}–{span[1]}]</span>
      )}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 mb-4">
      <h3 className="text-sm font-semibold text-gray-700 mb-3 uppercase tracking-wide">{title}</h3>
      {children}
    </div>
  );
}

export default function DocumentDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [doc, setDoc] = useState<Document | null>(null);
  const [extraction, setExtraction] = useState<ExtractionDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([documents.get(id), documents.getExtraction(id)])
      .then(([docRes, extRes]) => {
        setDoc(docRes.data);
        setExtraction(extRes.data);
      })
      .catch(() => setError("Could not load document. Are you logged in?"))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) return <p className="text-gray-400">Loading…</p>;
  if (error)   return <p className="text-red-500">{error}</p>;
  if (!doc)    return null;

  const data = extraction?.data as Record<string, unknown> | undefined;
  const patient = data?.patient as Record<string, { value: unknown; confidence: number; source_span: [number,number] | null }> | undefined;
  const visit   = data?.visit   as typeof patient;
  const vitals  = data?.vitals  as typeof patient;
  const meds    = data?.medications as Array<Record<string, { value: unknown; confidence: number }>> | undefined;
  const diagnoses = data?.diagnoses as Array<Record<string, { value: unknown; confidence: number }>> | undefined;
  const ap      = data?.assessment_plan as Record<string, { value: unknown; confidence: number }> | undefined;
  const ver     = extraction?.verification;

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <a href="/" className="text-sm text-gray-400 hover:text-brand-600">← Dashboard</a>
          <h1 className="text-xl font-semibold mt-1 truncate max-w-xl">{doc.original_filename}</h1>
          <div className="flex items-center gap-3 mt-2 text-sm text-gray-500">
            <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${STATUS_BADGE[doc.status] ?? "bg-gray-100"}`}>
              {doc.status}
            </span>
            {doc.ocr_engine && <span>OCR: {doc.ocr_engine} ({((doc.ocr_confidence ?? 0) * 100).toFixed(0)}%)</span>}
            <span>{new Date(doc.uploaded_at).toLocaleString()}</span>
          </div>
        </div>

        {extraction && (
          <div className="text-right">
            <p className="text-2xl font-bold text-gray-800">{(extraction.overall_confidence * 100).toFixed(0)}%</p>
            <p className="text-xs text-gray-400">confidence</p>
          </div>
        )}
      </div>

      {/* Verification banner */}
      {ver && (
        <div className={`rounded-xl border p-4 mb-4 flex items-start gap-4 ${VERDICT_COLOR[ver.verdict] ?? "bg-gray-50 border-gray-200"}`}>
          <div>
            <p className="font-bold text-lg">{ver.verdict}</p>
            <p className="text-sm">Verification score: {(ver.overall_score * 100).toFixed(0)}%</p>
          </div>
          {ver.rule_flags.length > 0 && (
            <div className="ml-4">
              <p className="text-xs font-semibold mb-1">Rule flags:</p>
              <ul className="text-xs space-y-0.5">
                {ver.rule_flags.map((f, i) => <li key={i}>• {f}</li>)}
              </ul>
            </div>
          )}
          {Object.keys(ver.field_scores).length > 0 && (
            <div className="ml-auto text-xs space-y-0.5">
              {Object.entries(ver.field_scores).map(([k, v]) => (
                <div key={k} className="flex gap-2">
                  <span className="text-gray-500 capitalize">{k}:</span>
                  <span className="font-mono">{(Number(v) * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {!extraction && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4 mb-4 text-yellow-700 text-sm">
          Extraction not ready yet — document may still be processing. Refresh in a moment.
        </div>
      )}

      {/* Patient */}
      {patient && (
        <Section title="Patient">
          <Field label="Name"         value={patient.name?.value}         confidence={patient.name?.confidence}         span={patient.name?.source_span} />
          <Field label="Date of Birth"value={patient.dob?.value}          confidence={patient.dob?.confidence}          span={patient.dob?.source_span} />
          <Field label="MRN"          value={patient.mrn?.value}          confidence={patient.mrn?.confidence}          span={patient.mrn?.source_span} />
          <Field label="Gender"       value={patient.gender?.value}       confidence={patient.gender?.confidence} />
          <Field label="Insurance ID" value={patient.insurance_id?.value} confidence={patient.insurance_id?.confidence} />
        </Section>
      )}

      {/* Visit */}
      {visit && (
        <Section title="Visit">
          <Field label="Visit Date"      value={visit.visit_date?.value}      confidence={visit.visit_date?.confidence}      span={visit.visit_date?.source_span} />
          <Field label="Provider"        value={visit.provider_name?.value}   confidence={visit.provider_name?.confidence}   span={visit.provider_name?.source_span} />
          <Field label="Facility"        value={visit.facility_name?.value}   confidence={visit.facility_name?.confidence} />
          <Field label="Visit Type"      value={visit.visit_type?.value}      confidence={visit.visit_type?.confidence} />
          <Field label="Chief Complaint" value={visit.chief_complaint?.value} confidence={visit.chief_complaint?.confidence} />
        </Section>
      )}

      {/* Vitals */}
      {vitals && (
        <Section title="Vital Signs">
          {Boolean(vitals.blood_pressure_systolic?.value) && (
            <Field label="Blood Pressure"
              value={`${vitals.blood_pressure_systolic.value}/${vitals.blood_pressure_diastolic?.value} mmHg`}
              confidence={vitals.blood_pressure_systolic.confidence} />
          )}
          <Field label="Heart Rate"        value={vitals.heart_rate?.value        ? `${vitals.heart_rate.value} bpm`        : null} confidence={vitals.heart_rate?.confidence} />
          <Field label="Respiratory Rate"  value={vitals.respiratory_rate?.value  ? `${vitals.respiratory_rate.value} /min`  : null} confidence={vitals.respiratory_rate?.confidence} />
          <Field label="Temperature"       value={vitals.temperature?.value       ? `${vitals.temperature.value} °F`         : null} confidence={vitals.temperature?.confidence} />
          <Field label="Weight"            value={vitals.weight?.value            ? `${vitals.weight.value} lbs`             : null} confidence={vitals.weight?.confidence} />
          <Field label="Height"            value={vitals.height?.value            ? `${vitals.height.value} in`              : null} confidence={vitals.height?.confidence} />
          <Field label="BMI"               value={vitals.bmi?.value}              confidence={vitals.bmi?.confidence} />
          <Field label="O₂ Saturation"     value={vitals.oxygen_saturation?.value ? `${vitals.oxygen_saturation.value}%`    : null} confidence={vitals.oxygen_saturation?.confidence} />
        </Section>
      )}

      {/* Medications */}
      {meds && meds.length > 0 && (
        <Section title={`Medications (${meds.length})`}>
          <div className="space-y-3">
            {meds.map((m, i) => (
              <div key={i} className="bg-gray-50 rounded-lg px-4 py-2">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-sm">{String(m.name?.value ?? "")}</span>
                  <span className="text-sm text-gray-600">{String(m.dose?.value ?? "")} {String(m.route?.value ?? "")} {String(m.frequency?.value ?? "")}</span>
                  <span className={`ml-auto text-xs font-mono ${(m.name?.confidence ?? 0) > 0.85 ? "text-green-600" : "text-orange-500"}`}>
                    {((m.name?.confidence ?? 0) * 100).toFixed(0)}%
                  </span>
                </div>
                {Boolean(m.indication?.value) && (
                  <p className="text-xs text-gray-400 mt-0.5">For: {String(m.indication.value)}</p>
                )}
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Diagnoses */}
      {diagnoses && diagnoses.length > 0 && (
        <Section title={`Diagnoses (${diagnoses.length})`}>
          <div className="space-y-2">
            {diagnoses.map((d, i) => (
              <div key={i} className="flex items-center gap-3">
                {Boolean(d.icd10_code?.value) && (
                  <span className="px-2 py-0.5 bg-blue-50 text-blue-700 rounded text-xs font-mono font-bold">
                    {String(d.icd10_code.value)}
                  </span>
                )}
                <span className="text-sm">{String(d.description?.value ?? "")}</span>
                {Boolean(d.status?.value) && (
                  <span className="text-xs text-gray-400">{String(d.status.value)}</span>
                )}
                <span className={`ml-auto text-xs font-mono ${(d.description?.confidence ?? 0) > 0.85 ? "text-green-600" : "text-orange-500"}`}>
                  {((d.description?.confidence ?? 0) * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Assessment & Plan */}
      {ap && (
        <Section title="Assessment & Plan">
          <Field label="Assessment" value={ap.assessment?.value} confidence={ap.assessment?.confidence} />
          <Field label="Plan"       value={ap.plan?.value}       confidence={ap.plan?.confidence} />
        </Section>
      )}
    </div>
  );
}
