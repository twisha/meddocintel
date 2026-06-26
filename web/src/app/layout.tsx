import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MedDocIntel",
  description: "Clinical document intelligence platform",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-gray-50 text-gray-900 antialiased">
        <nav className="bg-white border-b border-gray-200 px-6 py-3 flex items-center gap-6">
          <span className="font-bold text-brand-600 text-lg">MedDocIntel</span>
          <a href="/" className="text-sm text-gray-600 hover:text-brand-600">Dashboard</a>
          <a href="/upload" className="text-sm text-gray-600 hover:text-brand-600">Upload</a>
          <a href="/review" className="text-sm text-gray-600 hover:text-brand-600">Review Queue</a>
          <div className="ml-auto">
            <a href="/login" className="text-sm text-gray-600 hover:text-brand-600">Login</a>
          </div>
        </nav>
        <main className="max-w-6xl mx-auto px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
