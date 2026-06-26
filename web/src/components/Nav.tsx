"use client";

import { useEffect, useState } from "react";
import { auth } from "@/lib/api";

export default function Nav() {
  const [loggedIn, setLoggedIn] = useState(false);

  useEffect(() => {
    setLoggedIn(auth.isAuthenticated());
  }, []);

  const handleLogout = () => {
    auth.logout();
    window.location.href = "/login";
  };

  return (
    <nav className="bg-white border-b border-gray-200 px-6 py-3 flex items-center gap-6">
      <a href="/" className="font-bold text-brand-600 text-lg">MedDocIntel</a>
      {loggedIn && (
        <>
          <a href="/"       className="text-sm text-gray-600 hover:text-brand-600">Dashboard</a>
          <a href="/upload" className="text-sm text-gray-600 hover:text-brand-600">Upload</a>
          <a href="/review" className="text-sm text-gray-600 hover:text-brand-600">Review Queue</a>
        </>
      )}
      <div className="ml-auto">
        {loggedIn ? (
          <button onClick={handleLogout} className="text-sm text-gray-600 hover:text-red-500">
            Logout
          </button>
        ) : (
          <a href="/login" className="text-sm text-gray-600 hover:text-brand-600">Login</a>
        )}
      </div>
    </nav>
  );
}
