"use client";

import { useState } from "react";
import NavbarDemo from "@/components/resizable-navbar-demo";

export default function SimulatorPage() {
  const [loaded, setLoaded] = useState(false);

  return (
    <main className="relative min-h-screen w-full bg-black overflow-hidden">
      {/* Navbar overlay */}
      <NavbarDemo />
      
      {/* Full-screen iframe with loading state */}
      <div className="w-full h-screen pt-16 relative">
        {!loaded && (
          <div className="absolute inset-0 bg-slate-900 flex flex-col items-center justify-center z-10">
            <div className="w-10 h-10 border-4 border-slate-600 border-t-blue-500 rounded-full animate-spin" />
            <p className="mt-4 text-slate-400 text-sm">Loading simulator...</p>
          </div>
        )}
        <iframe
          src="https://ml-streamlit-1tma.onrender.com/?embed=true&embed_options=dark_theme"
          width="100%"
          height="100%"
          frameBorder="0"
          className="w-full h-full"
          title="Pricing Simulator"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          onLoad={() => setLoaded(true)}
        />
      </div>
    </main>
  );
}
