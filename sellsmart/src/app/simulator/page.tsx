"use client";

import NavbarDemo from "@/components/resizable-navbar-demo";

export default function SimulatorPage() {
  return (
    <main className="relative min-h-screen w-full bg-black overflow-hidden">
      {/* Navbar overlay */}
      <NavbarDemo />
      
      {/* Full-screen iframe */}
      <div className="w-full h-screen pt-16">
        <iframe
          src="https://ml-streamlit-1tma.onrender.com/?embed=true"
          width="100%"
          height="100%"
          frameBorder="0"
          className="w-full h-full"
          title="Pricing Simulator"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        />
      </div>
    </main>
  );
}
