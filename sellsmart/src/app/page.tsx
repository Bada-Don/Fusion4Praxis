"use client";
import React from "react";
import { BackgroundBeams } from "@/components/ui/background-beams";
import { BentoGrid, BentoGridItem } from "@/components/ui/bento-grid";
import { StickyScroll } from "@/components/ui/sticky-scroll-reveal";
import { BackgroundGradient } from "@/components/ui/background-gradient";
import {
  ShieldCheck,
  BarChart3,
  BrainCircuit,
  ChevronDown
} from "lucide-react";

// --- Section 1: Hero ---
function Hero() {
  const scrollToSimulator = () => {
    document.getElementById("simulator")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="h-[40rem] w-full rounded-md bg-slate-950 relative flex flex-col items-center justify-center antialiased">
      <div className="max-w-2xl mx-auto p-4 relative z-10 text-center">
        <h1 className="relative z-10 text-lg md:text-7xl  bg-clip-text text-transparent bg-gradient-to-b from-neutral-200 to-neutral-600  text-center font-sans font-bold">
          Retail Pricing, <br /> Anchored in Reality.
        </h1>
        <p className="text-neutral-500 max-w-lg mx-auto my-2 text-sm text-center relative z-10">
          The first AI pricing engine with enforced Economic Guardrails.
          No hallucinations, just profit.
        </p>
        <button
          onClick={scrollToSimulator}
          className="mt-8 px-6 py-3 rounded-full bg-neutral-900 border border-neutral-800 text-neutral-300 hover:bg-neutral-800 transition flex items-center gap-2 mx-auto"
        >
          Scroll to Simulate <ChevronDown size={16} />
        </button>
      </div>
      <BackgroundBeams className="opacity-50" />
    </div>
  );
}

// --- Section 2: Logic Layer (Bento) ---
function LogicLayer() {
  const items = [
    {
      title: "Monotonic Constraints",
      description: "We forced the AI to learn that Price Up = Demand Down. 0% Hallucination rate ensuring economic viability.",
      header: <div className="flex flex-1 w-full h-full min-h-[6rem] rounded-xl bg-gradient-to-br from-neutral-900 to-neutral-800" />,
      icon: <ShieldCheck className="h-4 w-4 text-neutral-500" />,
    },
    {
      title: "SEO Impact Visibility",
      description: "Quantifying exactly how Description Length and Keyword Density drives conversion and effective pricing.",
      header: <div className="flex flex-1 w-full h-full min-h-[6rem] rounded-xl bg-gradient-to-br from-neutral-900 to-neutral-800" />,
      icon: <BarChart3 className="h-4 w-4 text-neutral-500" />,
    },
    {
      title: "Real-time Brand Health",
      description: "NLP-driven sentiment analysis integrated directly into pricing elasticity models for reputation safety.",
      header: <div className="flex flex-1 w-full h-full min-h-[6rem] rounded-xl bg-gradient-to-br from-neutral-900 to-neutral-800" />,
      icon: <BrainCircuit className="h-4 w-4 text-neutral-500" />,
    },
  ];

  return (
    <div className="py-20 bg-slate-950">
      <div className="max-w-4xl mx-auto px-4">
        <h2 className="text-3xl font-bold text-center mb-12 text-white">
          The <span className="text-emerald-500">Logic</span> Layer
        </h2>
        <BentoGrid>
          {items.map((item, i) => (
            <BentoGridItem
              key={i}
              title={item.title}
              description={item.description}
              header={item.header}
              icon={item.icon}
              className={i === 3 || i === 6 ? "md:col-span-2" : ""}
            />
          ))}
        </BentoGrid>
      </div>
    </div>
  );
}

// --- Section 3: Metrics (Sticky Scroll) ---
function ValidationMetrics() {
  const content = [
    {
      title: "1.81 RMSE Accuracy",
      description:
        "Our model achieves a Root Mean Squared Error of 1.81 on the validation set, outperforming standard linear regression baselines by 40%. This ensures that our demand forecasts are tight and reliable.",
      content: (
        <div className="h-full w-full bg-[linear-gradient(to_bottom_right,var(--cyan-500),var(--emerald-500))] flex items-center justify-center text-white text-4xl font-bold">
          1.81 RMSE
        </div>
      ),
    },
    {
      title: "0 Logic Violations",
      description:
        "Thanks to XGBoost's monotonic constraints, we have eliminated 'economic hallucinations'. The model never predicts that raising prices will magically increase demand for standard goods.",
      content: (
        <div className="h-full w-full bg-[linear-gradient(to_bottom_right,var(--orange-500),var(--yellow-500))] flex items-center justify-center text-white text-4xl font-bold">
          0 Violations
        </div>
      ),
    },
    {
      title: "Unbiased Estimator",
      description:
        "Residual analysis confirms that our error distribution is centered at zero, meaning the model is not systematically overestimating or underestimating demand across different categories.",
      content: (
        <div className="h-full w-full bg-[linear-gradient(to_bottom_right,var(--purple-500),var(--pink-500))] flex items-center justify-center text-white text-4xl font-bold">
          Unbiased
        </div>
      ),
    },
  ];

  return (
    <div className="bg-slate-950 py-10">
      <h2 className="text-3xl font-bold text-center mb-10 text-white">
        Validated <span className="text-purple-500">Performance</span>
      </h2>
      <StickyScroll content={content} />
    </div>
  );
}

// --- Section 4: Simulator ---
function InteractiveSimulator() {
  return (
    <div id="simulator" className="py-20 bg-slate-950 min-h-screen flex flex-col items-center justify-center">
      <div className="text-center mb-10 max-w-2xl px-4">
        <h2 className="text-3xl md:text-5xl font-bold text-white mb-4">
          Interactive <span className="text-cyan-500">Simulator</span>
        </h2>
        <p className="text-neutral-400">
          Experience the power of constrained optimization. Adjust prices and see the
          demand curve react in real-time, enforcing economic logic.
        </p>
      </div>

      <div className="w-full max-w-7xl px-4 h-[85vh]">
        <BackgroundGradient containerClassName="h-full w-full p-1" className="h-full w-full bg-slate-900 rounded-[22px] overflow-hidden">
          <iframe
            src="https://ml-streamlit-1tma.onrender.com/?embed=true"
            width="100%"
            height="100%"
            frameBorder="0"
            className="w-full h-full rounded-[20px]"
            title="Pricing Simulator"
          />
        </BackgroundGradient>
      </div>
    </div>
  );
}

// --- Main Page ---
export default function Home() {
  return (
    <main className="bg-slate-950 min-h-screen antialiased selection:bg-cyan-500/30">
      <Hero />
      <LogicLayer />
      <ValidationMetrics />
      <InteractiveSimulator />
    </main>
  );
}
