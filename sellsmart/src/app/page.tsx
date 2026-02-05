"use client";
import { HeroHighlight, Highlight } from "@/components/ui/hero-highlight";
import { BentoGrid, BentoGridItem } from "@/components/ui/bento-grid";
import { StickyScroll } from "@/components/ui/sticky-scroll-reveal";
import { Simulator } from "@/components/Dashboard";
import { IconChartLine, IconCurrencyDollar, IconEye, IconBrain } from "@tabler/icons-react";
import { motion } from "framer-motion";

export default function Home() {
  return (
    <main className="min-h-screen bg-black antialiased selection:bg-cyan-500 selection:text-black">

      {/* 1. HERO SECTION */}
      <HeroHighlight>
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: [20, -5, 0] }}
          transition={{ duration: 0.5, ease: [0.4, 0.0, 0.2, 1] }}
          className="text-4xl px-4 md:text-5xl lg:text-6xl font-bold text-white max-w-4xl leading-relaxed lg:leading-snug text-center mx-auto"
        >
          Stop Guessing. <br />
          <Highlight className="text-white">Start Optimizing.</Highlight>
        </motion.h1>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5, duration: 0.8 }}
          className="text-neutral-400 text-center mt-4 max-w-xl mx-auto text-lg"
        >
          The first Retail AI that respects Economic Laws. <br />
          Optimized for Profitability, anchored in Reality.
        </motion.p>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="flex justify-center mt-10"
        >
          <button
            onClick={() => document.getElementById('simulator')?.scrollIntoView({ behavior: 'smooth' })}
            className="px-8 py-3 rounded-full bg-cyan-500 text-black font-bold text-lg hover:bg-cyan-400 transition shadow-[0_0_20px_rgba(6,182,212,0.5)]"
          >
            Launch Simulator
          </button>
        </motion.div>
      </HeroHighlight>

      {/* 2. BENTO GRID (PROBLEM vs SOLUTION) */}
      <section className="py-20 px-4 bg-neutral-950">
        <div className="max-w-4xl mx-auto mb-10 text-center">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Why Current AI Fails Retail</h2>
          <p className="text-neutral-500">Standard models hallucinate demand. We guarantee logic.</p>
        </div>
        <BentoGrid className="max-w-4xl mx-auto">
          <BentoGridItem
            title="Blind Pricing"
            description="Retailers hike prices ignoring elasticity, leading to revenue collapse."
            header={<div className="flex flex-1 w-full h-full min-h-[6rem] rounded-xl bg-gradient-to-br from-red-500/20 to-neutral-900 border border-red-500/10" />}
            icon={<IconCurrencyDollar className="h-6 w-6 text-red-500" />}
          />
          <BentoGridItem
            title="Monotonic Constraints"
            description="Our model enforces economic laws: Price UP = Demand DOWN. Zero Hallucinations."
            header={<div className="flex flex-1 w-full h-full min-h-[6rem] rounded-xl bg-gradient-to-br from-green-500/20 to-neutral-900 border border-green-500/10" />}
            icon={<IconChartLine className="h-6 w-6 text-green-500" />}
          />
          <BentoGridItem
            title="SEO Visibility"
            description="Quantifiable impact of Description Length on Sales Volume."
            header={<div className="flex flex-1 w-full h-full min-h-[6rem] rounded-xl bg-gradient-to-br from-purple-500/20 to-neutral-900 border border-purple-500/10" />}
            icon={<IconEye className="h-6 w-6 text-purple-500" />}
          />
        </BentoGrid>
      </section>

      {/* 3. STICKY SCROLL (MODEL INTELLIGENCE) */}
      <section className="py-20 bg-black">
        <div className="max-w-7xl mx-auto px-4">
          <h2 className="text-3xl md:text-5xl font-bold text-white mb-10 text-center">Model Validation</h2>
          <StickyScroll content={stickyContent} />
        </div>
      </section>

      {/* 4. SIMULATOR */}
      <section id="simulator" className="py-20 bg-neutral-950 min-h-screen flex flex-col items-center">
        <div className="text-center mb-10">
          <h2 className="text-4xl md:text-6xl font-bold text-white mb-4 bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-purple-500">
            Interactive Simulator
          </h2>
          <p className="text-neutral-400 max-w-2xl mx-auto">
            Adjust Price, SEO, and Sentiment to see real-time demand forecasts.
            Powered by our Monotonic XGBoost Model.
          </p>
        </div>

        <Simulator />
      </section>

      {/* FOOTER */}
      <footer className="py-10 bg-black border-t border-white/10 text-center text-neutral-600">
        <p>Built for Fusion4Praxis Hackathon • 2026</p>
      </footer>
    </main>
  );
}

const stickyContent = [
  {
    title: "RMSE: 1.81",
    description:
      "Our model achieves a Root Mean Square Error of 1.81 on the holdout set, ensuring high-fidelity demand prediction without overfitting.",
    content: (
      <div className="h-full w-full bg-[linear-gradient(to_bottom_right,var(--cyan-500),var(--emerald-500))] flex items-center justify-center text-white">
        <IconChartLine className="h-20 w-20" />
      </div>
    ),
  },
  {
    title: "Zero Violations",
    description:
      "Tested against 10,000+ synthetic scenarios. The model strictly adheres to Monotonic Constraints. No positive price elasticity anomalies.",
    content: (
      <div className="h-full w-full bg-[linear-gradient(to_bottom_right,var(--orange-500),var(--yellow-500))] flex items-center justify-center text-white">
        <div className="text-6xl font-black">0%</div>
      </div>
    ),
  },
  {
    title: "Unbiased Residuals",
    description:
      "Residual analysis confirms error distribution is centered at zero, meaning our pricing recommendations are not systematically skewed high or low.",
    content: (
      <div className="h-full w-full bg-[linear-gradient(to_bottom_right,var(--pink-500),var(--indigo-500))] flex items-center justify-center text-white">
        <IconBrain className="h-20 w-20" />
      </div>
    ),
  },
];
