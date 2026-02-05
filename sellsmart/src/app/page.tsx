"use client";

import NavbarDemo from "@/components/resizable-navbar-demo";
import { Hero } from "@/components/ui/hero-1";
import { Highlighter } from "@/components/ui/highlighter";
import { TextReveal } from "@/components/ui/text-reveal";
import { InfiniteSlider } from "@/components/ui/infinite-slider";
import { BackgroundRippleEffect } from "@/components/ui/background-ripple-effect";
import { ScrollVelocityContainer, ScrollVelocityRow } from "@/components/ui/scroll-based-velocity";
import { BackgroundGradient } from "@/components/ui/background-gradient";
import Image from "next/image";
import { BentoGrid, BentoGridItem } from "@/components/ui/bento-grid"; // Kept from previous step
import { ShieldCheck, BarChart3, BrainCircuit } from "lucide-react";
import { StickyScroll } from "@/components/ui/sticky-scroll-reveal"; // Kept from previous step

function AboutSection() {
  return (
    <section id="features" className="relative py-20 bg-black overflow-hidden">
      {/* Background Ripple Effect */}
      <div className="absolute inset-0 w-full h-full">
        <BackgroundRippleEffect rows={20} cols={30} cellSize={50} />
      </div>

      {/* Gradient overlay */}
      <div className="absolute inset-0 z-[4] pointer-events-none bg-gradient-to-b from-black via-transparent to-black" />

      {/* Scroll Velocity Text */}
      <div className="relative z-10 flex w-full flex-col items-center justify-center overflow-hidden mb-16">
        <ScrollVelocityContainer className="text-2xl font-bold tracking-[-0.02em] md:text-4xl md:leading-[3rem] text-white">
          <ScrollVelocityRow baseVelocity={3} direction={1}>
            Monotonic Constraints • XGBoost • Price Optimization •
          </ScrollVelocityRow>
          <ScrollVelocityRow baseVelocity={3} direction={-1}>
            Logic Guardrails • No Hallucations • Profit Maximization •
          </ScrollVelocityRow>
        </ScrollVelocityContainer>
        <div className="from-background pointer-events-none absolute inset-y-0 left-0 w-1/4 bg-gradient-to-r"></div>
        <div className="from-background pointer-events-none absolute inset-y-0 right-0 w-1/4 bg-gradient-to-l"></div>
      </div>

      {/* Text Reveal Animation */}
      <div className="relative z-10 container mx-auto px-6 md:px-8">
        <TextReveal>
          SellSmart bridges the gap between data science and retail strategy.
          Our constrained XGBoost engine prevents "economic hallucinations"
          by enforcing logical pricing rules, ensuring every recommendation
          is profitable and realistic.
        </TextReveal>
      </div>

      {/* Bento Grid Feature Layer (Mixed in) */}
      <div className="relative z-10 max-w-4xl mx-auto px-4 mt-20">
        <BentoGrid>
          <BentoGridItem
            title="Monotonic Constraints"
            description="Price Up = Demand Down. We force the AI to respect basic economic laws."
            header={<div className="flex flex-1 w-full h-full min-h-[6rem] rounded-xl bg-gradient-to-br from-neutral-900 to-neutral-800" />}
            icon={<ShieldCheck className="h-4 w-4 text-neutral-500" />}
            className="md:col-span-1"
          />
          <BentoGridItem
            title="SEO Visibility"
            description="Quantify how description length dictates conversion rates."
            header={<div className="flex flex-1 w-full h-full min-h-[6rem] rounded-xl bg-gradient-to-br from-neutral-900 to-neutral-800" />}
            icon={<BarChart3 className="h-4 w-4 text-neutral-500" />}
            className="md:col-span-1"
          />
          <BentoGridItem
            title="Sentiment Analysis"
            description="Real-time brand health monitoring integrated into pricing logic."
            header={<div className="flex flex-1 w-full h-full min-h-[6rem] rounded-xl bg-gradient-to-br from-neutral-900 to-neutral-800" />}
            icon={<BrainCircuit className="h-4 w-4 text-neutral-500" />}
            className="md:col-span-1"
          />
        </BentoGrid>
      </div>
    </section>
  );
}

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
    <div id="metrics" className="bg-black py-20">
      <h2 className="text-3xl font-bold text-center mb-10 text-white">
        Validated <Highlighter color="#a855f7" isView>Performance</Highlighter>
      </h2>
      <div className="p-10">
        <StickyScroll content={content} />
      </div>
    </div>
  )
}


function SimulatorSection() {
  return (
    <div id="simulator" className="py-20 bg-neutral-950 min-h-screen flex flex-col items-center justify-center relative overflow-hidden">
      <div className="absolute inset-0 w-full h-full bg-black/50 z-0" />

      <div className="text-center mb-10 max-w-2xl px-4 relative z-10">
        <h2 className="text-3xl md:text-5xl font-bold text-white mb-4">
          Interactive <Highlighter color="#06b6d4" isView>Simulator</Highlighter>
        </h2>
        <p className="text-neutral-400">
          Experience the power of constrained optimization. Adjust prices and see the
          demand curve react in real-time.
        </p>
      </div>

      <div className="w-full max-w-7xl px-4 h-[85vh] relative z-10">
        <BackgroundGradient containerClassName="h-full w-full p-1" className="h-full w-full bg-neutral-900 rounded-[22px] overflow-hidden">
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

export default function Home() {
  const scrollToSimulator = () => {
    document.getElementById("simulator")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <main className="relative min-h-screen bg-black">
      <NavbarDemo />
      <Hero
        title={
          <>
            Retail Pricing,{" "}
            <Highlighter action="highlight" color="#404040" isView>
              Anchored
            </Highlighter>{" "}
            in Reality
          </>
        }
        subtitle="The first AI pricing engine with enforced Economic Guardrails. No hallucinations, just profit."
        eyebrow="AI-Powered Optimization"
        ctaLabel="Launch Simulator"
        onCtaClick={scrollToSimulator}
      />
      <AboutSection />
      <ValidationMetrics />
      <SimulatorSection />
    </main>
  );
}
