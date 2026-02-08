"use client";

import NavbarDemo from "@/components/resizable-navbar-demo";
import { Highlighter } from "@/components/ui/highlighter";
import { MorphingText } from "@/components/ui/morphing-text";
import { EtheralShadow } from "@/components/ui/etheral-shadow";
import { PinContainer } from "@/components/ui/3d-pin";
import { FlipText } from "@/components/ui/flip-text";
import { SparklesText } from "@/components/ui/sparkles-text";
import { motion } from "framer-motion";
import { GlowCard } from "@/components/ui/spotlight-card";
import { ShieldCheck, Scale, Brain } from "lucide-react";

const features = [
  {
    icon: ShieldCheck,
    title: "Monotonic Guardrails",
    subtitle: "The USP",
    description: "We utilized XGBoost's monotone_constraints hyperparameter to enforce economic laws.",
    details: "As Price increases, the predicted Demand is forced to decrease. No \"Correlation Fallacy.\"",
    stat: "0% Violation Rate",
    glowColor: "blue" as const,
  },
  {
    icon: Scale,
    title: "Context-Aware Pricing",
    subtitle: "Relative Price Scaling",
    description: "A ₹500 price tag means nothing without context.",
    details: "We normalize price against the Sub-Category Average, enabling generalization across 200+ categories.",
    stat: "200+ Categories",
    glowColor: "purple" as const,
  },
  {
    icon: Brain,
    title: "Sentiment Analysis",
    subtitle: "NLP-Powered Insights",
    description: "\"Trust\" (Reviews) is a critical factor in purchase decisions.",
    details: "Polarity scores (-1 to +1) from review text weigh \"Brand Health\" impact on elasticity.",
    stat: "-1 to +1 Scale",
    glowColor: "green" as const,
  },
];

const techSpecs = [
  { label: "Algorithm", value: "XGBoost with Monotonic Constraints" },
  { label: "Objective", value: "reg:squarederror (RMSE)" },
  { label: "Target", value: "Log(Demand + 1)" },
];

const comparisonData = [
  {
    feature: "Algorithm",
    standard: "Linear Regression / Random Forest",
    praxis: "XGBoost with Monotonic Constraints",
  },
  {
    feature: "Logic",
    standard: "Black Box predictions",
    praxis: "Enforced economic laws",
  },
  {
    feature: "Features",
    standard: "Raw Price (₹)",
    praxis: "Relative Price Ratios",
  },
  {
    feature: "Output",
    standard: "Static Jupyter Notebook",
    praxis: "Deployed Simulator",
  },
];

export default function FeaturesPage() {
  return (
    <main className="relative min-h-screen bg-black">
      <NavbarDemo />
      
      {/* Hero Section with Etheral Shadow Background */}
      <section className="min-h-screen relative overflow-hidden">
        {/* Background Layer */}
        <div className="absolute inset-0 z-0">
          <EtheralShadow
            color="rgba(100, 100, 150, 0.6)"
            animation={{ scale: 50, speed: 60 }}
            noise={{ opacity: 0.3, scale: 1 }}
            sizing="fill"
          />
          {/* Gradient Overlay - transparent top to black bottom */}
          <div 
            className="absolute inset-0 pointer-events-none"
            style={{
              background: 'linear-gradient(to bottom, transparent 0%, transparent 50%, black 100%)'
            }}
          />
        </div>
        
        {/* Content Layer */}
        <div className="relative z-10 min-h-screen flex items-center justify-center px-6 pt-20">
          <div className="text-center max-w-4xl mx-auto">
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-sm uppercase tracking-widest text-gray-400 mb-4"
            >
              Economic Realism
            </motion.p>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="mb-6"
            >
              <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold text-white mb-2">
                The First Retail AI That
              </h1>
              <MorphingText 
                texts={["Obeys", "Respects", "Follows", "Honors"]} 
                className="h-12 md:h-20 lg:h-24 text-white"
              />
              <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold text-white mt-2">
                the Laws of Economics
              </h1>
            </motion.div>
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="text-lg md:text-xl text-gray-300 max-w-2xl mx-auto"
            >
              Most ML models hallucinate demand. Ours is mathematically constrained 
              to respect Price Elasticity. Zero hallucinations. 100% Business Logic.
            </motion.p>
          </div>
        </div>
      </section>

      {/* Features with GlowCards */}
      <section className="py-20 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-4 flex items-center justify-center gap-3">
              <span>Core</span>
              <SparklesText 
                text="Features" 
                className="text-3xl md:text-5xl"
                colors={{ first: "#6b7280", second: "#9ca3af" }}
                sparklesCount={8}
              />
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Built with economic principles at its core, not as an afterthought.
            </p>
          </div>

          <div className="flex flex-wrap justify-center gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <GlowCard 
                  glowColor={feature.glowColor}
                  size="lg"
                  className="flex flex-col"
                >
                  <div className="flex-1 flex flex-col">
                    <div className="flex items-center gap-3 mb-3">
                      <feature.icon className="w-6 h-6 text-white" />
                      <span className="text-xs uppercase tracking-wider text-gray-500">
                        {feature.subtitle}
                      </span>
                    </div>
                    <h3 className="text-xl font-bold text-white mb-2">{feature.title}</h3>
                    <p className="text-sm text-gray-400 mb-3">{feature.description}</p>
                    <p className="text-xs text-gray-500 flex-1">{feature.details}</p>
                  </div>
                  <div className="pt-4 border-t border-white/10">
                    <span className="text-lg font-bold text-white">{feature.stat}</span>
                  </div>
                </GlowCard>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Technical Specs */}
      <section className="py-32 px-6 bg-neutral-950 relative z-10">
        <div className="max-w-6xl mx-auto text-center">
          <h2 className="text-3xl md:text-5xl font-bold text-white mb-4 flex items-center justify-center gap-3">
            <FlipText 
              word="Technical" 
              className="text-3xl md:text-5xl font-bold text-white"
            />
            <FlipText 
              word="Specs" 
              className="text-3xl md:text-5xl font-bold text-white"
              delayMultiple={0.1}
            />
          </h2>
          <p className="text-gray-400 mb-20 max-w-2xl mx-auto">
            An XGBoost Regressor engineered to optimize Revenue, not just Accuracy.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-x-4 gap-y-48 place-items-center">
            {techSpecs.map((spec) => (
              <PinContainer
                key={spec.label}
                title={spec.label}
                containerClassName="w-full flex justify-center"
              >
                <div className="flex flex-col p-4 tracking-tight text-gray-100/80 w-[18rem] h-[14rem] bg-gradient-to-b from-neutral-800/60 to-neutral-900/40 backdrop-blur-sm border border-neutral-700/50 rounded-2xl">
                  <div className="flex items-center gap-2 mb-4">
                    <div className="size-2 rounded-full bg-gray-400 animate-pulse" />
                    <div className="text-xs text-gray-500">{spec.label}</div>
                  </div>
                  <div className="flex-1 flex flex-col justify-center">
                    <div className="text-xl font-bold text-white leading-tight">
                      {spec.value}
                    </div>
                  </div>
                  <div className="text-xs text-gray-600 mt-auto">
                    Optimized for production
                  </div>
                </div>
              </PinContainer>
            ))}
          </div>
        </div>
      </section>

      {/* Comparison Section */}
      <section className="py-20 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-4">
              Why <Highlighter color="#374151" isView>Praxis</Highlighter> Wins
            </h2>
          </div>

          <div className="overflow-hidden rounded-2xl border border-neutral-800">
            <table className="w-full">
              <thead>
                <tr className="bg-neutral-900">
                  <th className="text-left p-4 text-gray-400 font-medium">Feature</th>
                  <th className="text-left p-4 text-gray-400 font-medium">Standard</th>
                  <th className="text-left p-4 text-white font-medium">Praxis</th>
                </tr>
              </thead>
              <tbody>
                {comparisonData.map((row, index) => (
                  <tr 
                    key={row.feature}
                    className={index % 2 === 0 ? "bg-neutral-950" : "bg-neutral-900/50"}
                  >
                    <td className="p-4 text-white font-medium">{row.feature}</td>
                    <td className="p-4 text-gray-500">{row.standard}</td>
                    <td className="p-4 text-gray-300">{row.praxis}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Summary Quote */}
      <section className="py-20 px-6 bg-neutral-950">
        <div className="max-w-4xl mx-auto text-center">
          <blockquote className="text-xl md:text-2xl text-gray-300 leading-relaxed">
            <span className="text-white font-semibold">
              "We focused on Business Realism, not just R² scores.
            </span>{" "}
            A model that predicts a 50% price hike will double sales 
            is useless. We built a decision-support system."
          </blockquote>
        </div>
      </section>
    </main>
  );
}
