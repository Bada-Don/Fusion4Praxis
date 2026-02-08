"use client";

import NavbarDemo from "@/components/resizable-navbar-demo";
import { MorphingText } from "@/components/ui/morphing-text";
import { EtheralShadow } from "@/components/ui/etheral-shadow";
import { FlipText } from "@/components/ui/flip-text";
import { SparklesText } from "@/components/ui/sparkles-text";
import { GlowCard } from "@/components/ui/spotlight-card";
import { motion } from "framer-motion";
import { TrendingDown, Target, BarChart3, Zap, CheckCircle, XCircle } from "lucide-react";

const metrics = [
  {
    icon: BarChart3,
    label: "RMSE Score",
    value: "1.81",
    description: "Log Scale",
    detail: "Root Mean Squared Error measuring prediction accuracy on log-transformed demand",
    glowColor: "blue" as const,
  },
  {
    icon: Target,
    label: "Variance Explained",
    value: "~30%",
    description: "R² Approximation",
    detail: "Using only public listing data, without access to private ad-spend data",
    glowColor: "purple" as const,
  },
  {
    icon: TrendingDown,
    label: "Violation Rate",
    value: "0%",
    description: "Economic Law Compliance",
    detail: "100/100 products respected the Law of Demand in stress tests",
    glowColor: "green" as const,
  },
  {
    icon: Zap,
    label: "Categories",
    value: "200+",
    description: "Product Types",
    detail: "Generalizes across distinct product categories using relative price ratios",
    glowColor: "blue" as const,
  },
];

const performanceBreakdown = [
  {
    metric: "Target Variable",
    value: "Log(Demand + 1)",
    reason: "Sales data follows Power Law (Pareto Distribution). Log-transformation normalizes variance.",
  },
  {
    metric: "Objective Function",
    value: "reg:squarederror",
    reason: "Minimizes RMSE for regression accuracy on continuous demand predictions.",
  },
  {
    metric: "Algorithm",
    value: "Gradient Boosted Trees",
    reason: "XGBoost with monotonic constraints enforces economic realism.",
  },
  {
    metric: "Feature Scaling",
    value: "Relative Ratios",
    reason: "Price competitiveness normalized against sub-category average.",
  },
];

const comparisonMetrics = [
  {
    metric: "Accuracy Focus",
    standard: "Maximizes R² score only",
    praxis: "Balances accuracy with business realism",
    praxisWins: true,
  },
  {
    metric: "Prediction Validity",
    standard: "May predict price ↑ = demand ↑",
    praxis: "Guarantees price ↑ → demand ↓",
    praxisWins: true,
  },
  {
    metric: "Feature Engineering",
    standard: "Raw numerical values",
    praxis: "Context-aware relative ratios",
    praxisWins: true,
  },
  {
    metric: "Output Format",
    standard: "Static notebook metrics",
    praxis: "Interactive simulator",
    praxisWins: true,
  },
];

export default function MetricsPage() {
  return (
    <main className="relative min-h-screen bg-black">
      <NavbarDemo />
      
      {/* Hero Section */}
      <section className="min-h-screen relative overflow-hidden">
        <div className="absolute inset-0 z-0">
          <EtheralShadow
            color="rgba(80, 120, 100, 0.5)"
            animation={{ scale: 40, speed: 50 }}
            noise={{ opacity: 0.3, scale: 1 }}
            sizing="fill"
          />
          <div 
            className="absolute inset-0 pointer-events-none"
            style={{
              background: 'linear-gradient(to bottom, transparent 0%, transparent 50%, black 100%)'
            }}
          />
        </div>
        
        <div className="relative z-10 min-h-screen flex items-center justify-center px-6 pt-20">
          <div className="text-center max-w-4xl mx-auto">
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-sm uppercase tracking-widest text-gray-400 mb-4"
            >
              Performance Metrics
            </motion.p>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="mb-6"
            >
              <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold text-white mb-2">
                Measuring What
              </h1>
              <MorphingText 
                texts={["Matters", "Works", "Counts", "Delivers"]} 
                className="h-12 md:h-20 lg:h-24 text-white"
              />
            </motion.div>
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="text-lg md:text-xl text-gray-300 max-w-2xl mx-auto"
            >
              We optimize for business realism, not just R² scores. 
              A model that predicts a 50% price hike will double sales is useless.
            </motion.p>
          </div>
        </div>
      </section>

      {/* Key Metrics Grid */}
      <section className="py-20 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-4 flex items-center justify-center gap-3">
              <span>Key</span>
              <SparklesText 
                text="Metrics" 
                className="text-3xl md:text-5xl"
                colors={{ first: "#6b7280", second: "#9ca3af" }}
                sparklesCount={8}
              />
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Quantified performance across accuracy, compliance, and generalization.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {metrics.map((metric, index) => (
              <motion.div
                key={metric.label}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <GlowCard 
                  glowColor={metric.glowColor}
                  customSize
                  width={280}
                  height={220}
                  className="flex flex-col h-full"
                >
                  <div className="flex items-center gap-2 mb-4">
                    <metric.icon className="w-5 h-5 text-gray-400" />
                    <span className="text-xs uppercase tracking-wider text-gray-500">
                      {metric.label}
                    </span>
                  </div>
                  <div className="text-4xl font-bold text-white mb-1">{metric.value}</div>
                  <div className="text-sm text-gray-400 mb-3">{metric.description}</div>
                  <p className="text-xs text-gray-500 mt-auto">{metric.detail}</p>
                </GlowCard>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Performance Breakdown */}
      <section className="py-20 px-6 bg-neutral-950">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-4 flex items-center justify-center gap-3">
              <FlipText 
                word="Technical" 
                className="text-3xl md:text-5xl font-bold text-white"
              />
              <FlipText 
                word="Breakdown" 
                className="text-3xl md:text-5xl font-bold text-white"
                delayMultiple={0.1}
              />
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Why each design decision was made for business realism.
            </p>
          </div>

          <div className="space-y-4">
            {performanceBreakdown.map((item, index) => (
              <motion.div
                key={item.metric}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="bg-neutral-900 border border-neutral-800 rounded-xl p-6"
              >
                <div className="flex flex-col md:flex-row md:items-center gap-4">
                  <div className="md:w-1/4">
                    <span className="text-sm text-gray-500">{item.metric}</span>
                    <div className="text-xl font-bold text-white">{item.value}</div>
                  </div>
                  <div className="md:w-3/4 md:border-l md:border-neutral-700 md:pl-6">
                    <p className="text-gray-400">{item.reason}</p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Comparison Section */}
      <section className="py-20 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-4">
              Metrics That <span className="text-gray-500">Matter</span>
            </h2>
            <p className="text-gray-400">
              How Praxis compares to standard approaches.
            </p>
          </div>

          <div className="space-y-3">
            {comparisonMetrics.map((item, index) => (
              <motion.div
                key={item.metric}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="grid grid-cols-1 md:grid-cols-3 gap-4 bg-neutral-900/50 border border-neutral-800 rounded-xl p-4"
              >
                <div className="flex items-center">
                  <span className="text-white font-medium">{item.metric}</span>
                </div>
                <div className="flex items-center gap-2">
                  <XCircle className="w-4 h-4 text-red-500/60 shrink-0" />
                  <span className="text-gray-500 text-sm">{item.standard}</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-500/60 shrink-0" />
                  <span className="text-gray-300 text-sm">{item.praxis}</span>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Summary */}
      <section className="py-20 px-6 bg-neutral-950">
        <div className="max-w-4xl mx-auto text-center">
          <blockquote className="text-xl md:text-2xl text-gray-300 leading-relaxed">
            <span className="text-white font-semibold">
              "Our solution uses Monotonic XGBoost to bridge the gap between Data Science and Microeconomics.
            </span>{" "}
            We engineered features that capture 'Market Context' rather than raw numbers, 
            and we wrapped the engine in an interactive simulator that turns abstract predictions 
            into actionable revenue strategy."
          </blockquote>
        </div>
      </section>
    </main>
  );
}
