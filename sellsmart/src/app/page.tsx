"use client";

import NavbarDemo from "@/components/resizable-navbar-demo";
import { Hero } from "@/components/ui/hero-1";
import { Highlighter } from "@/components/ui/highlighter";
import { TextReveal } from "@/components/ui/text-reveal";
import { BackgroundRippleEffect } from "@/components/ui/background-ripple-effect";
import { ScrollVelocityContainer, ScrollVelocityRow } from "@/components/ui/scroll-based-velocity";
import { SkewCards } from "@/components/ui/gradient-card-showcase";
import { StickyScroll } from "@/components/ui/sticky-scroll-reveal";

const featureCards = [
  {
    title: 'Monotonic Constraints',
    desc: 'Price Up = Demand Down. We force the AI to respect basic economic laws, eliminating illogical predictions.',
    gradientFrom: '#374151',
    gradientTo: '#1f2937',
  },
  {
    title: 'SEO Visibility',
    desc: 'Quantify how description length dictates conversion rates. Optimize product listings for maximum impact.',
    gradientFrom: '#475569',
    gradientTo: '#334155',
  },
  {
    title: 'Sentiment Analysis',
    desc: 'Real-time brand health monitoring integrated into pricing logic. Adapt to market perception instantly.',
    gradientFrom: '#52525b',
    gradientTo: '#3f3f46',
  },
];

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
          Praxis bridges the gap between data science and retail strategy.
          Our constrained XGBoost engine prevents "economic hallucinations"
          by enforcing logical pricing rules, ensuring every recommendation
          is profitable and realistic.
        </TextReveal>
      </div>

      {/* Skew Cards Feature Layer */}
      <div className="relative z-10 mt-10">
        <SkewCards cards={featureCards} />
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
        <div className="h-full w-full relative overflow-hidden rounded-md">
          <img 
            src="https://media.istockphoto.com/id/1158762452/vector/team-business-goals-active-employees-social-media-marketing.jpg?s=612x612&w=0&k=20&c=snF7iP0bsaAxnarO73EX1-Cjdb8FwUiSocNW_YWpwY8=" 
            alt="RMSE Accuracy"
            className="w-full h-full object-cover"
          />
        </div>
      ),
    },
    {
      title: "0 Logic Violations",
      description:
        "Thanks to XGBoost's monotonic constraints, we have eliminated 'economic hallucinations'. The model never predicts that raising prices will magically increase demand for standard goods.",
      content: (
        <div className="h-full w-full relative overflow-hidden rounded-md">
          <img 
            src="https://img.freepik.com/premium-vector/financial-business-education-economics-study-finance-literacy-concept-books-economy-investment-knowledge-money-composition-flat-vector-illustration-isolated-white-background_198278-23758.jpg?semt=ais_hybrid&w=740&q=80" 
            alt="Zero Logic Violations"
            className="w-full h-full object-cover"
          />
        </div>
      ),
    },
    {
      title: "Unbiased Estimator",
      description:
        "Residual analysis confirms that our error distribution is centered at zero, meaning the model is not systematically overestimating or underestimating demand across different categories.",
      content: (
        <div className="h-full w-full relative overflow-hidden rounded-md">
          <img 
            src="https://pixelplex.io/wp-content/uploads/2023/11/ai-bias-examples-main.jpg" 
            alt="Unbiased Estimator"
            className="w-full h-full object-cover"
          />
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

export default function Home() {
  return (
    <main className="relative min-h-screen bg-black">
      <NavbarDemo />
      <Hero
        title={
          <>
            Retail Pricing,<br />{" "}
            <Highlighter action="highlight" color="#404040" isView>
              Anchored
            </Highlighter>{" "}
            in Reality
          </>
        }
        subtitle="The first AI pricing engine with enforced Economic Guardrails. No hallucinations, just profit."
        eyebrow="AI-Powered Optimization"
        ctaLabel="Launch Simulator"
        ctaHref="/simulator"
      />
      <AboutSection />
      <ValidationMetrics />
    </main>
  );
}
