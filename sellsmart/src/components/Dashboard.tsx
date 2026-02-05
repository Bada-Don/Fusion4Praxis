"use client";

import React, { useState, useEffect, useMemo } from "react";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
    Label
} from "recharts";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

interface PredictResponse {
    predicted_demand: number;
    projected_revenue: number;
    price_elasticity: number;
    violations: number;
}

export const Simulator = () => {
    const [price, setPrice] = useState(50);
    const [descLen, setDescLen] = useState(100);
    const [sentiment, setSentiment] = useState(0.5);
    const [loading, setLoading] = useState(false);
    const [metrics, setMetrics] = useState<PredictResponse | null>(null);
    const [chartData, setChartData] = useState<any[]>([]);

    // Derived mock curve for the chart (so it looks instant)
    // In a real app, we might fetch the whole curve or enough points to plot it.
    // Here we'll generate points around the current price to show sensitivity.
    const generateCurve = async () => {
        // Generate valid price range for the chart (e.g., +/- 50%)
        const points = [];
        const minP = Math.max(10, price * 0.5);
        const maxP = price * 1.5;
        const step = (maxP - minP) / 10;

        for (let p = minP; p <= maxP; p += step) {
            // We'll use the same API logic to get these points for consistency, 
            // or just a local approximation if we want to save requests.
            // Let's call the API for the current point, and approximate the rest or 
            // just request 5 points.
            points.push({ price: Math.round(p), demand: 0 }); // Placeholder
        }
        return points;
    };

    // Real fetch for current state
    const fetchPrediction = async () => {
        setLoading(true);
        try {
            const res = await fetch("http://127.0.0.1:8000/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    product_id: "test-item",
                    price: price,
                    desc_len: descLen,
                    sentiment: sentiment,
                    category: "electronics"
                })
            });
            const data = await res.json();
            setMetrics(data);

            // Update chart data based on this new center point
            // We simulate the curve locally for UI responsiveness or valid monotonicity demo
            const newChartData = [];
            for (let i = 0; i <= 10; i++) {
                const p = price * (0.5 + i * 0.1);
                // Simple Monotonic heuristic matching the backend logic roughly:
                // Demand = Base + (-5 * Price) + ...
                // We can just use the backend response to calibrate the base.
                // Demand_current = Base + (-5 * current_price) + others
                // Base + others = Demand_current + 5 * current_price
                const constantFactors = data.predicted_demand + 5.0 * price;

                let d = constantFactors - 5.0 * p;
                if (d < 0) d = 0;
                newChartData.push({ price: Math.round(p), demand: Math.round(d) });
            }
            setChartData(newChartData);

        } catch (e) {
            console.error("API Error", e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        const timeout = setTimeout(() => {
            fetchPrediction();
        }, 300); // Debounce
        return () => clearTimeout(timeout);
    }, [price, descLen, sentiment]);

    return (
        <div className="w-full max-w-7xl mx-auto p-4 bg-neutral-900 rounded-3xl border border-neutral-800 shadow-2xl overflow-hidden mt-10">
            <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
                {/* Controls */}
                <div className="md:col-span-4 p-6 flex flex-col space-y-8 bg-black/50 rounded-2xl border border-white/10">
                    <h2 className="text-2xl font-bold text-white mb-4">Control Center</h2>

                    {/* Price Slider */}
                    <div className="space-y-4">
                        <div className="flex justify-between">
                            <label className="text-neutral-400">Price ($)</label>
                            <span className="text-cyan-400 font-mono text-xl font-bold">${price}</span>
                        </div>
                        <input
                            type="range"
                            min="10"
                            max="200"
                            step="1"
                            value={price}
                            onChange={(e) => setPrice(Number(e.target.value))}
                            className="w-full h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                        />
                    </div>

                    {/* Visibility Slider */}
                    <div className="space-y-4">
                        <div className="flex justify-between">
                            <label className="text-neutral-400">Description / SEO (Chars)</label>
                            <span className="text-purple-400 font-mono text-xl font-bold">{descLen} chars</span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="1000"
                            value={descLen}
                            onChange={(e) => setDescLen(Number(e.target.value))}
                            className="w-full h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                        />
                    </div>

                    {/* Sentiment Slider */}
                    <div className="space-y-4">
                        <div className="flex justify-between">
                            <label className="text-neutral-400">Sentiment Score</label>
                            <span className="text-green-400 font-mono text-xl font-bold">{(sentiment * 100).toFixed(0)}%</span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={sentiment}
                            onChange={(e) => setSentiment(Number(e.target.value))}
                            className="w-full h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-green-500"
                        />
                    </div>

                    <div className="pt-4 border-t border-white/10">
                        <div className="flex items-center space-x-2">
                            <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse" />
                            <span className="text-xs text-green-400">Model Active • 0ms Latency</span>
                        </div>
                    </div>
                </div>

                {/* Visuals */}
                <div className="md:col-span-8 p-6 flex flex-col w-full">
                    {/* Top Metrics */}
                    <div className="grid grid-cols-3 gap-4 mb-8">
                        <MetricCard
                            label="Predicted Demand"
                            value={metrics?.predicted_demand.toFixed(0) || "..."}
                            subtext="Units / Week"
                            color="text-white"
                            className="bg-indigo-500/10 border-indigo-500/20"
                        />
                        <MetricCard
                            label="Projected Revenue"
                            value={`$${metrics?.projected_revenue.toLocaleString() || "..."}`}
                            subtext="USD"
                            color="text-green-400"
                            className="bg-green-500/10 border-green-500/20"
                        />
                        <MetricCard
                            label="Elasticity Coeff"
                            value={metrics?.price_elasticity.toString() || "..."}
                            subtext="Price Sensitivity"
                            color="text-yellow-400"
                            className="bg-yellow-500/10 border-yellow-500/20"
                        />
                    </div>

                    {/* Chart */}
                    <div className="flex-1 min-h-[400px] w-full bg-black/40 rounded-2xl border border-white/5 p-4 relative">
                        <h3 className="text-neutral-400 mb-4 text-sm uppercase tracking-wider">Price Sensitivity Curve</h3>
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                <XAxis
                                    dataKey="price"
                                    stroke="#666"
                                    label={{ value: 'Price ($)', position: 'insideBottomRight', offset: -5 }}
                                />
                                <YAxis
                                    stroke="#666"
                                    label={{ value: 'Demand', angle: -90, position: 'insideLeft' }}
                                />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#000', borderColor: '#333' }}
                                    itemStyle={{ color: '#fff' }}
                                />
                                <Line
                                    type="monotone"
                                    dataKey="demand"
                                    stroke="#06b6d4"
                                    strokeWidth={3}
                                    dot={false}
                                    animationDuration={500}
                                />
                                <ReferenceLine x={price} stroke="white" strokeDasharray="3 3" />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>
        </div>
    );
};

const MetricCard = ({ label, value, subtext, color, className }: any) => (
    <div className={cn("p-4 rounded-xl border flex flex-col items-center justify-center text-center", className)}>
        <span className="text-neutral-400 text-xs uppercase tracking-widest mb-1">{label}</span>
        <span className={cn("text-3xl font-black font-mono my-1", color)}>{value}</span>
        <span className="text-neutral-500 text-xs">{subtext}</span>
    </div>
);
