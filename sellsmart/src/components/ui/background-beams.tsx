"use client";
import React from "react";
import { cn } from "@/lib/utils";

export const BackgroundBeams = ({ className }: { className?: string }) => {
    return (
        <div
            className={cn(
                "absolute inset-0 z-0 h-full w-full overflow-hidden [mask-image:radial-gradient(ellipse_at_center,transparent_20%,black)]",
                className
            )}
        >
            <div className="absolute inset-0 bg-gradient-to-t from-slate-950 to-transparent z-10 pointer-events-none" />
            <svg
                className="absolute left-[-10%] top-[-10%] h-[120%] w-[120%] animate-pulse opacity-[0.05]"
                xmlns="http://www.w3.org/2000/svg"
            >
                <defs>
                    <pattern id="beams" width="40" height="40" patternUnits="userSpaceOnUse">
                        <path d="M0 40V.5H40" fill="none" stroke="white" strokeWidth="0.5" />
                    </pattern>
                </defs>
                <rect width="100%" height="100%" fill="url(#beams)" />
            </svg>
            {/* Additional Beams for effect */}
            <div className="absolute top-0 left-0 w-full h-full bg-[radial-gradient(circle_500px_at_50%_200px,#3b82f6,transparent)] opacity-20" />
        </div>
    );
};
