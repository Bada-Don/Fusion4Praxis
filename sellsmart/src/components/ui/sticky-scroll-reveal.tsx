"use client";
import React, { useRef, useEffect } from "react";
import { useScroll, useTransform, motion } from "framer-motion";
import { cn } from "@/lib/utils";

export const StickyScroll = ({
    content,
    contentClassName,
}: {
    content: {
        title: string;
        description: string;
        content?: React.ReactNode;
    }[];
    contentClassName?: string;
}) => {
    const [activeCard, setActiveCard] = React.useState(0);
    const ref = useRef<HTMLDivElement>(null);
    const cardRefs = useRef<(HTMLDivElement | null)[]>([]);
    
    const { scrollYProgress } = useScroll({
        container: ref,
        offset: ["start start", "end start"],
    });
    const cardLength = content.length;

    useTransform(
        scrollYProgress,
        [0, 1],
        [0, cardLength * 100]
    );

    // Track which card is in view
    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        const index = cardRefs.current.findIndex(
                            (ref) => ref === entry.target
                        );
                        if (index !== -1) {
                            setActiveCard(index);
                        }
                    }
                });
            },
            {
                root: ref.current,
                threshold: 0.5,
                rootMargin: "-20% 0px -20% 0px",
            }
        );

        cardRefs.current.forEach((cardRef) => {
            if (cardRef) observer.observe(cardRef);
        });

        return () => observer.disconnect();
    }, [content]);

    return (
        <motion.div
            className="h-[30rem] overflow-y-auto flex justify-center relative space-x-10 rounded-md p-10"
            ref={ref}
            style={{
                scrollbarWidth: 'none',
                msOverflowStyle: 'none',
                position: 'relative',
            }}
        >
            {/* Hide scrollbar for webkit browsers */}
            <style jsx>{`
                :global(.sticky-scroll-container::-webkit-scrollbar) {
                    display: none;
                }
            `}</style>
            
            <div className="div relative flex items-start px-4">
                <div className="max-w-2xl">
                    {content.map((item, index) => (
                        <div
                            key={item.title + index}
                            ref={(el) => { cardRefs.current[index] = el; }}
                            className="my-20"
                        >
                            <motion.h2
                                initial={{ opacity: 0.3 }}
                                animate={{
                                    opacity: activeCard === index ? 1 : 0.3,
                                    color: activeCard === index ? "#ffffff" : "#94a3b8",
                                }}
                                transition={{ duration: 0.3 }}
                                className="text-2xl font-bold"
                            >
                                {item.title}
                            </motion.h2>
                            <motion.p
                                initial={{ opacity: 0.3 }}
                                animate={{
                                    opacity: activeCard === index ? 1 : 0.3,
                                    color: activeCard === index ? "#e2e8f0" : "#64748b",
                                }}
                                transition={{ duration: 0.3 }}
                                className="text-lg max-w-sm mt-10"
                            >
                                {item.description}
                            </motion.p>
                        </div>
                    ))}
                    <div className="h-40" />
                </div>
            </div>
            <motion.div
                className={cn(
                    "hidden lg:block h-60 w-80 rounded-md bg-white sticky top-10 overflow-hidden",
                    contentClassName
                )}
            >
                {content[activeCard].content ?? null}
            </motion.div>
        </motion.div>
    );
};

