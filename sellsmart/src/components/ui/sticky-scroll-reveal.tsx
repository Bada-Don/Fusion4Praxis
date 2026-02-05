"use client";
import React, { useRef } from "react";
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
    const ref = useRef<any>(null);
    const { scrollYProgress } = useScroll({
        // target: ref,
        container: ref,
        offset: ["start start", "end start"],
    });
    const cardLength = content.length;

    useTransform(
        scrollYProgress,
        [0, 1],
        [0, cardLength * 100] // not mapped directly, just for triggers
    );

    return (
        <motion.div
            className="h-[30rem] overflow-y-auto flex justify-center relative space-x-10 rounded-md p-10 scrollbar-hide"
            ref={ref}
        >
            <div className="div relative flex items-start px-4">
                <div className="max-w-2xl">
                    {content.map((item, index) => (
                        <div key={item.title + index} className="my-20">
                            <motion.h2
                                initial={{
                                    opacity: 0,
                                }}
                                animate={{
                                    opacity: activeCard === index ? 1 : 0.3,
                                }}
                                className="text-2xl font-bold text-slate-100"
                            >
                                {item.title}
                            </motion.h2>
                            <motion.p
                                initial={{
                                    opacity: 0,
                                }}
                                animate={{
                                    opacity: activeCard === index ? 1 : 0.3,
                                }}
                                className="text-kg text-slate-300 max-w-sm mt-10"
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
