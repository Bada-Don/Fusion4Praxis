"use client";

import {
    Navbar,
    NavBody,
    NavItems,
    MobileNav,
    NavbarLogo,
    NavbarButton,
    MobileNavHeader,
    MobileNavToggle,
    MobileNavMenu,
} from "@/components/ui/resizable-navbar";
import { useState } from "react";

export default function NavbarDemo() {
    const navItems = [
        {
            name: "Features",
            link: "#features",
        },
        {
            name: "Simulator",
            link: "#simulator",
        },
        {
            name: "Metrics",
            link: "#metrics",
        },
    ];

    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

    return (
        <Navbar>
            {/* Desktop Navigation */}
            <NavBody>
                <NavbarLogo />
                <NavItems items={navItems} />
                <div className="flex items-center gap-4">
                    <NavbarButton
                        variant="secondary"
                        onClick={() => document.getElementById('metrics')?.scrollIntoView({ behavior: 'smooth' })}
                    >
                        Metrics
                    </NavbarButton>
                    <NavbarButton
                        variant="primary"
                        onClick={() => document.getElementById('simulator')?.scrollIntoView({ behavior: 'smooth' })}
                    >
                        Try Simulator
                    </NavbarButton>
                </div>
            </NavBody>

            {/* Mobile Navigation */}
            <MobileNav>
                <MobileNavHeader>
                    <NavbarLogo />
                    <MobileNavToggle
                        isOpen={isMobileMenuOpen}
                        onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                    />
                </MobileNavHeader>

                <MobileNavMenu
                    isOpen={isMobileMenuOpen}
                    onClose={() => setIsMobileMenuOpen(false)}
                >
                    {navItems.map((item, idx) => (
                        <a
                            key={`mobile-link-${idx}`}
                            href={item.link}
                            onClick={() => setIsMobileMenuOpen(false)}
                            className="relative text-neutral-300"
                        >
                            <span className="block">{item.name}</span>
                        </a>
                    ))}
                    <div className="flex w-full flex-col gap-4">
                        <NavbarButton
                            variant="primary"
                            onClick={() => {
                                setIsMobileMenuOpen(false);
                                document.getElementById('simulator')?.scrollIntoView({ behavior: 'smooth' });
                            }}
                        >
                            Try Simulator
                        </NavbarButton>
                    </div>
                </MobileNavMenu>
            </MobileNav>
        </Navbar>
    );
}
